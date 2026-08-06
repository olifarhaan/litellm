"""Rate limiting for the admin UI login endpoints.

Counts failed credential checks per source over a fixed window and rejects further
attempts from that source with 429 once the count passes the limit. The count is
applied before the password is verified. Counters live in the coordination Redis when
one is configured and in this process otherwise, and every counter operation fails
open.
"""

import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.network import TrustedProxyConfig, normalize_cidr_ranges, resolve_client_ip
from litellm.proxy.auth.trusted_proxy_utils import TRUSTED_PROXY_RANGES_KEY

if TYPE_CHECKING:
    from litellm.proxy.auth.login_utils import LoginResult

DEFAULT_LOGIN_RATE_LIMIT_MAX_FAILURES: Final = 15
DEFAULT_LOGIN_RATE_LIMIT_WINDOW_SECONDS: Final = 300

_CACHE_KEY_PREFIX: Final = "litellm_login_failures"
_UNAUTHORIZED_STATUS: Final = 401
_UNAUTHORIZED_CODE: Final = str(_UNAUTHORIZED_STATUS)
_NO_SETTINGS: Final = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class LoginRateLimitConfig:
    """Resolved settings. A ``max_failures`` of 0 disables the rate limit."""

    max_failures: int
    window_seconds: int
    trusted_proxy: TrustedProxyConfig


def _int_setting(name: str, value: object, default: int, minimum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        verbose_proxy_logger.warning(
            "general_settings.%s=%r is not an integer >= %s; falling back to the default of %s",
            name,
            value,
            minimum,
            default,
        )
        return default
    return value


def _as_count(cached: object) -> int:
    return int(cached) if isinstance(cached, int | float) and not isinstance(cached, bool) else 0


def login_rate_limit_config(general_settings: Mapping[str, object] | None) -> LoginRateLimitConfig:
    """Read the rate limit settings out of ``general_settings``, falling back to the defaults."""
    settings: Final = _NO_SETTINGS if general_settings is None else general_settings
    return LoginRateLimitConfig(
        max_failures=_int_setting(
            "login_rate_limit_max_failures",
            settings.get("login_rate_limit_max_failures"),
            DEFAULT_LOGIN_RATE_LIMIT_MAX_FAILURES,
            0,
        ),
        window_seconds=_int_setting(
            "login_rate_limit_window_seconds",
            settings.get("login_rate_limit_window_seconds"),
            DEFAULT_LOGIN_RATE_LIMIT_WINDOW_SECONDS,
            1,
        ),
        trusted_proxy=_trusted_proxy_config(settings),
    )


def _trusted_proxy_config(settings: Mapping[str, object]) -> TrustedProxyConfig:
    """Trust X-Forwarded-For exactly when ``trusted_proxy_ranges`` is configured.

    Matches the sibling call site in ``user_api_key_auth``; requiring a second flag as
    well would silently bucket every request behind an ingress under one address.
    """
    cidrs: Final = normalize_cidr_ranges(settings.get(TRUSTED_PROXY_RANGES_KEY), setting_name=TRUSTED_PROXY_RANGES_KEY)
    return TrustedProxyConfig(use_forwarded_for=bool(cidrs), trusted_proxy_cidrs=cidrs)


def login_rate_limited_html(retry_after_seconds: int) -> str:
    """Render the throttled response for the no-JS sign-in form."""
    return (
        "<html><head><title>Too many login attempts</title></head>"
        '<body style="font-family: system-ui, sans-serif; text-align: center; padding-top: 15vh">'
        "<h2>Too many failed login attempts</h2>"
        f"<p>Try again in about {retry_after_seconds} seconds.</p>"
        "</body></html>"
    )


@dataclass(frozen=True, slots=True)
class LoginRateLimiter:
    """Rate limits admin UI login attempts per source, over a fixed window."""

    cache: DualCache
    config: LoginRateLimitConfig
    redis_cache: RedisCache | None = None

    @property
    def enabled(self) -> bool:
        return self.config.max_failures > 0

    @property
    def retry_after_seconds(self) -> int:
        return self.config.window_seconds

    def _source_ip(self, request: Request) -> str:
        resolved, _ = resolve_client_ip(request, self.config.trusted_proxy)
        return resolved or "unknown"

    @staticmethod
    def _cache_key(source_ip: str) -> str:
        return f"{_CACHE_KEY_PREFIX}:ip:{hashlib.sha256(source_ip.encode('utf-8')).hexdigest()}"

    def _tier_count(self, operation: str, tier: str, value: object) -> int:
        """One tier's count, or 0 when that tier raised."""
        if isinstance(value, BaseException):
            verbose_proxy_logger.warning(
                "litellm_login_throttle_unavailable op=%s tier=%s error=%s", operation, tier, value
            )
            return 0
        return _as_count(value)

    def _resolve(self, operation: str, shared: object, local: object) -> int:
        """The highest count either tier reports.

        ``RedisCache.async_get_cache`` returns None for both an absent key and a
        swallowed connection error, so a shared answer is never trusted to be lower than
        this replica's own count.
        """
        return max(
            self._tier_count(operation, "shared", shared),
            self._tier_count(operation, "local", local),
        )

    @staticmethod
    async def _outcome(work: Awaitable[object]) -> object:
        """The value the backend returned, or the exception it raised, as a value."""
        try:
            return await work
        except Exception as exc:  # noqa: BLE001  # an unreachable backend must never deny a valid credential
            return exc

    async def _add(self, key: str, delta: int) -> tuple[bool, int]:
        """Apply ``delta`` to both tiers, reporting whether the shared tier took it."""
        redis_cache: Final = self.redis_cache
        shared: Final[object] = (
            None
            if redis_cache is None
            else await self._outcome(redis_cache.async_increment(key, delta, ttl=self.config.window_seconds))
        )
        local: Final[object] = await self._outcome(
            self.cache.async_increment_cache(key=key, value=delta, ttl=self.config.window_seconds)
        )
        shared_accepted: Final = redis_cache is None or not isinstance(shared, BaseException)
        return shared_accepted, self._resolve("increment", shared, local)

    async def _read(self, key: str) -> int:
        redis_cache: Final = self.redis_cache
        shared: Final[object] = (
            None if redis_cache is None else await self._outcome(redis_cache.async_get_cache(key=key))
        )
        local: Final[object] = await self._outcome(self.cache.async_get_cache(key=key))
        return self._resolve("read", shared, local)

    def _log_rate_limited(self, request: Request, source_ip: str, attempts: int) -> None:
        """Emit one line on the attempt that crosses the limit, not on every rejection."""
        if attempts != self.config.max_failures + 1:
            return
        verbose_proxy_logger.warning(
            "litellm_login_rate_limited source=%s attempts=%s limit=%s window_seconds=%s path=%s",
            source_ip,
            attempts,
            self.config.max_failures,
            self.config.window_seconds,
            request.url.path,
        )

    def too_many_attempts(self) -> ProxyException:
        retry_after: Final = {"retry-after": str(self.retry_after_seconds)}  # mutable-ok: ProxyException coerces values
        return ProxyException(
            message="Too many failed login attempts. Try again later.",
            type=ProxyErrorTypes.auth_error,
            param="login_rate_limit",
            code=429,
            headers=retry_after,
        )

    async def register_attempt(self, request: Request) -> bool:
        """Count this attempt and reject it when the source is over its limit.

        Returns whether the shared tier took the charge, which decides how it is given back.
        """
        if not self.enabled:
            return True
        source_ip: Final = self._source_ip(request)
        shared_charged, attempts = await self._add(self._cache_key(source_ip), 1)
        if attempts > self.config.max_failures:
            self._log_rate_limited(request, source_ip, attempts)
            raise self.too_many_attempts()
        return shared_charged

    async def _refund_shared(self, key: str) -> bool:
        """Give the shared tier its attempt back, once.

        Never retried. ``RedisCache.async_increment`` applies the INCRBYFLOAT before the
        TTL calls that can raise, so a failure does not mean the decrement was not
        applied, and repeating it would hand back two attempts and weaken the limit.
        """
        redis_cache: Final = self.redis_cache
        if redis_cache is None:
            return True
        outcome: Final = await self._outcome(redis_cache.async_increment(key, -1, ttl=self.config.window_seconds))
        if isinstance(outcome, BaseException):
            verbose_proxy_logger.warning(
                "litellm_login_throttle_refund_lost tier=shared error=%s window_seconds=%s",
                outcome,
                self.config.window_seconds,
            )
            return False
        return True

    async def refund(self, request: Request, shared_charged: bool) -> None:
        """Give back an attempt that was not a failed credential guess.

        A charge the shared tier never took is given back locally without touching the
        shared tier, so a Redis outage cannot make successful sign-ins accumulate. A
        charge it did take is given back locally only if it accepts the give-back, so the
        two tiers cannot disagree about what a sign-in cost.
        """
        if not self.enabled:
            return
        key: Final = self._cache_key(self._source_ip(request))
        if shared_charged and not await self._refund_shared(key):
            return
        await self._outcome(self.cache.async_increment_cache(key=key, value=-1, ttl=self.config.window_seconds))

    async def is_rate_limited(self, request: Request) -> bool:
        """Whether the source is over its limit, read without counting the caller."""
        if not self.enabled:
            return False
        return await self._read(self._cache_key(self._source_ip(request))) > self.config.max_failures


async def enforce_login_rate_limit(
    request: Request,
    limiter: LoginRateLimiter,
    authenticate: Callable[[], Awaitable["LoginResult"]],
) -> "LoginResult":
    """Run ``authenticate`` behind the rate limit.

    Counts the attempt up front, then refunds it unless it was a failed credential guess,
    so a successful sign-in and a misconfiguration 500 both cost nothing.
    """
    shared_charged: Final = await limiter.register_attempt(request=request)
    try:
        result: Final = await authenticate()
    except ProxyException as exc:
        if exc.code != _UNAUTHORIZED_CODE:
            await limiter.refund(request=request, shared_charged=shared_charged)
        raise
    except HTTPException as exc:
        if exc.status_code != _UNAUTHORIZED_STATUS:
            await limiter.refund(request=request, shared_charged=shared_charged)
        raise
    except Exception:
        await limiter.refund(request=request, shared_charged=shared_charged)
        raise
    await limiter.refund(request=request, shared_charged=shared_charged)
    return result
