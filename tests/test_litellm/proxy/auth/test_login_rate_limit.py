"""Unit tests for litellm/proxy/auth/login_rate_limit.py.

Route-level coverage of the login endpoints lives in
tests/test_litellm/proxy/proxy_server/test_routes_login_sso.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _login_request(client_ip: str):
    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = client_ip
    request.url = MagicMock()
    request.url.path = "/v2/login"
    return request


class _StubRedis:
    """Minimal stand-in for the coordination Redis, injected into the limiter."""

    def __init__(self, values: dict[str, int] | None = None):
        self.values = dict(values or {})

    async def async_get_cache(self, key, **kwargs):
        return self.values.get(key)

    async def async_increment(self, key, value, **kwargs):
        self.values[key] = self.values.get(key, 0) + int(value)
        return self.values[key]

    async def async_delete_cache(self, key):
        self.values.pop(key, None)


class _BrokenCache:
    """A cache backend that is down. Every operation raises, like an open circuit breaker."""

    def __init__(self):
        self.calls = 0

    async def async_get_cache(self, key, **kwargs):
        self.calls += 1
        raise Exception("Redis circuit breaker is open - skipping async_get_cache")

    async def async_increment(self, key, value, **kwargs):
        self.calls += 1
        raise Exception("Redis circuit breaker is open - skipping async_increment")

    async def async_delete_cache(self, key):
        self.calls += 1
        raise Exception("Redis circuit breaker is open - skipping async_delete_cache")


def _limiter(cache=None, redis_cache=None, **config):
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.auth.login_rate_limit import LoginRateLimitConfig, LoginRateLimiter
    from litellm.proxy.auth.network import TrustedProxyConfig

    settings = {
        "max_failures": 2,
        "window_seconds": 60,
        "trusted_proxy": TrustedProxyConfig(use_forwarded_for=False, trusted_proxy_cidrs=[]),
    }
    settings.update(config)
    return LoginRateLimiter(
        cache=cache if cache is not None else DualCache(),
        config=LoginRateLimitConfig(**settings),
        redis_cache=redis_cache,
    )


@pytest.mark.asyncio
async def test_the_limit_is_shared_across_replicas():
    """The counter lives in the shared cache, not in the process."""
    from litellm.proxy._types import ProxyException

    stub_redis = _StubRedis()
    replica_a = _limiter(redis_cache=stub_redis, max_failures=2)
    replica_b = _limiter(redis_cache=stub_redis, max_failures=2)
    request = _login_request("10.0.0.8")

    await replica_a.register_attempt(request=request)
    await replica_a.register_attempt(request=request)

    with pytest.raises(ProxyException) as blocked:
        await replica_b.register_attempt(request=request)
    assert blocked.value.code == "429"


class _SilentlyFailingRedis(_StubRedis):
    """Production RedisCache swallows connection errors on reads and returns None.

    That is indistinguishable from an absent key, which is exactly what makes the read
    path dangerous to trust outright.
    """

    async def async_get_cache(self, key, **kwargs):
        return None


@pytest.mark.asyncio
async def test_a_swallowed_redis_read_falls_back_to_the_local_count():
    """Regression: a Redis read that quietly returns None must not read as zero.

    RedisCache.async_get_cache catches connection errors and returns None rather than
    raising, so a read-only check that trusted the shared tier reported an unthrottled
    source while this replica's own counter had already passed the limit.
    """
    from litellm.proxy._types import ProxyException

    limiter = _limiter(redis_cache=_SilentlyFailingRedis(), max_failures=2)
    request = _login_request("10.0.0.15")

    await limiter.register_attempt(request=request)
    await limiter.register_attempt(request=request)
    with pytest.raises(ProxyException):
        await limiter.register_attempt(request=request)

    assert await limiter.is_rate_limited(request=request) is True


@pytest.mark.asyncio
async def test_a_redis_outage_degrades_to_per_process_throttling_not_to_none():
    """A dead Redis must not switch the limit off.

    Every attempt is counted locally as well as in Redis, and the local value is what the
    limiter falls back to, so a replica keeps enforcing its own budget while Redis is down.
    """
    from litellm.proxy._types import ProxyException

    limiter = _limiter(redis_cache=_BrokenCache(), max_failures=2)
    request = _login_request("10.0.0.11")

    await limiter.register_attempt(request=request)
    await limiter.register_attempt(request=request)

    with pytest.raises(ProxyException) as blocked:
        await limiter.register_attempt(request=request)
    assert blocked.value.code == "429"


@pytest.mark.asyncio
async def test_a_broken_backend_never_blocks_or_raises():
    """Every counter operation fails open when its backend is unreachable."""
    limiter = _limiter(redis_cache=_BrokenCache(), max_failures=1)
    request = _login_request("10.0.0.9")

    await limiter.register_attempt(request=request)
    await limiter.refund(request=request, shared_charged=True)
    assert await limiter.is_rate_limited(request=request) is False


@pytest.mark.asyncio
async def test_a_different_source_is_unaffected():
    """The throttle is per source, so one attacker cannot deny anyone else the dashboard."""
    from litellm.proxy._types import ProxyException

    limiter = _limiter(max_failures=2)
    attacker = _login_request("203.0.113.5")

    await limiter.register_attempt(request=attacker)
    await limiter.register_attempt(request=attacker)
    with pytest.raises(ProxyException):
        await limiter.register_attempt(request=attacker)

    await limiter.register_attempt(request=_login_request("198.51.100.7"))
    assert await limiter.is_rate_limited(request=_login_request("198.51.100.7")) is False


@pytest.mark.asyncio
async def test_x_forwarded_for_is_only_trusted_behind_a_configured_proxy():
    """An untrusted peer cannot pick its own counter key by setting the header."""
    from litellm.proxy.auth.network import TrustedProxyConfig

    limiter = _limiter(
        max_failures=2,
        trusted_proxy=TrustedProxyConfig(use_forwarded_for=True, trusted_proxy_cidrs=[]),
    )

    spoofed = _login_request("203.0.113.9")
    spoofed.headers = {"x-forwarded-for": "10.1.1.1"}
    spoofed_again = _login_request("203.0.113.9")
    spoofed_again.headers = {"x-forwarded-for": "10.2.2.2"}

    assert limiter._source_ip(spoofed) == "203.0.113.9"
    assert limiter._cache_key(limiter._source_ip(spoofed)) == limiter._cache_key(limiter._source_ip(spoofed_again)), (
        "rotating X-Forwarded-For must not mint a fresh counter"
    )


class _BrokenLocalCache:
    """The in-process tier is down too, so there is no working backend at all."""

    async def async_get_cache(self, key, **kwargs):
        raise Exception("in-memory cache exploded")

    async def async_increment_cache(self, key, value, **kwargs):
        raise Exception("in-memory cache exploded")

    async def async_delete_cache(self, key, **kwargs):
        raise Exception("in-memory cache exploded")


@pytest.mark.asyncio
async def test_a_broken_local_tier_also_fails_open():
    """With no Redis configured and the local tier raising, the limiter must still let a
    valid credential through rather than turning a cache bug into a 500 on every login."""
    limiter = _limiter(cache=_BrokenLocalCache(), max_failures=1)
    request = _login_request("10.0.0.14")

    await limiter.register_attempt(request=request)
    await limiter.register_attempt(request=request)
    await limiter.refund(request=request, shared_charged=True)
    assert await limiter.is_rate_limited(request=request) is False


@pytest.mark.asyncio
async def test_a_redis_that_lost_writes_does_not_lower_the_count():
    """After an outage the shared counter can be behind this replica's own.

    The higher of the two wins, so a Redis that came back having missed writes cannot
    hand a source back budget it already spent.
    """
    from litellm.proxy._types import ProxyException

    stub_redis = _StubRedis()
    limiter = _limiter(redis_cache=stub_redis, max_failures=2)
    request = _login_request("10.0.0.16")

    await limiter.register_attempt(request=request)
    await limiter.register_attempt(request=request)

    stub_redis.values.clear()
    stub_redis.values[limiter._cache_key(limiter._source_ip(request))] = 1

    with pytest.raises(ProxyException):
        await limiter.register_attempt(request=request)


class _RefusingRefundRedis(_StubRedis):
    """Redis that refuses the decrement outright, without applying it."""

    async def async_increment(self, key, value, **kwargs):
        if value < 0:
            raise Exception("redis refused the decrement")
        return await super().async_increment(key, value, **kwargs)


class _AmbiguousRefundRedis(_StubRedis):
    """Redis that applies the decrement and then raises, like a TTL call failing after INCRBYFLOAT."""

    def __init__(self):
        super().__init__()
        self.decrements = 0

    async def async_increment(self, key, value, **kwargs):
        if value < 0:
            self.decrements += 1
            self.values[key] = self.values.get(key, 0) + int(value)
            raise Exception("redis applied the write, then the TTL call failed")
        return await super().async_increment(key, value, **kwargs)


@pytest.mark.asyncio
async def test_an_ambiguous_refund_failure_is_never_retried():
    """Regression: retrying a refund can hand back two attempts and weaken the limit.

    RedisCache.async_increment applies INCRBYFLOAT before the TTL calls that can raise,
    so an exception does not mean the decrement was not applied. Repeating it decremented
    twice, letting a source spend more than its budget.
    """
    redis = _AmbiguousRefundRedis()
    limiter = _limiter(redis_cache=redis, max_failures=3)
    request = _login_request("10.0.0.19")
    key = limiter._cache_key(limiter._source_ip(request))

    await limiter.register_attempt(request=request)
    await limiter.refund(request=request, shared_charged=True)

    assert redis.decrements == 1, "an ambiguous shared refund must be issued exactly once"
    assert redis.values.get(key) == 0, "the counter must not be driven below what the attempt cost"


@pytest.mark.asyncio
async def test_the_tiers_do_not_diverge_when_a_refund_is_refused():
    """A refund the shared tier will not take is not applied locally either."""
    redis = _RefusingRefundRedis()
    limiter = _limiter(redis_cache=redis, max_failures=3)
    request = _login_request("10.0.0.18")
    key = limiter._cache_key(limiter._source_ip(request))

    await limiter.register_attempt(request=request)
    await limiter.refund(request=request, shared_charged=True)

    local = await limiter._outcome(limiter.cache.async_get_cache(key=key))
    assert redis.values.get(key) == 1
    assert local == 1, "the local tier must not drift below the shared one"


@pytest.mark.asyncio
async def test_successful_logins_do_not_accumulate_while_the_shared_tier_is_down():
    """Regression: a charge the shared tier never took must be given back locally.

    Refunding locally only when the shared tier accepted the give-back meant that during a
    Redis outage every successful sign-in kept its local charge, so enough good logins in
    one window started returning 429 to correct passwords.
    """
    from litellm.proxy.auth.login_rate_limit import enforce_login_rate_limit

    limiter = _limiter(redis_cache=_BrokenCache(), max_failures=3)
    request = _login_request("10.0.0.20")

    async def _succeeds():
        return "login-result"

    for _ in range(6):
        assert (
            await enforce_login_rate_limit(request=request, limiter=limiter, authenticate=_succeeds) == "login-result"
        )

    assert await limiter.is_rate_limited(request=request) is False
