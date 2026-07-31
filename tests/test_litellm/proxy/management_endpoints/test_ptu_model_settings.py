"""Tests for PTU config on the model deployment (v1 model-settings design)."""

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.model_management_endpoints import _validate_ptu_model_info
from litellm.types.router import ModelInfo


def test_model_info_accepts_valid_ptu_fields():
    info = ModelInfo(id="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=2.0)
    assert info.ptu_count == 5
    assert info.cost_per_ptu_per_hour == 2.0


def test_model_info_rejects_non_positive_count():
    with pytest.raises(ValueError):
        ModelInfo(id="x", team_id="t", ptu_count=0, cost_per_ptu_per_hour=2.0)


def test_model_info_rejects_negative_rate():
    with pytest.raises(ValueError):
        ModelInfo(id="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=-1.0)


def test_model_info_allows_partial_delta_for_patch():
    # A PATCH delta may carry only one field; bounds-only validation must not reject it.
    info = ModelInfo(id="x", ptu_count=5)
    assert info.ptu_count == 5
    assert info.cost_per_ptu_per_hour is None


def test_validate_helper_no_ptu_is_noop():
    _validate_ptu_model_info({"team_id": "t"})


def test_validate_helper_requires_both_fields():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info({"team_id": "t", "ptu_count": 5})
    assert exc.value.status_code == 400
    assert "set together" in exc.value.detail


def test_validate_helper_requires_team_id():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info({"ptu_count": 5, "cost_per_ptu_per_hour": 2.0})
    assert exc.value.status_code == 400
    assert "team_id" in exc.value.detail


def test_validate_helper_passes_full_config():
    _validate_ptu_model_info({"team_id": "t", "ptu_count": 5, "cost_per_ptu_per_hour": 2.0})


def test_model_info_rejects_effective_to_before_from():
    import datetime

    with pytest.raises(ValueError):
        ModelInfo(
            id="x",
            team_id="t",
            ptu_count=5,
            cost_per_ptu_per_hour=2.0,
            ptu_effective_from=datetime.datetime(2026, 7, 30, tzinfo=datetime.timezone.utc),
            ptu_effective_to=datetime.datetime(2026, 7, 29, tzinfo=datetime.timezone.utc),
        )


def test_model_info_accepts_valid_effective_window():
    import datetime

    info = ModelInfo(
        id="x",
        team_id="t",
        ptu_count=5,
        cost_per_ptu_per_hour=2.0,
        ptu_effective_from=datetime.datetime(2026, 7, 30, tzinfo=datetime.timezone.utc),
        ptu_effective_to=datetime.datetime(2026, 8, 30, tzinfo=datetime.timezone.utc),
    )
    assert info.ptu_effective_from is not None


def test_model_info_compares_mixed_naive_and_aware_timestamps():
    import datetime

    info = ModelInfo(
        id="x",
        team_id="t",
        ptu_count=5,
        cost_per_ptu_per_hour=2.0,
        ptu_effective_from=datetime.datetime(2026, 7, 30, 23, 0),
        ptu_effective_to=datetime.datetime(2026, 7, 31, 0, 0, tzinfo=datetime.timezone.utc),
    )
    assert info.ptu_effective_to is not None

    with pytest.raises(ValueError):
        ModelInfo(
            id="x",
            team_id="t",
            ptu_count=5,
            cost_per_ptu_per_hour=2.0,
            ptu_effective_from=datetime.datetime(2026, 7, 31, 2, 0),
            ptu_effective_to=datetime.datetime(2026, 7, 31, 0, 0, tzinfo=datetime.timezone.utc),
        )


def test_validate_helper_rejects_effective_to_before_from():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info(
            {
                "team_id": "t",
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "ptu_effective_from": "2026-07-30T00:00:00Z",
                "ptu_effective_to": "2026-07-29T00:00:00Z",
            }
        )
    assert exc.value.status_code == 400
    assert "ptu_effective_to" in exc.value.detail


def test_validate_helper_accepts_valid_window_on_merged_info():
    _validate_ptu_model_info(
        {
            "team_id": "t",
            "ptu_count": 5,
            "cost_per_ptu_per_hour": 2.0,
            "ptu_effective_from": "2026-07-30T00:00:00Z",
            "ptu_effective_to": "2026-08-30T00:00:00Z",
        }
    )


def test_validate_helper_rejects_inverted_window_without_count_or_rate():
    """A patch that touches only one end of the window merges to a model_info with no count
    or rate. Returning early on that shape let an inverted window reach the row, and the next
    load then failed to parse it and dropped the deployment out of the router."""
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info(
            {
                "team_id": "t",
                "ptu_effective_from": "2026-08-02T00:00:00Z",
                "ptu_effective_to": "2026-08-01T00:00:00Z",
            }
        )
    assert exc.value.status_code == 400
    assert "ptu_effective_to" in exc.value.detail


def test_validate_helper_rejects_equal_window_bounds_without_count_or_rate():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info(
            {
                "ptu_effective_from": "2026-08-01T00:00:00Z",
                "ptu_effective_to": "2026-08-01T00:00:00Z",
            }
        )
    assert exc.value.status_code == 400


def test_validate_helper_accepts_ordered_window_without_count_or_rate():
    """Window-only edits stay legal; only the ordering is enforced, and no team_id is
    demanded while the deployment carries no priced PTU config."""
    _validate_ptu_model_info(
        {
            "ptu_effective_from": "2026-08-01T00:00:00Z",
            "ptu_effective_to": "2026-08-02T00:00:00Z",
        }
    )


def test_validate_helper_accepts_a_single_open_ended_bound():
    _validate_ptu_model_info({"ptu_effective_from": "2026-08-01T00:00:00Z"})
    _validate_ptu_model_info({"ptu_effective_to": "2026-08-02T00:00:00Z"})
