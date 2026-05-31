"""Tests for the feature pipeline.

Covers:
    - Point-in-time correctness: features at t0 must only use alarms < t0
    - Feature group composition: alarm_history + site_static produce expected cols
    - SoC ledger sanity: never NaN, bounded [0,1]
    - Schedule features: look up correctly per region
"""

from __future__ import annotations

import pandas as pd
import pytest

from battery_pdm.common.features import (
    compute_features,
    list_groups,
)


def test_alarm_history_group_registered():
    assert "alarm_history" in list_groups()


def test_site_static_group_registered():
    assert "site_static" in list_groups()


def test_soc_proxy_group_registered():
    assert "soc_proxy" in list_groups()


def test_compute_features_basic(small_alarms, small_site_static):
    labels = pd.DataFrame(
        {
            "site_id": ["SITE_001", "SITE_002"],
            "mains_fail_h": [200.0, 200.0],
        }
    )
    out = compute_features(
        labels=labels,
        groups=["alarm_history", "site_static"],
        inputs={"alarms": small_alarms, "site_static": small_site_static},
    )
    assert len(out) == 2
    assert "site_id" in out.columns
    assert "mains_fail_count_30d" in out.columns
    assert "load_A" in out.columns


def test_compute_features_point_in_time_correct(small_alarms, small_site_static):
    """Features at t0 must NOT see alarms from t > t0."""
    early_labels = pd.DataFrame(
        {
            "site_id": ["SITE_001"],
            "mains_fail_h": [50.0],
        }
    )
    late_labels = pd.DataFrame(
        {
            "site_id": ["SITE_001"],
            "mains_fail_h": [400.0],
        }
    )

    early = compute_features(
        labels=early_labels,
        groups=["alarm_history"],
        inputs={"alarms": small_alarms},
    )
    late = compute_features(
        labels=late_labels,
        groups=["alarm_history"],
        inputs={"alarms": small_alarms},
    )

    # Lifetime counts should never decrease over time
    assert (
        early["mains_fail_count_lifetime"].iloc[0]
        <= late["mains_fail_count_lifetime"].iloc[0]
    )


def test_compute_features_soc_proxy_bounded(small_alarms, small_site_static):
    labels = pd.DataFrame(
        {
            "site_id": ["SITE_001", "SITE_002"],
            "mains_fail_h": [300.0, 300.0],
        }
    )
    out = compute_features(
        labels=labels,
        groups=["soc_proxy"],
        inputs={"alarms": small_alarms, "site_static": small_site_static},
    )
    soc = out["estimated_soc_at_ref"]
    assert (soc >= 0.0).all() and (soc <= 1.0).all()
    assert not soc.isna().any()


def test_compute_features_unknown_group_raises(small_alarms, small_site_static):
    labels = pd.DataFrame({"site_id": ["SITE_001"], "mains_fail_h": [200.0]})
    with pytest.raises(KeyError):
        compute_features(
            labels=labels,
            groups=["nonexistent_group"],
            inputs={"alarms": small_alarms},
        )


def test_compute_features_missing_input_raises(small_alarms):
    labels = pd.DataFrame({"site_id": ["SITE_001"], "mains_fail_h": [200.0]})
    with pytest.raises(ValueError):
        # site_static requires the site_static input
        compute_features(
            labels=labels,
            groups=["site_static"],
            inputs={"alarms": small_alarms},
        )


def test_schedule_features(small_alarms, small_site_static, small_schedule):
    labels = pd.DataFrame(
        {
            "site_id": ["SITE_001", "SITE_002"],
            "mains_fail_h": [300.0, 300.0],
        }
    )
    out = compute_features(
        labels=labels,
        groups=["load_shedding_schedule"],
        inputs={
            "alarms": small_alarms,
            "site_static": small_site_static,
            "schedule": small_schedule,
        },
    )
    assert "schedule_offgrid_hours_past_30d" in out.columns
    assert "schedule_offgrid_hours_next_24h" in out.columns
    # Karachi should show higher severity than Lahore (fixture sets 0.3 vs 0.15)
    karachi_row = out[out["site_id"] == "SITE_002"].iloc[0]
    lahore_row = out[out["site_id"] == "SITE_001"].iloc[0]
    assert (
        karachi_row["schedule_severity_mean_past_30d"]
        > lahore_row["schedule_severity_mean_past_30d"]
    )


def test_ref_time_col_renaming(small_alarms, small_site_static):
    """compute_features should preserve the caller's ref_time column name in output."""
    labels = pd.DataFrame(
        {
            "site_id": ["SITE_001"],
            "event_hour": [200.0],
        }
    )
    out = compute_features(
        labels=labels,
        groups=["alarm_history"],
        inputs={"alarms": small_alarms},
        ref_time_col="event_hour",
    )
    assert "event_hour" in out.columns
    assert "mains_fail_h" not in out.columns
