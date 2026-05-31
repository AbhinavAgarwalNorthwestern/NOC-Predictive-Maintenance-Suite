"""Tests for concept_drift module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_pdm.monitoring.concept_drift import (
    PerformanceWindow,
    fetch_realized_outcomes,
    compute_window_performance,
    detect_concept_drift,
    check_label_maturity,
)


def _make_predictions(n=200, scoring_hours=None):
    """Build a synthetic predictions DataFrame."""
    rng = np.random.default_rng(42)
    if scoring_hours is None:
        scoring_hours = np.linspace(0, 30 * 24, n)
    return pd.DataFrame(
        {
            "site_id": [f"SITE_{i % 5:03d}" for i in range(n)],
            "scoring_hour": scoring_hours,
            "predicted_score": rng.uniform(0, 1, n),
        }
    )


def _make_alarms_with_events(prediction_df, event_rate=0.3, horizon_h=48):
    """Build a synthetic alarm stream with realistic LVD events."""
    rng = np.random.default_rng(42)
    alarms = []
    for _, row in prediction_df.iterrows():
        if rng.random() < event_rate:
            # Event happens within the horizon
            event_time = row["scoring_hour"] + rng.uniform(0, horizon_h)
            alarms.append(
                {
                    "site_id": row["site_id"],
                    "timestamp_h": event_time,
                    "alarm_code": "LOAD_DISCONNECT",
                    "severity": "critical",
                }
            )
    return (
        pd.DataFrame(alarms)
        if alarms
        else pd.DataFrame(columns=["site_id", "timestamp_h", "alarm_code", "severity"])
    )


def test_fetch_realized_outcomes_basic():
    preds = _make_predictions(n=50)
    alarms = _make_alarms_with_events(preds, event_rate=0.4, horizon_h=48)

    enriched = fetch_realized_outcomes(preds, alarms, horizon_h=48)
    assert "realized_label" in enriched.columns
    assert "labels_mature_at_h" in enriched.columns
    assert (enriched["labels_mature_at_h"] == enriched["scoring_hour"] + 48).all()
    # ~40% should be positive (event_rate)
    assert 0.2 < enriched["realized_label"].mean() < 0.6


def test_fetch_realized_outcomes_empty_alarms():
    preds = _make_predictions(n=10)
    empty_alarms = pd.DataFrame(
        columns=["site_id", "timestamp_h", "alarm_code", "severity"]
    )
    enriched = fetch_realized_outcomes(preds, empty_alarms, horizon_h=48)
    assert enriched["realized_label"].sum() == 0


def test_compute_window_performance_insufficient_returns_nan_auc():
    preds = _make_predictions(n=20)
    alarms = _make_alarms_with_events(preds)
    enriched = fetch_realized_outcomes(preds, alarms, horizon_h=48)
    # Current time after all label maturity
    current_h = enriched["labels_mature_at_h"].max() + 1

    # Small window with <30 mature samples → AUC=nan
    result = compute_window_performance(enriched, 0, 100, current_h, horizon_h=48)
    assert np.isnan(result.auc)
    assert result.n_with_labels < 30


def test_compute_window_performance_with_signal():
    """Predictions correlated with labels should give AUC > 0.5."""
    rng = np.random.default_rng(42)
    n = 500
    scoring_hours = np.linspace(0, 30 * 24, n)
    y = (rng.uniform(0, 1, n) < 0.3).astype(int)
    # Correlated predictions: positives get higher scores
    p = rng.uniform(0, 1, n) * 0.5 + y * 0.4

    preds = pd.DataFrame(
        {
            "site_id": [f"S_{i:04d}" for i in range(n)],
            "scoring_hour": scoring_hours,
            "predicted_score": p,
            "realized_label": y,
            "labels_mature_at_h": scoring_hours + 48,
        }
    )
    current_h = preds["labels_mature_at_h"].max() + 1

    result = compute_window_performance(preds, 0, 1000, current_h, horizon_h=48)
    assert result.n_with_labels >= 30
    assert (
        result.auc > 0.65
    )  # signal correlated with labels — should beat random meaningfully
    assert 0 < result.brier < 0.3


def test_detect_concept_drift_no_drift_when_stable():
    windows = [
        PerformanceWindow(0, 100, 100, 100, auc=0.88, brier=0.1, positive_rate=0.3),
        PerformanceWindow(100, 200, 100, 100, auc=0.87, brier=0.1, positive_rate=0.3),
        PerformanceWindow(200, 300, 100, 100, auc=0.86, brier=0.1, positive_rate=0.3),
    ]
    result = detect_concept_drift(
        windows, baseline_auc=0.88, auc_degradation_threshold=0.05
    )
    assert not result["concept_drift_detected"]
    assert result["auc_degradation"] < 0.05


def test_detect_concept_drift_triggers_on_real_degradation():
    windows = [
        PerformanceWindow(0, 100, 100, 100, auc=0.88, brier=0.1, positive_rate=0.3),
        PerformanceWindow(100, 200, 100, 100, auc=0.75, brier=0.18, positive_rate=0.3),
        PerformanceWindow(200, 300, 100, 100, auc=0.70, brier=0.20, positive_rate=0.3),
    ]
    result = detect_concept_drift(
        windows, baseline_auc=0.88, auc_degradation_threshold=0.05
    )
    assert result["concept_drift_detected"]
    assert result["auc_degradation"] > 0.05


def test_check_label_maturity_empty():
    result = check_label_maturity(
        pd.DataFrame(), min_days_of_labels=30, horizon_h=48, current_h=10000
    )
    assert not result["ready_to_promote"]
    assert result["reason"] == "no_predictions_logged"


def test_check_label_maturity_not_enough_yet():
    """5 days of labels when we need 30 → not ready."""
    preds = pd.DataFrame(
        {
            "site_id": ["S1"] * 100,
            "scoring_hour": np.linspace(0, 5 * 24, 100),
            "predicted_score": [0.5] * 100,
            "realized_label": [0] * 100,
            "labels_mature_at_h": np.linspace(0, 5 * 24, 100) + 48,
        }
    )
    current_h = preds["labels_mature_at_h"].max() + 1
    result = check_label_maturity(
        preds, min_days_of_labels=30, horizon_h=48, current_h=current_h
    )
    assert not result["ready_to_promote"]
    assert result["days_of_labels"] < 10


def test_check_label_maturity_enough_labels():
    preds = pd.DataFrame(
        {
            "site_id": ["S1"] * 200,
            "scoring_hour": np.linspace(0, 40 * 24, 200),
            "predicted_score": [0.5] * 200,
            "realized_label": [0] * 200,
            "labels_mature_at_h": np.linspace(0, 40 * 24, 200) + 48,
        }
    )
    current_h = preds["labels_mature_at_h"].max() + 1
    result = check_label_maturity(
        preds, min_days_of_labels=30, horizon_h=48, current_h=current_h
    )
    assert result["ready_to_promote"]
    assert result["days_of_labels"] >= 30
