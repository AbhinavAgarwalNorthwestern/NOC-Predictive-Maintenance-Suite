"""Concept drift detection — needs labels.

Feature drift (PSI) can detect that inputs changed; it can't tell you the
model is genuinely worse. Concept drift detection compares model predictions
to REALIZED outcomes as labels arrive.

Pipeline:
    1. Every scoring run writes predictions to S3 alerts/
    2. After label horizon (48h for drain, 12h for autonomy, 6mo for failure),
       fetch the realized outcome from the alarm stream
    3. Compute rolling performance metric (AUC, Brier) on labeled feedback
    4. If performance degrades vs baseline → flag concept drift

This module is the missing 4th pillar of drift monitoring (after schema,
feature, and prediction drift).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceWindow:
    """Rolling performance metrics over a labeled window."""

    window_start_h: float
    window_end_h: float
    n_predictions: int
    n_with_labels: int
    auc: float
    brier: float
    positive_rate: float


def fetch_realized_outcomes(
    predictions: pd.DataFrame,
    alarms: pd.DataFrame,
    horizon_h: int,
    event_alarm_code: str = "LOAD_DISCONNECT",
) -> pd.DataFrame:
    """For each prediction, look up whether the event occurred within horizon.

    Args:
        predictions: DataFrame with columns [site_id, scoring_hour, predicted_score]
        alarms: Full alarm stream
        horizon_h: Hours after scoring to check for the event
        event_alarm_code: Which alarm marks the event (default LOAD_DISCONNECT)

    Returns:
        predictions augmented with [realized_label, labels_mature_at_h]
    """
    out = predictions.copy()
    out["labels_mature_at_h"] = out["scoring_hour"] + horizon_h

    # Filter alarms to just events
    events = alarms[alarms["alarm_code"] == event_alarm_code]
    events_by_site = {
        sid: g["timestamp_h"].values for sid, g in events.groupby("site_id")
    }

    realized = []
    for _, row in out.iterrows():
        site = row["site_id"]
        scoring_h = row["scoring_hour"]
        site_events = events_by_site.get(site, np.array([]))
        # Did any event happen in [scoring_h, scoring_h + horizon_h]?
        in_window = (site_events >= scoring_h) & (site_events < scoring_h + horizon_h)
        realized.append(int(in_window.any()))
    out["realized_label"] = realized
    return out


def compute_window_performance(
    predictions_with_labels: pd.DataFrame,
    window_start_h: float,
    window_end_h: float,
    current_h: float,
    horizon_h: int,
) -> PerformanceWindow | None:
    """Compute AUC/Brier on labeled predictions in a time window.

    Only includes predictions where labels are MATURE (label horizon passed).
    """
    from sklearn.metrics import roc_auc_score

    in_window = predictions_with_labels[
        (predictions_with_labels["scoring_hour"] >= window_start_h)
        & (predictions_with_labels["scoring_hour"] < window_end_h)
    ]
    mature = in_window[in_window["labels_mature_at_h"] <= current_h]

    if len(mature) < 30 or mature["realized_label"].nunique() < 2:
        return PerformanceWindow(
            window_start_h=window_start_h,
            window_end_h=window_end_h,
            n_predictions=len(in_window),
            n_with_labels=len(mature),
            auc=float("nan"),
            brier=float("nan"),
            positive_rate=float(mature["realized_label"].mean())
            if len(mature)
            else 0.0,
        )

    y = mature["realized_label"].values
    p = mature["predicted_score"].values
    return PerformanceWindow(
        window_start_h=window_start_h,
        window_end_h=window_end_h,
        n_predictions=len(in_window),
        n_with_labels=len(mature),
        auc=float(roc_auc_score(y, p)),
        brier=float(np.mean((p - y) ** 2)),
        positive_rate=float(y.mean()),
    )


def detect_concept_drift(
    rolling_windows: list[PerformanceWindow],
    baseline_auc: float,
    auc_degradation_threshold: float = 0.05,
) -> dict:
    """Detect concept drift by comparing rolling performance to training baseline.

    Triggers if recent windows' AUC drops more than threshold below baseline.

    Args:
        rolling_windows: List of PerformanceWindow, oldest first
        baseline_auc: AUC achieved at training time (from model meta.json)
        auc_degradation_threshold: trigger if rolling AUC < baseline - this

    Returns:
        Dict with detection result + reason
    """
    valid_windows = [w for w in rolling_windows if not np.isnan(w.auc)]
    if not valid_windows:
        return {
            "concept_drift_detected": False,
            "reason": "insufficient_labels",
            "n_valid_windows": 0,
        }

    recent = valid_windows[-3:]  # last 3 windows
    avg_recent_auc = float(np.mean([w.auc for w in recent]))
    delta = baseline_auc - avg_recent_auc

    detected = delta > auc_degradation_threshold
    return {
        "concept_drift_detected": detected,
        "baseline_auc": float(baseline_auc),
        "recent_avg_auc": avg_recent_auc,
        "auc_degradation": float(delta),
        "threshold": float(auc_degradation_threshold),
        "n_valid_windows": len(valid_windows),
        "reason": f"AUC dropped {delta:.4f} (threshold {auc_degradation_threshold})"
        if detected
        else "stable",
    }


# ---------------------------------------------------------------------------
# Label maturity gate — for the RetrainingFlow
# ---------------------------------------------------------------------------


def check_label_maturity(
    predictions_with_labels: pd.DataFrame,
    min_days_of_labels: int,
    horizon_h: int,
    current_h: float,
) -> dict:
    """Check if we have enough days of mature labels to validate a new model.

    For long-horizon models (failure: 6-12mo), this prevents promoting a
    challenger before we have realized outcomes to compare against.

    Returns:
        Dict with:
            ready_to_promote: bool
            days_of_labels: float (how many days of mature labels we have)
            required_days: int (min_days_of_labels)
            reason: str
    """
    if predictions_with_labels.empty:
        return {
            "ready_to_promote": False,
            "days_of_labels": 0.0,
            "required_days": min_days_of_labels,
            "reason": "no_predictions_logged",
        }

    mature = predictions_with_labels[
        predictions_with_labels["labels_mature_at_h"] <= current_h
    ]
    if mature.empty:
        return {
            "ready_to_promote": False,
            "days_of_labels": 0.0,
            "required_days": min_days_of_labels,
            "reason": "no_mature_labels_yet",
        }

    label_span_h = mature["scoring_hour"].max() - mature["scoring_hour"].min()
    days_of_labels = label_span_h / 24.0
    ready = days_of_labels >= min_days_of_labels
    return {
        "ready_to_promote": ready,
        "days_of_labels": float(days_of_labels),
        "required_days": int(min_days_of_labels),
        "n_mature_labels": int(len(mature)),
        "reason": f"have {days_of_labels:.1f} days of labels (need {min_days_of_labels})"
        if ready
        else f"need {min_days_of_labels - days_of_labels:.1f} more days of labels",
    }
