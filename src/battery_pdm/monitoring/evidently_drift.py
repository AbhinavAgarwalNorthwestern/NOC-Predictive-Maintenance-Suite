"""Evidently AI wrapper — drop-in alternative to our PSI implementation.

Why have both?
    - Our PSI (drift.py) is hand-rolled — shows we understand the math.
    - Evidently provides battle-tested implementations of PSI, KS, JS, Wasserstein.
    - Industry standard tooling that interviewers recognize.

This wrapper exposes the same interface as drift.detect_drift() so the
DriftMonitorFlow can swap between implementations via a parameter.
"""

from __future__ import annotations


import numpy as np
import pandas as pd


def detect_drift_evidently(
    reference_profile: dict,
    current_features: pd.DataFrame,
    current_predictions: np.ndarray | None = None,
    feature_cols: list[str] | None = None,
    stattest: str = "psi",
    threshold: float = 0.25,
) -> dict:
    """Run Evidently's DataDriftPreset on the current vs reference features.

    Returns the same shape as drift.detect_drift() for drop-in replacement.

    Args:
        reference_profile: our reference profile JSON (contains training samples)
        current_features: DataFrame of current scoring features
        current_predictions: optional model predictions for prediction drift
        feature_cols: which features to compare
        stattest: "psi" (default), "ks", "jensenshannon", or "wasserstein"
        threshold: detector threshold (default 0.25 for PSI)
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError as exc:
        raise ImportError(
            "Evidently not installed. Run: uv add 'evidently>=0.4,<0.5'"
        ) from exc

    feature_cols = feature_cols or list(reference_profile.get("features", {}).keys())

    # Reconstruct reference DataFrame from our stored sampled values
    ref_dict: dict[str, list[float]] = {}
    for col in feature_cols:
        if col in reference_profile["features"]:
            ref_dict[col] = list(reference_profile["features"][col]["values_sample"])

    if not ref_dict:
        return {
            "feature_drift": pd.DataFrame(),
            "prediction_drift": {},
            "summary": {
                "status": "NO_REFERENCE",
                "n_features_monitored": 0,
                "n_significant_drift": 0,
                "n_moderate_drift": 0,
                "retrain_recommended": False,
                "reasons": ["No reference features found"],
            },
        }

    min_len = min(len(v) for v in ref_dict.values())
    ref_df = pd.DataFrame({k: v[:min_len] for k, v in ref_dict.items()})
    cur_df = current_features[
        [c for c in feature_cols if c in current_features.columns]
    ].copy()

    # Align columns
    shared = [c for c in ref_df.columns if c in cur_df.columns]
    ref_df = ref_df[shared].dropna()
    cur_df = cur_df[shared].dropna()

    if len(ref_df) < 30 or len(cur_df) < 30:
        return {
            "feature_drift": pd.DataFrame(),
            "prediction_drift": {},
            "summary": {
                "status": "INSUFFICIENT_DATA",
                "n_features_monitored": len(shared),
                "n_significant_drift": 0,
                "n_moderate_drift": 0,
                "retrain_recommended": False,
                "reasons": ["Reference or current too small"],
            },
        }

    # Run Evidently (v0.4 API)
    report = Report(
        metrics=[
            DataDriftPreset(stattest=stattest, stattest_threshold=threshold),
        ]
    )
    report.run(reference_data=ref_df, current_data=cur_df)
    snap_dict = report.as_dict()

    # Parse Evidently's output — v0.4 returns metrics[0]['result']['drift_by_columns']
    feature_drift_rows = []
    n_drifted = 0
    for metric in snap_dict.get("metrics", []):
        result = metric.get("result", {})
        drift_by_cols = result.get("drift_by_columns", {})
        for col, info in drift_by_cols.items():
            drift_score = info.get("drift_score", 0.0)
            is_drift = bool(info.get("drift_detected", False))
            feature_drift_rows.append(
                {
                    "feature": col,
                    "psi": float(drift_score) if drift_score is not None else 0.0,
                    "drift_level": "SIGNIFICANT" if is_drift else "STABLE",
                    "ks_statistic": float("nan"),
                    "ks_pvalue": float("nan"),
                    "ref_mean": float(ref_df[col].mean()),
                    "cur_mean": float(cur_df[col].mean()),
                    "mean_shift_std": float(
                        abs(cur_df[col].mean() - ref_df[col].mean())
                        / max(ref_df[col].std(), 1e-6)
                    ),
                }
            )
            if is_drift:
                n_drifted += 1
        if drift_by_cols:
            break  # only one ColumnMapping-aware result expected

    feature_drift = (
        pd.DataFrame(feature_drift_rows)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
        if feature_drift_rows
        else pd.DataFrame()
    )

    # Prediction drift — use our existing implementation (Evidently version requires labels)
    pred_drift = {}
    if current_predictions is not None and "predictions" in reference_profile:
        from battery_pdm.monitoring.drift import compute_prediction_drift

        pred_drift = compute_prediction_drift(reference_profile, current_predictions)

    retrain = n_drifted >= 3 or pred_drift.get("drift_detected", False)
    reasons = []
    if n_drifted >= 3:
        reasons.append(
            f"{n_drifted}/{len(shared)} features show SIGNIFICANT drift (Evidently)"
        )
    if pred_drift.get("drift_detected", False):
        reasons.append(
            f"Prediction distribution shifted (PSI={pred_drift.get('prediction_psi', 0):.3f})"
        )

    return {
        "feature_drift": feature_drift,
        "prediction_drift": pred_drift,
        "summary": {
            "status": "DRIFT_DETECTED" if retrain else "STABLE",
            "n_features_monitored": len(shared),
            "n_significant_drift": int(n_drifted),
            "n_moderate_drift": 0,  # Evidently is binary; we map all drifted to significant
            "retrain_recommended": retrain,
            "reasons": reasons,
            "engine": "evidently",
            "stattest": stattest,
        },
        "_evidently_report": report,
    }


def generate_evidently_html_report(
    reference_profile: dict,
    current_features: pd.DataFrame,
    feature_cols: list[str] | None = None,
    stattest: str = "psi",
    threshold: float = 0.25,
) -> str | None:
    """Generate an Evidently HTML drift report for Metaflow card rendering.

    Returns the HTML string or None if Evidently is unavailable or data insufficient.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        return None

    feature_cols = feature_cols or list(reference_profile.get("features", {}).keys())

    ref_dict: dict[str, list[float]] = {}
    for col in feature_cols:
        if col in reference_profile["features"]:
            ref_dict[col] = list(reference_profile["features"][col]["values_sample"])

    if not ref_dict:
        return None

    min_len = min(len(v) for v in ref_dict.values())
    ref_df = pd.DataFrame({k: v[:min_len] for k, v in ref_dict.items()})
    cur_df = current_features[
        [c for c in feature_cols if c in current_features.columns]
    ].copy()

    shared = [c for c in ref_df.columns if c in cur_df.columns]
    ref_df = ref_df[shared].dropna()
    cur_df = cur_df[shared].dropna()

    if len(ref_df) < 30 or len(cur_df) < 30:
        return None

    report = Report(
        metrics=[
            DataDriftPreset(stattest=stattest, stattest_threshold=threshold),
        ]
    )
    report.run(reference_data=ref_df, current_data=cur_df)
    return report.get_html()
