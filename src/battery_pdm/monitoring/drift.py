"""Drift detection for battery PdM models.

Monitors feature and prediction distributions against a reference profile
(snapshot from training time). Detects when the production data has shifted
enough that model retraining is warranted.

Methods:
    - PSI (Population Stability Index): standard metric for tabular drift.
      PSI < 0.1 = stable, 0.1-0.25 = moderate drift, > 0.25 = significant.
    - KS test: p-value per feature, flags when distribution has shifted.
    - Prediction drift: compares score distribution vs training-time scores.

Usage:
    from battery_pdm.monitoring.drift import (
        build_reference_profile,
        compute_psi,
        detect_drift,
    )
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PSI_BINS = 10
PSI_THRESHOLD_MODERATE = 0.10
PSI_THRESHOLD_SIGNIFICANT = 0.25
KS_PVALUE_THRESHOLD = 0.01


def _psi_single(
    reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS
) -> float:
    """Compute PSI for a single feature using equal-frequency binning on reference."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) < bins or len(cur) < bins:
        return 0.0

    # Bin edges from reference quantiles
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(ref, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Ensure unique edges
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)

    # Avoid zero bins
    ref_pct = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))
    cur_pct = (cur_counts + 1) / (cur_counts.sum() + len(cur_counts))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def build_reference_profile(
    features: pd.DataFrame,
    feature_cols: list[str],
    predictions: np.ndarray | None = None,
    output_path: Path | str = "outputs/models/reference_profile.json",
) -> dict:
    """Build and save a reference profile from training-time features.

    Stores per-feature: mean, std, quantiles (for PSI binning), and
    optionally the prediction score distribution.
    """
    profile = {
        "n_samples": len(features),
        "features": {},
    }

    for col in feature_cols:
        vals = features[col].dropna().values.astype(float)
        if len(vals) == 0:
            continue
        quantiles = np.linspace(0, 100, PSI_BINS + 1)
        profile["features"][col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "quantile_edges": np.percentile(vals, quantiles).tolist(),
            "values_sample": vals[:: max(1, len(vals) // 500)].tolist(),
        }

    if predictions is not None:
        profile["predictions"] = {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "quantile_edges": np.percentile(
                predictions, np.linspace(0, 100, PSI_BINS + 1)
            ).tolist(),
            "values_sample": predictions[:: max(1, len(predictions) // 500)].tolist(),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2))
    return profile


def load_reference_profile(
    path: Path | str = "outputs/models/reference_profile.json",
) -> dict:
    return json.loads(Path(path).read_text())


def compute_psi(
    reference_profile: dict,
    current_features: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute PSI for each feature comparing current window to reference.

    Returns DataFrame with columns: feature, psi, drift_level, ks_statistic, ks_pvalue
    """
    if feature_cols is None:
        feature_cols = list(reference_profile["features"].keys())

    results = []
    for col in feature_cols:
        if col not in reference_profile["features"]:
            continue
        if col not in current_features.columns:
            continue

        ref_info = reference_profile["features"][col]
        ref_vals = np.array(ref_info["values_sample"])
        cur_vals = current_features[col].dropna().values.astype(float)

        if len(cur_vals) < 10:
            continue

        psi = _psi_single(ref_vals, cur_vals)

        # KS test
        ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)

        # Mean shift
        mean_shift = abs(cur_vals.mean() - ref_info["mean"]) / max(
            ref_info["std"], 1e-6
        )

        if psi >= PSI_THRESHOLD_SIGNIFICANT:
            drift_level = "SIGNIFICANT"
        elif psi >= PSI_THRESHOLD_MODERATE:
            drift_level = "MODERATE"
        else:
            drift_level = "STABLE"

        results.append(
            {
                "feature": col,
                "psi": psi,
                "drift_level": drift_level,
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "ref_mean": ref_info["mean"],
                "cur_mean": float(cur_vals.mean()),
                "mean_shift_std": float(mean_shift),
            }
        )

    return (
        pd.DataFrame(results).sort_values("psi", ascending=False).reset_index(drop=True)
    )


def compute_prediction_drift(
    reference_profile: dict,
    current_predictions: np.ndarray,
) -> dict:
    """Check if model output distribution has shifted."""
    if "predictions" not in reference_profile:
        return {"prediction_drift": "NO_REFERENCE"}

    ref_vals = np.array(reference_profile["predictions"]["values_sample"])
    psi = _psi_single(ref_vals, current_predictions)
    ks_stat, ks_pval = stats.ks_2samp(ref_vals, current_predictions)

    return {
        "prediction_psi": float(psi),
        "prediction_ks_stat": float(ks_stat),
        "prediction_ks_pval": float(ks_pval),
        "ref_mean_score": reference_profile["predictions"]["mean"],
        "cur_mean_score": float(np.mean(current_predictions)),
        "drift_detected": psi >= PSI_THRESHOLD_MODERATE,
    }


def detect_drift(
    reference_profile: dict,
    current_features: pd.DataFrame,
    current_predictions: np.ndarray | None = None,
    feature_cols: list[str] | None = None,
) -> dict:
    """Full drift report: feature-level PSI + prediction drift.

    Returns dict with:
        - feature_drift: DataFrame of per-feature PSI scores
        - prediction_drift: dict with prediction distribution shift
        - summary: overall drift status and recommendation
    """
    feature_drift = compute_psi(reference_profile, current_features, feature_cols)

    n_significant = (feature_drift["drift_level"] == "SIGNIFICANT").sum()
    n_moderate = (feature_drift["drift_level"] == "MODERATE").sum()
    n_total = len(feature_drift)

    pred_drift = {}
    if current_predictions is not None:
        pred_drift = compute_prediction_drift(reference_profile, current_predictions)

    # Decision logic
    retrain_recommended = False
    reasons = []

    if n_significant >= 3 or (n_significant / max(n_total, 1)) > 0.15:
        retrain_recommended = True
        reasons.append(
            f"{n_significant}/{n_total} features show SIGNIFICANT drift (PSI>0.25)"
        )

    if pred_drift.get("drift_detected", False):
        retrain_recommended = True
        reasons.append(
            f"Prediction distribution shifted (PSI={pred_drift.get('prediction_psi', 0):.3f})"
        )

    if n_moderate >= n_total * 0.3:
        retrain_recommended = True
        reasons.append(f"{n_moderate}/{n_total} features show MODERATE+ drift")

    summary = {
        "n_features_monitored": n_total,
        "n_significant_drift": int(n_significant),
        "n_moderate_drift": int(n_moderate),
        "retrain_recommended": retrain_recommended,
        "reasons": reasons,
        "status": "DRIFT_DETECTED" if retrain_recommended else "STABLE",
    }

    return {
        "feature_drift": feature_drift,
        "prediction_drift": pred_drift,
        "summary": summary,
    }
