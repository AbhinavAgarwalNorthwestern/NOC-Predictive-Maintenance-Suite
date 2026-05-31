"""Site-level anomaly detection — flags individual sites behaving abnormally.

Drift monitoring detects fleet-level distribution shifts. This module catches
single sites whose alarm patterns deviate from their own history and/or
their regional baseline. Operationally: "site KHI-042 is acting weird today."

Approach: Isolation Forest on the same alarm-history features used by the models,
with separate regional baselines so a high-outage region (Peshawar) doesn't
flag every site as anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyResult:
    site_id: str
    anomaly_score: float
    is_anomalous: bool
    top_deviations: dict[str, float]


def fit_regional_detector(
    features: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
    seed: int = 42,
) -> dict[str, IsolationForest]:
    """Fit one Isolation Forest per region.

    Parameters
    ----------
    features : DataFrame with columns site_id, region, + feature_cols
    feature_cols : alarm-history features to use for anomaly scoring
    contamination : expected fraction of anomalous sites (default 5%)
    seed : random state for reproducibility

    Returns
    -------
    Dict mapping region -> fitted IsolationForest
    """
    detectors = {}
    for region, group in features.groupby("region"):
        X = group[feature_cols].astype(float).fillna(0.0)
        if len(X) < 10:
            continue
        iso = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )
        iso.fit(X)
        detectors[region] = iso
    return detectors


def score_sites(
    features: pd.DataFrame,
    feature_cols: list[str],
    detectors: dict[str, IsolationForest],
    threshold: float = -0.5,
) -> list[AnomalyResult]:
    """Score each site against its regional detector.

    Parameters
    ----------
    features : current features with site_id, region, + feature_cols
    detectors : fitted regional detectors from fit_regional_detector
    threshold : anomaly score threshold (more negative = more anomalous)

    Returns
    -------
    List of AnomalyResult for all sites, sorted most-anomalous first.
    """
    results = []
    for _, row in features.iterrows():
        region = row.get("region", "unknown")
        detector = detectors.get(region)
        if detector is None:
            continue

        x = row[feature_cols].values.astype(float).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0)
        score = float(detector.score_samples(x)[0])
        is_anomalous = score < threshold

        # Compute which features deviate most from regional median
        top_devs = _top_deviations(row, feature_cols, features, region)

        results.append(
            AnomalyResult(
                site_id=row["site_id"],
                anomaly_score=score,
                is_anomalous=is_anomalous,
                top_deviations=top_devs,
            )
        )

    results.sort(key=lambda r: r.anomaly_score)
    return results


def _top_deviations(
    row: pd.Series,
    feature_cols: list[str],
    all_features: pd.DataFrame,
    region: str,
    top_n: int = 5,
) -> dict[str, float]:
    """Return top N features by z-score deviation from regional median."""
    regional = all_features[all_features["region"] == region][feature_cols]
    medians = regional.median()
    stds = regional.std().replace(0, 1.0)
    z_scores = {}
    for col in feature_cols:
        val = float(row[col]) if not pd.isna(row[col]) else 0.0
        z = (val - medians[col]) / stds[col]
        z_scores[col] = float(z)
    ranked = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    return {k: round(v, 2) for k, v in ranked[:top_n]}


def detect_anomalies(
    alarms: pd.DataFrame,
    site_static: pd.DataFrame,
    feature_cols: list[str] | None = None,
    current_h: float | None = None,
    contamination: float = 0.05,
    seed: int = 42,
) -> list[AnomalyResult]:
    """End-to-end anomaly detection: compute features, fit detector, score.

    Convenience function for the dashboard / NOC view.
    """
    from battery_pdm.common.features import compute_features

    if current_h is None:
        current_h = float(alarms["timestamp_h"].max())

    labels = pd.DataFrame(
        {
            "site_id": site_static["site_id"].values,
            "mains_fail_h": current_h,
        }
    )

    alarms_pit = alarms[alarms["timestamp_h"] <= current_h]

    features = compute_features(
        labels=labels,
        groups=["alarm_history", "site_static"],
        inputs={"alarms": alarms_pit, "site_static": site_static},
        ref_time_col="mains_fail_h",
    )
    features["region"] = features["site_id"].map(
        dict(zip(site_static["site_id"], site_static["region"]))
    )

    if feature_cols is None:
        feature_cols = [
            c
            for c in features.columns
            if c not in ("site_id", "mains_fail_h", "region")
        ]

    detectors = fit_regional_detector(features, feature_cols, contamination, seed)
    return score_sites(features, feature_cols, detectors)
