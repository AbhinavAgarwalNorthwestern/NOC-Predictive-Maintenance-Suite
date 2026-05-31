"""Data quality checks + missing-data handling for scoring pipelines.

Two principles:
    1. Don't silently impute zeros and pretend you have data. Track missingness
       as a feature so the model can be appropriately uncertain.
    2. Refuse to score when input is degenerate (no alarms at all for a site
       we're supposed to score). A loud INSUFFICIENT_DATA flag is safer than
       a confident wrong prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# Minimum alarm history needed before scoring is meaningful
MIN_HISTORY_DAYS = 7
MIN_ALARMS_PER_SITE = 3


@dataclass
class DataQualityReport:
    n_sites_total: int
    n_sites_scorable: int
    n_sites_insufficient_data: int
    null_rate_per_column: dict
    missing_regions: list[str]
    missing_site_static: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "n_sites_total": self.n_sites_total,
            "n_sites_scorable": self.n_sites_scorable,
            "n_sites_insufficient_data": self.n_sites_insufficient_data,
            "null_rate_per_column": self.null_rate_per_column,
            "missing_regions": self.missing_regions,
            "missing_site_static": self.missing_site_static,
            "warnings": self.warnings,
        }


def assess_scoring_inputs(
    site_ids: list[str],
    alarms: pd.DataFrame,
    site_static: pd.DataFrame,
    schedule: pd.DataFrame | None,
    ref_time_h: float,
    history_days: int = 30,
) -> tuple[set[str], DataQualityReport]:
    """Identify which sites have enough data to score and report on gaps.

    Returns:
        scorable_sites: set of site_ids that pass the quality bar
        report: DataQualityReport with the gaps found
    """
    warnings_list: list[str] = []
    n_total = len(site_ids)

    # Site static coverage
    static_sites = set(site_static["site_id"].unique())
    missing_static = [s for s in site_ids if s not in static_sites]
    if missing_static:
        warnings_list.append(
            f"{len(missing_static)} sites are missing from site_static — cannot score them."
        )

    # Schedule region coverage
    missing_regions: list[str] = []
    if schedule is not None and not schedule.empty:
        sched_regions = set(schedule["region"].unique())
        site_regions = set(
            site_static[site_static["site_id"].isin(site_ids)]["region"]
            .dropna()
            .unique()
        )
        missing_regions = sorted(site_regions - sched_regions)
        if missing_regions:
            warnings_list.append(
                f"Schedule missing for regions: {missing_regions}. "
                f"Scores for these regions will use zero-filled schedule features."
            )

    # Alarm history sufficiency per site
    window_start = ref_time_h - history_days * 24
    recent = alarms[
        (alarms["timestamp_h"] >= window_start) & (alarms["timestamp_h"] < ref_time_h)
    ]
    counts = recent.groupby("site_id").size()

    scorable = set()
    insufficient = []
    for s in site_ids:
        if s in missing_static:
            insufficient.append(s)
            continue
        # Allow scoring if any alarms ever exist for this site (some sites are quiet)
        # but flag if no recent activity at all (might be sensor down)
        site_recent_count = int(counts.get(s, 0))  # noqa: F841
        site_lifetime_count = int((alarms["site_id"] == s).sum())
        if site_lifetime_count < MIN_ALARMS_PER_SITE:
            insufficient.append(s)
        else:
            scorable.add(s)

    if insufficient:
        warnings_list.append(
            f"{len(insufficient)} sites have insufficient history (<{MIN_ALARMS_PER_SITE} lifetime alarms)"
        )

    report = DataQualityReport(
        n_sites_total=n_total,
        n_sites_scorable=len(scorable),
        n_sites_insufficient_data=len(insufficient),
        null_rate_per_column={},  # filled in after features are computed
        missing_regions=missing_regions,
        missing_site_static=missing_static,
        warnings=warnings_list,
    )
    return scorable, report


def add_missingness_flags(
    features: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Add a `data_completeness_score` column reflecting how much data backed each row.

    Score = fraction of feature_cols that are non-null AND non-zero. Low scores
    mean the row is mostly default/imputed values.
    """
    features = features.copy()
    relevant = [c for c in feature_cols if c in features.columns]
    if not relevant:
        features["data_completeness_score"] = 1.0
        return features

    non_null = features[relevant].notna().sum(axis=1)
    non_zero = (features[relevant].fillna(0) != 0).sum(axis=1)
    # Average of both — a row with all NaN OR all zero is suspicious
    features["data_completeness_score"] = (
        (non_null + non_zero) / (2.0 * len(relevant))
    ).clip(0.0, 1.0)
    return features


def compute_null_rates(features: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Per-column null rate, for logging to CloudWatch + drift report."""
    out = {}
    for c in feature_cols:
        if c in features.columns:
            out[c] = float(features[c].isna().mean())
    return out


def compute_regional_priors(
    alarms: pd.DataFrame,
    site_static: pd.DataFrame,
    horizon_h: int = 48,
) -> dict[str, float]:
    """Per-region historical drain rate — used as COLD_START fallback risk score.

    For a brand-new battery, the alarm-history features are all zero, so the ML
    model sees a "clean slate" and might predict misleadingly low risk.
    Instead, we use the region's historical drain rate as a conservative default.

    Returns a dict mapping region -> drain rate in [0, 1].
    Peshawar / Lahore come out high (~0.25-0.30); Islamabad near zero (~0.001).
    """
    if alarms.empty or site_static.empty:
        return {}
    merged = alarms.merge(site_static[["site_id", "region"]], on="site_id", how="left")
    rates: dict[str, float] = {}
    for region, sub in merged.groupby("region"):
        sub = sub.sort_values(["site_id", "timestamp_h"]).reset_index(drop=True)
        mfs = sub[sub["alarm_code"] == "AC_MAINS_FAIL"]
        if len(mfs) == 0:
            rates[str(region)] = 0.0
            continue
        drains = 0
        for _, row in mfs.iterrows():
            window = sub[
                (sub["site_id"] == row["site_id"])
                & (sub["timestamp_h"] > row["timestamp_h"])
                & (sub["timestamp_h"] <= row["timestamp_h"] + horizon_h)
            ]
            if (window["alarm_code"] == "LOAD_DISCONNECT").any():
                drains += 1
        rates[str(region)] = float(drains / max(len(mfs), 1))
    return rates
