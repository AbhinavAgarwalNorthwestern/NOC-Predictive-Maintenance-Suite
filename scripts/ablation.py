"""Ablation study: telemetry-only vs telemetry+alarms features.

Group-constrained CV: all lifecycles of the same physical site stay in
the same fold to prevent leakage from shared site properties.

NOTE: This script depends on the legacy telemetry-based feature_pipeline/
and training_pipeline/ packages which were removed during the production-style
restructure (2026-05). The results are documented in docs/PROJECT_BOOK.md.
To re-run, restore those packages from git history.
"""

import numpy as np
import pandas as pd

from battery_pdm.synth.simulator import simulate_fleet

import battery_pdm.feature_pipeline.compute  # noqa: F401
from battery_pdm.feature_pipeline.compute import set_alarms_data
from battery_pdm.feature_pipeline.prep import prepare_lifecycle_observations
from battery_pdm.feature_pipeline.registry import compute_all_features
from battery_pdm.training_pipeline.cv import temporal_group_split, encode_survival_labels
from battery_pdm.training_pipeline.train import train_survival_xgb, predict_risk, compute_cindex


TELEMETRY_ONLY_FEATURES = [
    "discharge_slope_mean",
    "discharge_depth_mean",
    "days_since_last_discharge",
    "recharge_time_to_float_mean",
    "arrhenius_weighted_age",
    "psoc_fraction_30d",
    "coulomb_throughput_30d",
    "lvd_count_30d",
    "deep_discharge_count_30d",
    "mean_time_between_outages",
    "time_since_full_recharge",
    "trend_discharge_slope_30d",
    "trend_float_voltage_30d",
    "battery_age_months",
    "region_encoded",
    "manufacturer_encoded",
]

ALARM_FEATURES = [
    "rectifier_fault_count_30d",
    "rectifier_fault_count_lifetime",
    "critical_alarm_count_30d",
    "alarm_rate_trend_30d",
    "cell_imbalance_count",
    "has_repeat_failure",
    "ticket_count_lifetime",
]


def run_cv(features: pd.DataFrame, labels: pd.DataFrame, telemetry: pd.DataFrame, tag: str, seed: int = 42):
    np.random.seed(seed)  # same splits across all ablations
    fold_metrics = []
    for i, split in enumerate(temporal_group_split(labels, telemetry)):
        y_train = encode_survival_labels(labels, split.train_site_ids)
        y_test = encode_survival_labels(labels, split.test_site_ids)

        X_train = features.loc[features.index.isin(split.train_site_ids)]
        X_test = features.loc[features.index.isin(split.test_site_ids)]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = train_survival_xgb(X_train, y_train, X_test, y_test)
        risk_scores = predict_risk(model, X_test)
        test_labels = labels[labels["site_id"].isin(split.test_site_ids)]
        cindex = compute_cindex(
            risk_scores,
            test_labels["time_to_event_months"].values,
            test_labels["event"].values,
        )
        fold_metrics.append(cindex)
        print(f"    Fold {i}: C-index = {cindex:.4f} (train={len(X_train)}, test={len(X_test)})")

    mean_c = np.mean(fold_metrics) if fold_metrics else 0.0
    std_c = np.std(fold_metrics) if fold_metrics else 0.0
    print(f"  [{tag}] Mean C-index: {mean_c:.4f} +/- {std_c:.4f}\n")
    return mean_c, std_c, fold_metrics


def main():
    print("Generating fleet data (200 sites, 36 months)...")
    data = simulate_fleet(n_sites=200, seed=42)
    telemetry, alarms, labels = data["telemetry"], data["alarms"], data["labels"]

    print(f"Raw labels: {len(labels)} rows, {labels['event'].sum()} events")
    print(f"Sites with multiple lifecycles: {(labels.groupby('site_id').size() > 1).sum()}")
    print(f"REPEAT_FAILURE_FLAG alarms: {(alarms['alarm_code'] == 'REPEAT_FAILURE_FLAG').sum()}")

    # Prepare per-lifecycle observations (group-constrained)
    telemetry, alarms, labels = prepare_lifecycle_observations(telemetry, alarms, labels)
    print(f"After lifecycle prep: {len(labels)} observations, {labels['event'].sum()} events")
    print("  (CV splits by original_site_id so all lifecycles of one site stay together)\n")

    set_alarms_data(alarms)
    all_features = compute_all_features(telemetry, labels)
    all_features.index = labels["site_id"].values

    print(f"Feature matrix: {all_features.shape}")
    print(f"has_repeat_failure nonzero: {(all_features['has_repeat_failure'] > 0).sum()} obs\n")

    # --- Ablation 1: Telemetry only ---
    print("=" * 60)
    print("ABLATION 1: Telemetry-only features (16)")
    print("=" * 60)
    telem_feats = all_features[[c for c in TELEMETRY_ONLY_FEATURES if c in all_features.columns]]
    m1, s1, _ = run_cv(telem_feats, labels, telemetry, "telemetry-only")

    # --- Ablation 2: All features (telemetry + alarms) ---
    print("=" * 60)
    print("ABLATION 2: Telemetry + Alarm features (22)")
    print("=" * 60)
    m2, s2, _ = run_cv(all_features, labels, telemetry, "telemetry+alarms")

    # --- Ablation 3: Alarm features only ---
    print("=" * 60)
    print("ABLATION 3: Alarm-only features (6)")
    print("=" * 60)
    alarm_feats = all_features[[c for c in ALARM_FEATURES if c in all_features.columns]]
    m3, s3, _ = run_cv(alarm_feats, labels, telemetry, "alarms-only")

    # --- Summary ---
    print("=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"  Telemetry only (16):     C-index = {m1:.4f} +/- {s1:.4f}")
    print(f"  Telemetry + Alarms (22): C-index = {m2:.4f} +/- {s2:.4f}")
    print(f"  Alarms only (6):         C-index = {m3:.4f} +/- {s3:.4f}")
    print(f"\n  Lift from adding alarms:  {m2 - m1:+.4f}")
    if m3 > 0.5:
        print(f"  Alarms alone beat random: {m3:.4f} > 0.50 (independent signal confirmed)")


if __name__ == "__main__":
    main()
