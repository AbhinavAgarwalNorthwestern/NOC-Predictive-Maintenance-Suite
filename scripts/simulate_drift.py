"""Simulate a realistic drift scenario and demonstrate detection.

Scenario: "Peshawar Grid Stabilizer Upgrade"
    At month 24, Peshawar installs grid stabilizers. Effect:
    - AC_MAINS_FAIL alarms drop by 60% for Peshawar sites
    - RECTIFIER_FAULT alarms drop by 40%
    - The drain predictor model was trained on pre-upgrade data, so its
      feature distributions no longer match production.

This script:
    1. Builds a reference profile from "training time" (months 0-24)
    2. Creates a drifted dataset (months 24-36 with reduced Peshawar alarms)
    3. Runs drift detection and shows the monitor catching it
    4. Shows that model performance degrades on the shifted data

Usage:
    uv run python scripts/simulate_drift.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from battery_pdm.monitoring.drift import (
    build_reference_profile,
    detect_drift,
    load_reference_profile,
)
from battery_pdm.common.features import compute_features
from battery_pdm.synth.load_shedding import build_load_shedding_schedule


FEATURE_GROUPS = ["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"]
DRIFT_START_H = 24 * 30 * 24  # month 24 in hours (17280h)


def inject_peshawar_upgrade(alarms: pd.DataFrame, site_static: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simulate grid stabilizer upgrade: drop alarm rates for Peshawar sites after DRIFT_START_H."""
    rng = np.random.default_rng(seed)
    peshawar_sites = set(site_static[site_static["region"] == "peshawar"]["site_id"])

    # Identify post-upgrade alarms from Peshawar
    mask_peshawar_post = (
        alarms["site_id"].isin(peshawar_sites)
        & (alarms["timestamp_h"] >= DRIFT_START_H)
    )

    # Drop 60% of AC_MAINS_FAIL and 40% of RECTIFIER_FAULT for these sites
    drop_mask = pd.Series(False, index=alarms.index)

    mf_mask = mask_peshawar_post & (alarms["alarm_code"] == "AC_MAINS_FAIL")
    mf_drop = rng.random(mf_mask.sum()) < 0.60
    drop_mask.loc[alarms.index[mf_mask][mf_drop]] = True

    rf_mask = mask_peshawar_post & (alarms["alarm_code"] == "RECTIFIER_FAULT")
    rf_drop = rng.random(rf_mask.sum()) < 0.40
    drop_mask.loc[alarms.index[rf_mask][rf_drop]] = True

    # Also drop some downstream alarms (BATT_UNDERVOLTAGE, LOAD_DISCONNECT)
    uv_mask = mask_peshawar_post & (alarms["alarm_code"] == "BATT_UNDERVOLTAGE")
    uv_drop = rng.random(uv_mask.sum()) < 0.30
    drop_mask.loc[alarms.index[uv_mask][uv_drop]] = True

    lvd_mask = mask_peshawar_post & (alarms["alarm_code"] == "LOAD_DISCONNECT")
    lvd_drop = rng.random(lvd_mask.sum()) < 0.25
    drop_mask.loc[alarms.index[lvd_mask][lvd_drop]] = True

    alarms_drifted = alarms[~drop_mask].reset_index(drop=True)

    n_dropped = drop_mask.sum()
    n_peshawar_post = mask_peshawar_post.sum()
    print(f"  Peshawar upgrade: dropped {n_dropped:,} / {n_peshawar_post:,} "
          f"post-upgrade alarms ({n_dropped/max(n_peshawar_post,1):.0%})")

    return alarms_drifted


def main(seed: int = 42, n_sites: int = 100):
    print("=" * 60)
    print("DRIFT SIMULATION: Peshawar Grid Stabilizer Upgrade")
    print("=" * 60)

    print("\nLoading data...")
    alarms = pd.read_parquet("outputs/alarms.parquet")
    site_static = pd.read_parquet("outputs/site_static.parquet")

    if n_sites and n_sites < len(site_static):
        sample_sites = site_static["site_id"].sample(n_sites, random_state=seed).tolist()
        alarms = alarms[alarms["site_id"].isin(sample_sites)].reset_index(drop=True)
        site_static = site_static[site_static["site_id"].isin(sample_sites)].reset_index(drop=True)

    schedule = build_load_shedding_schedule(n_months=36, seed=seed)
    print(f"  Alarms: {len(alarms):,}, Sites: {len(site_static)}")

    n_peshawar = (site_static["region"] == "peshawar").sum()
    print(f"  Peshawar sites: {n_peshawar}")

    # --- Step 1: Build reference profile from pre-upgrade data (month 18-24) ---
    print("\n--- Step 1: Build reference profile (months 18-24, pre-upgrade) ---")
    ref_time_h = DRIFT_START_H  # score at month 24
    labels_ref = pd.DataFrame({
        "site_id": site_static["site_id"].values,
        "ref_time_h": float(ref_time_h),
    })
    alarms_pre = alarms[alarms["timestamp_h"] <= ref_time_h]
    features_ref = compute_features(
        labels=labels_ref.rename(columns={"ref_time_h": "mains_fail_h"}),
        groups=FEATURE_GROUPS,
        inputs={"alarms": alarms_pre, "site_static": site_static, "schedule": schedule},
        ref_time_col="mains_fail_h",
    )
    features_ref = features_ref.rename(columns={"mains_fail_h": "ref_time_h"})

    feature_cols = [
        c for c in features_ref.columns
        if c not in ("site_id", "ref_time_h") and c not in ("charger_misconfigured", "aging_multiplier")
    ]

    # Score with drain model for prediction baseline
    import xgboost as xgb
    model_path = Path("outputs/models/drain_predictor_48h")
    meta = json.loads((model_path / "meta.json").read_text())
    booster = xgb.Booster()
    booster.load_model(str(model_path / "booster.json"))
    model_feature_cols = meta["feature_cols"]
    X_ref = features_ref[model_feature_cols].astype(float).fillna(0.0)
    pred_ref = booster.predict(xgb.DMatrix(X_ref))

    profile_path = "outputs/models/reference_profile_drift_demo.json"
    build_reference_profile(features_ref, feature_cols, predictions=pred_ref, output_path=profile_path)
    print(f"  Reference profile saved ({len(features_ref)} samples)")
    print(f"  Reference prediction mean: {pred_ref.mean():.4f}")

    # --- Step 2: Inject drift (Peshawar upgrade) ---
    print("\n--- Step 2: Inject Peshawar grid upgrade at month 24 ---")
    alarms_drifted = inject_peshawar_upgrade(alarms, site_static, seed=seed)

    # --- Step 3: Compute features at month 30 (6 months post-upgrade) ---
    print("\n--- Step 3: Compute features at month 30 (post-upgrade) ---")
    post_time_h = 30 * 30 * 24  # month 30
    labels_post = pd.DataFrame({
        "site_id": site_static["site_id"].values,
        "ref_time_h": float(post_time_h),
    })
    alarms_post_pit = alarms_drifted[alarms_drifted["timestamp_h"] <= post_time_h]
    features_post = compute_features(
        labels=labels_post.rename(columns={"ref_time_h": "mains_fail_h"}),
        groups=FEATURE_GROUPS,
        inputs={"alarms": alarms_post_pit, "site_static": site_static, "schedule": schedule},
        ref_time_col="mains_fail_h",
    )
    features_post = features_post.rename(columns={"mains_fail_h": "ref_time_h"})

    X_post = features_post[model_feature_cols].astype(float).fillna(0.0)
    pred_post = booster.predict(xgb.DMatrix(X_post))
    print(f"  Post-upgrade prediction mean: {pred_post.mean():.4f} (was {pred_ref.mean():.4f})")

    # --- Step 4: Run drift detection ---
    print("\n--- Step 4: Drift detection ---")
    reference_profile = load_reference_profile(profile_path)
    drift_report = detect_drift(
        reference_profile=reference_profile,
        current_features=features_post,
        current_predictions=pred_post,
        feature_cols=feature_cols,
    )

    summary = drift_report["summary"]
    print(f"\n  Status: {summary['status']}")
    print(f"  Features monitored: {summary['n_features_monitored']}")
    print(f"  Significant drift: {summary['n_significant_drift']}")
    print(f"  Moderate drift: {summary['n_moderate_drift']}")
    print(f"  Retrain recommended: {summary['retrain_recommended']}")
    if summary["reasons"]:
        for r in summary["reasons"]:
            print(f"    -> {r}")

    # Show top drifted features
    feature_drift = drift_report["feature_drift"]
    drifted = feature_drift[feature_drift["drift_level"] != "STABLE"]
    if not drifted.empty:
        print("\n  Drifted features:")
        print(drifted[["feature", "psi", "drift_level", "ref_mean", "cur_mean", "mean_shift_std"]]
              .head(15).to_string(index=False))

    # Prediction drift
    pred_drift = drift_report["prediction_drift"]
    if pred_drift:
        print("\n  Prediction drift:")
        print(f"    PSI: {pred_drift.get('prediction_psi', 0):.4f}")
        print(f"    Ref mean score: {pred_drift.get('ref_mean_score', 0):.4f}")
        print(f"    Cur mean score: {pred_drift.get('cur_mean_score', 0):.4f}")
        print(f"    Drift detected: {pred_drift.get('drift_detected', False)}")

    # --- Step 5: Show Peshawar vs other regions ---
    print("\n--- Step 5: Regional breakdown ---")
    peshawar_sites = set(site_static[site_static["region"] == "peshawar"]["site_id"])
    other_sites = set(site_static[~site_static["region"].isin(["peshawar"])]["site_id"])

    pesh_mask = features_post["site_id"].isin(peshawar_sites)
    other_mask = features_post["site_id"].isin(other_sites)

    key_features = ["mains_fail_count_30d", "rectifier_fault_count_30d", "lvd_count_30d", "estimated_soc_at_ref"]
    print(f"\n  {'Feature':<30} {'Peshawar (post)':<18} {'Others (post)':<18} {'Peshawar (ref)':<18}")
    for f in key_features:
        if f in features_post.columns and f in features_ref.columns:
            pesh_ref = features_ref.loc[features_ref["site_id"].isin(peshawar_sites), f].mean()
            pesh_post = features_post.loc[pesh_mask, f].mean()
            other_post = features_post.loc[other_mask, f].mean()
            print(f"  {f:<30} {pesh_post:<18.2f} {other_post:<18.2f} {pesh_ref:<18.2f}")

    print("\n  Risk scores:")
    print(f"    Peshawar pre-upgrade:  {pred_ref[features_ref['site_id'].isin(peshawar_sites)].mean():.4f}")
    print(f"    Peshawar post-upgrade: {pred_post[pesh_mask.values].mean():.4f}")
    print(f"    Other regions (stable): {pred_post[other_mask.values].mean():.4f}")

    # Save output
    out_dir = Path("outputs/drift_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "drift_simulation_report.json"
    report_file.write_text(json.dumps({
        "scenario": "peshawar_grid_stabilizer_upgrade",
        "drift_start_month": 24,
        "detection_month": 30,
        "summary": summary,
        "prediction_drift": pred_drift,
        "top_drifted_features": feature_drift.head(10).to_dict(orient="records") if not feature_drift.empty else [],
    }, indent=2))
    print(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()
