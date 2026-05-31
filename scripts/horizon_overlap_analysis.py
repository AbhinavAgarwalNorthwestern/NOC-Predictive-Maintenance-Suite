"""Do 24h and 48h models predict shared events similarly?

For drain events that occur within 24h (which are ALSO within the 48h window),
compare: does the 48h model catch them just as well as the dedicated 24h model?

Also: are there events the 48h model catches that 24h misses (drains at hours 25-48)?

Usage:
    uv run python scripts/horizon_overlap_analysis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

from battery_pdm.common.features import compute_features
from battery_pdm.synth.load_shedding import build_load_shedding_schedule


FEATURE_GROUPS = ["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"]
SAMPLE_INTERVAL_DAYS = 7


def build_labels_with_time(alarms: pd.DataFrame) -> pd.DataFrame:
    """Daily labels with exact hours_to_next_lvd for each observation."""
    alarms = alarms.sort_values(["site_id", "timestamp_h"]).reset_index(drop=True)
    max_h = alarms["timestamp_h"].max()

    lvd_events = alarms[alarms["alarm_code"] == "LOAD_DISCONNECT"][["site_id", "timestamp_h"]].copy()
    lvd_by_site = {}
    for _, row in lvd_events.iterrows():
        lvd_by_site.setdefault(row["site_id"], []).append(row["timestamp_h"])
    for site in lvd_by_site:
        lvd_by_site[site] = np.array(sorted(lvd_by_site[site]))

    labels = []
    for site_id, group in alarms.groupby("site_id"):
        site_start = group["timestamp_h"].min()
        site_lvds = lvd_by_site.get(site_id, np.array([]))
        obs_start = site_start + 30 * 24

        for ref_h in np.arange(obs_start, max_h - 48, SAMPLE_INTERVAL_DAYS * 24):
            # Find next LVD after ref_h
            future_lvds = site_lvds[site_lvds >= ref_h]
            if len(future_lvds) > 0:
                hours_to_next_lvd = float(future_lvds[0] - ref_h)
            else:
                hours_to_next_lvd = 9999.0

            labels.append({
                "site_id": site_id,
                "ref_time_h": float(ref_h),
                "hours_to_next_lvd": hours_to_next_lvd,
                "drain_24h": int(hours_to_next_lvd <= 24),
                "drain_48h": int(hours_to_next_lvd <= 48),
                "drain_24_to_48h": int(24 < hours_to_next_lvd <= 48),
            })

    return pd.DataFrame(labels)


def main(seed: int = 42, n_sites: int = 100):
    print("=" * 60)
    print("OVERLAP ANALYSIS: 24h vs 48h model on shared events")
    print("=" * 60)

    print("\nLoading data...")
    alarms = pd.read_parquet("outputs/alarms.parquet")
    site_static = pd.read_parquet("outputs/site_static.parquet")

    if n_sites and n_sites < len(site_static):
        sample_sites = site_static["site_id"].sample(n_sites, random_state=seed).tolist()
        alarms = alarms[alarms["site_id"].isin(sample_sites)].reset_index(drop=True)
        site_static = site_static[site_static["site_id"].isin(sample_sites)].reset_index(drop=True)

    print(f"  Alarms: {len(alarms):,}, Sites: {len(site_static)}")

    schedule = build_load_shedding_schedule(n_months=36, seed=seed)

    print("\nBuilding labels with exact hours-to-LVD...")
    labels = build_labels_with_time(alarms)
    print(f"  Total observations: {len(labels):,}")
    print(f"  Drain within 24h: {labels['drain_24h'].sum():,} ({labels['drain_24h'].mean():.2%})")
    print(f"  Drain within 48h: {labels['drain_48h'].sum():,} ({labels['drain_48h'].mean():.2%})")
    print(f"  Drain in 24-48h only: {labels['drain_24_to_48h'].sum():,} ({labels['drain_24_to_48h'].mean():.2%})")

    print("\nComputing features...")
    labels_for_feat = labels[["site_id", "ref_time_h"]].rename(columns={"ref_time_h": "mains_fail_h"})
    features = compute_features(
        labels=labels_for_feat, groups=FEATURE_GROUPS,
        inputs={"alarms": alarms, "site_static": site_static, "schedule": schedule},
        ref_time_col="mains_fail_h",
    )
    features = features.rename(columns={"mains_fail_h": "ref_time_h"})

    feature_cols = [
        c for c in features.columns
        if c not in ("site_id", "ref_time_h") and c not in ("charger_misconfigured", "aging_multiplier")
    ]

    merged = features.copy()
    for col in ["drain_24h", "drain_48h", "drain_24_to_48h", "hours_to_next_lvd"]:
        merged[col] = labels[col].values

    # Group-constrained split
    rng = np.random.default_rng(seed)
    unique_sites = merged["site_id"].unique()
    rng.shuffle(unique_sites)
    split_idx = int(len(unique_sites) * 0.75)
    train_sites = set(unique_sites[:split_idx])
    test_sites = set(unique_sites[split_idx:])
    train_mask = merged["site_id"].isin(train_sites).values
    test_mask = merged["site_id"].isin(test_sites).values

    X_train = merged.loc[train_mask, feature_cols].astype(float).fillna(0.0)
    X_test = merged.loc[test_mask, feature_cols].astype(float).fillna(0.0)

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Train 24h model
    y_train_24 = labels["drain_24h"].values[train_mask]
    scale_24 = max(1.0, (len(y_train_24) - y_train_24.sum()) / max(y_train_24.sum(), 1))
    params_24 = {**xgb_params, "scale_pos_weight": scale_24}
    booster_24 = xgb.train(
        params_24, xgb.DMatrix(X_train, label=y_train_24),
        num_boost_round=300, evals=[(xgb.DMatrix(X_train, label=y_train_24), "train")],
        early_stopping_rounds=30, verbose_eval=False,
    )

    # Train 48h model
    y_train_48 = labels["drain_48h"].values[train_mask]
    scale_48 = max(1.0, (len(y_train_48) - y_train_48.sum()) / max(y_train_48.sum(), 1))
    params_48 = {**xgb_params, "scale_pos_weight": scale_48}
    booster_48 = xgb.train(
        params_48, xgb.DMatrix(X_train, label=y_train_48),
        num_boost_round=300, evals=[(xgb.DMatrix(X_train, label=y_train_48), "train")],
        early_stopping_rounds=30, verbose_eval=False,
    )

    # Score test set
    scores_24 = booster_24.predict(xgb.DMatrix(X_test))
    scores_48 = booster_48.predict(xgb.DMatrix(X_test))

    y_test_24 = labels["drain_24h"].values[test_mask]
    y_test_48 = labels["drain_48h"].values[test_mask]
    y_test_24to48 = labels["drain_24_to_48h"].values[test_mask]
    hours_test = labels["hours_to_next_lvd"].values[test_mask]

    # --- Analysis ---
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    # 1. Overall AUC for each model on its own task
    print("\n--- Each model on its own task ---")
    print(f"  24h model on 24h labels: AUC={roc_auc_score(y_test_24, scores_24):.4f}  "
          f"AP={average_precision_score(y_test_24, scores_24):.4f}")
    print(f"  48h model on 48h labels: AUC={roc_auc_score(y_test_48, scores_48):.4f}  "
          f"AP={average_precision_score(y_test_48, scores_48):.4f}")

    # 2. Both models on the SHARED events (drain within 24h)
    print("\n--- Both models on SHARED events (drain within 24h) ---")
    print(f"  These {y_test_24.sum()} events are in BOTH the 24h and 48h positive sets")
    print(f"  24h model on 24h events: AUC={roc_auc_score(y_test_24, scores_24):.4f}")
    print(f"  48h model on 24h events: AUC={roc_auc_score(y_test_24, scores_48):.4f}")

    # 3. Mean predicted probability for shared events
    shared_mask = y_test_24 == 1
    print(f"\n--- Mean P(drain) for the {shared_mask.sum()} shared drain events ---")
    print(f"  24h model: {scores_24[shared_mask].mean():.4f}")
    print(f"  48h model: {scores_48[shared_mask].mean():.4f}")

    # 4. Events ONLY in 48h window (hours 24-48) - can 48h model catch these?
    only_48_mask = y_test_24to48 == 1
    no_drain_mask = (y_test_48 == 0)
    print(f"\n--- Events in 24-48h window ONLY ({only_48_mask.sum()} events) ---")
    print(f"  48h model mean P(drain): {scores_48[only_48_mask].mean():.4f}")
    print(f"  24h model mean P(drain): {scores_24[only_48_mask].mean():.4f}")
    print("  (24h model shouldn't flag these - they're outside its horizon)")

    # 5. Agreement at threshold 0.5
    pred_24 = (scores_24 >= 0.5).astype(int)
    pred_48 = (scores_48 >= 0.5).astype(int)

    both_flag = (pred_24 == 1) & (pred_48 == 1)
    only_24_flags = (pred_24 == 1) & (pred_48 == 0)
    only_48_flags = (pred_24 == 0) & (pred_48 == 1)
    neither = (pred_24 == 0) & (pred_48 == 0)

    print("\n--- Agreement (threshold=0.5) ---")
    print(f"  Both flag:       {both_flag.sum():,}")
    print(f"  Only 24h flags:  {only_24_flags.sum():,}")
    print(f"  Only 48h flags:  {only_48_flags.sum():,}")
    print(f"  Neither flags:   {neither.sum():,}")

    # Of the shared events (drain<24h), what fraction does each model catch?
    print("\n--- Recall on shared events (drain within 24h) at threshold 0.5 ---")
    print(f"  24h model catches: {pred_24[shared_mask].sum()}/{shared_mask.sum()} "
          f"({pred_24[shared_mask].mean():.1%})")
    print(f"  48h model catches: {pred_48[shared_mask].sum()}/{shared_mask.sum()} "
          f"({pred_48[shared_mask].mean():.1%})")

    # 6. Breakdown by hours-to-event bucket
    print("\n--- Score by hours-to-LVD bucket ---")
    print(f"  {'Bucket':<15} {'N':<6} {'24h_model_mean':<16} {'48h_model_mean':<16}")
    buckets = [(0, 6), (6, 12), (12, 24), (24, 36), (36, 48), (48, 168), (168, 9999)]
    for lo, hi in buckets:
        bucket_mask = (hours_test >= lo) & (hours_test < hi)
        n = bucket_mask.sum()
        if n > 0:
            label = f"{lo}-{hi}h"
            print(f"  {label:<15} {n:<6} {scores_24[bucket_mask].mean():<16.4f} "
                  f"{scores_48[bucket_mask].mean():<16.4f}")


if __name__ == "__main__":
    main()
