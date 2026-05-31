"""Compare signal strength at 24h vs 48h BEFORE the same drain events.

The question: for the SAME battery that eventually drains, is our model more
confident when we ask 24h before vs 48h before?

Approach:
  1. Find all actual LVD events in the alarm stream
  2. For each LVD, create two observations: one at (lvd_time - 24h) and one at (lvd_time - 48h)
  3. Compute features at both points
  4. Score with the same model — compare P(drain) at 24h-before vs 48h-before
  5. Also add negative controls (random times where no LVD occurred)

Usage:
    uv run python scripts/horizon_signal_strength.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from battery_pdm.common.features import compute_features
from battery_pdm.synth.load_shedding import build_load_shedding_schedule


FEATURE_GROUPS = ["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"]


def main(seed: int = 42, n_sites: int = 100):
    print("=" * 60)
    print("SIGNAL STRENGTH: 24h vs 48h before the SAME drain event")
    print("=" * 60)

    print("\nLoading data...")
    alarms = pd.read_parquet("outputs/alarms.parquet")
    site_static = pd.read_parquet("outputs/site_static.parquet")

    if n_sites and n_sites < len(site_static):
        rng = np.random.default_rng(seed)
        sample_sites = site_static["site_id"].sample(n_sites, random_state=seed).tolist()
        alarms = alarms[alarms["site_id"].isin(sample_sites)].reset_index(drop=True)
        site_static = site_static[site_static["site_id"].isin(sample_sites)].reset_index(drop=True)

    print(f"  Alarms: {len(alarms):,}, Sites: {len(site_static)}")

    schedule = build_load_shedding_schedule(n_months=36, seed=seed)

    # Find all LVD events (minimum 48h after first alarm for that site, so features exist)
    lvd_events = alarms[alarms["alarm_code"] == "LOAD_DISCONNECT"].copy()
    site_first_alarm = alarms.groupby("site_id")["timestamp_h"].min().to_dict()
    lvd_events = lvd_events[
        lvd_events.apply(lambda r: r["timestamp_h"] - site_first_alarm.get(r["site_id"], 0) > 30 * 24, axis=1)
    ].reset_index(drop=True)
    print(f"\n  LVD events (after 30d warmup): {len(lvd_events):,}")

    # For each LVD, create observations at T-24h and T-48h
    obs_24h = []
    obs_48h = []
    for _, row in lvd_events.iterrows():
        obs_24h.append({"site_id": row["site_id"], "ref_time_h": row["timestamp_h"] - 24.0, "label": 1})
        obs_48h.append({"site_id": row["site_id"], "ref_time_h": row["timestamp_h"] - 48.0, "label": 1})

    # Add negative controls: random times where no LVD occurred within 48h
    rng = np.random.default_rng(seed)
    max_h = alarms["timestamp_h"].max()
    lvd_lookup = {}
    for _, row in lvd_events.iterrows():
        lvd_lookup.setdefault(row["site_id"], []).append(row["timestamp_h"])

    n_negatives = len(lvd_events)
    neg_count = 0
    attempts = 0
    while neg_count < n_negatives and attempts < n_negatives * 10:
        site = rng.choice(site_static["site_id"].values)
        t = float(rng.uniform(30 * 24, max_h - 48))
        # Check no LVD within 48h of this time
        site_lvds = lvd_lookup.get(site, [])
        if any(abs(t - lvd_t) < 48 for lvd_t in site_lvds):
            attempts += 1
            continue
        obs_24h.append({"site_id": site, "ref_time_h": t, "label": 0})
        obs_48h.append({"site_id": site, "ref_time_h": t, "label": 0})
        neg_count += 1
        attempts += 1

    df_24h = pd.DataFrame(obs_24h)
    df_48h = pd.DataFrame(obs_48h)
    print(f"  Observations per horizon: {len(df_24h):,} ({df_24h['label'].sum()} positives, {neg_count} negatives)")

    # Compute features at 24h-before
    print("\n  Computing features at T-24h...")
    labels_24 = df_24h[["site_id", "ref_time_h"]].rename(columns={"ref_time_h": "mains_fail_h"})
    feat_24 = compute_features(
        labels=labels_24, groups=FEATURE_GROUPS,
        inputs={"alarms": alarms, "site_static": site_static, "schedule": schedule},
        ref_time_col="mains_fail_h",
    )

    # Compute features at 48h-before
    print("  Computing features at T-48h...")
    labels_48 = df_48h[["site_id", "ref_time_h"]].rename(columns={"ref_time_h": "mains_fail_h"})
    feat_48 = compute_features(
        labels=labels_48, groups=FEATURE_GROUPS,
        inputs={"alarms": alarms, "site_static": site_static, "schedule": schedule},
        ref_time_col="mains_fail_h",
    )

    feature_cols = [
        c for c in feat_24.columns
        if c not in ("site_id", "mains_fail_h") and c not in ("charger_misconfigured", "aging_multiplier")
    ]

    # Train a single model on 48h data, then evaluate on both horizons
    import xgboost as xgb

    y_24 = df_24h["label"].values
    y_48 = df_48h["label"].values
    X_24 = feat_24[feature_cols].astype(float).fillna(0.0)
    X_48 = feat_48[feature_cols].astype(float).fillna(0.0)

    # Group split for training (train on 75% of sites, test on 25%)
    unique_sites = df_24h["site_id"].unique()
    rng.shuffle(unique_sites)
    split_idx = int(len(unique_sites) * 0.75)
    train_sites = set(unique_sites[:split_idx])
    test_sites = set(unique_sites[split_idx:])

    train_mask = df_24h["site_id"].isin(train_sites).values
    test_mask = df_24h["site_id"].isin(test_sites).values

    # Train on 48h observations (more conservative — model learns "at risk" patterns)
    X_train = X_48.loc[train_mask]
    y_train = y_48[train_mask]

    scale_pos = max(1.0, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
    dtrain = xgb.DMatrix(X_train, label=y_train)
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "max_depth": 5,
            "learning_rate": 0.05,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos,
        },
        dtrain, num_boost_round=200,
        evals=[(dtrain, "train")],
        verbose_eval=False,
    )

    # Score both 24h and 48h test sets with the SAME model
    X_test_24 = X_24.loc[test_mask]
    X_test_48 = X_48.loc[test_mask]
    y_test = y_24[test_mask]  # same labels for same events

    scores_24 = booster.predict(xgb.DMatrix(X_test_24))
    scores_48 = booster.predict(xgb.DMatrix(X_test_48))

    # Compare
    print(f"\n{'='*60}")
    print("SAME MODEL, SAME EVENTS -- scored at two time points:")
    print(f"{'='*60}")

    auc_24 = roc_auc_score(y_test, scores_24) if len(np.unique(y_test)) > 1 else float("nan")
    auc_48 = roc_auc_score(y_test, scores_48) if len(np.unique(y_test)) > 1 else float("nan")
    ap_24 = average_precision_score(y_test, scores_24) if len(np.unique(y_test)) > 1 else float("nan")
    ap_48 = average_precision_score(y_test, scores_48) if len(np.unique(y_test)) > 1 else float("nan")

    print(f"\n  {'Metric':<20} {'At T-24h':<12} {'At T-48h':<12} {'Lift (24h over 48h)'}")
    print(f"  {'ROC AUC':<20} {auc_24:<12.4f} {auc_48:<12.4f} {auc_24 - auc_48:+.4f}")
    print(f"  {'Avg Precision':<20} {ap_24:<12.4f} {ap_48:<12.4f} {ap_24 - ap_48:+.4f}")

    # Mean predicted probability for TRUE positives at each horizon
    pos_mask = y_test == 1
    neg_mask = y_test == 0
    print("\n  Mean P(drain) for TRUE drain events:")
    print(f"    At T-24h (closer):  {scores_24[pos_mask].mean():.4f}")
    print(f"    At T-48h (further): {scores_48[pos_mask].mean():.4f}")
    print("\n  Mean P(drain) for TRUE negatives:")
    print(f"    At T-24h: {scores_24[neg_mask].mean():.4f}")
    print(f"    At T-48h: {scores_48[neg_mask].mean():.4f}")

    # Separation: difference in mean score between positives and negatives
    sep_24 = scores_24[pos_mask].mean() - scores_24[neg_mask].mean()
    sep_48 = scores_48[pos_mask].mean() - scores_48[neg_mask].mean()
    print("\n  Score separation (pos_mean - neg_mean):")
    print(f"    At T-24h: {sep_24:.4f}")
    print(f"    At T-48h: {sep_48:.4f}")
    print(f"    --> {'24h has stronger signal' if sep_24 > sep_48 else '48h has stronger signal'}")

    # Key features that differ between 24h and 48h observations for the SAME events
    print("\n  Feature differences (positives only, T-24h vs T-48h):")
    pos_24 = X_24.loc[test_mask][pos_mask]
    pos_48 = X_48.loc[test_mask][pos_mask]
    diffs = (pos_24.mean() - pos_48.mean()).abs().sort_values(ascending=False)
    for feat in diffs.head(10).index:
        print(f"    {feat:<35} 24h_mean={pos_24[feat].mean():.3f}  48h_mean={pos_48[feat].mean():.3f}  "
              f"diff={pos_24[feat].mean() - pos_48[feat].mean():+.3f}")


if __name__ == "__main__":
    main()
