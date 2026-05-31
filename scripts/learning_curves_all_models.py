"""Learning curves: would more training events improve each model?

For each of the 3 models (drain predictor, failure, autonomy):
    - Hold out a FIXED test set (60 sites)
    - Train on subsets of remaining sites: 10%, 25%, 50%, 75%, 100%
    - Measure test performance at each fraction
    - Plot: if curve still rising → more data would help
            if curve plateaued → at task ceiling

Output: outputs/reports/learning_curves/curves.png + curves.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from battery_pdm.common.features import compute_features  # noqa: E402
from battery_pdm.common.survival import _cindex  # noqa: E402
from battery_pdm.synth.load_shedding import build_load_shedding_schedule  # noqa: E402

OUTPUTS = Path("outputs")
SEED = 42
N_SITES = 300
TRAIN_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]


def step(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


def main():
    print("Loading data...")
    alarms = pd.read_parquet(OUTPUTS / "alarms.parquet")
    sites_all = pd.read_parquet(OUTPUTS / "site_static.parquet")
    schedule = build_load_shedding_schedule(n_months=36, seed=SEED)

    sample = sites_all["site_id"].sample(N_SITES, random_state=SEED).tolist()
    alarms = alarms[alarms["site_id"].isin(sample)].reset_index(drop=True)
    sites = sites_all[sites_all["site_id"].isin(sample)].reset_index(drop=True)

    # FIXED test set: last 20% of sites (60 sites)
    rng = np.random.default_rng(SEED)
    sample_arr = np.array(sample)
    rng.shuffle(sample_arr)
    test_sites = set(sample_arr[int(N_SITES * 0.80):])      # 60 sites
    pool_sites = list(sample_arr[: int(N_SITES * 0.80)])    # 240 train pool

    print(f"  Total sites: {N_SITES}, Test set: {len(test_sites)} (fixed), Train pool: {len(pool_sites)}")

    results = []

    # =================================================================
    # MODEL 1: DRAIN PREDICTOR (binary 48h)
    # =================================================================
    step("MODEL 1: Drain predictor learning curve")

    # Build labels for ALL sites (we'll subset by site later)
    alarms_s = alarms.sort_values(["site_id", "timestamp_h"]).reset_index(drop=True)
    max_h = alarms_s["timestamp_h"].max()
    lvd_by_site = {}
    for _, r in alarms_s[alarms_s["alarm_code"] == "LOAD_DISCONNECT"].iterrows():
        lvd_by_site.setdefault(r["site_id"], []).append(r["timestamp_h"])
    for s in lvd_by_site:
        lvd_by_site[s] = np.array(sorted(lvd_by_site[s]))

    drain_labels = []
    for sid, g in alarms_s.groupby("site_id"):
        for ref_h in np.arange(g["timestamp_h"].min() + 30 * 24, max_h - 48, 7 * 24):
            future = lvd_by_site.get(sid, np.array([]))
            future = future[(future >= ref_h) & (future < ref_h + 48)]
            drain_labels.append({"site_id": sid, "ref_time_h": float(ref_h),
                                 "drain_event": int(len(future) > 0)})
    drain_labels = pd.DataFrame(drain_labels)
    drain_feats = compute_features(
        labels=drain_labels.rename(columns={"ref_time_h": "mains_fail_h"}),
        groups=["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"],
        inputs={"alarms": alarms, "site_static": sites, "schedule": schedule},
        ref_time_col="mains_fail_h",
    ).rename(columns={"mains_fail_h": "ref_time_h"})
    drain_feat_cols = [c for c in drain_feats.columns
                       if c not in ("site_id", "ref_time_h")
                       and c not in ("charger_misconfigured", "aging_multiplier")]

    test_mask = drain_feats["site_id"].isin(test_sites).values
    X_test = drain_feats.loc[test_mask, drain_feat_cols].astype(float).fillna(0.0)
    y_test = drain_labels["drain_event"].values[test_mask]
    print(f"  Test observations: {len(y_test)} ({y_test.sum()} positives)")

    for frac in TRAIN_FRACTIONS:
        n_train_sites = max(1, int(len(pool_sites) * frac))
        train_sites_subset = set(pool_sites[:n_train_sites])
        train_mask = drain_feats["site_id"].isin(train_sites_subset).values
        X_train = drain_feats.loc[train_mask, drain_feat_cols].astype(float).fillna(0.0)
        y_train = drain_labels["drain_event"].values[train_mask]

        if y_train.sum() < 5 or y_test.sum() == 0:
            continue
        scale_pos = max(1.0, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        booster = xgb.train(
            {"objective": "binary:logistic", "eval_metric": "aucpr",
             "tree_method": "hist", "max_depth": 5, "learning_rate": 0.05,
             "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
             "scale_pos_weight": scale_pos},
            xgb.DMatrix(X_train, label=y_train), num_boost_round=300,
            evals=[(xgb.DMatrix(X_test, label=y_test), "val")],
            early_stopping_rounds=30, verbose_eval=False,
        )
        scores = booster.predict(xgb.DMatrix(X_test))
        auc = roc_auc_score(y_test, scores)
        results.append({
            "model": "drain_predictor", "fraction": frac,
            "n_train_sites": n_train_sites, "n_train_obs": int(train_mask.sum()),
            "n_train_positives": int(y_train.sum()),
            "metric_name": "AUC", "metric_value": float(auc),
        })
        print(f"  frac={frac:.2f}: {n_train_sites} sites, {train_mask.sum()} obs, AUC={auc:.4f}")

    # =================================================================
    # MODEL 2: FAILURE MODEL (survival cox)
    # =================================================================
    step("MODEL 2: Failure model learning curve")
    labels_raw = pd.read_parquet(OUTPUTS / "labels.parquet")
    site_ids_set = set(sites["site_id"].values)
    if "original_site_id" not in labels_raw.columns and "lifecycle_id" in labels_raw.columns:
        labels_raw["original_site_id"] = labels_raw["site_id"]
        labels_raw["site_id"] = labels_raw["site_id"] + "_L" + labels_raw["lifecycle_id"].astype(int).astype(str)
    labels_raw = labels_raw[labels_raw.get("original_site_id", labels_raw["site_id"]).isin(site_ids_set)].reset_index(drop=True)

    alarm_parts = []
    for _, row in labels_raw.iterrows():
        original_sid = row.get("original_site_id", row["site_id"])
        win = alarms[
            (alarms["site_id"] == original_sid)
            & (alarms["timestamp_h"] >= row.get("lifecycle_start_h", 0))
            & (alarms["timestamp_h"] <= row["event_hour"])
        ].copy()
        win["site_id"] = row["site_id"]
        alarm_parts.append(win)
    alarms_w = pd.concat(alarm_parts, ignore_index=True) if alarm_parts else alarms.iloc[:0]
    static_lc = labels_raw[["site_id", "original_site_id"]].merge(
        sites, left_on="original_site_id", right_on="site_id", suffixes=("", "_static"),
    ).drop(columns=["site_id_static"])

    fail_feats = compute_features(
        labels=labels_raw, groups=["alarm_history", "site_static", "soc_proxy"],
        inputs={"alarms": alarms_w, "site_static": static_lc}, ref_time_col="event_hour",
    )
    fail_cols = [c for c in fail_feats.columns
                 if c not in ("site_id", "event_hour")
                 and c not in ("charger_misconfigured", "aging_multiplier")]
    label_only = [c for c in labels_raw.columns
                  if c not in fail_feats.columns or c in ("site_id", "event_hour")]
    merged_fail = fail_feats.merge(labels_raw[label_only], on=["site_id", "event_hour"], how="inner")

    groups_orig = merged_fail.get("original_site_id", merged_fail["site_id"]).values
    test_m_fail = np.array([g in test_sites for g in groups_orig])
    X_te_fail = merged_fail.loc[test_m_fail, fail_cols].astype(float).fillna(0.0)
    times_te = merged_fail.loc[test_m_fail, "time_to_event_months"].values
    events_te = merged_fail.loc[test_m_fail, "event"].values
    print(f"  Test lifecycles: {test_m_fail.sum()} ({events_te.sum()} failures)")

    for frac in TRAIN_FRACTIONS:
        n_train_sites = max(1, int(len(pool_sites) * frac))
        train_sites_subset = set(pool_sites[:n_train_sites])
        train_m_fail = np.array([g in train_sites_subset for g in groups_orig])
        X_tr_fail = merged_fail.loc[train_m_fail, fail_cols].astype(float).fillna(0.0)
        y_tr = merged_fail.loc[train_m_fail, "time_to_event_months"].values.astype(float).copy()
        y_tr[merged_fail.loc[train_m_fail, "event"].values == 0] *= -1

        if train_m_fail.sum() < 10 or test_m_fail.sum() == 0:
            continue
        booster = xgb.train(
            {"objective": "survival:cox", "eval_metric": "cox-nloglik",
             "tree_method": "hist", "max_depth": 4, "learning_rate": 0.05,
             "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8},
            xgb.DMatrix(X_tr_fail, label=y_tr), num_boost_round=200, verbose_eval=False,
        )
        risk = booster.predict(xgb.DMatrix(X_te_fail))
        cindex = _cindex(risk, times_te, events_te)
        results.append({
            "model": "failure_model", "fraction": frac,
            "n_train_sites": n_train_sites, "n_train_obs": int(train_m_fail.sum()),
            "n_train_positives": int(merged_fail.loc[train_m_fail, "event"].sum()),
            "metric_name": "C-index", "metric_value": float(cindex),
        })
        print(f"  frac={frac:.2f}: {n_train_sites} sites, {train_m_fail.sum()} lifecycles, "
              f"C-index={cindex:.4f}")

    # Autonomy model was decommissioned — drain predictor subsumes its functionality.

    # =================================================================
    # PLOT
    # =================================================================
    step("PLOTTING LEARNING CURVES")
    df = pd.DataFrame(results)
    out_dir = OUTPUTS / "reports" / "learning_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].sort_values("n_train_obs")
        ax.plot(sub["n_train_obs"], sub["metric_value"], "o-",
                linewidth=2, markersize=8, label=f"{model} ({sub['metric_name'].iloc[0]})")
    ax.set_xlabel("Training observations")
    ax.set_ylabel("Test metric (AUC for binary, C-index for survival)")
    ax.set_title("Learning curves — would MORE data help?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(out_dir / "curves.png", dpi=120)
    plt.close()
    print(f"  Saved curves.png + curves.csv to {out_dir}/")

    print("\n  INTERPRETATION GUIDE:")
    print("    - Curve still rising at 100% → MORE EVENTS WOULD HELP (data-limited)")
    print("    - Curve plateaued → NOT DATA-LIMITED (feature/task ceiling)")
    print("    - Look at the slope at the rightmost point in the plot")


if __name__ == "__main__":
    main()
