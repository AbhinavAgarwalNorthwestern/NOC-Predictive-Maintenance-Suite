"""Decision strategy comparison across all three models.

Binary classifier (drain predictor): threshold-based decisions make sense.
Survival models (failure, autonomy): output hazard scores; the right operational
question is top-K ranking, not thresholding.

Outputs a unified table comparing:
    - Drain predictor: default 0.5 / F1-max / cost-aware / top-K(200, 500)
    - Failure model:   top-K(20, 50, 100) replacements per month — precision/recall@K
    - Autonomy model:  top-K(10, 25, 50) dispatch slots — precision/recall@K

All numbers are auditable (TP/FP/FN reported).
"""

from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix,
)

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from battery_pdm.common.features import compute_features
from battery_pdm.monitoring.threshold import (
    CostMatrix, optimal_threshold, top_k_dispatch,
)
from battery_pdm.synth.load_shedding import build_load_shedding_schedule

OUTPUTS = Path("outputs")
SEED = 42
N_SITES = 300   # MUST match run_end_to_end.py so the drain model on disk fits this sample


def step(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


def main():
    rng = np.random.default_rng(SEED)
    print("Loading data...")
    alarms = pd.read_parquet(OUTPUTS / "alarms.parquet")
    sites = pd.read_parquet(OUTPUTS / "site_static.parquet")
    schedule = build_load_shedding_schedule(n_months=36, seed=SEED)

    sample = sites["site_id"].sample(N_SITES, random_state=SEED).tolist()
    alarms = alarms[alarms["site_id"].isin(sample)].reset_index(drop=True)
    sites = sites[sites["site_id"].isin(sample)].reset_index(drop=True)
    print(f"  {len(alarms):,} alarms, {len(sites)} sites")

    # ==================================================================
    # MODEL 1: DRAIN PREDICTOR (binary classifier) — threshold applies
    # ==================================================================
    step("MODEL 1: DRAIN PREDICTOR (48h binary)")

    # Load trained model + calibrator
    import json
    from battery_pdm.monitoring.model_registry import load_calibrator, apply_calibrator
    model_dir = OUTPUTS / "models" / "drain_predictor_48h"
    meta = json.loads((model_dir / "meta.json").read_text())
    booster = xgb.Booster(); booster.load_model(str(model_dir / "booster.json"))
    calibrator = load_calibrator(model_dir)

    # Build daily labels + features (matching the training pipeline)
    alarms_s = alarms.sort_values(["site_id", "timestamp_h"]).reset_index(drop=True)
    max_h = alarms_s["timestamp_h"].max()
    lvd_by_site = {}
    for _, r in alarms_s[alarms_s["alarm_code"] == "LOAD_DISCONNECT"].iterrows():
        lvd_by_site.setdefault(r["site_id"], []).append(r["timestamp_h"])
    for s in lvd_by_site:
        lvd_by_site[s] = np.array(sorted(lvd_by_site[s]))
    labels = []
    for sid, g in alarms_s.groupby("site_id"):
        for ref_h in np.arange(g["timestamp_h"].min() + 30*24, max_h - 48, 7*24):
            future = lvd_by_site.get(sid, np.array([]))
            future = future[(future >= ref_h) & (future < ref_h + 48)]
            labels.append({"site_id": sid, "ref_time_h": float(ref_h),
                           "drain_event": int(len(future) > 0)})
    labels = pd.DataFrame(labels)
    features = compute_features(
        labels=labels.rename(columns={"ref_time_h": "mains_fail_h"}),
        groups=["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"],
        inputs={"alarms": alarms, "site_static": sites, "schedule": schedule},
        ref_time_col="mains_fail_h",
    ).rename(columns={"mains_fail_h": "ref_time_h"})
    feature_cols = meta["feature_cols"]
    X = features[feature_cols].astype(float).fillna(0.0)

    # Group-constrained test set (last 20% of sites)
    rng.shuffle(sample)
    test_sites = set(sample[int(len(sample) * 0.80):])
    test_mask = features["site_id"].isin(test_sites).values
    X_test, y_test = X[test_mask], labels["drain_event"].values[test_mask]
    raw = booster.predict(xgb.DMatrix(X_test))
    scored = apply_calibrator(calibrator, raw)
    auc = roc_auc_score(y_test, scored) if len(np.unique(y_test)) > 1 else float("nan")
    print(f"  Test size: {len(y_test)}, positives: {y_test.sum()}, AUC: {auc:.4f}")

    drain_rows = []
    cost = CostMatrix(false_positive=200.0, false_negative=1000.0)

    # Strategy 1: Default 0.5
    for label_strategy, thr_chooser in [
        ("Default 0.5", lambda: 0.5),
        ("F1-max", lambda: float(np.arange(0.01, 1.0, 0.01)[
            int(np.argmax([f1_score(y_test, (scored >= t).astype(int), zero_division=0)
                           for t in np.arange(0.01, 1.0, 0.01)]))])),
        ("Cost-optimal (FP=$200,FN=$1000)", lambda: optimal_threshold(scored, y_test, cost)["best_threshold"]),
    ]:
        thr = thr_chooser()
        pred = (scored >= thr).astype(int)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        drain_rows.append({
            "model": "drain_predictor",
            "strategy": label_strategy,
            "param": f"thr={thr:.3f}",
            "n_flagged": int(pred.sum()),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "cost_usd": float(fp * 200 + fn * 1000),
        })

    # Strategy 4: Top-K
    for k in [50, 200, 500]:
        if k > len(y_test):
            continue
        r = top_k_dispatch(scored, y_test, k, cost)
        drain_rows.append({
            "model": "drain_predictor",
            "strategy": "Top-K",
            "param": f"K={k} (thr={r['effective_threshold']:.3f})",
            "n_flagged": k,
            "TP": r["true_positives"], "FP": r["false_positives"],
            "FN": r["false_negatives"], "TN": len(y_test) - k - r["false_negatives"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "cost_usd": r["cost_at_top_k"],
        })

    drain_df = pd.DataFrame(drain_rows)
    print(drain_df.to_string(index=False))

    # ==================================================================
    # MODEL 2: FAILURE MODEL (survival:cox) — top-K only, no threshold
    # ==================================================================
    step("MODEL 2: FAILURE MODEL (lifetime replacement ranking)")
    print("  Threshold concept does NOT apply (output is a hazard score, not P).")
    print("  Operational use: top-K monthly replacement budget.")

    labels_raw = pd.read_parquet(OUTPUTS / "labels.parquet")
    site_ids = set(sites["site_id"].values)
    if "original_site_id" not in labels_raw.columns and "lifecycle_id" in labels_raw.columns:
        labels_raw["original_site_id"] = labels_raw["site_id"]
        labels_raw["site_id"] = labels_raw["site_id"] + "_L" + labels_raw["lifecycle_id"].astype(int).astype(str)
    labels_raw = labels_raw[labels_raw.get("original_site_id", labels_raw["site_id"]).isin(site_ids)].reset_index(drop=True)
    print(f"  Lifecycles: {len(labels_raw)}, events: {labels_raw['event'].sum()}")

    # Window alarms per lifecycle (matching production)
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
    alarms_windowed = pd.concat(alarm_parts, ignore_index=True) if alarm_parts else alarms.iloc[:0]
    static_lc = labels_raw[["site_id", "original_site_id"]].merge(
        sites, left_on="original_site_id", right_on="site_id", suffixes=("", "_static"),
    ).drop(columns=["site_id_static"])

    fail_features = compute_features(
        labels=labels_raw, groups=["alarm_history", "site_static", "soc_proxy"],
        inputs={"alarms": alarms_windowed, "site_static": static_lc},
        ref_time_col="event_hour",
    )
    fail_feature_cols = [c for c in fail_features.columns
                        if c not in ("site_id", "event_hour")
                        and c not in ("charger_misconfigured", "aging_multiplier")]
    label_only = [c for c in labels_raw.columns if c not in fail_features.columns or c in ("site_id", "event_hour")]
    merged = fail_features.merge(labels_raw[label_only], on=["site_id", "event_hour"], how="inner")

    # Train fresh Cox model on 80% of groups (sites), evaluate on 20%
    groups = merged.get("original_site_id", merged["site_id"]).values
    unique_g = np.array(sorted(set(groups)))
    rng2 = np.random.default_rng(SEED)
    rng2.shuffle(unique_g)
    train_g = set(unique_g[:int(len(unique_g) * 0.80)])
    train_m = np.array([g in train_g for g in groups])
    test_m = ~train_m

    def encode_y(mask):
        y = merged.loc[mask, "time_to_event_months"].values.astype(float).copy()
        y[merged.loc[mask, "event"].values == 0] *= -1
        return y

    Xf_train = merged.loc[train_m, fail_feature_cols].astype(float).fillna(0.0)
    Xf_test = merged.loc[test_m, fail_feature_cols].astype(float).fillna(0.0)
    yf_train = encode_y(train_m)

    booster_fail = xgb.train(
        {"objective": "survival:cox", "eval_metric": "cox-nloglik", "tree_method": "hist",
         "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5,
         "subsample": 0.8, "colsample_bytree": 0.8},
        xgb.DMatrix(Xf_train, label=yf_train), num_boost_round=200, verbose_eval=False,
    )
    risk = booster_fail.predict(xgb.DMatrix(Xf_test))
    events_test = merged.loc[test_m, "event"].values
    times_test = merged.loc[test_m, "time_to_event_months"].values

    # "Real failure within HORIZON months" — the operational question
    HORIZON_MONTHS = 6  # rolling 6-month replacement planning window
    y_fail = ((events_test == 1) & (times_test <= HORIZON_MONTHS)).astype(int)
    print(f"  Test lifecycles: {len(y_fail)}, failures within {HORIZON_MONTHS}mo: {y_fail.sum()}")

    fail_rows = []
    for k in [10, 25, 50, 100]:
        if k > len(risk):
            continue
        # Rank by risk descending, take top K
        order = np.argsort(-risk)
        pred = np.zeros_like(y_fail)
        pred[order[:k]] = 1
        tp = int(((pred == 1) & (y_fail == 1)).sum())
        fp = int(((pred == 1) & (y_fail == 0)).sum())
        fn = int(((pred == 0) & (y_fail == 1)).sum())
        fail_rows.append({
            "model": "failure_model",
            "strategy": "Top-K (monthly replacement)",
            "param": f"K={k}",
            "n_flagged": k,
            "TP": tp, "FP": fp, "FN": fn, "TN": len(y_fail) - k - fn,
            "precision_at_k": tp / k,
            "recall_at_k": tp / max(y_fail.sum(), 1),
            "f1_at_k": (2 * tp / k * tp / max(y_fail.sum(), 1)) / (tp / k + tp / max(y_fail.sum(), 1)) if (tp / k + tp / max(y_fail.sum(), 1)) > 0 else 0,
        })
    fail_df = pd.DataFrame(fail_rows)
    print(fail_df.to_string(index=False))
    print(f"\n  Insight: with K={fail_rows[1]['param'][2:]} (typical monthly budget), we'd catch "
          f"{fail_rows[1]['recall_at_k']:.0%} of failures.")

    # ==================================================================
    # MODEL 3: AUTONOMY (survival:cox per AC_MAINS_FAIL event)
    # ==================================================================
    step("MODEL 3: AUTONOMY MODEL (dispatch order during outage)")
    print("  Threshold concept does NOT apply (output is a hazard score).")
    print("  Operational use: top-K dispatch order when multiple sites lose power simultaneously.")

    auto_labels = extract_autonomy_labels(alarms)
    auto_labels = auto_labels.merge(sites[["site_id"]], on="site_id")
    print(f"  AC_MAINS_FAIL events: {len(auto_labels)}, "
          f"LVD events: {auto_labels['event'].sum()} ({auto_labels['event'].mean():.1%})")

    if len(auto_labels) == 0:
        print("  No autonomy labels — skipping")
    else:
        auto_features = compute_features(
            labels=auto_labels, groups=["alarm_history", "site_static"],
            inputs={"alarms": alarms, "site_static": sites},
            ref_time_col="mains_fail_h",
        )
        auto_feature_cols = [c for c in auto_features.columns
                             if c not in ("site_id", "mains_fail_h")
                             and c not in ("charger_misconfigured", "aging_multiplier")]
        test_sites_set = set(sample[int(len(sample) * 0.80):])
        auto_test_mask = auto_features["site_id"].isin(test_sites_set).values
        auto_train_mask = ~auto_test_mask

        Xa_train = auto_features.loc[auto_train_mask, auto_feature_cols].astype(float).fillna(0.0)
        Xa_test = auto_features.loc[auto_test_mask, auto_feature_cols].astype(float).fillna(0.0)
        ya_train = auto_labels.loc[auto_train_mask, "hours_to_lvd"].values.astype(float).copy()
        ya_train[auto_labels.loc[auto_train_mask, "event"].values == 0] *= -1

        booster_auto = xgb.train(
            {"objective": "survival:cox", "eval_metric": "cox-nloglik", "tree_method": "hist",
             "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5,
             "subsample": 0.8, "colsample_bytree": 0.8},
            xgb.DMatrix(Xa_train, label=ya_train), num_boost_round=200, verbose_eval=False,
        )
        auto_risk = booster_auto.predict(xgb.DMatrix(Xa_test))
        auto_events = auto_labels.loc[auto_test_mask, "event"].values
        auto_hours = auto_labels.loc[auto_test_mask, "hours_to_lvd"].values
        print(f"  Test events: {len(auto_events)}, LVDs: {auto_events.sum()}")

        # "Real LVD within HORIZON hours" — the operational question
        HORIZON_HOURS = 12
        y_auto = ((auto_events == 1) & (auto_hours <= HORIZON_HOURS)).astype(int)

        auto_rows = []
        for k in [10, 25, 50, 100]:
            if k > len(auto_risk):
                continue
            order = np.argsort(-auto_risk)
            pred = np.zeros_like(y_auto)
            pred[order[:k]] = 1
            tp = int(((pred == 1) & (y_auto == 1)).sum())
            fp = int(((pred == 1) & (y_auto == 0)).sum())
            fn = int(((pred == 0) & (y_auto == 1)).sum())
            auto_rows.append({
                "model": "autonomy_model",
                "strategy": "Top-K (dispatch during multi-site outage)",
                "param": f"K={k}",
                "n_flagged": k,
                "TP": tp, "FP": fp, "FN": fn, "TN": len(y_auto) - k - fn,
                "precision_at_k": tp / k,
                "recall_at_k": tp / max(y_auto.sum(), 1),
            })
        auto_df = pd.DataFrame(auto_rows)
        print(auto_df.to_string(index=False))

    # ==================================================================
    # SAVE
    # ==================================================================
    out_dir = OUTPUTS / "reports" / "decision_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    drain_df.to_csv(out_dir / "drain_predictor_decisions.csv", index=False)
    fail_df.to_csv(out_dir / "failure_model_decisions.csv", index=False)
    print(f"\nSaved comparisons to {out_dir}/")


if __name__ == "__main__":
    main()
