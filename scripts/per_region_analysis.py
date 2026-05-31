"""Per-region model analysis with full statistics + graphs.

Trains and compares THREE modeling strategies and persists all stats + plots:
    1. Global model (one model for all regions, baseline)
    2. Per-region models (5 separate models, specialized)
    3. Hybrid: global model with region as primary feature

Outputs (all under outputs/reports/per_region/):
    - stats.parquet       (per-region AUC, prevalence, n, etc.)
    - roc_curves.png      (overlay ROC for all approaches)
    - per_region_auc.png  (bar chart comparing strategies)
    - calibration.png     (reliability diagram per strategy)
    - confusion.png       (heatmap per strategy at threshold 0.5)
    - drift_psi.png       (top-10 features PSI bar chart)
    - performance_log.png (depletion over time from append-only log)
    - report.md           (markdown summary)

Usage:
    uv run python scripts/per_region_analysis.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Headless plotting (works on AWS, no GUI needed)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)


HORIZON_H = 48
FEATURE_GROUPS = ["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"]
N_SITES_PER_REGION = 50  # apples-to-apples per-region sample size
SEED = 42

OUT_DIR = Path("outputs/reports/per_region")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_labels_for_sites(alarms: pd.DataFrame, sample_interval_days: int = 7) -> pd.DataFrame:
    """Daily screening labels per site."""
    alarms_s = alarms.sort_values(["site_id", "timestamp_h"]).reset_index(drop=True)
    max_h = alarms_s["timestamp_h"].max()
    lvd_by_site = {}
    for _, r in alarms_s[alarms_s["alarm_code"] == "LOAD_DISCONNECT"].iterrows():
        lvd_by_site.setdefault(r["site_id"], []).append(r["timestamp_h"])
    for s in lvd_by_site:
        lvd_by_site[s] = np.array(sorted(lvd_by_site[s]))

    rows = []
    for site_id, g in alarms_s.groupby("site_id"):
        site_start = g["timestamp_h"].min()
        site_lvds = lvd_by_site.get(site_id, np.array([]))
        for ref_h in np.arange(site_start + 30 * 24, max_h - HORIZON_H, sample_interval_days * 24):
            future = site_lvds[(site_lvds >= ref_h) & (site_lvds < ref_h + HORIZON_H)]
            rows.append({"site_id": site_id, "ref_time_h": float(ref_h),
                         "drain_event": int(len(future) > 0)})
    return pd.DataFrame(rows)


def train_xgb_binary(X_train, y_train, X_test, y_test):
    """Train an XGBoost binary classifier with sane defaults."""
    if y_train.sum() == 0:
        return None
    scale_pos = max(1.0, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    booster = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
         "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 10,
         "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": scale_pos},
        dtrain, num_boost_round=300,
        evals=[(dtest, "val")], early_stopping_rounds=30, verbose_eval=False,
    )
    return booster


def evaluate(y_true, scores, name: str) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"name": name, "n": len(y_true), "positives": int(y_true.sum()),
                "auc": np.nan, "ap": np.nan,
                "precision@0.5": np.nan, "recall@0.5": np.nan, "f1@0.5": np.nan}
    pred = (scores >= 0.5).astype(int)
    return {
        "name": name, "n": len(y_true), "positives": int(y_true.sum()),
        "auc": float(roc_auc_score(y_true, scores)),
        "ap": float(average_precision_score(y_true, scores)),
        "precision@0.5": float(precision_score(y_true, pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_true, pred, zero_division=0)),
        "f1@0.5": float(f1_score(y_true, pred, zero_division=0)),
    }


def main():
    print("=" * 70)
    print("PER-REGION MODEL ANALYSIS")
    print("=" * 70)

    from battery_pdm.common.features import compute_features
    from battery_pdm.synth.load_shedding import build_load_shedding_schedule

    alarms_full = pd.read_parquet("outputs/alarms.parquet")
    site_static_full = pd.read_parquet("outputs/site_static.parquet")
    schedule = build_load_shedding_schedule(n_months=36, seed=SEED)

    # Stratified sample: N_SITES_PER_REGION from each region
    rng = np.random.default_rng(SEED)
    sampled_sites = []
    for region, group in site_static_full.groupby("region"):
        ids = group["site_id"].sample(min(N_SITES_PER_REGION, len(group)), random_state=SEED).tolist()
        sampled_sites.extend(ids)
    alarms = alarms_full[alarms_full["site_id"].isin(sampled_sites)].reset_index(drop=True)
    site_static = site_static_full[site_static_full["site_id"].isin(sampled_sites)].reset_index(drop=True)

    print(f"\nDataset: {len(alarms):,} alarms, {len(site_static)} sites "
          f"({N_SITES_PER_REGION}/region)")

    # Labels + features
    labels = build_labels_for_sites(alarms)
    feat_input = labels.rename(columns={"ref_time_h": "mains_fail_h"})
    features = compute_features(
        labels=feat_input, groups=FEATURE_GROUPS,
        inputs={"alarms": alarms, "site_static": site_static, "schedule": schedule},
        ref_time_col="mains_fail_h",
    ).rename(columns={"mains_fail_h": "ref_time_h"})

    feature_cols = [c for c in features.columns if c not in ("site_id", "ref_time_h")
                    and c not in ("charger_misconfigured", "aging_multiplier")]

    # Merge region into features for downstream analysis
    site_region = dict(zip(site_static["site_id"], site_static["region"]))
    features["region"] = features["site_id"].map(site_region)
    features["drain_event"] = labels["drain_event"].values

    # Group-constrained 60/20/20 split by site (train / calibration / test)
    rng.shuffle(sampled_sites)
    n = len(sampled_sites)
    train_sites = set(sampled_sites[: int(n * 0.60)])
    cal_sites   = set(sampled_sites[int(n * 0.60) : int(n * 0.80)])
    test_sites  = set(sampled_sites[int(n * 0.80) :])
    train_mask = features["site_id"].isin(train_sites).values
    cal_mask   = features["site_id"].isin(cal_sites).values
    test_mask  = features["site_id"].isin(test_sites).values

    X_all = features[feature_cols].astype(float).fillna(0.0)
    y_all = features["drain_event"].values
    X_train, X_cal, X_test = X_all[train_mask], X_all[cal_mask], X_all[test_mask]
    y_train, y_cal, y_test = y_all[train_mask], y_all[cal_mask], y_all[test_mask]
    regions_test = features.loc[test_mask, "region"].values

    print(f"  Train: {train_mask.sum()} obs, {y_train.sum()} drains")
    print(f"  Cal:   {cal_mask.sum()} obs, {y_cal.sum()} drains")
    print(f"  Test:  {test_mask.sum()} obs, {y_test.sum()} drains")

    # Helper: fit isotonic calibrator + apply to test
    from battery_pdm.monitoring.model_registry import (
        train_isotonic_calibrator, brier_score,
    )

    def calibrate_and_eval(booster, X_cal_in, y_cal_in, X_test_in, y_test_in):
        cal_raw = booster.predict(xgb.DMatrix(X_cal_in))
        calibrator = train_isotonic_calibrator(cal_raw, y_cal_in)
        raw_test = booster.predict(xgb.DMatrix(X_test_in))
        cal_test = calibrator.predict(raw_test)
        return raw_test, cal_test, calibrator, brier_score(y_test_in, raw_test), brier_score(y_test_in, cal_test)

    # === STRATEGY 1: Global model ===
    print("\n--- Strategy 1: Global model ---")
    global_booster = train_xgb_binary(X_train, y_train, X_test, y_test)
    global_scores, global_scores_cal, _, global_brier_raw, global_brier_cal = (
        calibrate_and_eval(global_booster, X_cal, y_cal, X_test, y_test)
    )
    print(f"  Brier (uncalibrated): {global_brier_raw:.4f}")
    print(f"  Brier (calibrated):   {global_brier_cal:.4f} (delta {global_brier_cal - global_brier_raw:+.4f})")

    # === STRATEGY 2: Per-region models (each fits its own calibrator) ===
    print("\n--- Strategy 2: Per-region models (calibrated per region) ---")
    per_region_scores = np.zeros(len(y_test))
    per_region_scores_cal = np.zeros(len(y_test))
    per_region_boosters = {}
    train_regions = features.loc[train_mask, "region"].values
    cal_regions = features.loc[cal_mask, "region"].values
    for region in sorted(features["region"].unique()):
        tr_mask = train_regions == region
        cl_mask = cal_regions == region
        te_mask = regions_test == region
        if tr_mask.sum() < 50 or y_train[tr_mask].sum() < 5:
            print(f"  {region}: insufficient training data ({tr_mask.sum()} obs, "
                  f"{y_train[tr_mask].sum()} pos), skipping")
            per_region_scores[te_mask] = global_scores[te_mask]
            per_region_scores_cal[te_mask] = global_scores_cal[te_mask]
            continue
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_cl = X_cal[cl_mask]
        y_cl = y_cal[cl_mask]
        X_te = X_test[te_mask]
        y_te = y_test[te_mask]
        booster = train_xgb_binary(X_tr, y_tr, X_te, y_te)
        if booster is None:
            per_region_scores[te_mask] = global_scores[te_mask]
            per_region_scores_cal[te_mask] = global_scores_cal[te_mask]
            continue
        per_region_boosters[region] = booster
        raw_te = booster.predict(xgb.DMatrix(X_te))
        per_region_scores[te_mask] = raw_te
        # Fit per-region calibrator if we have enough cal samples
        if cl_mask.sum() >= 30 and y_cl.sum() >= 3:
            region_calibrator = train_isotonic_calibrator(
                booster.predict(xgb.DMatrix(X_cl)), y_cl,
            )
            per_region_scores_cal[te_mask] = region_calibrator.predict(raw_te)
        else:
            # Not enough cal data — keep raw scores for this region
            per_region_scores_cal[te_mask] = raw_te
        print(f"  {region}: trained on {tr_mask.sum()} obs ({y_tr.sum()} pos), "
              f"cal {cl_mask.sum()} obs ({y_cl.sum()} pos)")

    per_region_brier_raw = brier_score(y_test, per_region_scores)
    per_region_brier_cal = brier_score(y_test, per_region_scores_cal)
    print(f"  Brier (uncalibrated): {per_region_brier_raw:.4f}")
    print(f"  Brier (calibrated):   {per_region_brier_cal:.4f} "
          f"(delta {per_region_brier_cal - per_region_brier_raw:+.4f})")

    # === STRATEGY 3: Hybrid (global + region prior offset) ===
    print("\n--- Strategy 3: Hybrid (global + region prior) ---")
    region_prior = features.loc[train_mask].groupby("region")["drain_event"].mean().to_dict()
    region_prior_test = np.array([region_prior.get(r, 0.15) for r in regions_test])
    hybrid_scores = 0.7 * global_scores + 0.3 * region_prior_test
    # Calibrate the hybrid on the cal set
    region_prior_cal = np.array([
        region_prior.get(r, 0.15) for r in features.loc[cal_mask, "region"].values
    ])
    hybrid_cal_raw = 0.7 * global_booster.predict(xgb.DMatrix(X_cal)) + 0.3 * region_prior_cal
    hybrid_calibrator = train_isotonic_calibrator(hybrid_cal_raw, y_cal)
    hybrid_scores_cal = hybrid_calibrator.predict(hybrid_scores)
    hybrid_brier_raw = brier_score(y_test, hybrid_scores)
    hybrid_brier_cal = brier_score(y_test, hybrid_scores_cal)
    print(f"  Brier (uncalibrated): {hybrid_brier_raw:.4f}")
    print(f"  Brier (calibrated):   {hybrid_brier_cal:.4f} "
          f"(delta {hybrid_brier_cal - hybrid_brier_raw:+.4f})")

    # ===== Stats collection =====
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    brier_lookup = {
        "global": (global_brier_raw, global_brier_cal),
        "per_region": (per_region_brier_raw, per_region_brier_cal),
        "hybrid": (hybrid_brier_raw, hybrid_brier_cal),
    }
    stats_rows = []
    for strategy, scores in [("global", global_scores),
                              ("per_region", per_region_scores),
                              ("hybrid", hybrid_scores)]:
        overall = evaluate(y_test, scores, f"{strategy}_overall")
        br, bc = brier_lookup[strategy]
        overall["brier_raw"] = br
        overall["brier_cal"] = bc
        stats_rows.append({"strategy": strategy, "region": "ALL", **overall})
        for region in sorted(set(regions_test)):
            m = regions_test == region
            if m.sum() < 20:
                continue
            sub = evaluate(y_test[m], scores[m], f"{strategy}_{region}")
            stats_rows.append({"strategy": strategy, "region": region, **sub})

    stats = pd.DataFrame(stats_rows)
    stats.to_parquet(OUT_DIR / "stats.parquet", index=False)
    print(stats.to_string(index=False))

    # ===== GRAPH 1: Per-region AUC comparison =====
    print("\nGenerating graphs...")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = stats[stats["region"] != "ALL"].pivot(index="region", columns="strategy", values="auc")
    pivot.plot(kind="bar", ax=ax, color=["#3b82f6", "#10b981", "#f59e0b"])
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (AUC=0.5)")
    ax.set_ylabel("AUC")
    ax.set_title("Per-region AUC: global vs per-region vs hybrid")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_region_auc.png", dpi=120)
    plt.close()

    # ===== GRAPH 2: ROC curves overlay =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (strat_name, scores) in zip(axes, [("Global", global_scores),
                                                  ("Per-region", per_region_scores),
                                                  ("Hybrid", hybrid_scores)]):
        for region in sorted(set(regions_test)):
            m = regions_test == region
            if m.sum() < 20 or len(np.unique(y_test[m])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_test[m], scores[m])
            auc = roc_auc_score(y_test[m], scores[m])
            ax.plot(fpr, tpr, label=f"{region} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(f"{strat_name} model")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curves.png", dpi=120)
    plt.close()

    # ===== GRAPH 3: Calibration — raw vs isotonic-calibrated =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bins = np.linspace(0, 1, 11)
    triples = [
        ("Global", global_scores, global_scores_cal, global_brier_raw, global_brier_cal),
        ("Per-region", per_region_scores, per_region_scores_cal, per_region_brier_raw, per_region_brier_cal),
        ("Hybrid", hybrid_scores, hybrid_scores_cal, hybrid_brier_raw, hybrid_brier_cal),
    ]
    for ax, (strat_name, raw_scores, cal_scores, br_raw, br_cal) in zip(axes, triples):
        def _bin(s):
            xs, ys = [], []
            for i in range(10):
                lo, hi = bins[i], bins[i + 1]
                m = (s >= lo) & (s < hi)
                if m.sum() > 5:
                    xs.append(s[m].mean())
                    ys.append(y_test[m].mean())
            return xs, ys

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfectly calibrated")
        rx, ry = _bin(raw_scores)
        cx, cy = _bin(cal_scores)
        ax.plot(rx, ry, "o-", color="#ef4444", label=f"Raw (Brier={br_raw:.3f})")
        ax.plot(cx, cy, "o-", color="#10b981", label=f"Isotonic (Brier={br_cal:.3f})")
        ax.set_title(f"{strat_name} calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Actual drain rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "calibration.png", dpi=120)
    plt.close()

    # ===== GRAPH 4: Confusion matrices =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (strat_name, scores) in zip(axes, [("Global", global_scores),
                                                  ("Per-region", per_region_scores),
                                                  ("Hybrid", hybrid_scores)]):
        pred = (scores >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No drain", "Drain"])
        ax.set_yticklabels(["No drain", "Drain"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{strat_name} confusion (@0.5)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion.png", dpi=120)
    plt.close()

    # ===== GRAPH 5: Drift PSI from simulation =====
    perf_log_path = Path("outputs/model_performance_log.parquet")
    if perf_log_path.exists():
        perf = pd.read_parquet(perf_log_path)
        if not perf.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            perf_train = perf[perf["metric_name"].isin(["roc_auc", "mean_cindex"])].copy()
            perf_train["logged_at"] = pd.to_datetime(perf_train["logged_at"])
            for name, sub in perf_train.groupby("model_name"):
                ax.plot(sub["logged_at"], sub["metric_value"], "o-", label=name)
            ax.set_xlabel("Time")
            ax.set_ylabel("Test metric (AUC / C-index)")
            ax.set_title("Model performance over time (depletion chart)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "performance_log.png", dpi=120)
            plt.close()

    # ===== GRAPH 6: Drift PSI =====
    drift_path = Path("outputs/drift_reports")
    drift_files = list(drift_path.glob("drift_report_h*.json")) if drift_path.exists() else []
    if drift_files:
        latest = max(drift_files, key=lambda p: p.stat().st_mtime)
        drift_data = json.loads(latest.read_text())
        top_drifted = drift_data.get("top_drifted_features", [])
        if top_drifted:
            fig, ax = plt.subplots(figsize=(10, 5))
            df = pd.DataFrame(top_drifted).head(10)
            colors = [
                "#ef4444" if level == "SIGNIFICANT" else "#f59e0b" if level == "MODERATE" else "#10b981"
                for level in df["drift_level"]
            ]
            ax.barh(df["feature"], df["psi"], color=colors)
            ax.axvline(0.10, color="orange", linestyle="--", label="Moderate threshold")
            ax.axvline(0.25, color="red", linestyle="--", label="Significant threshold")
            ax.set_xlabel("PSI")
            ax.set_title("Top drifted features (most recent drift report)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(OUT_DIR / "drift_psi.png", dpi=120)
            plt.close()

    # ===== Markdown summary =====
    overall = stats[stats["region"] == "ALL"][["strategy", "auc", "ap", "f1@0.5"]]
    report = [
        "# Per-region modeling analysis",
        "",
        f"Sites per region: {N_SITES_PER_REGION} | horizon: {HORIZON_H}h | test sites: {test_mask.sum()}",
        "",
        "## Overall comparison",
        "",
        overall.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Per-region AUC",
        "",
        stats[stats["region"] != "ALL"].pivot(index="region", columns="strategy", values="auc")
            .to_markdown(floatfmt=".4f"),
        "",
        "## Graphs",
        "",
        "- `per_region_auc.png` — AUC bar chart by strategy and region",
        "- `roc_curves.png` — ROC curves per region for each strategy",
        "- `calibration.png` — reliability diagrams",
        "- `confusion.png` — confusion matrices at threshold 0.5",
        "- `drift_psi.png` — top drifted features from latest drift report",
        "- `performance_log.png` — model metric over time (depletion chart)",
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report))
    print(f"\nReport saved to {OUT_DIR}/report.md")
    print(f"Graphs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
