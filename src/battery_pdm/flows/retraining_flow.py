"""RetrainingFlow — champion/challenger with CV-gated promotion (proactive parallel path).

Implements the ml.school continuous retraining pattern:
    - Inference path: champion serves daily (DrainPredictorFlow, FailureScoringFlow)
    - Training path: runs WEEKLY on latest data (Saturday cron, --force true).
      Trains challenger, compares vs champion on same held-out set, promotes
      immediately if CV gate passes. Training is cheap (~5min Fargate Spot).

Why proactive (not just drift-triggered):
    - Drift monitor is a SAFETY NET for sudden shifts between weekly runs.
    - Weekly training catches GRADUAL degradation before PSI crosses threshold.
    - The CV gate prevents regressions, so there's no risk in always training.

Promotion strategy (model-specific):
    - Drain predictor (48h labels): immediate promotion (labels mature fast,
      held-out CV is a fair evaluation). Rollback flow as safety net.
    - Failure model (6-12mo labels): shadow mode (held-out CV may not reflect
      current fleet composition; needs realized production labels).

Run:
    python -m battery_pdm.flows.retraining_flow run --force true
    python -m battery_pdm.flows.retraining_flow run --model-name failure_alarms_only --shadow-mode true
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from metaflow import FlowSpec, Parameter, project, step

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@project(name="battery_pdm")
class RetrainingFlow(FlowSpec):
    """Champion/challenger retraining with auto-promotion."""

    model_name = Parameter(
        "model-name",
        default="drain_predictor_48h",
        help="Which model to retrain: drain_predictor_48h | failure_alarms_only",
    )
    models_root = Parameter("models-root", default="outputs/models")
    alarms_path = Parameter("alarms-path", default="outputs/alarms.parquet")
    site_static_path = Parameter(
        "site-static-path", default="outputs/site_static.parquet"
    )
    promotion_margin = Parameter(
        "promotion-margin",
        type=float,
        default=0.005,
        help="Challenger must beat champion by this margin",
    )
    force = Parameter(
        "force",
        type=bool,
        default=False,
        help="Force retraining even without drift trigger",
    )
    seed = Parameter("seed", type=int, default=42)
    n_sites = Parameter("n-sites", type=int, default=100)
    # Label maturity gate — prevents promoting a challenger before enough
    # post-drift labels accumulate. Critical for slow-feedback models (failure).
    # Set to 0 to disable (useful for short-horizon models or local dev).
    min_label_maturity_days = Parameter(
        "min-label-maturity-days",
        type=int,
        default=0,
        help="Minimum days of post-drift labels needed to promote. 0=disabled. "
        "Drain predictor: ~7 days. Failure model: ~180 days.",
    )
    alerts_dir = Parameter(
        "alerts-dir",
        default="outputs/drain_alerts",
        help="Where prior predictions are logged (S3 in prod). Used for label maturity check.",
    )
    shadow_mode = Parameter(
        "shadow-mode",
        type=bool,
        default=True,
        help="If true, save trained challenger to _shadow/ for parallel scoring rather than immediate promotion. "
        "This is the DEFAULT for continuous retraining (proactive parallel path): "
        "always produce a challenger, let ShadowPromotionFlow validate on realized labels. "
        "Set to false only for reactive drift-triggered promotion (legacy mode).",
    )

    @step
    def start(self):
        """Atomically claim the drift trigger (or run forced)."""
        from battery_pdm.monitoring.model_registry import claim_retrain_trigger

        trigger = claim_retrain_trigger()
        if trigger:
            print(f"Claimed retrain trigger: {trigger.get('reasons', [])}")
            self.triggered = True
            self.trigger_payload = trigger
        elif self.force:
            print("Forced retraining (no trigger required)")
            self.triggered = True
            self.trigger_payload = {"reasons": ["forced"]}
        else:
            print("No retrain trigger and --force not set. Exiting.")
            self.triggered = False
            self.trigger_payload = {}

        self.next(self.load_data)

    @step
    def load_data(self):
        """Load data for retraining."""
        if not self.triggered:
            self.next(self.train_challenger)
            return

        self.alarms = pd.read_parquet(self.alarms_path)
        self.site_static = pd.read_parquet(self.site_static_path)

        if self.n_sites and self.n_sites < len(self.site_static):
            sample_sites = (
                self.site_static["site_id"]
                .sample(self.n_sites, random_state=self.seed)
                .tolist()
            )
            self.alarms = self.alarms[
                self.alarms["site_id"].isin(sample_sites)
            ].reset_index(drop=True)
            self.site_static = self.site_static[
                self.site_static["site_id"].isin(sample_sites)
            ].reset_index(drop=True)

        print(f"Data: {len(self.alarms):,} alarms, {len(self.site_static)} sites")
        self.next(self.train_challenger)

    @step
    def train_challenger(self):
        """Train challenger model on current data."""
        if not self.triggered:
            self.next(self.compare)
            return

        from battery_pdm.synth.load_shedding import build_load_shedding_schedule

        schedule = build_load_shedding_schedule(n_months=36, seed=self.seed)

        if self.model_name == "drain_predictor_48h":
            self.challenger_metrics = self._train_drain_challenger(schedule)
        elif self.model_name == "failure_alarms_only":
            self.challenger_metrics = self._train_failure_challenger()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        print(f"Challenger trained: {self.challenger_metrics}")
        self.next(self.compare)

    def _train_drain_challenger(self, schedule) -> dict:
        """Train drain predictor challenger."""
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
        from battery_pdm.common.features import compute_features

        horizon_h = 48
        alarms_sorted = self.alarms.sort_values(["site_id", "timestamp_h"]).reset_index(
            drop=True
        )
        max_h = alarms_sorted["timestamp_h"].max()

        lvd_by_site = {}
        lvd_events = alarms_sorted[alarms_sorted["alarm_code"] == "LOAD_DISCONNECT"]
        for _, row in lvd_events.iterrows():
            lvd_by_site.setdefault(row["site_id"], []).append(row["timestamp_h"])
        for site in lvd_by_site:
            lvd_by_site[site] = np.array(sorted(lvd_by_site[site]))

        labels_list = []
        for site_id, group in alarms_sorted.groupby("site_id"):
            site_start = group["timestamp_h"].min()
            site_lvds = lvd_by_site.get(site_id, np.array([]))
            for ref_h in np.arange(site_start + 30 * 24, max_h - horizon_h, 7 * 24):
                future_lvds = site_lvds[
                    (site_lvds >= ref_h) & (site_lvds < ref_h + horizon_h)
                ]
                labels_list.append(
                    {
                        "site_id": site_id,
                        "ref_time_h": float(ref_h),
                        "drain_event": int(len(future_lvds) > 0),
                    }
                )
        labels = pd.DataFrame(labels_list)

        feature_groups = [
            "alarm_history",
            "site_static",
            "soc_proxy",
            "load_shedding_schedule",
        ]
        features = compute_features(
            labels=labels.rename(columns={"ref_time_h": "mains_fail_h"}),
            groups=feature_groups,
            inputs={
                "alarms": self.alarms,
                "site_static": self.site_static,
                "schedule": schedule,
            },
            ref_time_col="mains_fail_h",
        )
        features = features.rename(columns={"mains_fail_h": "ref_time_h"})

        feature_cols = [
            c
            for c in features.columns
            if c
            not in (
                "site_id",
                "ref_time_h",
                "charger_misconfigured",
                "aging_multiplier",
            )
        ]
        merged = features.copy()
        merged["drain_event"] = labels["drain_event"].values

        # Group-constrained 60/20/20 split: train / calibration / test
        rng = np.random.default_rng(self.seed)
        unique_sites = merged["site_id"].unique()
        rng.shuffle(unique_sites)
        n = len(unique_sites)
        train_mask = merged["site_id"].isin(set(unique_sites[: int(n * 0.60)])).values
        cal_mask = (
            merged["site_id"]
            .isin(set(unique_sites[int(n * 0.60) : int(n * 0.80)]))
            .values
        )
        test_mask = merged["site_id"].isin(set(unique_sites[int(n * 0.80) :])).values

        y_train = merged["drain_event"].values[train_mask]
        y_cal = merged["drain_event"].values[cal_mask]
        y_test = merged["drain_event"].values[test_mask]
        X_train = merged.loc[train_mask, feature_cols].astype(float).fillna(0.0)
        X_cal = merged.loc[cal_mask, feature_cols].astype(float).fillna(0.0)
        X_test = merged.loc[test_mask, feature_cols].astype(float).fillna(0.0)

        scale_pos = max(1.0, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
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
            xgb.DMatrix(X_train, label=y_train),
            num_boost_round=300,
            evals=[(xgb.DMatrix(X_test, label=y_test), "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        # Fit isotonic calibrator on the strictly held-out calibration set
        from battery_pdm.monitoring.model_registry import (
            save_model_artifacts,
            save_reference_profile_for_model,
            train_isotonic_calibrator,
            save_calibrator,
            brier_score,
        )

        cal_raw = booster.predict(xgb.DMatrix(X_cal))
        calibrator = train_isotonic_calibrator(cal_raw, y_cal)

        raw_test = booster.predict(xgb.DMatrix(X_test))
        cal_test = calibrator.predict(raw_test)
        auc = float(roc_auc_score(y_test, raw_test))
        brier_uncal = brier_score(y_test, raw_test)
        brier_cal = brier_score(y_test, cal_test)

        # Save challenger via registry (includes feature_hash, reference profile, calibrator)
        challenger_dir = Path(self.models_root) / f"{self.model_name}_challenger"
        save_model_artifacts(
            booster=booster,
            feature_cols=feature_cols,
            feature_groups=feature_groups,
            metrics={
                "roc_auc": auc,
                "brier_uncalibrated": brier_uncal,
                "brier_calibrated": brier_cal,
            },
            model_dir=challenger_dir,
            extras={"horizon_h": 48, "calibration_method": "isotonic"},
        )
        save_calibrator(calibrator, challenger_dir)
        save_reference_profile_for_model(
            training_features=merged.loc[train_mask, feature_cols + ["site_id"]],
            feature_cols=feature_cols,
            predictions=calibrator.predict(booster.predict(xgb.DMatrix(X_train))),
            model_dir=challenger_dir,
        )

        # Stash held-out test set so `compare` step can re-evaluate BOTH champion + challenger on it
        self.test_set_features = X_test
        self.test_set_labels = y_test
        self.test_set_feature_cols = feature_cols

        return {"metric": "roc_auc", "value": auc}

    def _train_failure_challenger(self) -> dict:
        """Train failure model challenger."""
        import xgboost as xgb
        from battery_pdm.common.features import compute_features
        from battery_pdm.common.survival import _cindex

        labels_raw = pd.read_parquet("outputs/labels.parquet")
        site_ids = set(self.site_static["site_id"].values)
        if (
            "original_site_id" not in labels_raw.columns
            and "lifecycle_id" in labels_raw.columns
        ):
            labels_raw["original_site_id"] = labels_raw["site_id"]
            labels_raw["site_id"] = (
                labels_raw["site_id"]
                + "_L"
                + labels_raw["lifecycle_id"].astype(int).astype(str)
            )
        labels_raw = labels_raw[
            labels_raw.get("original_site_id", labels_raw["site_id"]).isin(site_ids)
        ].reset_index(drop=True)

        alarm_parts = []
        for _, row in labels_raw.iterrows():
            original_sid = row.get("original_site_id", row["site_id"])
            windowed = self.alarms[
                (self.alarms["site_id"] == original_sid)
                & (self.alarms["timestamp_h"] >= row.get("lifecycle_start_h", 0))
                & (self.alarms["timestamp_h"] <= row["event_hour"])
            ].copy()
            windowed["site_id"] = row["site_id"]
            alarm_parts.append(windowed)
        alarms_w = (
            pd.concat(alarm_parts, ignore_index=True)
            if alarm_parts
            else self.alarms.iloc[:0].copy()
        )

        static_lc = (
            labels_raw[["site_id", "original_site_id"]]
            .merge(
                self.site_static,
                left_on="original_site_id",
                right_on="site_id",
                suffixes=("", "_static"),
            )
            .drop(columns=["site_id_static"])
        )

        feature_groups = ["alarm_history", "site_static", "soc_proxy"]
        features = compute_features(
            labels=labels_raw,
            groups=feature_groups,
            inputs={"alarms": alarms_w, "site_static": static_lc},
            ref_time_col="event_hour",
        )
        feature_cols = [
            c
            for c in features.columns
            if c
            not in (
                "site_id",
                "event_hour",
                "charger_misconfigured",
                "aging_multiplier",
            )
        ]

        label_only = [
            c
            for c in labels_raw.columns
            if c not in features.columns or c in ("site_id", "event_hour")
        ]
        merged = features.merge(
            labels_raw[label_only], on=["site_id", "event_hour"], how="inner"
        )

        rng = np.random.default_rng(self.seed)
        groups = merged.get("original_site_id", merged["site_id"]).values
        unique_g = np.array(sorted(set(groups)))
        rng.shuffle(unique_g)
        split_idx = int(len(unique_g) * 0.75)
        train_mask = np.array([g in set(unique_g[:split_idx]) for g in groups])
        test_mask = ~train_mask

        train_df, test_df = merged[train_mask], merged[test_mask]
        y_train = train_df["time_to_event_months"].values.astype(float).copy()
        y_train[train_df["event"].values == 0] = -np.abs(
            y_train[train_df["event"].values == 0]
        )
        y_test_raw = test_df["time_to_event_months"].values.astype(float).copy()
        y_test_raw[test_df["event"].values == 0] = -np.abs(
            y_test_raw[test_df["event"].values == 0]
        )

        booster = xgb.train(
            {
                "objective": "survival:cox",
                "eval_metric": "cox-nloglik",
                "tree_method": "hist",
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            xgb.DMatrix(
                train_df[feature_cols].astype(float).fillna(0.0), label=y_train
            ),
            num_boost_round=200,
            verbose_eval=False,
        )
        risk = booster.predict(
            xgb.DMatrix(test_df[feature_cols].astype(float).fillna(0.0))
        )
        cindex = _cindex(
            risk, test_df["time_to_event_months"].values, test_df["event"].values
        )

        from battery_pdm.monitoring.model_registry import (
            save_model_artifacts,
            save_reference_profile_for_model,
        )

        challenger_dir = Path(self.models_root) / f"{self.model_name}_challenger"
        save_model_artifacts(
            booster=booster,
            feature_cols=feature_cols,
            feature_groups=feature_groups,
            metrics={"mean_cindex": float(cindex)},
            model_dir=challenger_dir,
        )
        save_reference_profile_for_model(
            training_features=train_df[feature_cols + ["site_id"]],
            feature_cols=feature_cols,
            predictions=booster.predict(
                xgb.DMatrix(train_df[feature_cols].astype(float).fillna(0.0))
            ),
            model_dir=challenger_dir,
        )

        # Stash held-out test set so `compare` step can re-evaluate champion on it.
        # For survival models we need features + time-to-event + event indicator.
        self.test_set_features = test_df[feature_cols].astype(float).fillna(0.0)
        self.test_set_labels = test_df["time_to_event_months"].values
        self.test_set_events = test_df["event"].values
        self.test_set_feature_cols = feature_cols

        return {"metric": "cindex", "value": float(cindex)}

    @step
    def compare(self):
        """Compare champion vs challenger on the SAME held-out test set."""
        if not self.triggered:
            self.next(self.promote_or_reject)
            return

        import xgboost as xgb
        from sklearn.metrics import roc_auc_score

        champion_dir = Path(self.models_root) / self.model_name
        champion_meta_path = champion_dir / "meta.json"
        challenger_dir = Path(self.models_root) / f"{self.model_name}_challenger"

        if not champion_meta_path.exists():
            print("No existing champion. Promoting challenger.")
            self.promote = True
            self.next(self.promote_or_reject)
            return

        champion_meta = json.loads(champion_meta_path.read_text())
        champion_booster = xgb.Booster()
        champion_booster.load_model(str(champion_dir / "booster.json"))

        challenger_booster = xgb.Booster()
        challenger_booster.load_model(str(challenger_dir / "booster.json"))

        # Re-evaluate champion on challenger's test set (apples to apples)
        champion_feature_cols = champion_meta["feature_cols"]
        # Build the champion-flavor X from challenger's test features (pad/reorder)
        X_test = self.test_set_features
        X_champion = pd.DataFrame(index=X_test.index)
        for col in champion_feature_cols:
            X_champion[col] = X_test[col] if col in X_test.columns else 0.0

        metric_name = self.challenger_metrics["metric"]
        challenger_val = self.challenger_metrics["value"]

        if metric_name == "roc_auc":
            champion_scores = champion_booster.predict(xgb.DMatrix(X_champion))
            champion_val_shared = float(
                roc_auc_score(self.test_set_labels, champion_scores)
            )
        elif metric_name == "cindex":
            # Survival model: compute C-index on shared test set
            from battery_pdm.common.survival import _cindex

            champion_risk = champion_booster.predict(xgb.DMatrix(X_champion))
            champion_val_shared = float(
                _cindex(champion_risk, self.test_set_labels, self.test_set_events)
            )
        else:
            champion_val_shared = champion_meta.get("metrics", {}).get(
                "mean_cindex", 0.0
            )

        champion_val_stored = champion_meta.get("metrics", {}).get(metric_name, 0.0)
        margin = challenger_val - champion_val_shared

        print(f"\nChampion (re-eval on shared test): {champion_val_shared:.4f}")
        print(f"Champion (stored at training time): {champion_val_stored:.4f}")
        print(f"Challenger (on shared test): {challenger_val:.4f}")
        print(
            f"Margin (shared test): {margin:+.4f} (required: +{self.promotion_margin})"
        )

        self.champion_metric_shared = champion_val_shared
        margin_check_passes = margin >= self.promotion_margin

        # Label-maturity gate: refuse to promote if we don't have enough days of
        # post-drift LABELED feedback to validate the new model. Critical for
        # slow-feedback models (failure: 6-12mo labels). Skipped if min_days=0.
        maturity_ok = True
        if self.min_label_maturity_days > 0:
            maturity_ok = self._check_label_maturity()
            if not maturity_ok:
                print(
                    f"Label maturity gate FAILED → will NOT promote even though "
                    f"margin check {'passes' if margin_check_passes else 'fails'}"
                )

        self.promote = margin_check_passes and maturity_ok

        self.next(self.promote_or_reject)

    def _check_label_maturity(self) -> bool:
        """Verify enough days of post-prediction labels exist to validate challenger."""
        from battery_pdm.monitoring.concept_drift import (
            fetch_realized_outcomes,
            check_label_maturity,
        )

        alerts_dir = Path(self.alerts_dir)
        if not alerts_dir.exists():
            print(
                f"  Maturity check: no alerts dir at {alerts_dir} — assuming early deployment"
            )
            return False  # safer default

        # Aggregate prior alerts (each parquet file = one scoring run)
        alert_files = sorted(alerts_dir.glob("*.parquet"))
        if not alert_files:
            print("  Maturity check: no alert files yet — assuming early deployment")
            return False

        dfs = []
        for f in alert_files[-30:]:  # last 30 batches
            try:
                df = pd.read_parquet(f)
                if "scoring_hour" in df.columns and "drain_risk_48h" in df.columns:
                    df = df.rename(columns={"drain_risk_48h": "predicted_score"})
                    dfs.append(df[["site_id", "scoring_hour", "predicted_score"]])
            except Exception as exc:
                print(f"  Maturity check: skip {f.name} ({exc})")
        if not dfs:
            return False

        predictions = pd.concat(dfs, ignore_index=True)
        # Horizon depends on model — 48h for drain predictor by default
        horizon_h = (
            48 if self.model_name == "drain_predictor_48h" else 12 * 30 * 24
        )  # 12mo for failure
        enriched = fetch_realized_outcomes(
            predictions, self.alarms, horizon_h=horizon_h
        )
        current_h = float(self.alarms["timestamp_h"].max())
        maturity = check_label_maturity(
            enriched,
            min_days_of_labels=self.min_label_maturity_days,
            horizon_h=horizon_h,
            current_h=current_h,
        )
        print(f"  Label maturity: {maturity['reason']}")
        return bool(maturity["ready_to_promote"])

    @step
    def promote_or_reject(self):
        """Promote challenger to champion, OR write as shadow for parallel scoring."""
        from battery_pdm.monitoring.model_registry import (
            complete_retrain_trigger,
            release_retrain_trigger,
            append_performance_log,
        )

        if not self.triggered:
            self.next(self.end)
            return

        champion_dir = Path(self.models_root) / self.model_name
        challenger_dir = Path(self.models_root) / f"{self.model_name}_challenger"

        # Shadow mode: don't promote immediately. Save as <model>_shadow/ so
        # scoring flow runs both in parallel and accumulates labeled comparisons.
        # A separate ShadowPromotionFlow eventually promotes shadow → champion
        # when labels confirm improvement.
        if self.shadow_mode:
            shadow_dir = Path(self.models_root) / f"{self.model_name}_shadow"
            shadow_dir.mkdir(parents=True, exist_ok=True)
            # Move challenger files into shadow location (atomic via copy + delete)
            for f in challenger_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, shadow_dir / f.name)
            shutil.rmtree(challenger_dir)
            print(
                f"SHADOW MODE: challenger written to {shadow_dir} for parallel scoring."
            )
            print(
                f"  Champion {champion_dir.name} stays live. Shadow gets validated in production runs."
            )
            print(
                "  Use ShadowPromotionFlow to evaluate + promote shadow once labels mature."
            )
            complete_retrain_trigger(promoted=False, reason="written_as_shadow")
            self.next(self.end)
            return

        if self.promote:
            try:
                # Archive current champion
                archive_dir = (
                    Path(self.models_root)
                    / "archive"
                    / f"{self.model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                )
                if champion_dir.exists():
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    for f in champion_dir.iterdir():
                        if f.is_file():
                            shutil.copy2(f, archive_dir / f.name)
                    print(f"Archived champion to {archive_dir}")

                # Promote challenger atomically — copy all files first, then commit
                champion_dir.mkdir(parents=True, exist_ok=True)
                for f in challenger_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, champion_dir / f.name)
                print(f"PROMOTED challenger to {champion_dir}")

                # Reference profile travels with the challenger — already in challenger_dir
                # (set during train_challenger), so copying meta + booster + reference is enough

                # Log performance entry for the promotion event
                challenger_meta = json.loads((champion_dir / "meta.json").read_text())
                metric_name = (
                    "roc_auc"
                    if "roc_auc" in challenger_meta["metrics"]
                    else "mean_cindex"
                )
                append_performance_log(
                    model_name=self.model_name,
                    model_version=challenger_meta.get("model_version", "unknown"),
                    metric_name=metric_name,
                    metric_value=challenger_meta["metrics"][metric_name],
                    n_observations=0,
                    feature_hash=challenger_meta.get("feature_hash", ""),
                    extras={
                        "event": "promotion",
                        "trigger_reasons": str(self.trigger_payload.get("reasons", [])),
                    },
                )

                complete_retrain_trigger(promoted=True, reason="challenger_won")

                # Promote in MLflow Registry: find latest Staging version -> Production
                try:
                    from battery_pdm.monitoring.model_registry import (
                        transition_model_stage,
                    )
                    import mlflow

                    client = mlflow.MlflowClient()
                    versions = client.search_model_versions(
                        f"name='{self.model_name}'", order_by=["version_number DESC"]
                    )
                    if versions:
                        transition_model_stage(
                            self.model_name, versions[0].version, "Production"
                        )
                        print(
                            f"  MLflow Registry: {self.model_name} v{versions[0].version} -> Production"
                        )
                except Exception as exc:
                    print(f"  (MLflow Registry promotion skipped: {exc})")

                # Emit CloudWatch metrics for the dashboard
                try:
                    from battery_pdm.aws.metrics import (
                        emit_model_performance,
                        emit_metric,
                    )

                    emit_model_performance(
                        model_name=self.model_name,
                        auc=float(challenger_meta["metrics"].get(metric_name, 0.0)),
                        n_obs=0,
                    )
                    emit_metric(
                        "RetrainPromoted",
                        1,
                        "Count",
                        dimensions={"ModelName": self.model_name},
                    )
                except Exception as exc:
                    print(f"  (CloudWatch metric emission skipped: {exc})")
            except Exception as exc:
                # If anything fails, release the trigger so a retry can pick it up
                release_retrain_trigger()
                raise RuntimeError(
                    f"Promotion failed mid-flight, trigger released: {exc}"
                )
        else:
            print("Challenger REJECTED (did not beat champion by required margin)")
            complete_retrain_trigger(promoted=False, reason="margin_not_met")
            try:
                from battery_pdm.aws.metrics import emit_metric

                emit_metric(
                    "RetrainPromoted",
                    0,
                    "Count",
                    dimensions={"ModelName": self.model_name},
                )
            except Exception:
                pass

        # Clean up challenger dir
        if challenger_dir.exists():
            shutil.rmtree(challenger_dir)

        self.next(self.end)

    @step
    def end(self):
        if not self.triggered:
            print("RetrainingFlow: no action taken (no trigger).")
        elif self.promote:
            print(f"RetrainingFlow complete: {self.model_name} PROMOTED")
        else:
            print(f"RetrainingFlow complete: {self.model_name} challenger rejected")


if __name__ == "__main__":
    RetrainingFlow()
