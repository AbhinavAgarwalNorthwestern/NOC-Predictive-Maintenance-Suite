"""TrainingFlow — trains battery PdM models with CV gating.

Two models, two decisions, no overlap:
    1. Failure model (survival:cox) — "should this battery be replaced?"
    2. Drain predictor (binary:logistic) — "will this site drain in next 48h?"

The autonomy model was decommissioned: the drain predictor (AUC 0.90, with
schedule + SoC features) covers both daily planning AND real-time triage via
the FastAPI /predict endpoint. The autonomy model (C-index 0.73, no schedule
features) added no unique operational value. See docs/PROJECT_BOOK.md §10.

DAG: start -> train_failure -> train_drain -> upload_models -> end

Run:
    python -m battery_pdm.flows.training_flow run --n-sites 50
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from metaflow import FlowSpec, Parameter, project, step

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@project(name="battery_pdm")
class TrainingFlow(FlowSpec):
    """Train and promote all battery PdM models."""

    n_sites = Parameter(
        "n-sites", type=int, default=100, help="Subset of sites for training"
    )
    seed = Parameter("seed", type=int, default=42)
    alarms_path = Parameter("alarms-path", default="outputs/alarms.parquet")
    site_static_path = Parameter(
        "site-static-path", default="outputs/site_static.parquet"
    )
    labels_path = Parameter("labels-path", default="outputs/labels.parquet")
    models_output = Parameter(
        "models-output",
        default="",
        help="S3 URI to upload models (e.g. s3://bucket/models/)",
    )
    min_cindex_failure = Parameter("min-cindex-failure", type=float, default=0.70)
    min_auc_drain = Parameter("min-auc-drain", type=float, default=0.70)

    @step
    def start(self):
        """Load shared data artifacts."""
        from battery_pdm.synth.load_shedding import build_load_shedding_schedule
        from battery_pdm.aws.s3_io import read_parquet

        alarms = read_parquet(self.alarms_path)
        site_static = read_parquet(self.site_static_path)

        if self.n_sites and self.n_sites < len(site_static):
            sample_sites = (
                site_static["site_id"]
                .sample(self.n_sites, random_state=self.seed)
                .tolist()
            )
            alarms = alarms[alarms["site_id"].isin(sample_sites)].reset_index(drop=True)
            site_static = site_static[
                site_static["site_id"].isin(sample_sites)
            ].reset_index(drop=True)

        self.alarms = alarms
        self.site_static = site_static
        self.schedule = build_load_shedding_schedule(n_months=36, seed=self.seed)

        print(f"Data loaded: {len(alarms):,} alarms, {len(site_static)} sites")
        self.next(self.train_failure)

    @step
    def train_failure(self):
        """Train long-term failure prediction model (survival:cox)."""
        import xgboost as xgb
        from battery_pdm.common.features import compute_features
        from battery_pdm.common.models import (
            FAILURE_FEATURE_GROUPS,
            FAILURE_LEAKY_FEATURES,
            train_failure_model,
            evaluate_failure_model,
        )
        from battery_pdm.common.survival import _encode_survival

        from battery_pdm.aws.s3_io import read_parquet as _rp

        labels_raw = _rp(self.labels_path)
        feature_groups = FAILURE_FEATURE_GROUPS
        leaky = FAILURE_LEAKY_FEATURES

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

        # Window alarms per lifecycle
        alarm_parts = []
        for _, row in labels_raw.iterrows():
            original_sid = row.get("original_site_id", row["site_id"])
            start_h = row.get("lifecycle_start_h", 0)
            end_h = row["event_hour"]
            windowed = self.alarms[
                (self.alarms["site_id"] == original_sid)
                & (self.alarms["timestamp_h"] >= start_h)
                & (self.alarms["timestamp_h"] <= end_h)
            ].copy()
            windowed["site_id"] = row["site_id"]
            alarm_parts.append(windowed)
        alarms_windowed = (
            pd.concat(alarm_parts, ignore_index=True)
            if alarm_parts
            else self.alarms.iloc[:0].copy()
        )

        static_per_lifecycle = (
            labels_raw[["site_id", "original_site_id"]]
            .merge(
                self.site_static,
                left_on="original_site_id",
                right_on="site_id",
                suffixes=("", "_static"),
            )
            .drop(columns=["site_id_static"])
        )

        features = compute_features(
            labels=labels_raw,
            groups=feature_groups,
            inputs={"alarms": alarms_windowed, "site_static": static_per_lifecycle},
            ref_time_col="event_hour",
        )
        feature_cols = [
            c
            for c in features.columns
            if c not in ("site_id", "event_hour") and c not in leaky
        ]

        label_only_cols = [
            c
            for c in labels_raw.columns
            if c not in features.columns or c in ("site_id", "event_hour")
        ]
        merged = features.merge(
            labels_raw[label_only_cols], on=["site_id", "event_hour"], how="inner"
        )

        # Group-constrained 3-fold CV
        rng = np.random.default_rng(self.seed)
        if "original_site_id" in merged.columns:
            groups = merged["original_site_id"].values
        else:
            groups = merged["site_id"].values
        unique_groups = np.array(sorted(set(groups)))

        fold_results = []
        for fold in range(3):
            rng.shuffle(unique_groups)
            split_idx = int(len(unique_groups) * 0.75)
            train_groups = set(unique_groups[:split_idx])
            train_mask = np.array([g in train_groups for g in groups])
            test_mask = ~train_mask

            train_df, test_df = merged[train_mask], merged[test_mask]
            if len(test_df) == 0 or test_df["event"].sum() == 0:
                continue

            X_train = train_df[feature_cols].astype(float).fillna(0.0)
            X_test = test_df[feature_cols].astype(float).fillna(0.0)
            y_train = _encode_survival(
                train_df["time_to_event_months"].values, train_df["event"].values
            )
            y_test = _encode_survival(
                test_df["time_to_event_months"].values, test_df["event"].values
            )

            booster = train_failure_model(X_train, y_train, X_test, y_test)
            cindex = evaluate_failure_model(
                booster,
                X_test,
                test_df["time_to_event_months"].values,
                test_df["event"].values,
            )
            fold_results.append(cindex)

        self.failure_mean_cindex = float(np.mean(fold_results)) if fold_results else 0.0
        self.failure_passed = self.failure_mean_cindex >= self.min_cindex_failure

        if self.failure_passed:
            from battery_pdm.monitoring.model_registry import (
                save_model_artifacts,
                save_reference_profile_for_model,
                append_performance_log,
                log_to_mlflow,
                setup_mlflow_tracking,
            )

            X_all = merged[feature_cols].astype(float).fillna(0.0)
            y_all = _encode_survival(
                merged["time_to_event_months"].values, merged["event"].values
            )
            final = train_failure_model(X_all, y_all)
            out_dir = Path("outputs/models/failure_alarms_only")

            meta = save_model_artifacts(
                booster=final,
                feature_cols=feature_cols,
                feature_groups=feature_groups,
                metrics={"mean_cindex": self.failure_mean_cindex},
                model_dir=out_dir,
            )
            # Reference profile from training-time features (critical for drift monitor)
            save_reference_profile_for_model(
                training_features=merged[feature_cols + ["site_id"]],
                feature_cols=feature_cols,
                predictions=final.predict(xgb.DMatrix(X_all)),
                model_dir=out_dir,
            )
            append_performance_log(
                model_name="failure_alarms_only",
                model_version=meta["model_version"],
                metric_name="mean_cindex",
                metric_value=self.failure_mean_cindex,
                n_observations=len(merged),
                feature_hash=meta["feature_hash"],
                extras={"event": "training"},
            )
            if setup_mlflow_tracking("battery-pdm-failure"):
                log_to_mlflow(
                    run_name=f"training_{meta['model_version']}",
                    params={
                        "feature_groups": feature_groups,
                        "n_features": len(feature_cols),
                    },
                    metrics={"mean_cindex": self.failure_mean_cindex},
                    model=final,
                    tags={"model_type": "survival_cox", "promoted": "true"},
                )

        print(
            f"Failure model: C-index={self.failure_mean_cindex:.4f} "
            f"({'PROMOTED' if self.failure_passed else 'REJECTED'})"
        )
        self.next(self.train_drain)

    @step
    def train_drain(self):
        """Train 48h drain predictor (binary classifier)."""
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
        from battery_pdm.common.features import compute_features
        from battery_pdm.common.models import (
            DRAIN_FEATURE_GROUPS,
            DRAIN_LEAKY_FEATURES,
            DRAIN_HORIZON_H,
            DRAIN_SAMPLE_INTERVAL_DAYS,
            train_drain_model,
        )

        horizon_h = DRAIN_HORIZON_H
        sample_interval_days = DRAIN_SAMPLE_INTERVAL_DAYS

        # Build daily labels
        alarms_sorted = self.alarms.sort_values(["site_id", "timestamp_h"]).reset_index(
            drop=True
        )
        max_h = alarms_sorted["timestamp_h"].max()
        lvd_events = alarms_sorted[alarms_sorted["alarm_code"] == "LOAD_DISCONNECT"]
        lvd_by_site = {}
        for _, row in lvd_events.iterrows():
            lvd_by_site.setdefault(row["site_id"], []).append(row["timestamp_h"])
        for site in lvd_by_site:
            lvd_by_site[site] = np.array(sorted(lvd_by_site[site]))

        labels_list = []
        for site_id, group in alarms_sorted.groupby("site_id"):
            site_start = group["timestamp_h"].min()
            site_lvds = lvd_by_site.get(site_id, np.array([]))
            obs_start = site_start + 30 * 24
            for ref_h in np.arange(
                obs_start, max_h - horizon_h, sample_interval_days * 24
            ):
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

        # Compute features
        feature_groups = DRAIN_FEATURE_GROUPS
        labels_for_feat = labels[["site_id", "ref_time_h"]].rename(
            columns={"ref_time_h": "mains_fail_h"}
        )
        features = compute_features(
            labels=labels_for_feat,
            groups=feature_groups,
            inputs={
                "alarms": self.alarms,
                "site_static": self.site_static,
                "schedule": self.schedule,
            },
            ref_time_col="mains_fail_h",
        )
        features = features.rename(columns={"mains_fail_h": "ref_time_h"})

        feature_cols = [
            c
            for c in features.columns
            if c not in ("site_id", "ref_time_h") and c not in DRAIN_LEAKY_FEATURES
        ]

        merged = features.copy()
        merged["drain_event"] = labels["drain_event"].values

        # Group-constrained 60/20/20 split: train / calibration / test
        # The calibration set is held out so isotonic regression doesn't peek at
        # the model's training distribution (would over-fit the calibrator).
        rng = np.random.default_rng(self.seed)
        unique_sites = merged["site_id"].unique()
        rng.shuffle(unique_sites)
        n = len(unique_sites)
        train_sites = set(unique_sites[: int(n * 0.60)])
        cal_sites = set(unique_sites[int(n * 0.60) : int(n * 0.80)])
        test_sites = set(unique_sites[int(n * 0.80) :])

        train_mask = merged["site_id"].isin(train_sites).values
        cal_mask = merged["site_id"].isin(cal_sites).values
        test_mask = merged["site_id"].isin(test_sites).values

        y_train = merged["drain_event"].values[train_mask]
        y_cal = merged["drain_event"].values[cal_mask]
        y_test = merged["drain_event"].values[test_mask]
        X_train = merged.loc[train_mask, feature_cols].astype(float).fillna(0.0)
        X_cal = merged.loc[cal_mask, feature_cols].astype(float).fillna(0.0)
        X_test = merged.loc[test_mask, feature_cols].astype(float).fillna(0.0)

        scale_pos = max(1.0, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        booster = train_drain_model(
            X_train, y_train, X_test, y_test, scale_pos_weight=scale_pos
        )

        # Fit isotonic calibrator on the held-out calibration set
        from battery_pdm.monitoring.model_registry import (
            train_isotonic_calibrator,
            brier_score,
        )

        raw_cal_scores = booster.predict(xgb.DMatrix(X_cal))
        calibrator = train_isotonic_calibrator(raw_cal_scores, y_cal)

        # Evaluate calibration impact on the test set
        raw_test_scores = booster.predict(xgb.DMatrix(X_test))
        cal_test_scores = calibrator.predict(raw_test_scores)
        brier_before = brier_score(y_test, raw_test_scores)
        brier_after = brier_score(y_test, cal_test_scores)
        print(
            f"  Calibration: Brier {brier_before:.4f} -> {brier_after:.4f} "
            f"(delta {brier_after - brier_before:+.4f}, lower is better)"
        )

        scores = raw_test_scores  # AUC is invariant to monotonic calibration, so report on raw
        self.drain_auc = float(roc_auc_score(y_test, scores))
        self.drain_brier_before = brier_before
        self.drain_brier_after = brier_after
        self.drain_passed = self.drain_auc >= self.min_auc_drain

        if self.drain_passed:
            from battery_pdm.monitoring.model_registry import (
                save_model_artifacts,
                save_reference_profile_for_model,
                append_performance_log,
                log_to_mlflow,
                setup_mlflow_tracking,
                save_calibrator,
            )

            # Train final on train+cal combined (test set still untouched, so test metrics
            # remain honest as an out-of-sample estimate of production performance).
            train_and_cal_mask = train_mask | cal_mask
            X_fit = (
                merged.loc[train_and_cal_mask, feature_cols].astype(float).fillna(0.0)
            )
            y_fit = merged["drain_event"].values[train_and_cal_mask]
            scale_all = max(1.0, (len(y_fit) - y_fit.sum()) / max(y_fit.sum(), 1))
            final = train_drain_model(X_fit, y_fit, scale_pos_weight=scale_all)

            # Use the calibrator fit on the CV-phase booster's predictions (which
            # never saw the cal set). Refitting on `final` would leak because final
            # was trained on train+cal combined.
            final_calibrator = calibrator

            out_dir = Path("outputs/models/drain_predictor_48h")
            meta = save_model_artifacts(
                booster=final,
                feature_cols=feature_cols,
                feature_groups=feature_groups,
                metrics={
                    "roc_auc": self.drain_auc,
                    "brier_uncalibrated": self.drain_brier_before,
                    "brier_calibrated": self.drain_brier_after,
                },
                model_dir=out_dir,
                extras={"horizon_h": horizon_h, "calibration_method": "isotonic"},
            )
            save_calibrator(final_calibrator, out_dir)

            # Register in MLflow Model Registry + transition to Staging
            from battery_pdm.monitoring.model_registry import (
                register_model,
                transition_model_stage,
            )

            version = register_model(final, model_name="drain_predictor_48h")
            if version:
                transition_model_stage("drain_predictor_48h", version, "Staging")
                print(f"  Registered drain_predictor_48h v{version} -> Staging")

            save_reference_profile_for_model(
                training_features=merged.loc[
                    train_and_cal_mask, feature_cols + ["site_id"]
                ],
                feature_cols=feature_cols,
                predictions=final_calibrator.predict(final.predict(xgb.DMatrix(X_fit))),
                model_dir=out_dir,
            )
            append_performance_log(
                model_name="drain_predictor_48h",
                model_version=meta["model_version"],
                metric_name="roc_auc",
                metric_value=self.drain_auc,
                n_observations=len(merged),
                feature_hash=meta["feature_hash"],
                extras={
                    "event": "training",
                    "horizon_h": str(horizon_h),
                    "brier_calibrated": str(self.drain_brier_after),
                },
            )
            if setup_mlflow_tracking("battery-pdm-drain"):
                log_to_mlflow(
                    run_name=f"training_{meta['model_version']}",
                    params={
                        "feature_groups": feature_groups,
                        "horizon_h": horizon_h,
                        "n_features": len(feature_cols),
                        "calibration_method": "isotonic",
                    },
                    metrics={
                        "roc_auc": self.drain_auc,
                        "brier_uncalibrated": self.drain_brier_before,
                        "brier_calibrated": self.drain_brier_after,
                    },
                    model=final,
                    tags={"model_type": "binary_logistic", "promoted": "true"},
                )

        print(
            f"Drain predictor (48h): AUC={self.drain_auc:.4f} "
            f"({'PROMOTED' if self.drain_passed else 'REJECTED'})"
        )
        self.next(self.upload_models)

    @step
    def upload_models(self):
        """Upload trained models to S3 if models_output is set."""
        if not self.models_output:
            print("No models_output set — skipping S3 upload (local only).")
            self.next(self.end)
            return

        import boto3
        from battery_pdm.aws.s3_io import _split_s3

        s3 = boto3.client("s3")
        models_dir = Path("outputs/models")
        if not models_dir.exists():
            print("No outputs/models directory — nothing to upload.")
            self.next(self.end)
            return

        uploaded = 0
        for local_file in models_dir.rglob("*"):
            if local_file.is_dir():
                continue
            relative = local_file.relative_to(models_dir)
            s3_key_base = self.models_output.rstrip("/")
            dest = f"{s3_key_base}/{relative.as_posix()}"
            bucket, key = _split_s3(dest)
            s3.upload_file(str(local_file), bucket, key)
            uploaded += 1

        print(f"Uploaded {uploaded} files to {self.models_output}")
        self.next(self.end)

    @step
    def end(self):
        """Report summary."""
        print("\n" + "=" * 60)
        print("TRAINING FLOW SUMMARY")
        print("=" * 60)
        print(
            f"  Failure model:    C-index={self.failure_mean_cindex:.4f}  "
            f"{'PROMOTED' if self.failure_passed else 'REJECTED'}"
        )
        print(
            f"  Drain predictor:  AUC={self.drain_auc:.4f}      "
            f"{'PROMOTED' if self.drain_passed else 'REJECTED'}"
        )
        print("=" * 60)


if __name__ == "__main__":
    TrainingFlow()
