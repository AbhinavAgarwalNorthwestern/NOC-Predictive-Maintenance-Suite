"""DriftMonitorFlow — daily check for feature/prediction drift.

Compares current scoring-window features against the training-time reference
profile. If drift exceeds thresholds, writes a retrain trigger.

DAG: start -> compute_current_features -> detect_drift -> decide -> end

Run:
    python -m battery_pdm.flows.drift_monitor_flow run

Scheduled: daily, after DrainPredictorFlow completes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, Parameter, card, current, project, step
from metaflow.cards import Markdown, Table

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@project(name="battery_pdm")
class DriftMonitorFlow(FlowSpec):
    """Daily drift monitoring for battery PdM models."""

    alarms_path = Parameter("alarms-path", default="outputs/alarms.parquet")
    site_static_path = Parameter(
        "site-static-path", default="outputs/site_static.parquet"
    )
    reference_profile_path = Parameter(
        "reference-profile",
        default="outputs/models/reference_profile.json",
    )
    model_dir = Parameter("model-dir", default="outputs/models/drain_predictor_48h")
    drift_output_dir = Parameter("drift-output-dir", default="outputs/drift_reports")
    window_days = Parameter(
        "window-days", type=int, default=7, help="Days of recent data to compare"
    )
    use_evidently = Parameter(
        "use-evidently",
        type=bool,
        default=False,
        help="If true, use Evidently AI's drift implementation instead of our hand-rolled PSI. "
        "Both produce the same output format; Evidently is the industry-standard tool.",
    )
    sim_hour = Parameter("sim-hour", type=float, default=0.0)

    @step
    def start(self):
        """Load reference profile and determine monitoring window."""
        from battery_pdm.aws.s3_io import exists, read_json, read_parquet

        if not exists(self.reference_profile_path):
            print(
                f"No reference profile at {self.reference_profile_path}. Building from current data."
            )
            self.has_reference = False
        else:
            self.reference_profile = read_json(self.reference_profile_path)
            self.has_reference = True
            print(
                f"Reference profile loaded: {self.reference_profile['n_samples']} training samples"
            )

        self.alarms = read_parquet(self.alarms_path)
        self.site_static = read_parquet(self.site_static_path)

        if self.sim_hour > 0:
            self.current_h = self.sim_hour
        else:
            self.current_h = float(self.alarms["timestamp_h"].max())

        self.window_start_h = self.current_h - self.window_days * 24
        print(
            f"Monitoring window: [{self.window_start_h:.0f}h, {self.current_h:.0f}h] "
            f"({self.window_days} days)"
        )
        self.next(self.compute_current_features)

    @step
    def compute_current_features(self):
        """Compute features for recent observations (same pipeline as scoring)."""
        from battery_pdm.common.features import compute_features
        from battery_pdm.synth.load_shedding import build_load_shedding_schedule

        schedule = build_load_shedding_schedule(n_months=36, seed=42)

        # Score all sites at current_h (same as DrainPredictorFlow)
        labels = pd.DataFrame(
            {
                "site_id": self.site_static["site_id"].values,
                "ref_time_h": self.current_h,
            }
        )

        alarms_pit = self.alarms[self.alarms["timestamp_h"] <= self.current_h]
        feature_groups = [
            "alarm_history",
            "site_static",
            "soc_proxy",
            "load_shedding_schedule",
        ]

        self.current_features = compute_features(
            labels=labels.rename(columns={"ref_time_h": "mains_fail_h"}),
            groups=feature_groups,
            inputs={
                "alarms": alarms_pit,
                "site_static": self.site_static,
                "schedule": schedule,
            },
            ref_time_col="mains_fail_h",
        )
        self.current_features = self.current_features.rename(
            columns={"mains_fail_h": "ref_time_h"}
        )
        print(f"Computed features for {len(self.current_features)} sites")

        # Also get model predictions for prediction drift
        from battery_pdm.aws.s3_io import (
            exists as s3_exists,
            read_json as s3_read_json,
            read_bytes as s3_read_bytes,
            join_path as s3_join,
        )

        booster_path = s3_join(self.model_dir, "booster.json")
        if s3_exists(booster_path):
            import xgboost as xgb

            meta = s3_read_json(s3_join(self.model_dir, "meta.json"))
            feature_cols = meta["feature_cols"]
            booster = xgb.Booster()
            booster.load_model(bytearray(s3_read_bytes(booster_path)))

            available = [c for c in feature_cols if c in self.current_features.columns]
            X = self.current_features[available].astype(float).fillna(0.0)
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[feature_cols]
            self.current_predictions = booster.predict(xgb.DMatrix(X))
        else:
            self.current_predictions = None

        self.next(self.detect_drift)

    @card(type="html", id="evidently_report")
    @card(type="default", id="summary")
    @step
    def detect_drift(self):
        """Compare current features/predictions against reference profile."""
        from battery_pdm.monitoring.drift import (
            detect_drift,
        )

        if not self.has_reference:
            raise FileNotFoundError(
                f"No reference_profile.json at {self.reference_profile_path}. "
                f"Reference profile MUST be created at training time and saved "
                f"with the model. Run TrainingFlow first; verify it called "
                f"save_reference_profile_for_model(). Do NOT bootstrap from production."
            )

        from battery_pdm.aws.s3_io import (
            read_json as s3_read_json,
            join_path as s3_join,
        )

        meta = s3_read_json(s3_join(self.model_dir, "meta.json"))
        feature_cols = meta["feature_cols"]
        available = [c for c in feature_cols if c in self.current_features.columns]

        if self.use_evidently:
            print("  Using Evidently AI drift detection (industry-standard tooling)")
            from battery_pdm.monitoring.evidently_drift import detect_drift_evidently

            self.drift_report = detect_drift_evidently(
                reference_profile=self.reference_profile,
                current_features=self.current_features,
                current_predictions=self.current_predictions,
                feature_cols=available,
            )
        else:
            self.drift_report = detect_drift(
                reference_profile=self.reference_profile,
                current_features=self.current_features,
                current_predictions=self.current_predictions,
                feature_cols=available,
            )

        summary = self.drift_report["summary"]
        print(f"\nDrift Status: {summary['status']}")
        print(f"  Features monitored: {summary['n_features_monitored']}")
        print(f"  Significant drift:  {summary['n_significant_drift']}")
        print(f"  Moderate drift:     {summary['n_moderate_drift']}")

        # Emit CloudWatch metrics for the dashboard
        try:
            from battery_pdm.aws.metrics import emit_drift_report

            pred_drift = self.drift_report.get("prediction_drift", {}) or {}
            emit_drift_report(
                model_name="drain_predictor_48h",
                n_significant=int(summary.get("n_significant_drift", 0)),
                n_moderate=int(summary.get("n_moderate_drift", 0)),
                prediction_psi=float(pred_drift.get("prediction_psi", 0.0)),
                retrain_recommended=bool(summary.get("retrain_recommended", False)),
            )
        except Exception as exc:
            print(f"  (CloudWatch metric emission skipped: {exc})")
        if summary["reasons"]:
            print("  Reasons:")
            for r in summary["reasons"]:
                print(f"    - {r}")

        feature_drift = self.drift_report["feature_drift"]
        if not feature_drift.empty:
            drifted = feature_drift[feature_drift["drift_level"] != "STABLE"]
            if not drifted.empty:
                print("\n  Drifted features:")
                print(
                    drifted[
                        ["feature", "psi", "drift_level", "ref_mean", "cur_mean"]
                    ].to_string(index=False)
                )

        # Render Evidently HTML report as interactive card (when using Evidently)
        if self.use_evidently:
            from battery_pdm.monitoring.evidently_drift import (
                generate_evidently_html_report,
            )

            html_content = generate_evidently_html_report(
                reference_profile=self.reference_profile,
                current_features=self.current_features,
                feature_cols=[
                    c
                    for c in meta["feature_cols"]
                    if c in self.current_features.columns
                ],
            )
            if html_content:
                current.card["evidently_report"].append(html_content)

        current.card["summary"].append(
            Markdown(f"# Drift Monitor — Status: {summary['status']}")
        )
        current.card["summary"].append(
            Markdown(
                f"**Significant drift features:** {summary['n_significant_drift']}"
            )
        )
        current.card["summary"].append(
            Markdown(f"**Moderate drift features:** {summary['n_moderate_drift']}")
        )
        current.card["summary"].append(
            Markdown(f"**Retrain recommended:** {summary['retrain_recommended']}")
        )
        fd = self.drift_report["feature_drift"]
        if not fd.empty:
            top_drift = fd.head(10)[["feature", "psi", "drift_level"]]
            current.card["summary"].append(
                Table(headers=list(top_drift.columns), data=top_drift.values.tolist())
            )

        self.next(self.decide)

    @step
    def decide(self):
        """Write drift report and optionally trigger retraining."""
        from battery_pdm.aws.s3_io import join_path, write_json, mkdir

        mkdir(self.drift_output_dir)
        summary = self.drift_report["summary"]

        report_path = join_path(
            self.drift_output_dir, f"drift_report_h{int(self.current_h):08d}.json"
        )
        report_data = {
            "scoring_hour": self.current_h,
            "window_days": self.window_days,
            "summary": summary,
            "prediction_drift": self.drift_report.get("prediction_drift", {}),
        }
        if not self.drift_report["feature_drift"].empty:
            report_data["top_drifted_features"] = (
                self.drift_report["feature_drift"]
                .head(10)[["feature", "psi", "drift_level", "mean_shift_std"]]
                .to_dict(orient="records")
            )
        write_json(report_data, report_path)
        print(f"Drift report saved to {report_path}")

        if summary.get("retrain_recommended", False):
            from battery_pdm.monitoring.model_registry import TRIGGER_PATH

            TRIGGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            TRIGGER_PATH.write_text(
                json.dumps(
                    {
                        "triggered_at_h": self.current_h,
                        "reasons": summary["reasons"],
                    },
                    indent=2,
                )
            )
            print(f"\n*** RETRAIN TRIGGERED *** -> {TRIGGER_PATH}")
        else:
            print("No retraining needed.")

        self.next(self.end)

    @step
    def end(self):
        status = self.drift_report["summary"]["status"]
        print(f"DriftMonitorFlow complete. Status: {status}")


if __name__ == "__main__":
    DriftMonitorFlow()
