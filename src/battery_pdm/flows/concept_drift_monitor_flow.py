"""ConceptDriftMonitorFlow — labeled-feedback drift detection.

Complements feature drift (drift_monitor_flow.py). This flow:
    1. Loads prior predictions from alerts/
    2. Fetches realized outcomes from the alarm stream
    3. Computes rolling AUC/Brier on labeled feedback
    4. Compares to training baseline → flags concept drift
    5. Logs to MLflow (platform-agnostic, S3 backend in AWS, file local)
    6. Emits CloudWatch metrics when running in AWS

Where feature drift catches "inputs changed", concept drift catches
"the X→Y relationship changed" — only fireable when ground-truth labels
have matured.

Run:
    python -m battery_pdm.flows.concept_drift_monitor_flow run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from metaflow import FlowSpec, Parameter, project, step

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@project(name="battery_pdm")
class ConceptDriftMonitorFlow(FlowSpec):
    """Labeled-feedback concept drift detection — daily check on accumulated labels."""

    model_name = Parameter("model-name", default="drain_predictor_48h")
    model_dir = Parameter("model-dir", default="outputs/models/drain_predictor_48h")
    alarms_path = Parameter("alarms-path", default="outputs/alarms.parquet")
    alerts_dir = Parameter(
        "alerts-dir",
        default="outputs/drain_alerts",
        help="Where prior predictions are logged",
    )
    reports_dir = Parameter("reports-dir", default="outputs/concept_drift_reports")
    horizon_h = Parameter(
        "horizon-h",
        type=int,
        default=48,
        help="Label maturity horizon (48 for drain, 4320 for failure)",
    )
    window_days = Parameter(
        "window-days", type=int, default=7, help="Rolling performance window size"
    )
    auc_degradation_threshold = Parameter(
        "auc-degradation-threshold",
        type=float,
        default=0.05,
        help="Trigger concept drift if rolling AUC drops more than this below baseline",
    )

    @step
    def start(self):
        """Load model metadata + alerts + alarms."""
        model_path = Path(self.model_dir)
        meta_path = model_path / "meta.json"
        if not meta_path.exists():
            print(f"No model meta at {meta_path}. Cannot establish baseline.")
            self.baseline_auc = None
            self.proceed = False
            self.next(self.end)
            return

        self.meta = json.loads(meta_path.read_text())
        self.baseline_auc = float(self.meta.get("metrics", {}).get("roc_auc", 0.0))
        if self.baseline_auc == 0.0:
            # Survival model with C-index instead — adapt
            self.baseline_auc = float(
                self.meta.get("metrics", {}).get("mean_cindex", 0.0)
            )
        print(f"Baseline {self.model_name} AUC/C-index: {self.baseline_auc:.4f}")

        self.alarms = pd.read_parquet(self.alarms_path)
        self.current_h = float(self.alarms["timestamp_h"].max())

        # Load all prior alerts
        alerts_path = Path(self.alerts_dir)
        if not alerts_path.exists():
            print(f"No alerts dir at {alerts_path} — no predictions to validate yet.")
            self.proceed = False
            self.next(self.end)
            return

        alert_files = sorted(alerts_path.glob("*.parquet"))
        if not alert_files:
            print("No alert files yet — no predictions to validate.")
            self.proceed = False
            self.next(self.end)
            return

        dfs = []
        for f in alert_files:
            try:
                df = pd.read_parquet(f)
                if "scoring_hour" in df.columns and "drain_risk_48h" in df.columns:
                    df = df.rename(columns={"drain_risk_48h": "predicted_score"})
                    dfs.append(df[["site_id", "scoring_hour", "predicted_score"]])
            except Exception as exc:
                print(f"  skip {f.name}: {exc}")
        if not dfs:
            self.proceed = False
            self.next(self.end)
            return
        self.predictions = pd.concat(dfs, ignore_index=True)
        print(
            f"Loaded {len(self.predictions):,} prior predictions across {len(alert_files)} batches"
        )
        self.proceed = True
        self.next(self.fetch_labels)

    @step
    def fetch_labels(self):
        """For each past prediction, fetch the realized outcome (LVD within horizon)."""
        from battery_pdm.monitoring.concept_drift import fetch_realized_outcomes

        self.predictions_with_labels = fetch_realized_outcomes(
            self.predictions,
            self.alarms,
            horizon_h=self.horizon_h,
        )
        n_mature = int(
            (self.predictions_with_labels["labels_mature_at_h"] <= self.current_h).sum()
        )
        print(f"  Total predictions: {len(self.predictions_with_labels):,}")
        print(
            f"  Mature labels:     {n_mature:,} ({100 * n_mature / len(self.predictions_with_labels):.1f}%)"
        )
        self.next(self.compute_rolling_performance)

    @step
    def compute_rolling_performance(self):
        """Compute rolling AUC/Brier across the labeled window."""
        from battery_pdm.monitoring.concept_drift import compute_window_performance

        # Window = window_days, slide by 1 day, last 4 windows
        window_h = self.window_days * 24
        windows = []
        for end_h in np.arange(
            self.current_h, self.current_h - 4 * window_h, -window_h
        ):
            start_h = end_h - window_h
            w = compute_window_performance(
                self.predictions_with_labels,
                start_h,
                end_h,
                current_h=self.current_h,
                horizon_h=self.horizon_h,
            )
            windows.append(w)

        # Reverse so oldest first
        self.windows = list(reversed(windows))
        valid = [w for w in self.windows if not np.isnan(w.auc)]
        if valid:
            print(f"  Valid rolling windows: {len(valid)} / {len(self.windows)}")
            for w in valid:
                print(
                    f"    [{w.window_start_h:.0f}h, {w.window_end_h:.0f}h] "
                    f"n={w.n_with_labels} AUC={w.auc:.4f} Brier={w.brier:.4f}"
                )
        else:
            print("  No valid windows yet (need ≥30 mature labels per window).")
        self.next(self.detect_and_decide)

    @step
    def detect_and_decide(self):
        """Compare rolling perf to baseline → flag concept drift if degraded."""
        from battery_pdm.monitoring.concept_drift import detect_concept_drift

        self.drift_result = detect_concept_drift(
            self.windows,
            baseline_auc=self.baseline_auc,
            auc_degradation_threshold=self.auc_degradation_threshold,
        )
        print("\n  Concept drift status:")
        for k, v in self.drift_result.items():
            print(f"    {k}: {v}")
        self.next(self.log_and_emit)

    @step
    def log_and_emit(self):
        """Write report + log to MLflow + emit CloudWatch metrics."""
        out_dir = Path(self.reports_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"concept_drift_h{int(self.current_h):08d}.json"
        # Convert dataclass windows to dicts for JSON
        windows_dump = [
            {
                "window_start_h": w.window_start_h,
                "window_end_h": w.window_end_h,
                "n_predictions": w.n_predictions,
                "n_with_labels": w.n_with_labels,
                "auc": (None if np.isnan(w.auc) else w.auc),
                "brier": (None if np.isnan(w.brier) else w.brier),
                "positive_rate": w.positive_rate,
            }
            for w in self.windows
        ]
        report = {
            "scoring_hour": self.current_h,
            "model_name": self.model_name,
            "baseline_auc": self.baseline_auc,
            "drift_result": self.drift_result,
            "windows": windows_dump,
        }
        report_path.write_text(json.dumps(report, indent=2))
        print(f"  Wrote report to {report_path}")

        # MLflow logging (platform-agnostic — works locally + S3 backend in AWS)
        try:
            from battery_pdm.monitoring.model_registry import (
                setup_mlflow_tracking,
                log_to_mlflow,
            )

            if setup_mlflow_tracking(f"battery-pdm-concept-drift-{self.model_name}"):
                log_to_mlflow(
                    run_name=f"concept_drift_h{int(self.current_h):08d}",
                    params={
                        "model_name": self.model_name,
                        "horizon_h": self.horizon_h,
                        "window_days": self.window_days,
                    },
                    metrics={
                        "baseline_auc": float(self.baseline_auc),
                        "recent_avg_auc": float(
                            self.drift_result.get("recent_avg_auc", 0)
                        ),
                        "auc_degradation": float(
                            self.drift_result.get("auc_degradation", 0)
                        ),
                    },
                    tags={
                        "event": "concept_drift_check",
                        "drift_detected": str(
                            self.drift_result.get("concept_drift_detected", False)
                        ),
                    },
                    artifacts={"report": str(report_path)},
                )
                print("  Logged to MLflow")
        except Exception as exc:
            print(f"  MLflow logging skipped: {exc}")

        # CloudWatch metric emission (no-ops if not in AWS)
        try:
            from battery_pdm.aws.metrics import emit_metric

            emit_metric(
                "ConceptDriftAUC",
                float(self.drift_result.get("recent_avg_auc", 0)),
                dimensions={"ModelName": self.model_name},
            )
            emit_metric(
                "ConceptDriftDegradation",
                float(self.drift_result.get("auc_degradation", 0)),
                dimensions={"ModelName": self.model_name},
            )
            emit_metric(
                "ConceptDriftDetected",
                1 if self.drift_result.get("concept_drift_detected", False) else 0,
                dimensions={"ModelName": self.model_name},
            )
        except Exception as exc:
            print(f"  CloudWatch emission skipped: {exc}")

        # If concept drift detected → augment any existing retrain trigger
        if self.drift_result.get("concept_drift_detected", False):
            trigger_path = Path("outputs/drift_reports/retrain_trigger.json")
            trigger_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "triggered_at_h": self.current_h,
                "reasons": [
                    f"concept_drift: AUC dropped {self.drift_result['auc_degradation']:.4f}"
                ],
                "source": "concept_drift_monitor",
            }
            trigger_path.write_text(json.dumps(payload, indent=2))
            print("  *** Concept drift detected → wrote retrain_trigger.json ***")

        self.next(self.end)

    @step
    def end(self):
        print(f"ConceptDriftMonitorFlow complete for {self.model_name}.")


if __name__ == "__main__":
    ConceptDriftMonitorFlow()
