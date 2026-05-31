"""RollbackMonitorFlow — automated rollback if promoted model degrades.

Safety net for proactive parallel-path pattern. Runs daily (after scoring).
If a model was promoted within the last 48h and its production Brier/AUC is
worse than the archived champion's, auto-reverts.

Why 48h window:
    - Drain predictor labels mature in 48h. After promotion, we have exactly
      one day's worth of labeled feedback by the next rollback check.
    - If the new model is genuinely worse, 1 day of scored predictions with
      realized labels is enough signal (200+ sites × 1 day = 200+ observations).

Decision logic:
    1. Check if a promotion happened in the last 48h (read performance log)
    2. Fetch realized labels for the post-promotion scoring window
    3. Compute Brier score on those labels
    4. If Brier > champion_brier_at_training × (1 + tolerance) → ROLLBACK
    5. Rollback = restore archived champion, delete current model dir, re-copy

This flow is idempotent: if no recent promotion exists, it no-ops.

Run:
    python -m battery_pdm.flows.rollback_monitor_flow run
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
class RollbackMonitorFlow(FlowSpec):
    """Automated rollback for recently promoted models."""

    model_name = Parameter("model-name", default="drain_predictor_48h")
    models_root = Parameter("models-root", default="outputs/models")
    alarms_path = Parameter("alarms-path", default="outputs/alarms.parquet")
    alerts_dir = Parameter("alerts-dir", default="outputs/drain_alerts")
    rollback_window_h = Parameter(
        "rollback-window-h",
        type=int,
        default=48,
        help="Only check models promoted within this many hours",
    )
    brier_tolerance = Parameter(
        "brier-tolerance",
        type=float,
        default=0.10,
        help="Rollback if production Brier > training Brier × (1 + tolerance). "
        "0.10 = allow 10% degradation before rolling back.",
    )
    min_observations = Parameter(
        "min-observations",
        type=int,
        default=50,
        help="Need at least this many labeled observations to make a rollback decision",
    )

    @step
    def start(self):
        """Check if a promotion happened recently."""
        self.champion_dir = Path(self.models_root) / self.model_name
        self.needs_check = False
        self.should_rollback = False
        meta_path = self.champion_dir / "meta.json"

        if not meta_path.exists():
            print(f"No model at {self.champion_dir}. Nothing to monitor.")
            self.next(self.evaluate_production)
            return

        meta = json.loads(meta_path.read_text())
        trained_at = meta.get("trained_at", "")

        if not trained_at:
            print("No trained_at timestamp in meta. Skipping rollback check.")
            self.next(self.evaluate_production)
            return

        from datetime import datetime, timezone

        try:
            promotion_time = datetime.fromisoformat(trained_at)
        except ValueError:
            print(f"Cannot parse trained_at={trained_at}. Skipping.")
            self.next(self.evaluate_production)
            return

        now = datetime.now(timezone.utc)
        hours_since_promotion = (now - promotion_time).total_seconds() / 3600

        if hours_since_promotion > self.rollback_window_h:
            print(
                f"Last promotion was {hours_since_promotion:.0f}h ago "
                f"(> {self.rollback_window_h}h window). No rollback check needed."
            )
            self.next(self.evaluate_production)
            return

        self.meta = meta
        self.training_brier = meta.get("metrics", {}).get("brier_calibrated", None)
        if self.training_brier is None:
            print("No brier_calibrated in model metrics. Cannot evaluate rollback.")
            self.next(self.evaluate_production)
            return

        self.needs_check = True
        print(
            f"Model promoted {hours_since_promotion:.1f}h ago. "
            f"Training Brier: {self.training_brier:.4f}. Checking production performance."
        )
        self.next(self.evaluate_production)

    @step
    def evaluate_production(self):
        """Compute production Brier from post-promotion scored predictions."""
        if not self.needs_check:
            self.next(self.decide)
            return

        from battery_pdm.monitoring.concept_drift import fetch_realized_outcomes

        alerts_path = Path(self.alerts_dir)
        if not alerts_path.exists():
            print(f"No alerts dir at {alerts_path}. Cannot evaluate.")
            self.should_rollback = False
            self.next(self.decide)
            return

        alert_files = sorted(alerts_path.glob("*.parquet"))
        if not alert_files:
            print("No alert files yet. Cannot evaluate.")
            self.should_rollback = False
            self.next(self.decide)
            return

        # Load recent predictions (last few batches)
        dfs = []
        for f in alert_files[-14:]:
            try:
                df = pd.read_parquet(f)
                if "scoring_hour" in df.columns and "drain_risk_48h" in df.columns:
                    df = df.rename(columns={"drain_risk_48h": "predicted_score"})
                    dfs.append(df[["site_id", "scoring_hour", "predicted_score"]])
            except Exception:
                continue

        if not dfs:
            print("No valid alert files with predictions. Cannot evaluate.")
            self.should_rollback = False
            self.next(self.decide)
            return

        predictions = pd.concat(dfs, ignore_index=True)
        alarms = pd.read_parquet(self.alarms_path)
        current_h = float(alarms["timestamp_h"].max())

        enriched = fetch_realized_outcomes(predictions, alarms, horizon_h=48)
        mature = enriched[enriched["labels_mature_at_h"] <= current_h]

        if len(mature) < self.min_observations:
            print(
                f"Only {len(mature)} mature observations (need {self.min_observations}). "
                f"Waiting for more data."
            )
            self.should_rollback = False
            self.next(self.decide)
            return

        y = mature["realized_label"].values
        p = mature["predicted_score"].values
        self.production_brier = float(np.mean((p - y) ** 2))
        self.n_observations = len(mature)

        threshold = self.training_brier * (1 + self.brier_tolerance)
        self.should_rollback = self.production_brier > threshold

        print(
            f"Production Brier: {self.production_brier:.4f} "
            f"(on {self.n_observations} observations)"
        )
        print(f"Training Brier:   {self.training_brier:.4f}")
        print(
            f"Threshold:        {threshold:.4f} "
            f"(training × {1 + self.brier_tolerance:.2f})"
        )
        print(f"Decision:         {'ROLLBACK' if self.should_rollback else 'KEEP'}")

        self.next(self.decide)

    @step
    def decide(self):
        """Execute rollback if needed."""
        if not self.needs_check or not self.should_rollback:
            self.next(self.end)
            return

        archive_dir = Path(self.models_root) / "archive"
        if not archive_dir.exists():
            print(
                "No archive directory. Cannot rollback (no previous champion to restore)."
            )
            self.next(self.end)
            return

        # Find most recent archive for this model
        archives = sorted(
            [
                d
                for d in archive_dir.iterdir()
                if d.is_dir() and d.name.startswith(self.model_name)
            ],
            reverse=True,
        )
        if not archives:
            print(f"No archived versions of {self.model_name}. Cannot rollback.")
            self.next(self.end)
            return

        restore_from = archives[0]
        print("\n*** ROLLING BACK ***")
        print(f"  Restoring from: {restore_from}")
        print(f"  Overwriting:    {self.champion_dir}")

        # Save current (bad) model to a "rolled_back" dir for post-mortem
        rollback_archive = (
            archive_dir
            / f"{self.model_name}_rolled_back_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        rollback_archive.mkdir(parents=True, exist_ok=True)
        for f in self.champion_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, rollback_archive / f.name)

        # Restore archived champion
        for f in restore_from.iterdir():
            if f.is_file():
                shutil.copy2(f, self.champion_dir / f.name)

        print(f"  Rolled-back model archived to: {rollback_archive}")
        print(f"  Champion restored from: {restore_from}")

        # Emit CloudWatch metric
        try:
            from battery_pdm.aws.metrics import emit_metric

            emit_metric(
                "ModelRollback", 1, "Count", dimensions={"ModelName": self.model_name}
            )
        except Exception:
            pass

        # Log to performance log
        try:
            from battery_pdm.monitoring.model_registry import append_performance_log

            append_performance_log(
                model_name=self.model_name,
                model_version=self.meta.get("model_version", "unknown"),
                metric_name="brier_calibrated",
                metric_value=self.production_brier,
                n_observations=self.n_observations,
                feature_hash=self.meta.get("feature_hash", ""),
                extras={
                    "event": "rollback",
                    "reason": f"production_brier={self.production_brier:.4f} > threshold",
                },
            )
        except Exception:
            pass

        self.next(self.end)

    @step
    def end(self):
        if not self.needs_check:
            print("RollbackMonitorFlow: no recent promotion, nothing to check.")
        elif self.should_rollback:
            print(f"RollbackMonitorFlow: ROLLED BACK {self.model_name}")
        else:
            print(f"RollbackMonitorFlow: {self.model_name} performing well, no action.")


if __name__ == "__main__":
    RollbackMonitorFlow()
