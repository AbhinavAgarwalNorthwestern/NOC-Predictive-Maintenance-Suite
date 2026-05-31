"""ShadowPromotionFlow — validates shadow predictions against realized labels.

This is the **blue/green decision flow** for batch ML. It runs daily and:
    1. Reads shadow_comparisons/ (champion + shadow predictions side by side)
    2. Fetches realized labels from alarm stream
    3. Computes AUC/Brier of champion AND shadow on the same labeled feedback
    4. If shadow beats champion by margin AND enough labels accumulated:
       PROMOTES shadow to champion (atomic swap)
       Archives old champion
    5. Logs everything to MLflow (platform-agnostic)
    6. Emits CloudWatch metrics

This is what enables "challenger waits in wings until labels prove it's better"
- the production pattern at most mature ML orgs.
"""

from __future__ import annotations

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
class ShadowPromotionFlow(FlowSpec):
    """Validate shadow model on labeled feedback + promote if proven better."""

    model_name = Parameter("model-name", default="drain_predictor_48h")
    models_root = Parameter("models-root", default="outputs/models")
    alarms_path = Parameter("alarms-path", default="outputs/alarms.parquet")
    alerts_dir = Parameter("alerts-dir", default="outputs/drain_alerts")
    horizon_h = Parameter("horizon-h", type=int, default=48)
    min_days_of_labels = Parameter(
        "min-days-of-labels",
        type=int,
        default=7,
        help="Don't promote shadow until N days of labeled feedback exist",
    )
    promotion_margin = Parameter(
        "promotion-margin",
        type=float,
        default=0.005,
        help="Shadow must beat champion by this AUC margin on shared labeled data",
    )

    @step
    def start(self):
        """Check if a shadow model exists for this model."""
        self.champion_dir = Path(self.models_root) / self.model_name
        self.shadow_dir = Path(self.models_root) / f"{self.model_name}_shadow"
        self.has_shadow = False
        self.has_data = False
        self.ready_to_decide = False
        self.decision = "no_action"

        if (
            not self.shadow_dir.exists()
            or not (self.shadow_dir / "booster.json").exists()
        ):
            print(f"No shadow model at {self.shadow_dir}. Nothing to validate.")
        elif (
            not self.champion_dir.exists()
            or not (self.champion_dir / "booster.json").exists()
        ):
            print(f"No champion at {self.champion_dir}. Cannot compare.")
        else:
            self.has_shadow = True
            print(f"Champion: {self.champion_dir}")
            print(f"Shadow:   {self.shadow_dir}")

        self.next(self.load_shadow_comparisons)

    @step
    def load_shadow_comparisons(self):
        """Load all logged shadow vs champion predictions."""
        if not self.has_shadow:
            self.next(self.fetch_realized_labels)
            return

        shadow_dir = Path(self.alerts_dir) / "shadow_comparisons"
        if not shadow_dir.exists():
            print(f"No shadow comparisons at {shadow_dir} -- waiting for scoring flow.")
        elif not sorted(shadow_dir.glob("shadow_h*.parquet")):
            print("No shadow files yet.")
        else:
            shadow_files = sorted(shadow_dir.glob("shadow_h*.parquet"))
            dfs = []
            for f in shadow_files:
                try:
                    dfs.append(pd.read_parquet(f))
                except Exception as exc:
                    print(f"  skip {f.name}: {exc}")
            if dfs:
                self.shadow_data = pd.concat(dfs, ignore_index=True)
                print(
                    f"Loaded {len(self.shadow_data):,} shadow predictions "
                    f"across {len(shadow_files)} files"
                )
                self.has_data = True

        self.next(self.fetch_realized_labels)

    @step
    def fetch_realized_labels(self):
        """For each shadow prediction, fetch realized outcome."""
        if not self.has_data:
            self.next(self.evaluate)
            return

        from battery_pdm.monitoring.concept_drift import fetch_realized_outcomes

        alarms = pd.read_parquet(self.alarms_path)
        self.current_h = float(alarms["timestamp_h"].max())

        adapted = self.shadow_data.rename(
            columns={"predicted_score_champion": "predicted_score"}
        )
        enriched = fetch_realized_outcomes(adapted, alarms, horizon_h=self.horizon_h)
        self.labeled = enriched.copy()
        self.labeled["predicted_score_champion"] = self.shadow_data[
            "predicted_score_champion"
        ].values
        self.labeled["predicted_score_shadow"] = self.shadow_data[
            "predicted_score_shadow"
        ].values

        self.mature = self.labeled[
            self.labeled["labels_mature_at_h"] <= self.current_h
        ].copy()
        n_mature = len(self.mature)
        n_positive = int(self.mature["realized_label"].sum()) if n_mature else 0
        print(f"Mature labels: {n_mature:,} ({n_positive} positives)")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Compute AUC/Brier on the labeled shadow comparisons."""
        if not self.has_data:
            self.next(self.decide)
            return

        from sklearn.metrics import roc_auc_score

        if len(self.mature) < 30 or self.mature["realized_label"].nunique() < 2:
            print(f"Not enough mature labels yet ({len(self.mature)} labeled obs).")
            self.next(self.decide)
            return

        y = self.mature["realized_label"].values
        p_champ = self.mature["predicted_score_champion"].values
        p_shadow = self.mature["predicted_score_shadow"].values

        self.champion_auc = float(roc_auc_score(y, p_champ))
        self.shadow_auc = float(roc_auc_score(y, p_shadow))
        self.champion_brier = float(np.mean((p_champ - y) ** 2))
        self.shadow_brier = float(np.mean((p_shadow - y) ** 2))

        label_span_h = (
            self.mature["scoring_hour"].max() - self.mature["scoring_hour"].min()
        )
        self.days_of_labels = float(label_span_h / 24)

        print(
            f"\n  Champion: AUC={self.champion_auc:.4f}  Brier={self.champion_brier:.4f}"
        )
        print(f"  Shadow:   AUC={self.shadow_auc:.4f}  Brier={self.shadow_brier:.4f}")
        print(
            f"  Margin: {self.shadow_auc - self.champion_auc:+.4f} "
            f"(required: +{self.promotion_margin})"
        )
        print(
            f"  Label maturity: {self.days_of_labels:.1f} days "
            f"(required: {self.min_days_of_labels})"
        )
        self.ready_to_decide = True
        self.next(self.decide)

    @step
    def decide(self):
        """Promote, keep waiting, or discard shadow."""
        if not self.ready_to_decide:
            self.decision = "insufficient_labels"
        else:
            maturity_met = self.days_of_labels >= self.min_days_of_labels
            margin_met = (self.shadow_auc - self.champion_auc) >= self.promotion_margin

            if maturity_met and margin_met:
                self.decision = "promote_shadow"
            elif maturity_met and not margin_met:
                self.decision = "discard_shadow"
            else:
                self.decision = "wait_for_more_labels"

        print(f"  Decision: {self.decision}")
        self.next(self.log_and_act)

    @step
    def log_and_act(self):
        """Execute the decision + log to MLflow + CloudWatch."""
        if not self.has_shadow or not self.has_data:
            self.next(self.end)
            return

        try:
            from battery_pdm.monitoring.model_registry import (
                setup_mlflow_tracking,
                log_to_mlflow,
            )

            if setup_mlflow_tracking(f"battery-pdm-shadow-promotion-{self.model_name}"):
                metrics = {}
                if self.ready_to_decide:
                    metrics = {
                        "champion_auc": self.champion_auc,
                        "shadow_auc": self.shadow_auc,
                        "champion_brier": self.champion_brier,
                        "shadow_brier": self.shadow_brier,
                        "days_of_labels": self.days_of_labels,
                        "margin": self.shadow_auc - self.champion_auc,
                    }
                log_to_mlflow(
                    run_name=f"shadow_decision_h{int(self.current_h):08d}",
                    params={
                        "model_name": self.model_name,
                        "min_days_of_labels": self.min_days_of_labels,
                        "promotion_margin": self.promotion_margin,
                    },
                    metrics=metrics,
                    tags={"decision": self.decision},
                )
                print("  Logged decision to MLflow")
        except Exception as exc:
            print(f"  MLflow logging skipped: {exc}")

        if self.decision == "promote_shadow":
            archive = (
                Path(self.models_root)
                / "archive"
                / f"{self.model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )
            archive.mkdir(parents=True, exist_ok=True)
            for f in self.champion_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, archive / f.name)
            print(f"  Archived old champion -> {archive}")

            for f in self.shadow_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, self.champion_dir / f.name)
            print("  PROMOTED shadow -> champion")

            shutil.rmtree(self.shadow_dir)
            print(f"  Cleaned up {self.shadow_dir}")

            try:
                from battery_pdm.aws.metrics import emit_metric

                emit_metric(
                    "ShadowPromoted",
                    1,
                    "Count",
                    dimensions={"ModelName": self.model_name},
                )
            except Exception:
                pass

        elif self.decision == "discard_shadow":
            shutil.rmtree(self.shadow_dir)
            print("  DISCARDED shadow (didn't beat champion by margin after maturity)")
            try:
                from battery_pdm.aws.metrics import emit_metric

                emit_metric(
                    "ShadowDiscarded",
                    1,
                    "Count",
                    dimensions={"ModelName": self.model_name},
                )
            except Exception:
                pass

        else:
            print("  Shadow stays, will re-evaluate tomorrow")
            try:
                from battery_pdm.aws.metrics import emit_metric

                emit_metric(
                    "ShadowWaiting",
                    1,
                    "Count",
                    dimensions={"ModelName": self.model_name},
                )
            except Exception:
                pass

        self.next(self.end)

    @step
    def end(self):
        print(f"ShadowPromotionFlow complete for {self.model_name}.")


if __name__ == "__main__":
    ShadowPromotionFlow()
