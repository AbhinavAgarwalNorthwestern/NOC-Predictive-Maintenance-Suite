"""TrafficFlow — synthetic load against a deployed model + label generation.

Two modes:
    --mode=traffic: send synthetic requests to a deployed model (validates serving)
    --mode=labels: generate labels for captured predictions (validates monitoring loop)

This is the missing piece that ml.school has: it exercises the WHOLE pipeline
including the deployed model, not just the offline scoring path.

Run:
    python -m battery_pdm.flows.traffic_flow run --mode traffic
    python -m battery_pdm.flows.traffic_flow run --mode labels
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
class TrafficFlow(FlowSpec):
    """Synthetic traffic + label generation for monitoring loop validation."""

    mode = Parameter("mode", default="traffic", help="'traffic' or 'labels'")
    backend_spec = Parameter(
        "backend",
        default="battery_pdm.backends.local.LocalBackend",
        help="Python path to backend class",
    )
    model_name = Parameter("model-name", default="drain_predictor_48h")
    n_samples = Parameter("n-samples", type=int, default=10)
    inject_drift = Parameter(
        "inject-drift",
        type=bool,
        default=False,
        help="Add synthetic drift to numerical features",
    )
    alarms_path = Parameter("alarms-path", default="outputs/alarms.parquet")
    site_static_path = Parameter(
        "site-static-path", default="outputs/site_static.parquet"
    )

    @step
    def start(self):
        if self.mode not in ("traffic", "labels"):
            raise ValueError(f"Invalid mode: {self.mode}. Use 'traffic' or 'labels'.")
        print(f"Mode: {self.mode}")
        if self.mode == "traffic":
            self.next(self.send_traffic)
        else:
            self.next(self.generate_labels)

    @step
    def send_traffic(self):
        """Send synthetic prediction requests to deployed model."""
        from battery_pdm.backends import load_backend, BackendConfig
        from battery_pdm.common.features import compute_features
        from battery_pdm.synth.load_shedding import build_load_shedding_schedule

        alarms = pd.read_parquet(self.alarms_path)
        sites = pd.read_parquet(self.site_static_path)
        schedule = build_load_shedding_schedule(n_months=36, seed=42)

        sample_sites = sites["site_id"].sample(self.n_samples, random_state=42).tolist()
        labels_df = pd.DataFrame(
            {"site_id": sample_sites, "ref_time_h": float(alarms["timestamp_h"].max())}
        )

        features = compute_features(
            labels=labels_df.rename(columns={"ref_time_h": "mains_fail_h"}),
            groups=[
                "alarm_history",
                "site_static",
                "soc_proxy",
                "load_shedding_schedule",
            ],
            inputs={"alarms": alarms, "site_static": sites, "schedule": schedule},
            ref_time_col="mains_fail_h",
        ).rename(columns={"mains_fail_h": "ref_time_h"})

        if self.inject_drift:
            for col in features.select_dtypes(include=[np.number]).columns:
                if col not in ("ref_time_h",):
                    features[col] = features[col] * (
                        1 + 0.3 * np.random.randn(len(features))
                    )
            print("  Drift injected: 30% noise on numerical features")

        backend = load_backend(self.backend_spec, BackendConfig(name="traffic"))
        backend.deploy("ignored", self.model_name)

        for batch_start in range(0, len(features), 10):
            batch = features.iloc[batch_start : batch_start + 10]
            result = backend.invoke(self.model_name, batch)
            print(
                f"  Batch {batch_start}: invoked {len(batch)} samples, result: {type(result).__name__}"
            )

        self.n_sent = len(features)
        self.next(self.end)

    @step
    def generate_labels(self):
        """Generate synthetic labels for predictions logged earlier."""
        alerts_dir = Path("outputs/drain_alerts")
        if not alerts_dir.exists():
            print("No alerts directory found")
            self.n_labeled = 0
            self.next(self.end)
            return

        alarms = pd.read_parquet(self.alarms_path)
        from battery_pdm.monitoring.concept_drift import fetch_realized_outcomes

        alert_files = sorted(alerts_dir.glob("drain_alerts_*.parquet"))
        if not alert_files:
            self.n_labeled = 0
            self.next(self.end)
            return

        dfs = [pd.read_parquet(f) for f in alert_files[-30:]]
        predictions = pd.concat(dfs, ignore_index=True)
        if "scoring_hour" not in predictions.columns:
            self.n_labeled = 0
            self.next(self.end)
            return

        predictions = predictions.rename(columns={"drain_risk_48h": "predicted_score"})
        labeled = fetch_realized_outcomes(predictions, alarms, horizon_h=48)
        labels_path = Path("outputs/drain_alerts_with_labels.parquet")
        labeled.to_parquet(labels_path, index=False)
        self.n_labeled = len(labeled)
        print(f"  Labeled {self.n_labeled} predictions -> {labels_path}")
        self.next(self.end)

    @step
    def end(self):
        if self.mode == "traffic":
            print(f"TrafficFlow complete: sent {self.n_sent} samples")
        else:
            print(f"TrafficFlow complete: labeled {self.n_labeled} predictions")


if __name__ == "__main__":
    TrafficFlow()
