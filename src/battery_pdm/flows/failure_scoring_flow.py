"""FailureScoringFlow — weekly batch: flags batteries due for replacement.

Runs once per week. For each battery lifecycle (active sites), computes:
    - Alarm-derived features + SoC proxy at current time
    - Cox risk score from the failure model
    - Ranks sites by replacement urgency

Outputs a prioritized replacement list for maintenance planning.

Run:
    python -m battery_pdm.flows.failure_scoring_flow run

Scheduled: weekly (Sunday night, before Monday planning)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, Parameter, project, step

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@project(name="battery_pdm")
class FailureScoringFlow(FlowSpec):
    """Weekly batch scoring: battery replacement risk for all active sites."""

    alarms_path = Parameter("alarms-path", default="outputs/alarms.parquet")
    site_static_path = Parameter(
        "site-static-path", default="outputs/site_static.parquet"
    )
    model_dir = Parameter("model-dir", default="outputs/models/failure_alarms_only")
    alerts_dir = Parameter("alerts-dir", default="outputs/failure_alerts")
    sim_hour = Parameter(
        "sim-hour",
        type=float,
        default=0.0,
        help="Simulated current hour (0=use max from alarms).",
    )

    @step
    def start(self):
        """Load model and data."""
        import xgboost as xgb
        from battery_pdm.aws.s3_io import (
            exists,
            read_json,
            read_bytes,
            join_path,
            read_parquet,
        )

        booster_path = join_path(self.model_dir, "booster.json")
        if not exists(booster_path):
            raise FileNotFoundError(
                f"No failure model at {self.model_dir}. Run TrainingFlow first."
            )

        self.booster = xgb.Booster()
        self.booster.load_model(bytearray(read_bytes(booster_path)))
        self.meta = read_json(join_path(self.model_dir, "meta.json"))

        self.alarms = read_parquet(self.alarms_path)
        self.site_static = read_parquet(self.site_static_path)

        if self.sim_hour > 0:
            self.current_h = self.sim_hour
        else:
            self.current_h = float(self.alarms["timestamp_h"].max())

        print(f"Scoring time: {self.current_h:.1f}h")
        print(f"Sites: {len(self.site_static)}")
        self.next(self.compute_features)

    @step
    def compute_features(self):
        """Compute features for all sites at current time."""
        from battery_pdm.common.features import compute_features

        # One observation per site, scored at current_h
        labels = pd.DataFrame(
            {
                "site_id": self.site_static["site_id"].values,
                "event_hour": self.current_h,
            }
        )

        alarms_pit = self.alarms[self.alarms["timestamp_h"] <= self.current_h]
        feature_groups = self.meta.get(
            "feature_groups", ["alarm_history", "site_static", "soc_proxy"]
        )

        self.features = compute_features(
            labels=labels,
            groups=feature_groups,
            inputs={"alarms": alarms_pit, "site_static": self.site_static},
            ref_time_col="event_hour",
        )
        print(f"Features computed for {len(self.features)} sites")
        self.next(self.score)

    @step
    def score(self):
        """Score all sites with the failure model."""
        import xgboost as xgb
        from battery_pdm.monitoring.model_registry import validate_feature_hash

        feature_cols = self.meta["feature_cols"]
        try:
            validate_feature_hash(self.meta, feature_cols)
        except ValueError as exc:
            raise RuntimeError(f"Feature contract violation: {exc}")

        available_cols = [c for c in feature_cols if c in self.features.columns]
        X = self.features[available_cols].astype(float).fillna(0.0)

        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_cols]

        dmat = xgb.DMatrix(X)
        risk_scores = self.booster.predict(dmat)

        self.alerts = pd.DataFrame(
            {
                "site_id": self.features["site_id"].values,
                "scoring_hour": self.current_h,
                "failure_risk_score": risk_scores,
            }
        )
        self.alerts = self.alerts.sort_values(
            "failure_risk_score", ascending=False
        ).reset_index(drop=True)
        self.alerts["risk_rank"] = range(1, len(self.alerts) + 1)
        self.alerts["risk_percentile"] = self.alerts["failure_risk_score"].rank(
            pct=True
        )
        self.alerts["priority"] = self.alerts["risk_percentile"].apply(
            lambda p: "REPLACE_NOW" if p >= 0.90 else ("MONITOR" if p >= 0.70 else "OK")
        )

        self.alerts = self.alerts.merge(
            self.site_static[
                ["site_id", "region", "manufacturer", "load_A", "install_month"]
            ],
            on="site_id",
            how="left",
        )

        n_replace = (self.alerts["priority"] == "REPLACE_NOW").sum()
        n_monitor = (self.alerts["priority"] == "MONITOR").sum()
        print(
            f"Scored {len(self.alerts)} sites: {n_replace} REPLACE_NOW, {n_monitor} MONITOR"
        )
        self.next(self.emit_alerts)

    @step
    def emit_alerts(self):
        """Write replacement priority list."""
        from battery_pdm.aws.s3_io import join_path, write_parquet, mkdir

        mkdir(self.alerts_dir)
        out_file = join_path(
            self.alerts_dir, f"failure_alerts_h{int(self.current_h):08d}.parquet"
        )
        write_parquet(self.alerts, out_file)
        print(f"Wrote {out_file}")

        replace = self.alerts[self.alerts["priority"] == "REPLACE_NOW"]
        if not replace.empty:
            print("\nREPLACE_NOW sites (top 10):")
            print(
                replace[["site_id", "failure_risk_score", "region", "install_month"]]
                .head(10)
                .to_string(index=False)
            )
        self.next(self.end)

    @step
    def end(self):
        print(f"FailureScoringFlow complete. Scored at hour {self.current_h:.1f}")


if __name__ == "__main__":
    FailureScoringFlow()
