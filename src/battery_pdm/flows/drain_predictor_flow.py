"""DrainPredictorFlow — daily batch: scores all sites for 48h drain risk.

Runs once per day. For each active site, computes:
    - Current alarm-derived features (SoC proxy, alarm counts)
    - Upcoming 48h schedule features
    - P(drain in next 48h) from the trained XGBoost model

Outputs alerts ranked by risk for NOC dashboard consumption.

Run:
    python -m battery_pdm.flows.drain_predictor_flow run

Scheduled: daily at 00:00 (via @schedule or cron trigger on AWS)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, Parameter, card, current, project, step
from metaflow.cards import Markdown, Table

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@project(name="battery_pdm")
class DrainPredictorFlow(FlowSpec):
    """Daily batch scoring: 48h drain risk for all active sites."""

    alarms_path = Parameter("alarms-path", default="outputs/alarms.parquet")
    site_static_path = Parameter(
        "site-static-path", default="outputs/site_static.parquet"
    )
    model_dir = Parameter("model-dir", default="outputs/models/drain_predictor_48h")
    alerts_dir = Parameter("alerts-dir", default="outputs/drain_alerts")
    sim_hour = Parameter(
        "sim-hour",
        type=float,
        default=0.0,
        help="Simulated current hour (0=use max from alarms). For backfill/testing.",
    )
    schedule_path = Parameter(
        "schedule-path",
        default="",
        help="S3 or local path to load_shedding_schedule.parquet. "
        "If empty, regenerates from build_load_shedding_schedule() (dev mode).",
    )

    @step
    def start(self):
        """Load model, data, and determine scoring time."""
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
                f"No drain model at {self.model_dir}. Run TrainingFlow first."
            )

        self.booster = xgb.Booster()
        self.booster.load_model(bytearray(read_bytes(booster_path)))
        self.meta = read_json(join_path(self.model_dir, "meta.json"))

        # Load isotonic calibrator if the model has one.
        from battery_pdm.monitoring.model_registry import load_calibrator_from_bytes
        from battery_pdm.aws.s3_io import read_bytes as _rb

        cal_path = join_path(self.model_dir, "calibrator.pkl")
        if exists(cal_path):
            self.calibrator = load_calibrator_from_bytes(_rb(cal_path))
            print(
                f"  Loaded calibrator: {self.meta.get('calibration_method', 'isotonic')}"
            )
        else:
            self.calibrator = None
            print("  No calibrator found — emitting raw probabilities")

        self.alarms = read_parquet(self.alarms_path)
        self.site_static = read_parquet(self.site_static_path)

        # Schema validation — fail loud on upstream data quality issues
        from battery_pdm.schema_validation import validate_scoring_inputs

        validate_scoring_inputs(self.alarms, self.site_static, strict=True)

        if self.sim_hour > 0:
            self.current_h = self.sim_hour
        else:
            self.current_h = float(self.alarms["timestamp_h"].max())

        print(f"Scoring time: {self.current_h:.1f}h")
        print(f"Active sites: {self.site_static['site_id'].nunique()}")
        self.next(self.compute_features)

    @step
    def compute_features(self):
        """Compute alarm + schedule features for all sites at current time."""
        from battery_pdm.common.features import compute_features
        from battery_pdm.synth.load_shedding import build_load_shedding_schedule
        from battery_pdm.aws.s3_io import read_parquet

        schedule = pd.DataFrame()
        if self.schedule_path:
            try:
                schedule = read_parquet(self.schedule_path)
                if schedule.empty:
                    print(
                        f"  schedule_path {self.schedule_path} returned empty — falling back to generator"
                    )
            except Exception as exc:
                print(
                    f"  Could not read schedule from {self.schedule_path}: {exc} — falling back to generator"
                )
                schedule = pd.DataFrame()
        if schedule.empty:
            schedule = build_load_shedding_schedule(n_months=36, seed=42)

        # One observation per site at current_h
        labels = pd.DataFrame(
            {
                "site_id": self.site_static["site_id"].values,
                "ref_time_h": self.current_h,
            }
        )

        # Only use alarms up to current_h (point-in-time correctness)
        self.alarms_pit = self.alarms[self.alarms["timestamp_h"] <= self.current_h]

        feature_groups = self.meta.get(
            "feature_groups",
            ["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"],
        )
        self.features = compute_features(
            labels=labels.rename(columns={"ref_time_h": "mains_fail_h"}),
            groups=feature_groups,
            inputs={
                "alarms": self.alarms_pit,
                "site_static": self.site_static,
                "schedule": schedule,
            },
            ref_time_col="mains_fail_h",
        )
        self.features = self.features.rename(columns={"mains_fail_h": "ref_time_h"})
        print(f"Features computed for {len(self.features)} sites")
        self.next(self.score)

    @card(type="default")
    @step
    def score(self):
        """Score all sites with the drain predictor model."""
        import xgboost as xgb
        from battery_pdm.monitoring.model_registry import validate_feature_hash
        from battery_pdm.monitoring.data_quality import (
            assess_scoring_inputs,
            add_missingness_flags,
            compute_null_rates,
            compute_regional_priors,
        )

        feature_cols = self.meta["feature_cols"]
        # Validate that the feature contract hasn't drifted between training and inference
        try:
            validate_feature_hash(self.meta, feature_cols)
        except ValueError as exc:
            raise RuntimeError(f"Feature contract violation: {exc}")

        # Data quality assessment — reuse the schedule loaded in compute_features step
        # via the same path-or-generate pattern
        from battery_pdm.synth.load_shedding import build_load_shedding_schedule
        from battery_pdm.aws.s3_io import read_parquet

        schedule = pd.DataFrame()
        if self.schedule_path:
            try:
                schedule = read_parquet(self.schedule_path)
            except Exception:
                schedule = pd.DataFrame()
        if schedule.empty:
            schedule = build_load_shedding_schedule(n_months=36, seed=42)
        scorable_sites, dq_report = assess_scoring_inputs(
            site_ids=self.site_static["site_id"].tolist(),
            alarms=self.alarms,
            site_static=self.site_static,
            schedule=schedule,
            ref_time_h=self.current_h,
        )
        self.dq_report = dq_report.to_dict()
        if dq_report.warnings:
            for w in dq_report.warnings:
                print(f"  [DATA QUALITY] {w}")

        # Add missingness flags
        features_with_flags = add_missingness_flags(self.features, feature_cols)

        available_cols = [c for c in feature_cols if c in features_with_flags.columns]
        X = features_with_flags[available_cols].astype(float).fillna(0.0)
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_cols]

        self.null_rates = compute_null_rates(features_with_flags, feature_cols)

        dmat = xgb.DMatrix(X)
        raw_scores = self.booster.predict(dmat)

        # Apply isotonic calibration if available (production probabilities)
        from battery_pdm.monitoring.model_registry import apply_calibrator

        risk_scores = apply_calibrator(self.calibrator, raw_scores)

        self.alerts = pd.DataFrame(
            {
                "site_id": features_with_flags["site_id"].values,
                "scoring_hour": self.current_h,
                "drain_risk_48h": risk_scores,
                "data_completeness_score": features_with_flags[
                    "data_completeness_score"
                ].values,
                "scorable": features_with_flags["site_id"].isin(scorable_sites).values,
            }
        )

        # COLD-START FALLBACK: sites without enough alarm history get the regional
        # historical drain rate as their risk, instead of an ML score from zeros.
        regional_prior = compute_regional_priors(
            self.alarms_pit, self.site_static, horizon_h=48
        )
        site_to_region = dict(
            zip(self.site_static["site_id"], self.site_static["region"])
        )
        cold_mask = ~self.alerts["scorable"]
        for idx in self.alerts.index[cold_mask]:
            site = self.alerts.at[idx, "site_id"]
            region = site_to_region.get(site, "unknown")
            self.alerts.at[idx, "drain_risk_48h"] = regional_prior.get(region, 0.15)

        # Alert levels: COLD_START signals operator to monitor manually
        self.alerts["alert_level"] = self.alerts.apply(
            lambda r: (
                "COLD_START"
                if not r["scorable"]
                else (
                    "HIGH"
                    if r["drain_risk_48h"] >= 0.6
                    else ("MEDIUM" if r["drain_risk_48h"] >= 0.4 else "LOW")
                )
            ),
            axis=1,
        )

        self.alerts = self.alerts.sort_values(
            "drain_risk_48h", ascending=False
        ).reset_index(drop=True)
        self.alerts["risk_rank"] = range(1, len(self.alerts) + 1)

        self.alerts = self.alerts.merge(
            self.site_static[
                ["site_id", "region", "manufacturer", "load_A", "nominal_capacity_ah"]
            ],
            on="site_id",
            how="left",
        )

        n_high = int((self.alerts["alert_level"] == "HIGH").sum())
        n_med = int((self.alerts["alert_level"] == "MEDIUM").sum())
        n_low = int((self.alerts["alert_level"] == "LOW").sum())
        n_cold = int((self.alerts["alert_level"] == "COLD_START").sum())
        n_insuf = int((self.alerts["alert_level"] == "INSUFFICIENT_DATA").sum())
        print(
            f"Scored {len(self.alerts)} sites: {n_high} HIGH, {n_med} MEDIUM, "
            f"{n_low} LOW, {n_cold} COLD_START, {n_insuf} INSUFFICIENT_DATA"
        )

        # Emit CloudWatch metrics so the dashboard has data
        try:
            from battery_pdm.aws.metrics import emit_alert_counts

            emit_alert_counts(
                high=n_high, medium=n_med, low=n_low, insufficient=n_cold + n_insuf
            )
        except Exception as exc:
            print(f"  (CloudWatch metric emission skipped: {exc})")

        current.card.append(
            Markdown(f"# Drain Predictor Scoring — {self.current_h:.0f}h")
        )
        current.card.append(Markdown(f"**Sites scored:** {len(self.alerts)}"))
        current.card.append(Markdown(f"**HIGH alerts:** {n_high}"))
        current.card.append(Markdown(f"**MEDIUM alerts:** {n_med}"))
        current.card.append(Markdown(f"**COLD_START:** {n_cold}"))
        top = self.alerts.nlargest(10, "drain_risk_48h")[
            ["site_id", "drain_risk_48h", "alert_level", "region"]
        ]
        current.card.append(Table(headers=list(top.columns), data=top.values.tolist()))

        self.next(self.emit_alerts)

    @step
    def emit_alerts(self):
        """Write alerts + (if exists) score with shadow challenger for parallel validation."""
        from battery_pdm.aws.s3_io import (
            join_path,
            write_parquet,
            exists,
            read_json,
            read_bytes,
            mkdir,
        )

        mkdir(self.alerts_dir)
        out_file = join_path(
            self.alerts_dir, f"drain_alerts_h{int(self.current_h):08d}.parquet"
        )
        write_parquet(self.alerts, out_file)
        print(f"Wrote {out_file}")

        # SHADOW DEPLOYMENT: if a challenger model exists, score with it too.
        shadow_booster_path = join_path(self.model_dir + "_shadow", "booster.json")
        if exists(shadow_booster_path):
            import xgboost as xgb
            from battery_pdm.monitoring.model_registry import (
                load_calibrator_from_bytes,
                apply_calibrator,
            )

            try:
                shadow_meta = read_json(
                    join_path(self.model_dir + "_shadow", "meta.json")
                )
                shadow_booster = xgb.Booster()
                shadow_booster.load_model(bytearray(read_bytes(shadow_booster_path)))

                cal_path = join_path(self.model_dir + "_shadow", "calibrator.pkl")
                shadow_calibrator = (
                    load_calibrator_from_bytes(read_bytes(cal_path))
                    if exists(cal_path)
                    else None
                )

                feature_cols = shadow_meta["feature_cols"]
                X_shadow = self.features[feature_cols].astype(float).fillna(0.0)
                shadow_raw = shadow_booster.predict(xgb.DMatrix(X_shadow))
                shadow_scores = apply_calibrator(shadow_calibrator, shadow_raw)

                shadow_df = pd.DataFrame(
                    {
                        "site_id": self.features["site_id"].values,
                        "scoring_hour": self.current_h,
                        "predicted_score_champion": self.alerts[
                            "drain_risk_48h"
                        ].values,
                        "predicted_score_shadow": shadow_scores,
                        "champion_version": self.meta.get("model_version", "unknown"),
                        "shadow_version": shadow_meta.get("model_version", "unknown"),
                    }
                )
                shadow_out = join_path(
                    self.alerts_dir,
                    "shadow_comparisons",
                    f"shadow_h{int(self.current_h):08d}.parquet",
                )
                write_parquet(shadow_df, shadow_out)
                print(f"  Shadow scored {len(shadow_df)} sites — saved to {shadow_out}")
            except Exception as exc:
                print(f"  Shadow scoring failed: {exc}")

        high = self.alerts[self.alerts["alert_level"] == "HIGH"]
        if not high.empty:
            print("\nHIGH-risk sites (top 10):")
            print(
                high[["site_id", "drain_risk_48h", "region", "load_A"]]
                .head(10)
                .to_string(index=False)
            )
        self.next(self.end)

    @step
    def end(self):
        print(f"DrainPredictorFlow complete. Scored at hour {self.current_h:.1f}")


if __name__ == "__main__":
    DrainPredictorFlow()
