"""Drift Monitor — displays results from the last DriftMonitorFlow run.

Shows the ACTUAL drift status from the daily scheduled flow, not a live
re-computation (which produces false positives due to natural temporal evolution
of the synthetic data within the simulation window).
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

st.set_page_config(page_title="Drift Monitor", page_icon="📊", layout="wide")
st.title("📊 Feature & Prediction Drift Monitor")
st.markdown(
    "Shows results from the **last scheduled DriftMonitorFlow run**. "
    "PSI > 0.25 = significant drift requiring investigation."
)

DATA_DIR = Path("/app/data") if Path("/app/data").exists() else Path("../outputs")
DRIFT_DIR = DATA_DIR / "drift_reports"

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_latest_drift_report() -> dict | None:
    """Load the most recent drift report written by DriftMonitorFlow."""
    if not DRIFT_DIR.exists():
        return None
    reports = sorted(DRIFT_DIR.glob("drift_report_h*.json"), reverse=True)
    if not reports:
        return None
    return json.loads(reports[0].read_text()), reports[0].name


def run_live_drift_check():
    """Fallback: run drift detection live (for local dev when no saved reports exist)."""
    try:
        from battery_pdm.monitoring.drift import load_reference_profile, detect_drift
        from battery_pdm.common.features import compute_features
        from battery_pdm.monitoring.model_registry import load_calibrator, apply_calibrator
        from battery_pdm.synth.load_shedding import build_load_shedding_schedule
        import xgboost as xgb
    except ImportError as e:
        st.error(f"Import error: {e}")
        return None

    model_dir = DATA_DIR / "models" / "drain_predictor_48h"
    if not (model_dir / "reference_profile.json").exists():
        return None

    alarms = pd.read_parquet(DATA_DIR / "alarms.parquet")
    sites = pd.read_parquet(DATA_DIR / "site_static.parquet")
    schedule = build_load_shedding_schedule(n_months=36, seed=42)

    booster = xgb.Booster()
    booster.load_model(str(model_dir / "booster.json"))
    meta = json.loads((model_dir / "meta.json").read_text())
    ref_profile = load_reference_profile(model_dir / "reference_profile.json")
    calibrator = load_calibrator(model_dir)

    max_h = int(alarms["timestamp_h"].max())
    labels = pd.DataFrame({"site_id": sites["site_id"].values, "mains_fail_h": float(max_h)})
    alarms_pit = alarms[alarms["timestamp_h"] <= max_h]

    features = compute_features(
        labels=labels,
        groups=["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"],
        inputs={"alarms": alarms_pit, "site_static": sites, "schedule": schedule},
        ref_time_col="mains_fail_h",
    )

    X = features[meta["feature_cols"]].astype(float).fillna(0.0)
    raw_preds = booster.predict(xgb.DMatrix(X))
    preds = apply_calibrator(calibrator, raw_preds)

    drift_report = detect_drift(
        reference_profile=ref_profile,
        current_features=features,
        current_predictions=preds,
        feature_cols=meta["feature_cols"],
    )
    return drift_report


# Try loading saved report first (production path)
saved = load_latest_drift_report()

if saved is not None:
    report_data, report_name = saved
    summary = report_data["summary"]
    st.caption(f"Source: `{report_name}` (from scheduled DriftMonitorFlow)")
else:
    st.info("No saved drift reports found. Running live drift check (dev mode).")
    drift_report = run_live_drift_check()
    if drift_report is None:
        st.error("Cannot compute drift: missing reference profile or model.")
        st.stop()
    summary = drift_report["summary"]
    st.caption("Source: live computation (no saved DriftMonitorFlow reports available)")

# Status banner
if summary["status"] == "DRIFT_DETECTED":
    st.error(f"🚨 **DRIFT DETECTED** — {summary['n_significant_drift']} features significantly drifted")
elif summary.get("n_moderate_drift", 0) > 0:
    st.warning(f"⚠️ **MODERATE DRIFT** — {summary['n_moderate_drift']} features show moderate shift")
else:
    st.success("✅ **STABLE** — No significant drift detected")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Features Monitored", summary["n_features_monitored"])
col2.metric("Significant (PSI>0.25)", summary["n_significant_drift"])
col3.metric("Moderate (PSI 0.10-0.25)", summary.get("n_moderate_drift", 0))
col4.metric("Retrain Recommended", "YES" if summary.get("retrain_recommended") else "NO")

st.markdown("---")

# Feature drift details (from saved report or live)
if saved is not None:
    report_data, _ = saved
    if "top_drifted_features" in report_data:
        st.subheader("Top Drifted Features")
        fd = pd.DataFrame(report_data["top_drifted_features"])
        if not fd.empty:
            import plotly.express as px
            fd_sorted = fd.sort_values("psi", ascending=True)
            fig = px.bar(
                fd_sorted, x="psi", y="feature", orientation="h",
                color="drift_level",
                color_discrete_map={"SIGNIFICANT": "#ef4444", "MODERATE": "#f59e0b", "STABLE": "#22c55e"},
                title="Feature PSI (from last DriftMonitorFlow run)",
            )
            fig.add_vline(x=0.10, line_dash="dash", line_color="orange", annotation_text="Moderate")
            fig.add_vline(x=0.25, line_dash="dash", line_color="red", annotation_text="Significant")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Drift details table"):
                st.dataframe(fd, use_container_width=True, hide_index=True)

    # Prediction drift
    pred_drift = report_data.get("prediction_drift", {})
    if pred_drift and isinstance(pred_drift, dict):
        st.subheader("Prediction Distribution Shift")
        psi_val = pred_drift.get("prediction_psi", 0)
        st.metric("Prediction PSI", f"{psi_val:.3f}")

elif drift_report is not None:
    # Live computation fallback — show full feature drift
    st.subheader("Feature Drift Details")
    fd = drift_report["feature_drift"]
    if not fd.empty:
        fd_sorted = fd.sort_values("psi", ascending=False)

        import plotly.express as px
        top_drift = fd_sorted.head(15).iloc[::-1]
        fig = px.bar(
            top_drift, x="psi", y="feature", orientation="h",
            color="drift_level",
            color_discrete_map={"SIGNIFICANT": "#ef4444", "MODERATE": "#f59e0b", "STABLE": "#22c55e"},
            title="Top 15 Features by PSI (live computation — dev mode)",
        )
        fig.add_vline(x=0.10, line_dash="dash", line_color="orange", annotation_text="Moderate")
        fig.add_vline(x=0.25, line_dash="dash", line_color="red", annotation_text="Significant")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full drift report table"):
            st.dataframe(fd_sorted, use_container_width=True, hide_index=True)

    pred_drift = drift_report.get("prediction_drift", {})
    if pred_drift:
        st.subheader("Prediction Distribution")
        st.metric("Prediction PSI", f"{pred_drift.get('prediction_psi', 0):.3f}")

# Reasons
if summary.get("reasons"):
    st.subheader("Drift Reasons")
    for reason in summary["reasons"]:
        st.markdown(f"- {reason}")

# Historical drift reports
st.markdown("---")
st.subheader("📜 Drift Report History")
if DRIFT_DIR.exists():
    reports = sorted(DRIFT_DIR.glob("drift_report_h*.json"), reverse=True)
    if reports:
        history_data = []
        for r in reports[:30]:
            try:
                d = json.loads(r.read_text())
                s = d["summary"]
                history_data.append({
                    "report": r.name,
                    "hour": d.get("scoring_hour", 0),
                    "status": s["status"],
                    "significant": s["n_significant_drift"],
                    "moderate": s.get("n_moderate_drift", 0),
                    "retrain": "YES" if s.get("retrain_recommended") else "NO",
                })
            except Exception:
                continue
        if history_data:
            st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
        else:
            st.info("No parseable drift reports found.")
    else:
        st.info("No drift reports in outputs/drift_reports/. Run DriftMonitorFlow first.")
else:
    st.info("Drift reports directory not found. Run DriftMonitorFlow to generate reports.")
