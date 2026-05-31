"""Battery PdM — NOC Operations Dashboard.

Main entry point for the multi-page Streamlit app.
Deployed on ECS Fargate behind an ALB.
"""

import streamlit as st

st.set_page_config(
    page_title="Battery PdM — NOC Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Battery PdM")
st.sidebar.markdown("**NOC Operations Console**")
st.sidebar.markdown("Real-time monitoring for 500 telecom battery sites across 5 regions.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Pages:**
    - ⚡ Drain Risk — 48h drain probability (XGBoost + isotonic calibration)
    - 🔧 Replacement — Cox survival model for hardware failure
    - 🚨 Anomalies — Isolation Forest on alarm patterns
    - 📊 Drift — PSI monitoring vs training baseline
    - 🧪 Simulation — Simulate grid upgrades, watch drift respond
    - 📈 Model Health — Version history, calibration, feature importance
    """
)

st.title("🔋 Battery PdM — Fleet Overview")
st.markdown("""
This system predicts two failure modes for telecom battery sites:
- **Drain (48h):** Will load shedding exhaust this battery before mains returns? (XGBoost binary classifier, calibrated)
- **Hardware failure:** Is this battery degrading and due for replacement? (Cox proportional hazards survival model)

Scoring runs daily on AWS Batch (Fargate Spot). Drift detection compares current feature distributions
against the training baseline using Population Stability Index (PSI).
""")
st.markdown("---")

# Load fleet-level stats
import pandas as pd
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path("/app/data") if Path("/app/data").exists() else Path("../outputs")

try:
    alarms = pd.read_parquet(DATA_DIR / "alarms.parquet")
    sites = pd.read_parquet(DATA_DIR / "site_static.parquet")
    HAS_DATA = True
except Exception:
    HAS_DATA = False
    st.error("⚠️ No data found. Ensure S3 sync has populated /app/data/")
    st.stop()

# Fleet KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sites", f"{len(sites):,}")
col2.metric("Active Alarms (30d)", f"{len(alarms[alarms['timestamp_h'] > alarms['timestamp_h'].max() - 30*24]):,}")
col3.metric("Regions", f"{sites['region'].nunique()}")

# Load latest model meta
drain_meta_path = DATA_DIR / "models" / "drain_predictor_48h" / "meta.json"
if drain_meta_path.exists():
    drain_meta = json.loads(drain_meta_path.read_text())
    col4.metric("Model Version", drain_meta["model_version"][:8])
else:
    col4.metric("Model Version", "—")

st.markdown("---")

# Regional summary
st.subheader("Regional Alarm Rates (last 30 days)")
merged = alarms.merge(sites[["site_id", "region"]], on="site_id")
recent = merged[merged["timestamp_h"] > merged["timestamp_h"].max() - 30 * 24]
regional_summary = recent.groupby("region").agg(
    total_alarms=("alarm_code", "count"),
    sites_affected=("site_id", "nunique"),
    lvd_events=("alarm_code", lambda x: (x == "LOAD_DISCONNECT").sum()),
).reset_index()
regional_summary["lvd_rate"] = (regional_summary["lvd_events"] / regional_summary["sites_affected"]).round(2)

st.dataframe(
    regional_summary.style.background_gradient(subset=["lvd_rate"], cmap="YlOrRd"),
    use_container_width=True,
    hide_index=True,
)

# Alarm trend chart
st.subheader("Alarm Volume — 36-Month Trend")
alarms["month"] = (alarms["timestamp_h"] / (24 * 30)).astype(int)
monthly = alarms.groupby("month").size().reset_index(name="count")

import plotly.express as px

fig = px.area(monthly, x="month", y="count", title="Monthly Alarm Volume",
              labels={"month": "Month", "count": "Alarms"})
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# Quick system status
st.markdown("---")
st.subheader("System Status")
st.caption("Models retrained daily via AWS Batch. Data refreshed every 5 min from S3.")
status_col1, status_col2, status_col3 = st.columns(3)

drift_report_path = DATA_DIR / "drift_reports"
if drift_report_path.exists():
    reports = list(drift_report_path.glob("drift_report_*.json"))
    if reports:
        latest_report = json.loads(sorted(reports)[-1].read_text())
        drift_status = latest_report.get("summary", {}).get("status", "UNKNOWN")
        if drift_status == "DRIFT_DETECTED":
            status_col1.error(f"🚨 Drift: {drift_status}")
        else:
            status_col1.success(f"✅ Drift: STABLE")
    else:
        status_col1.info("ℹ️ No drift reports yet")
else:
    status_col1.info("ℹ️ No drift reports yet")

# Model freshness
if drain_meta_path.exists():
    trained_at = drain_meta.get("trained_at", "Unknown")
    status_col2.success(f"✅ Model trained: {trained_at[:10]}")
else:
    status_col2.warning("⚠️ No model found")

# Alert pipeline
drain_alerts_dir = DATA_DIR / "drain_alerts"
failure_alerts_dir = DATA_DIR / "failure_alerts"
has_drain_alerts = drain_alerts_dir.exists() and list(drain_alerts_dir.glob("*.parquet"))
has_failure_alerts = failure_alerts_dir.exists() and list(failure_alerts_dir.glob("*.parquet"))
if has_drain_alerts or has_failure_alerts:
    n_drain = len(list(drain_alerts_dir.glob("*.parquet"))) if has_drain_alerts else 0
    n_fail = len(list(failure_alerts_dir.glob("*.parquet"))) if has_failure_alerts else 0
    status_col3.success(f"✅ Alerts: {n_drain} drain, {n_fail} failure runs")
else:
    status_col3.info("ℹ️ No alerts generated yet (run Batch flows)")

# How to use
st.markdown("---")
with st.expander("📖 How to Use This Dashboard", expanded=False):
    st.markdown("""
    **Pages explained:**

    | Page | What it does | When to use |
    |------|-------------|-------------|
    | **⚡ Drain Risk** | Scores every site: "Will this battery go flat in the next 48h?" Uses XGBoost + isotonic calibration. Shows the top feature driving each site's score. | Daily morning check. Action HIGH-risk sites immediately. |
    | **🔧 Replacement Priority** | Ranks sites by long-term hardware degradation (Cox survival model). Not about tomorrow — about which batteries are aging out. | Monthly procurement planning. Export the top-K list. |
    | **🚨 Anomaly Detection** | Isolation Forest finds sites behaving differently from their regional peers — even if their risk score is normal. Catches emerging issues before the model does. | Weekly scan. Investigate flagged sites manually. |
    | **📊 Drift Monitor** | Compares today's feature distributions against what the model was trained on (PSI). If drift > 0.25, the model's assumptions may no longer hold. | After interventions (grid upgrades, new firmware). If RED: don't trust scores blindly. |
    | **🧪 Drift Simulation** | Simulates a "Peshawar grid upgrade" — drops alarm rates in one region and shows how drift detection responds. Demonstrates the full monitoring loop. | Demo/education. Shows why drift monitoring matters. |
    | **📈 Model Health** | Version history, calibration quality (Brier score improvement), feature importance comparison across models. | After retraining. Verify new model is better than old. |

    **Key concepts:**
    - **Calibrated probability:** The drain risk score IS a probability (e.g., 0.65 = 65% chance of drain). We use isotonic regression to make raw model outputs match observed frequencies.
    - **Top driver:** Each site's score is decomposed into feature contributions (SHAP-style). The "top driver" is the feature most responsible for that site's risk.
    - **PSI (Population Stability Index):** Measures how much a feature's distribution has shifted. >0.25 = significant. >0.10 = worth watching.
    - **Cold start:** New sites with <30 days of alarm history get the regional average risk instead of an ML score.

    **Architecture:**
    - Scoring: AWS Batch (Fargate Spot) — runs daily, writes alerts to S3
    - Dashboard: ECS Fargate behind ALB — refreshes data from S3 every 5 min
    - API: FastAPI endpoint for programmatic access (POST /predict)
    """)

