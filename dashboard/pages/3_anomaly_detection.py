"""Anomaly Detection — sites behaving abnormally relative to their regional peers."""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")
st.title("🚨 Site Anomaly Detection")
st.markdown("Flags individual sites whose alarm patterns deviate from their "
            "regional baseline. These sites may need investigation even if their "
            "drain risk score isn't high yet.")

DATA_DIR = Path("/app/data") if Path("/app/data").exists() else Path("../outputs")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from battery_pdm.monitoring.anomaly import detect_anomalies
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

@st.cache_data(ttl=300)
def load_data():
    alarms = pd.read_parquet(DATA_DIR / "alarms.parquet")
    sites = pd.read_parquet(DATA_DIR / "site_static.parquet")
    return alarms, sites

try:
    alarms, sites = load_data()
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

st.sidebar.subheader("Detection Parameters")
contamination = st.sidebar.slider("Expected anomaly rate", 0.01, 0.20, 0.05, 0.01)
current_h = st.sidebar.slider(
    "Evaluation time (hours)",
    int(alarms["timestamp_h"].min() + 30 * 24),
    int(alarms["timestamp_h"].max()),
    int(alarms["timestamp_h"].max()),
)

with st.spinner("Running anomaly detection..."):
    results = detect_anomalies(
        alarms=alarms,
        site_static=sites,
        current_h=float(current_h),
        contamination=contamination,
    )

anomalous = [r for r in results if r.is_anomalous]
normal = [r for r in results if not r.is_anomalous]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Sites Scanned", len(results))
col2.metric("Anomalies Detected", len(anomalous))
col3.metric("Anomaly Rate", f"{len(anomalous)/max(len(results),1):.1%}")

st.markdown("---")

# Anomaly table
if anomalous:
    st.subheader(f"⚠️ Anomalous Sites ({len(anomalous)})")
    anomaly_df = pd.DataFrame([
        {
            "site_id": r.site_id,
            "anomaly_score": round(r.anomaly_score, 3),
            "top_deviation_1": f"{list(r.top_deviations.keys())[0]}: {list(r.top_deviations.values())[0]:+.1f}σ" if r.top_deviations else "",
            "top_deviation_2": f"{list(r.top_deviations.keys())[1]}: {list(r.top_deviations.values())[1]:+.1f}σ" if len(r.top_deviations) > 1 else "",
            "top_deviation_3": f"{list(r.top_deviations.keys())[2]}: {list(r.top_deviations.values())[2]:+.1f}σ" if len(r.top_deviations) > 2 else "",
        }
        for r in anomalous
    ])
    st.dataframe(anomaly_df, use_container_width=True, hide_index=True)

    # Detail view for selected site
    st.subheader("🔍 Site Detail")
    selected = st.selectbox("Select anomalous site", [r.site_id for r in anomalous])
    site_result = next(r for r in anomalous if r.site_id == selected)

    st.markdown(f"**Anomaly Score:** {site_result.anomaly_score:.3f}")
    st.markdown("**Top Feature Deviations from Regional Norm:**")
    for feat, z in site_result.top_deviations.items():
        direction = "above" if z > 0 else "below"
        severity = "🔴" if abs(z) > 3 else "🟡" if abs(z) > 2 else "⚪"
        st.markdown(f"  {severity} `{feat}`: {z:+.1f}σ {direction} regional median")

    # Site alarm history
    site_alarms = alarms[alarms["site_id"] == selected].copy()
    site_alarms["month"] = (site_alarms["timestamp_h"] / (24 * 30)).astype(int)
    monthly = site_alarms.groupby(["month", "alarm_code"]).size().reset_index(name="count")

    import plotly.express as px
    fig = px.bar(monthly, x="month", y="count", color="alarm_code",
                 title=f"Alarm History — {selected}")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("✅ No anomalous sites detected at current parameters.")

# Score distribution
st.markdown("---")
st.subheader("Anomaly Score Distribution (all sites)")
import plotly.express as px

score_df = pd.DataFrame([
    {"site_id": r.site_id, "score": r.anomaly_score, "anomalous": r.is_anomalous}
    for r in results
])
fig = px.histogram(score_df, x="score", color="anomalous", nbins=40,
                   title="Isolation Forest Anomaly Scores (lower = more anomalous)",
                   color_discrete_map={True: "#ef4444", False: "#3b82f6"})
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)
