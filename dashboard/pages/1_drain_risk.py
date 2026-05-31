"""Drain Risk (48h) — which sites are most likely to experience LVD in the next 48 hours."""

import streamlit as st
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from pathlib import Path

st.set_page_config(page_title="Drain Risk (48h)", page_icon="⚡", layout="wide")
st.title("⚡ Drain Risk — Next 48 Hours")
st.markdown("Sites ranked by probability of battery drain (LVD) within 48h. "
            "Calibrated probabilities via isotonic regression.")

DATA_DIR = Path("/app/data") if Path("/app/data").exists() else Path("../outputs")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from battery_pdm.common.features import compute_features
    from battery_pdm.monitoring.model_registry import load_calibrator, apply_calibrator
    from battery_pdm.synth.load_shedding import build_load_shedding_schedule
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

@st.cache_data(ttl=300)
def load_data():
    alarms = pd.read_parquet(DATA_DIR / "alarms.parquet")
    sites = pd.read_parquet(DATA_DIR / "site_static.parquet")
    schedule = build_load_shedding_schedule(n_months=36, seed=42)
    return alarms, sites, schedule

@st.cache_resource
def load_model():
    model_dir = DATA_DIR / "models" / "drain_predictor_48h"
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "booster.json"))
    meta = json.loads((model_dir / "meta.json").read_text())
    calibrator = load_calibrator(model_dir)
    return booster, meta, calibrator

try:
    alarms, sites, schedule = load_data()
    booster, meta, calibrator = load_model()
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# Scoring controls
st.sidebar.subheader("Scoring Parameters")
current_h = st.sidebar.slider(
    "Evaluation time (hours)",
    int(alarms["timestamp_h"].min() + 30 * 24),
    int(alarms["timestamp_h"].max()),
    int(alarms["timestamp_h"].max()),
)
threshold = st.sidebar.slider("Alert threshold", 0.0, 1.0, 0.34, 0.01)

# Score all sites
labels = pd.DataFrame({
    "site_id": sites["site_id"].values,
    "mains_fail_h": float(current_h),
})
alarms_pit = alarms[alarms["timestamp_h"] <= current_h]

features = compute_features(
    labels=labels,
    groups=["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"],
    inputs={"alarms": alarms_pit, "site_static": sites, "schedule": schedule},
    ref_time_col="mains_fail_h",
)

X = features[meta["feature_cols"]].astype(float).fillna(0.0)
raw_scores = booster.predict(xgb.DMatrix(X))
calibrated_scores = apply_calibrator(calibrator, raw_scores)

results = features[["site_id"]].copy()
results["region"] = results["site_id"].map(dict(zip(sites["site_id"], sites["region"])))
results["risk_score"] = calibrated_scores
results["alert"] = results["risk_score"] >= threshold

# Per-site top driver via SHAP-style feature contributions
contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
feature_contribs = contribs[:, :-1]  # exclude bias
top_driver_idx = np.abs(feature_contribs).argmax(axis=1)
results["top_driver"] = [meta["feature_cols"][i] for i in top_driver_idx]
results["driver_contribution"] = [float(feature_contribs[row, top_driver_idx[row]]) for row in range(len(results))]

results = results.sort_values("risk_score", ascending=False).reset_index(drop=True)

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sites Scored", len(results))
col2.metric("HIGH Risk (above threshold)", int(results["alert"].sum()))
col3.metric("Mean Risk", f"{results['risk_score'].mean():.3f}")
col4.metric("Max Risk", f"{results['risk_score'].max():.3f}")

st.markdown("---")

# Alert list
st.subheader(f"🚨 Sites Above Threshold ({threshold:.2f})")
alerts = results[results["alert"]].copy()
if len(alerts) > 0:
    st.dataframe(
        alerts[["site_id", "region", "risk_score", "top_driver", "driver_contribution"]].style.background_gradient(
            subset=["risk_score"], cmap="YlOrRd"
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success("No sites above threshold.")

# Regional top drivers
st.subheader("Top Risk Driver by Region")
region_drivers = results.groupby("region")["top_driver"].agg(lambda x: x.value_counts().index[0]).reset_index()
region_drivers.columns = ["region", "most_common_driver"]
region_risk = results.groupby("region")["risk_score"].mean().reset_index()
region_drivers = region_drivers.merge(region_risk, on="region")
region_drivers.columns = ["region", "primary_risk_driver", "mean_risk"]
region_drivers = region_drivers.sort_values("mean_risk", ascending=False)
st.dataframe(region_drivers, use_container_width=True, hide_index=True)

# Regional breakdown
st.subheader("Risk Distribution by Region")
import plotly.express as px

fig = px.box(results, x="region", y="risk_score", color="region",
             title="Calibrated Drain Risk by Region")
fig.add_hline(y=threshold, line_dash="dash", line_color="red",
              annotation_text=f"Threshold={threshold}")
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Full ranked table
st.subheader("Full Site Rankings")
st.dataframe(
    results[["site_id", "region", "risk_score", "alert", "top_driver", "driver_contribution"]],
    use_container_width=True, hide_index=True, height=400,
)
