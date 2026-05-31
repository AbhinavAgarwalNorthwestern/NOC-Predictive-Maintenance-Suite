"""Replacement Priority — long-term failure risk ranking for procurement planning."""

import streamlit as st
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from pathlib import Path

st.set_page_config(page_title="Replacement Priority", page_icon="🔧", layout="wide")
st.title("🔧 Battery Replacement Priority")
st.markdown("Sites ranked by long-term failure risk (survival model). "
            "Use this for monthly replacement procurement planning.")

DATA_DIR = Path("/app/data") if Path("/app/data").exists() else Path("../outputs")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from battery_pdm.common.features import compute_features
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

@st.cache_data(ttl=300)
def load_data():
    alarms = pd.read_parquet(DATA_DIR / "alarms.parquet")
    sites = pd.read_parquet(DATA_DIR / "site_static.parquet")
    return alarms, sites

@st.cache_resource
def load_model():
    model_dir = DATA_DIR / "models" / "failure_alarms_only"
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "booster.json"))
    meta = json.loads((model_dir / "meta.json").read_text())
    return booster, meta

try:
    alarms, sites = load_data()
    booster, meta = load_model()
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

st.sidebar.subheader("Parameters")
top_k = st.sidebar.slider("Top-K for replacement list", 5, 100, 20)

# Score all sites at current time
current_h = float(alarms["timestamp_h"].max())
labels = pd.DataFrame({
    "site_id": sites["site_id"].values,
    "event_hour": current_h,
})

features = compute_features(
    labels=labels,
    groups=["alarm_history", "site_static", "soc_proxy"],
    inputs={"alarms": alarms, "site_static": sites},
    ref_time_col="event_hour",
)

feature_cols = meta["feature_cols"]
available_cols = [c for c in feature_cols if c in features.columns]
missing_cols = [c for c in feature_cols if c not in features.columns]

X = features[available_cols].astype(float).fillna(0.0)
for col in missing_cols:
    X[col] = 0.0
X = X[feature_cols]

risk_scores = booster.predict(xgb.DMatrix(X))

results = features[["site_id"]].copy()
results["region"] = results["site_id"].map(dict(zip(sites["site_id"], sites["region"])))
results["failure_risk"] = risk_scores
results = results.sort_values("failure_risk", ascending=False).reset_index(drop=True)
results["rank"] = range(1, len(results) + 1)

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Sites Assessed", len(results))
col2.metric(f"Top-{top_k} Mean Risk", f"{results.head(top_k)['failure_risk'].mean():.3f}")
col3.metric("Fleet Mean Risk", f"{results['failure_risk'].mean():.3f}")

st.markdown("---")

# Top-K replacement list
st.subheader(f"📋 Top-{top_k} Replacement Candidates")
top_list = results.head(top_k)[["rank", "site_id", "region", "failure_risk"]].copy()
st.dataframe(
    top_list.style.background_gradient(subset=["failure_risk"], cmap="YlOrRd"),
    use_container_width=True,
    hide_index=True,
)

# Download button
csv = top_list.to_csv(index=False)
st.download_button("📥 Download Replacement List (CSV)", csv,
                   file_name="replacement_priority.csv", mime="text/csv")

# Regional distribution
st.subheader("Failure Risk by Region")
import plotly.express as px

fig = px.histogram(results, x="failure_risk", color="region", nbins=30,
                   title="Failure Risk Score Distribution",
                   barmode="overlay", opacity=0.7)
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)

# Feature importance
st.subheader("Top Failure Predictors")
importance = meta.get("feature_importance", {})
if importance:
    imp_df = pd.DataFrame([
        {"feature": k, "gain": v} for k, v in importance.items()
    ]).sort_values("gain", ascending=False).head(10)
    fig2 = px.bar(imp_df, x="gain", y="feature", orientation="h",
                  title="Feature Importance (gain)")
    fig2.update_layout(height=350, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)
