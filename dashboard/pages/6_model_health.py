"""Model Health — performance tracking, version history, and calibration status."""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

st.set_page_config(page_title="Model Health", page_icon="📈", layout="wide")
st.title("📈 Model Health & Version History")
st.markdown("Track model performance over time, version history, and calibration quality.")

DATA_DIR = Path("/app/data") if Path("/app/data").exists() else Path("../outputs")

# Load performance log
perf_log_path = DATA_DIR / "model_performance_log.parquet"
if perf_log_path.exists():
    perf_log = pd.read_parquet(perf_log_path)
else:
    st.warning("No performance log found.")
    perf_log = pd.DataFrame()

# Load model metadata
models = {}
for model_name in ["drain_predictor_48h", "failure_alarms_only"]:
    meta_path = DATA_DIR / "models" / model_name / "meta.json"
    if meta_path.exists():
        models[model_name] = json.loads(meta_path.read_text())

# Current model versions
st.subheader("Current Model Versions")
if models:
    version_data = []
    for name, meta in models.items():
        version_data.append({
            "Model": name,
            "Version": meta["model_version"],
            "Trained": meta.get("trained_at", "Unknown")[:19],
            "Feature Hash": meta["feature_hash"],
            "Features": len(meta["feature_cols"]),
        })
    st.dataframe(pd.DataFrame(version_data), use_container_width=True, hide_index=True)
else:
    st.warning("No models found.")

st.markdown("---")

# Performance over time
st.subheader("Performance History")
if not perf_log.empty:
    perf_log["logged_at"] = pd.to_datetime(perf_log["logged_at"], format="ISO8601")

    # Filter to training events
    training_log = perf_log[perf_log["event"] == "training"] if "event" in perf_log.columns else perf_log

    for model_name in training_log["model_name"].unique():
        model_perf = training_log[training_log["model_name"] == model_name].sort_values("logged_at")
        if len(model_perf) > 0:
            import plotly.express as px
            fig = px.line(model_perf, x="logged_at", y="metric_value",
                         title=f"{model_name} — {model_perf['metric_name'].iloc[0]} over time",
                         markers=True)
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Performance log will populate after training runs.")

st.markdown("---")

# Calibration status
st.subheader("Calibration Status — Drain Predictor")
drain_meta = models.get("drain_predictor_48h", {})
metrics = drain_meta.get("metrics", {})
if metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC", f"{metrics.get('roc_auc', 'N/A'):.4f}" if isinstance(metrics.get('roc_auc'), float) else "N/A")
    col2.metric("Brier (raw)", f"{metrics.get('brier_uncalibrated', 'N/A'):.4f}" if isinstance(metrics.get('brier_uncalibrated'), float) else "N/A")
    col3.metric("Brier (calibrated)", f"{metrics.get('brier_calibrated', 'N/A'):.4f}" if isinstance(metrics.get('brier_calibrated'), float) else "N/A")

    brier_raw = metrics.get("brier_uncalibrated", 0)
    brier_cal = metrics.get("brier_calibrated", 0)
    if brier_raw and brier_cal:
        improvement = (brier_raw - brier_cal) / brier_raw * 100
        st.success(f"✅ Isotonic calibration reduces Brier by **{improvement:.0f}%** — "
                   f"calibrated probabilities are operationally usable.")

    calibrator_path = DATA_DIR / "models" / "drain_predictor_48h" / "calibrator.pkl"
    if calibrator_path.exists():
        st.info(f"📦 Calibrator: `{calibrator_path.name}` ({calibrator_path.stat().st_size / 1024:.1f} KB)")
    else:
        st.warning("⚠️ No calibrator.pkl found — raw scores only")

st.markdown("---")

# Version history
st.subheader("Model Version History")
for model_name in ["drain_predictor_48h", "failure_alarms_only"]:
    model_dir = DATA_DIR / "models" / model_name
    if not model_dir.exists():
        continue
    versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("v_")])
    if versions:
        with st.expander(f"{model_name} — {len(versions)} versions"):
            for v in reversed(versions):
                v_meta_path = model_dir / v / "meta.json"
                if v_meta_path.exists():
                    v_meta = json.loads(v_meta_path.read_text())
                    st.markdown(f"**{v}** — trained {v_meta.get('trained_at', 'Unknown')[:19]} | "
                                f"hash: `{v_meta.get('feature_hash', '?')}`")
                else:
                    st.markdown(f"**{v}**")

# Feature importance comparison
st.markdown("---")
st.subheader("Feature Importance — All Models")
tabs = st.tabs(list(models.keys()))
for tab, (name, meta) in zip(tabs, models.items()):
    with tab:
        importance = meta.get("feature_importance", {})
        if importance:
            imp_df = pd.DataFrame([
                {"feature": k, "gain": v} for k, v in importance.items()
            ]).sort_values("gain", ascending=False).head(12)

            import plotly.express as px
            fig = px.bar(imp_df, x="gain", y="feature", orientation="h",
                         title=f"{name} — Top 12 Features by Gain")
            fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature importance data.")
