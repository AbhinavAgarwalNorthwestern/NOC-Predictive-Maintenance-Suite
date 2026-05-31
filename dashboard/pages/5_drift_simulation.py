"""Drift Simulation — trigger a simulated regional grid upgrade and watch drift detection respond."""

import streamlit as st
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from pathlib import Path

st.set_page_config(page_title="Drift Simulation", page_icon="🧪", layout="wide")
st.title("🧪 Drift Simulation — Peshawar Grid Upgrade")
st.markdown("""
**Scenario:** The Peshawar utility installs grid stabilizers at a chosen month.
This reduces outage frequency, changing the alarm distribution. Watch how the
drift monitor detects it and how model scores shift.

This demonstrates the full monitoring loop:
1. Intervention changes the data
2. PSI catches the distribution shift
3. Model scores become unreliable
4. System recommends retraining
""")

DATA_DIR = Path("/app/data") if Path("/app/data").exists() else Path("../outputs")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from battery_pdm.monitoring.drift import load_reference_profile, detect_drift
    from battery_pdm.common.features import compute_features
    from battery_pdm.monitoring.model_registry import load_calibrator, apply_calibrator
    from battery_pdm.synth.load_shedding import build_load_shedding_schedule
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

@st.cache_data(ttl=600)
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
    ref_profile = load_reference_profile(model_dir / "reference_profile.json")
    calibrator = load_calibrator(model_dir)
    return booster, meta, ref_profile, calibrator

try:
    alarms, sites, schedule = load_data()
    booster, meta, ref_profile, calibrator = load_model()
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# Simulation controls
st.sidebar.subheader("Simulation Parameters")
drift_month = st.sidebar.slider("Grid upgrade month", 12, 30, 24)
eval_month = st.sidebar.slider("Evaluation month", drift_month + 1, 36, min(drift_month + 6, 35))
ac_mains_drop = st.sidebar.slider("AC_MAINS_FAIL reduction", 0.0, 0.90, 0.60, 0.05)
rectifier_drop = st.sidebar.slider("RECTIFIER_FAULT reduction", 0.0, 0.90, 0.40, 0.05)
undervoltage_drop = st.sidebar.slider("BATT_UNDERVOLTAGE reduction", 0.0, 0.90, 0.30, 0.05)
lvd_drop = st.sidebar.slider("LOAD_DISCONNECT reduction", 0.0, 0.90, 0.25, 0.05)

if st.sidebar.button("🚀 Run Drift Simulation", type="primary"):
    with st.spinner("Simulating grid upgrade + scoring + drift detection..."):
        # Apply the intervention
        drift_start_h = drift_month * 30 * 24
        eval_h = eval_month * 30 * 24
        peshawar_sites = set(sites[sites["region"] == "peshawar"]["site_id"])
        rng = np.random.default_rng(42)

        mask_post = (alarms["site_id"].isin(peshawar_sites)
                     & (alarms["timestamp_h"] >= drift_start_h))
        drop_mask = pd.Series(False, index=alarms.index)
        for code, prob in [("AC_MAINS_FAIL", ac_mains_drop),
                           ("RECTIFIER_FAULT", rectifier_drop),
                           ("BATT_UNDERVOLTAGE", undervoltage_drop),
                           ("LOAD_DISCONNECT", lvd_drop)]:
            m = mask_post & (alarms["alarm_code"] == code)
            drop = rng.random(m.sum()) < prob
            drop_mask.loc[alarms.index[m][drop]] = True

        alarms_drifted = alarms[~drop_mask].reset_index(drop=True)
        n_dropped = drop_mask.sum()

        # Score at evaluation time
        labels = pd.DataFrame({
            "site_id": sites["site_id"].values,
            "mains_fail_h": float(eval_h),
        })
        alarms_pit = alarms_drifted[alarms_drifted["timestamp_h"] <= eval_h]

        features = compute_features(
            labels=labels,
            groups=["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"],
            inputs={"alarms": alarms_pit, "site_static": sites, "schedule": schedule},
            ref_time_col="mains_fail_h",
        )

        X = features[meta["feature_cols"]].astype(float).fillna(0.0)
        raw_preds = booster.predict(xgb.DMatrix(X))
        preds = apply_calibrator(calibrator, raw_preds)

        # Run drift detection
        drift_report = detect_drift(
            reference_profile=ref_profile,
            current_features=features,
            current_predictions=preds,
            feature_cols=meta["feature_cols"],
        )

        summary = drift_report["summary"]

    # Results
    st.markdown("---")
    st.subheader("Simulation Results")

    # Impact summary
    st.info(f"**Intervention:** Dropped {n_dropped:,} Peshawar alarms after month {drift_month}. "
            f"Evaluating at month {eval_month}.")

    # Drift status
    if summary["status"] == "DRIFT_DETECTED":
        st.error(f"🚨 **DRIFT DETECTED** — {summary['n_significant_drift']} features drifted significantly")
    else:
        st.success("✅ No significant drift detected")

    col1, col2, col3 = st.columns(3)
    col1.metric("Significant Drift", summary["n_significant_drift"])
    col2.metric("Moderate Drift", summary["n_moderate_drift"])
    col3.metric("Retrain?", "YES" if summary["retrain_recommended"] else "NO")

    # Before vs after risk comparison
    st.subheader("Risk Score Shift — Peshawar vs Others")
    features["region"] = features["site_id"].map(dict(zip(sites["site_id"], sites["region"])))
    features["risk_score"] = preds

    comparison = features.groupby("region")["risk_score"].agg(["mean", "max"]).round(3)
    st.dataframe(comparison, use_container_width=True)

    # Drift chart
    st.subheader("Feature PSI — Top Drifted")
    fd = drift_report["feature_drift"].sort_values("psi", ascending=False).head(12).iloc[::-1]

    import plotly.express as px
    fig = px.bar(fd, x="psi", y="feature", orientation="h", color="drift_level",
                 color_discrete_map={"SIGNIFICANT": "#ef4444", "MODERATE": "#f59e0b", "STABLE": "#22c55e"},
                 title=f"Feature Drift After Peshawar Grid Upgrade (month {drift_month} → eval month {eval_month})")
    fig.add_vline(x=0.25, line_dash="dash", line_color="red")
    fig.add_vline(x=0.10, line_dash="dash", line_color="orange")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Reasons
    if summary.get("reasons"):
        st.subheader("Drift Reasons")
        for reason in summary["reasons"]:
            st.markdown(f"- {reason}")

    # The lesson
    st.markdown("---")
    st.subheader("💡 Why This Matters")
    st.markdown("""
    **The counterintuitive finding:** Peshawar risk scores go *up* after the grid
    upgrade, even though batteries are objectively healthier.

    **Why:** The model's top feature (`lvd_count_30d`) captures downstream symptoms,
    not root causes. After the intervention, worn batteries still occasionally drain
    (symptom persists) while the cause (frequent outages) has improved. The model
    can't tell the difference.

    **The correct response:** Don't blindly retrain. Investigate the cause, add
    a feature for the intervention, accumulate new labels under the new regime,
    then retrain with causal feature engineering.
    """)
else:
    st.info("👈 Configure simulation parameters and click **Run Drift Simulation**")
