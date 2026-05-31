"""Battery PdM — FastAPI inference + batch trigger API.

Endpoints:
    POST /predict          — real-time drain risk for a single site
    POST /predict/batch    — drain risk for all sites (or a list)
    POST /predict/failure  — failure risk for a single site
    POST /run-flow         — trigger an AWS Batch flow run
    GET  /health           — liveness check
    GET  /models           — current model versions
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["alarms"] = pd.read_parquet(DATA_DIR / "alarms.parquet")
    _state["sites"] = pd.read_parquet(DATA_DIR / "site_static.parquet")

    drain_dir = DATA_DIR / "models" / "drain_predictor_48h"
    _state["drain_booster"] = xgb.Booster()
    _state["drain_booster"].load_model(str(drain_dir / "booster.json"))
    _state["drain_meta"] = json.loads((drain_dir / "meta.json").read_text())

    cal_path = drain_dir / "calibrator.pkl"
    if cal_path.exists():
        import pickle
        _state["drain_calibrator"] = pickle.loads(cal_path.read_bytes())
    else:
        _state["drain_calibrator"] = None

    failure_dir = DATA_DIR / "models" / "failure_alarms_only"
    if (failure_dir / "booster.json").exists():
        _state["failure_booster"] = xgb.Booster()
        _state["failure_booster"].load_model(str(failure_dir / "booster.json"))
        _state["failure_meta"] = json.loads((failure_dir / "meta.json").read_text())
    else:
        _state["failure_booster"] = None

    from battery_pdm.synth.load_shedding import build_load_shedding_schedule
    _state["schedule"] = build_load_shedding_schedule(n_months=36, seed=42)

    yield
    _state.clear()


app = FastAPI(
    title="Battery PdM API",
    version="1.0.0",
    description="Real-time scoring and batch trigger for battery predictive maintenance.",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    site_id: str
    sim_hour: float | None = None


class BatchPredictRequest(BaseModel):
    site_ids: list[str] | None = None
    sim_hour: float | None = None


class FlowTriggerRequest(BaseModel):
    flow_name: str
    overrides: dict | None = None


def _compute_features_for_sites(site_ids: list[str], sim_hour: float | None) -> pd.DataFrame:
    from battery_pdm.common.features import compute_features

    alarms = _state["alarms"]
    sites = _state["sites"]
    schedule = _state["schedule"]

    current_h = sim_hour or float(alarms["timestamp_h"].max())
    site_df = sites[sites["site_id"].isin(site_ids)]
    if site_df.empty:
        raise HTTPException(status_code=404, detail=f"No matching sites found")

    labels = pd.DataFrame({"site_id": site_df["site_id"].values, "mains_fail_h": current_h})
    alarms_pit = alarms[(alarms["site_id"].isin(site_ids)) & (alarms["timestamp_h"] <= current_h)]

    features = compute_features(
        labels=labels,
        groups=["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"],
        inputs={"alarms": alarms_pit, "site_static": site_df, "schedule": schedule},
        ref_time_col="mains_fail_h",
    )
    return features


def _score_drain(features: pd.DataFrame) -> dict:
    from battery_pdm.monitoring.model_registry import apply_calibrator

    meta = _state["drain_meta"]
    feature_cols = meta["feature_cols"]
    X = features[feature_cols].astype(float).fillna(0.0)
    raw = _state["drain_booster"].predict(xgb.DMatrix(X))
    scores = apply_calibrator(_state["drain_calibrator"], raw)

    importance = _state["drain_booster"].get_score(importance_type="gain")
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

    results = []
    for i, row in features.iterrows():
        score = float(scores[i] if i < len(scores) else scores[features.index.get_loc(i)])
        alert_level = "HIGH" if score >= 0.6 else ("MEDIUM" if score >= 0.4 else "LOW")
        results.append({
            "site_id": row["site_id"],
            "drain_risk_48h": round(score, 4),
            "alert_level": alert_level,
        })

    return {
        "predictions": results,
        "model_version": meta["model_version"],
        "top_model_features": [{"feature": f, "gain": round(g, 2)} for f, g in top_features],
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(k for k in _state if "booster" in k)}


@app.get("/models")
def models():
    info = {}
    if "drain_meta" in _state:
        m = _state["drain_meta"]
        info["drain_predictor_48h"] = {
            "version": m["model_version"],
            "auc": m["metrics"].get("roc_auc"),
            "features": len(m["feature_cols"]),
            "calibrated": _state["drain_calibrator"] is not None,
        }
    if "failure_meta" in _state:
        m = _state["failure_meta"]
        info["failure_alarms_only"] = {
            "version": m["model_version"],
            "cindex": m["metrics"].get("mean_cindex"),
            "features": len(m["feature_cols"]),
        }
    return info


@app.post("/predict")
def predict(req: PredictRequest):
    features = _compute_features_for_sites([req.site_id], req.sim_hour)
    return _score_drain(features)


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    site_ids = req.site_ids or _state["sites"]["site_id"].tolist()
    features = _compute_features_for_sites(site_ids, req.sim_hour)
    return _score_drain(features)


@app.post("/predict/failure")
def predict_failure(req: PredictRequest):
    if not _state.get("failure_booster"):
        raise HTTPException(status_code=503, detail="Failure model not loaded")

    from battery_pdm.common.features import compute_features

    alarms = _state["alarms"]
    sites = _state["sites"]
    current_h = req.sim_hour or float(alarms["timestamp_h"].max())

    site_df = sites[sites["site_id"] == req.site_id]
    if site_df.empty:
        raise HTTPException(status_code=404, detail=f"Site {req.site_id} not found")

    labels = pd.DataFrame({"site_id": [req.site_id], "event_hour": current_h})
    alarms_pit = alarms[(alarms["site_id"] == req.site_id) & (alarms["timestamp_h"] <= current_h)]

    meta = _state["failure_meta"]
    feature_groups = meta.get("feature_groups", ["alarm_history", "site_static", "soc_proxy"])
    features = compute_features(
        labels=labels, groups=feature_groups,
        inputs={"alarms": alarms_pit, "site_static": site_df},
        ref_time_col="event_hour",
    )

    feature_cols = meta["feature_cols"]
    X = features[feature_cols].astype(float).fillna(0.0)
    risk = float(_state["failure_booster"].predict(xgb.DMatrix(X))[0])
    priority = "REPLACE_NOW" if risk > 2.0 else ("MONITOR" if risk > 1.0 else "OK")

    return {
        "site_id": req.site_id,
        "failure_risk_score": round(risk, 4),
        "priority": priority,
        "model_version": meta["model_version"],
    }


@app.post("/run-flow")
def run_flow(req: FlowTriggerRequest):
    import boto3

    valid_flows = ["training", "drain_predictor", "failure_scoring", "drift_monitor", "retraining"]
    if req.flow_name not in valid_flows:
        raise HTTPException(status_code=400, detail=f"Invalid flow. Choose from: {valid_flows}")

    region = os.getenv("AWS_REGION", "ap-south-1")
    queue = os.getenv("BATCH_QUEUE", "battery-pdm-dev-queue")
    job_def = os.getenv("BATCH_JOB_PREFIX", "battery-pdm-dev") + f"-{req.flow_name}"

    client = boto3.client("batch", region_name=region)
    params = {"jobName": f"{req.flow_name}-api-trigger", "jobQueue": queue, "jobDefinition": job_def}

    if req.overrides:
        env = [{"name": k, "value": str(v)} for k, v in req.overrides.items()]
        params["containerOverrides"] = {"environment": env}

    resp = client.submit_job(**params)
    return {
        "job_id": resp["jobId"],
        "job_name": resp["jobName"],
        "flow": req.flow_name,
        "status": "SUBMITTED",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
