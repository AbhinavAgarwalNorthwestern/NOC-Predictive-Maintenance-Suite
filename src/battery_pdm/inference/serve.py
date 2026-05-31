"""SageMaker inference handler.

SageMaker's PyTorch/XGBoost serving container looks for these functions:
    model_fn(model_dir)   - load model artifacts
    input_fn(body, ctype) - parse the request body
    predict_fn(input, m)  - run prediction
    output_fn(pred, ctype)- format the response
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb


def model_fn(model_dir):
    """Load model artifacts from /opt/ml/model/."""
    from battery_pdm.monitoring.model_registry import load_calibrator

    model_dir = Path(model_dir)
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "booster.json"))
    meta = json.loads((model_dir / "meta.json").read_text())
    calibrator = load_calibrator(model_dir)
    return {"booster": booster, "meta": meta, "calibrator": calibrator}


def input_fn(request_body, content_type):
    """Parse JSON request → DataFrame."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    if "features" in payload:
        return pd.DataFrame(payload["features"])
    raise ValueError("Request body must include 'features' key with list of records")


def predict_fn(input_df, model):
    """Validate features + score + calibrate."""
    from battery_pdm.monitoring.model_registry import (
        validate_feature_hash,
        apply_calibrator,
    )

    feature_cols = model["meta"]["feature_cols"]
    validate_feature_hash(model["meta"], feature_cols)

    # Ensure all required columns exist + ordered correctly
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0.0
    X = input_df[feature_cols].astype(float).fillna(0.0)

    raw = model["booster"].predict(xgb.DMatrix(X))
    calibrated = apply_calibrator(model["calibrator"], raw)
    return {
        "predictions": [float(p) for p in calibrated],
        "model_version": model["meta"].get("model_version", "unknown"),
    }


def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
