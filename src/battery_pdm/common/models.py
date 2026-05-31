"""Model definitions — hyperparameters, training, and prediction.

Two models, two operational decisions:

    FailureModel (survival:cox)
        Question: "Should this battery be replaced?"
        Horizon:  6-24 months
        Metric:   C-index

    DrainPredictor (binary:logistic + isotonic calibration)
        Question: "Will this site drain in the next 48h?"
        Horizon:  48 hours
        Metric:   AUC + calibrated Brier

To change the model (e.g., swap XGBoost for LightGBM), edit THIS file.
The flows in flows/ are orchestration — they call these classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from battery_pdm.common.survival import _cindex


# ---------------------------------------------------------------------------
# Failure model — long-term battery replacement ranking
# ---------------------------------------------------------------------------

FAILURE_PARAMS: dict = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

FAILURE_FEATURE_GROUPS: list[str] = ["alarm_history", "site_static", "soc_proxy"]
FAILURE_NUM_BOOST_ROUND: int = 200
FAILURE_EARLY_STOPPING: int = 20
FAILURE_LEAKY_FEATURES: set[str] = {"charger_misconfigured", "aging_multiplier"}


def train_failure_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
) -> xgb.Booster:
    """Train a single failure model (survival:cox)."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    evals = [(dtrain, "train")]
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, "val"))

    return xgb.train(
        FAILURE_PARAMS,
        dtrain,
        num_boost_round=FAILURE_NUM_BOOST_ROUND,
        evals=evals,
        early_stopping_rounds=FAILURE_EARLY_STOPPING if X_val is not None else None,
        verbose_eval=False,
    )


def evaluate_failure_model(
    booster: xgb.Booster,
    X_test: pd.DataFrame,
    times: np.ndarray,
    events: np.ndarray,
) -> float:
    """Evaluate failure model — returns C-index."""
    risk = booster.predict(xgb.DMatrix(X_test))
    return _cindex(risk, times, events)


# ---------------------------------------------------------------------------
# Drain predictor — 48h binary classifier with calibration
# ---------------------------------------------------------------------------

DRAIN_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

DRAIN_FEATURE_GROUPS: list[str] = [
    "alarm_history",
    "site_static",
    "soc_proxy",
    "load_shedding_schedule",
]
DRAIN_HORIZON_H: int = 48
DRAIN_SAMPLE_INTERVAL_DAYS: int = 7
DRAIN_NUM_BOOST_ROUND: int = 300
DRAIN_EARLY_STOPPING: int = 30
DRAIN_LEAKY_FEATURES: set[str] = {"charger_misconfigured", "aging_multiplier"}


def train_drain_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    scale_pos_weight: float = 1.0,
) -> xgb.Booster:
    """Train a single drain predictor (binary:logistic)."""
    params = {**DRAIN_PARAMS, "scale_pos_weight": scale_pos_weight}
    dtrain = xgb.DMatrix(X_train, label=y_train)
    evals = [(dtrain, "train")]
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, "val"))

    return xgb.train(
        params,
        dtrain,
        num_boost_round=DRAIN_NUM_BOOST_ROUND,
        evals=evals,
        early_stopping_rounds=DRAIN_EARLY_STOPPING if X_val is not None else None,
        verbose_eval=False,
    )
