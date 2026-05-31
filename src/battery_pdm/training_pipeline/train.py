"""XGBoost survival model training and evaluation.

Trains XGBoost with survival:cox objective and evaluates with C-index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional


def compute_cindex(
    risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray
) -> float:
    """Harrell's concordance index. O(n^2) but fine for <5000 samples."""
    concordant = 0
    discordant = 0
    tied = 0
    n = len(times)

    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[j] > times[i]:
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
                else:
                    tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


def train_survival_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
) -> xgb.Booster:
    """Train XGBoost with survival:cox objective.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix (n_samples, n_features)
    y_train : np.ndarray
        Survival labels: positive = failure time, negative = censored time
    X_val : optional validation features
    y_val : optional validation labels
    params : optional XGBoost params override

    Returns
    -------
    xgb.Booster : trained model
    """
    default_params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method": "hist",
        "seed": 42,
    }
    if params:
        default_params.update(params)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    watchlist = [(dtrain, "train")]

    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        watchlist.append((dval, "val"))

    model = xgb.train(
        default_params,
        dtrain,
        num_boost_round=200,
        evals=watchlist,
        early_stopping_rounds=20,
        verbose_eval=25,
    )

    return model


def predict_risk(model: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Get risk scores from trained model. Higher = more risky."""
    dmatrix = xgb.DMatrix(X)
    return model.predict(dmatrix)
