"""XGBoost survival model training and evaluation.

Trains XGBoost with survival:cox objective and evaluates with C-index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional


def compute_cindex(risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """Harrell's concordance index.

    C = P(risk_i > risk_j | t_i < t_j, event_i = 1)

    Among all comparable pairs (one failed, one survived longer),
    what fraction did the model rank correctly?

    Parameters
    ----------
    risk_scores : np.ndarray
        Model output (higher = more risky)
    times : np.ndarray
        Observed times (positive)
    events : np.ndarray
        Event indicators (1 = failure, 0 = censored)

    Returns
    -------
    float : C-index in [0, 1]. 0.5 = random, 1.0 = perfect.

    TODO (USER): implement. This is the survival-model equivalent of AUC.
    The math is: iterate over all pairs, count concordant vs discordant.
    """
    raise NotImplementedError("USER: implement C-index")


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
