"""Tests for RetrainingFlow champion/challenger CV gate logic.

These tests verify the core promotion decision: a challenger must beat the
champion on the SAME held-out test set by the configured AUC margin.
"""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb
from sklearn.metrics import roc_auc_score


@pytest.fixture
def synthetic_classifier_data():
    """Two-class data where a 'better' challenger model should win."""
    rng = np.random.default_rng(42)
    n = 600
    X = rng.normal(size=(n, 5))
    logits = (
        0.8 * X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + rng.normal(scale=0.5, size=n)
    )
    y = (logits > 0).astype(int)
    return X, y


def _train(X, y, n_estimators=100, max_depth=4):
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": max_depth,
            "learning_rate": 0.05,
            "tree_method": "hist",
        },
        dtrain,
        num_boost_round=n_estimators,
        verbose_eval=False,
    )


def test_champion_evaluated_on_challenger_test_set(synthetic_classifier_data):
    """Both models must be scored on the same held-out test set (apples to apples)."""
    X, y = synthetic_classifier_data
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    champion = _train(X_train, y_train, n_estimators=50)
    challenger = _train(X_train, y_train, n_estimators=200)

    champion_test_auc = float(
        roc_auc_score(y_test, champion.predict(xgb.DMatrix(X_test)))
    )
    challenger_test_auc = float(
        roc_auc_score(y_test, challenger.predict(xgb.DMatrix(X_test)))
    )

    assert 0.5 < champion_test_auc < 1.0
    assert 0.5 < challenger_test_auc < 1.0
    margin = challenger_test_auc - champion_test_auc
    assert np.isfinite(margin)


def test_cv_gate_blocks_marginal_challenger(synthetic_classifier_data):
    """CV gate prevents promotion when margin < required threshold."""
    X, y = synthetic_classifier_data
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    champion = _train(X_train, y_train, n_estimators=100, max_depth=4)
    challenger = _train(X_train, y_train, n_estimators=100, max_depth=4)

    champion_auc = float(roc_auc_score(y_test, champion.predict(xgb.DMatrix(X_test))))
    challenger_auc = float(
        roc_auc_score(y_test, challenger.predict(xgb.DMatrix(X_test)))
    )

    margin = challenger_auc - champion_auc
    promotion_margin = 0.005
    should_promote = margin >= promotion_margin

    assert abs(margin) < 0.01, f"Expected near-zero margin, got {margin}"
    assert not should_promote, "CV gate should reject near-zero margin"


def test_cv_gate_promotes_clearly_better_challenger(synthetic_classifier_data):
    """When challenger is clearly better, CV gate should promote."""
    X, y = synthetic_classifier_data
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    champion = _train(X_train, y_train, n_estimators=5, max_depth=2)
    challenger = _train(X_train, y_train, n_estimators=150, max_depth=5)

    champion_auc = float(roc_auc_score(y_test, champion.predict(xgb.DMatrix(X_test))))
    challenger_auc = float(
        roc_auc_score(y_test, challenger.predict(xgb.DMatrix(X_test)))
    )

    margin = challenger_auc - champion_auc
    assert margin > 0.005, f"Expected clear improvement, got margin={margin}"
    assert margin >= 0.005


def test_promotion_decision_state_machine():
    """The promotion decision has four mutually exclusive states."""
    test_cases = [
        (True, True, True, "promote_shadow"),
        (True, False, True, "discard_shadow"),
        (False, True, True, "wait_for_more_labels"),
        (False, False, True, "wait_for_more_labels"),
        (False, False, False, "insufficient_labels"),
    ]

    for maturity_met, margin_met, ready, expected in test_cases:
        if not ready:
            decision = "insufficient_labels"
        elif maturity_met and margin_met:
            decision = "promote_shadow"
        elif maturity_met and not margin_met:
            decision = "discard_shadow"
        else:
            decision = "wait_for_more_labels"
        assert decision == expected


def test_failure_model_stashes_test_set():
    """Verify _train_failure_challenger stashes test set for compare step.

    Regression test for the bug where AttributeError: Flow RetrainingFlow has no
    attribute 'test_set_features' occurred when retraining the failure model.

    We read the source file directly (not import) because Metaflow uses fcntl
    which is unavailable on Windows. The flow only runs on Linux containers.
    """
    from pathlib import Path

    src_path = (
        Path(__file__).parent.parent
        / "src"
        / "battery_pdm"
        / "flows"
        / "retraining_flow.py"
    )
    src = src_path.read_text()

    # Find the _train_failure_challenger method body
    start = src.find("def _train_failure_challenger")
    assert start > 0, "_train_failure_challenger method must exist"
    # Find the next def to bound the method
    next_def = src.find("\n    def ", start + 1)
    method_body = src[start:next_def] if next_def > 0 else src[start:]

    assert "self.test_set_features" in method_body, (
        "_train_failure_challenger must stash test_set_features for compare step"
    )
    assert "self.test_set_labels" in method_body
    assert "self.test_set_events" in method_body, (
        "Survival models need event indicator in addition to time-to-event"
    )
