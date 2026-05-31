"""Tests for cost-aware threshold optimization."""

from __future__ import annotations

import numpy as np
import pytest

from battery_pdm.monitoring.threshold import (
    CostMatrix,
    expected_cost,
    optimal_threshold,
    sensitivity_analysis,
    top_k_dispatch,
)


def test_expected_cost_zero_when_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    cost = expected_cost(
        0.5, scores, y, CostMatrix(false_positive=50, false_negative=1000)
    )
    assert cost == 0.0


def test_expected_cost_high_when_threshold_too_high():
    y = np.array([1, 1, 1, 1])
    scores = np.array([0.3, 0.4, 0.5, 0.6])
    cost = expected_cost(0.7, scores, y, CostMatrix(false_negative=1000))
    assert cost == 4000


def test_optimal_threshold_finds_lower_value_for_high_fn_cost():
    """If missing a positive is very expensive, optimal threshold should be low."""
    rng = np.random.default_rng(42)
    n = 1000
    y = (rng.uniform(0, 1, n) < 0.15).astype(int)
    scores = rng.uniform(0, 1, n) * 0.5 + y * 0.3 + rng.normal(0, 0.1, n)
    scores = np.clip(scores, 0, 1)

    high_fn_result = optimal_threshold(
        scores, y, CostMatrix(false_positive=1, false_negative=100)
    )
    low_fn_result = optimal_threshold(
        scores, y, CostMatrix(false_positive=1, false_negative=2)
    )
    assert high_fn_result["best_threshold"] < low_fn_result["best_threshold"], (
        f"Expected lower threshold for higher FN cost; got "
        f"high_fn={high_fn_result['best_threshold']}, low_fn={low_fn_result['best_threshold']}"
    )


def test_sensitivity_analysis_monotonic_in_ratio():
    """As FN/FP ratio rises, optimal threshold should monotonically decrease (or stay same)."""
    rng = np.random.default_rng(42)
    n = 500
    y = (rng.uniform(0, 1, n) < 0.15).astype(int)
    scores = rng.uniform(0, 1, n) * 0.5 + y * 0.3 + rng.normal(0, 0.1, n)
    scores = np.clip(scores, 0, 1)

    results = sensitivity_analysis(scores, y, fn_fp_ratios=[1, 5, 20, 100])
    thresholds = [r["best_threshold"] for r in results]
    for i in range(len(thresholds) - 1):
        assert thresholds[i + 1] <= thresholds[i] + 0.01, (
            f"Threshold should decrease with higher FN/FP ratio; got {thresholds}"
        )


def test_optimal_threshold_includes_metric_breakdown():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.3, 0.55, 0.8, 0.4])
    result = optimal_threshold(scores, y, CostMatrix())
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1


def test_top_k_dispatch_picks_highest_scores():
    """top-K should flag exactly K highest-scoring sites."""
    rng = np.random.default_rng(42)
    n = 100
    y = (rng.uniform(0, 1, n) < 0.20).astype(int)
    scores = rng.uniform(0, 1, n) * 0.5 + y * 0.4 + rng.normal(0, 0.05, n)
    scores = np.clip(scores, 0, 1)

    for k in [5, 10, 25, 50]:
        result = top_k_dispatch(scores, y, k, CostMatrix())
        # Sum of TP + FP should equal K (we dispatched exactly K sites)
        assert result["true_positives"] + result["false_positives"] == k, (
            f"top-K=K invariant violated at k={k}"
        )


def test_top_k_dispatch_effective_threshold_is_kth_highest_score():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    y = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0])
    result = top_k_dispatch(scores, y, k=4, cost_matrix=CostMatrix())
    # The 4th-highest score (index 3) is 0.6
    assert result["effective_threshold"] == 0.6


def test_top_k_dispatch_validates_k():
    scores = np.array([0.5, 0.6, 0.7])
    y = np.array([1, 0, 1])
    with pytest.raises(ValueError):
        top_k_dispatch(scores, y, k=0)
    with pytest.raises(ValueError):
        top_k_dispatch(scores, y, k=4)
