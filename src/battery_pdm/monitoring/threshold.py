"""Cost-aware decision threshold + budget-aware top-K dispatch optimization.

Two complementary operational decisions:

1. THRESHOLD: minimum risk worth considering. Below this, we don't even look.
2. CAPACITY (top-K): given finite dispatch resources, take the highest-risk K.

Real NOC operations use BOTH. Threshold alone (without capacity) can recommend
flagging 100% of sites if the cost asymmetry is high enough — mathematically
correct but operationally absurd because you can't actually dispatch 3,300
sites in one day.

Realistic default cost matrix for the drain predictor:
    FP cost = 200  (dispatch + operator time + opportunity + alert-fatigue tax)
    FN cost = 1000 (SLA penalty + customer churn + emergency response)

The "FP includes hidden costs" framing is critical — a naive FP of $50 (just
fuel + truck) understates the true operational cost by 4-5×. The hidden tax
includes: operator hours opportunity cost, alert fatigue degrading future
response, site-visit disruption.

For each customer, set FP cost to capture all observable + hidden costs of an
unnecessary dispatch, and FN cost to capture the worst-case drain consequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostMatrix:
    """Operational cost of each error type.

    All in the same currency unit (USD per event).
    Default values include hidden costs (alert fatigue, opportunity cost,
    capacity tax) — direct dispatch fuel cost alone would be ~$50; the $200
    default captures the full operational impact.
    """

    false_positive: float = 200.0  # full operational cost of unnecessary dispatch
    false_negative: float = 1000.0  # SLA + churn cost of a missed drain
    true_positive: float = 0.0  # cost when we correctly dispatch (offset by saved FN)
    true_negative: float = 0.0  # cost when we correctly skip


def expected_cost(
    threshold: float,
    scores: np.ndarray,
    y_true: np.ndarray,
    cost_matrix: CostMatrix | None = None,
) -> float:
    """Total expected cost across the dataset at this threshold."""
    cm = cost_matrix or CostMatrix()
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return float(
        tp * cm.true_positive
        + fp * cm.false_positive
        + fn * cm.false_negative
        + tn * cm.true_negative
    )


def optimal_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    cost_matrix: CostMatrix | None = None,
    grid: np.ndarray | None = None,
) -> dict:
    """Find the threshold that minimizes expected operational cost.

    Returns a dict with:
        - best_threshold
        - best_cost
        - cost_curve_thresholds, cost_curve_costs
        - precision, recall, f1 at best_threshold
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    cm = cost_matrix or CostMatrix()
    if grid is None:
        grid = np.arange(0.01, 1.0, 0.01)

    costs = np.array([expected_cost(t, scores, y_true, cm) for t in grid])
    best_idx = int(np.argmin(costs))
    best_t = float(grid[best_idx])
    pred = (scores >= best_t).astype(int)
    return {
        "best_threshold": best_t,
        "best_cost": float(costs[best_idx]),
        "cost_curve_thresholds": grid,
        "cost_curve_costs": costs,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "cost_matrix": {
            "false_positive": cm.false_positive,
            "false_negative": cm.false_negative,
        },
    }


def top_k_dispatch(
    scores: np.ndarray,
    y_true: np.ndarray,
    k: int,
    cost_matrix: CostMatrix | None = None,
) -> dict:
    """Capacity-constrained dispatch: take the top-K sites by score.

    This is what real NOCs do. Capacity (K) is set by operator availability;
    you don't get to dispatch more than you have crews for.

    Returns:
        - effective_threshold: the score at the K-th highest position
        - cost_at_top_k: realized cost with this strategy
        - cost_at_unconstrained_optimum: what unconstrained optimization would give
        - capacity_savings_vs_unconstrained: positive = better than just unconstrained
        - precision, recall, f1 at top-K
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    cm = cost_matrix or CostMatrix()
    if k <= 0 or k > len(scores):
        raise ValueError(f"k must be in (0, {len(scores)}], got {k}")

    # Sort indices by score descending
    order = np.argsort(-scores)
    pred = np.zeros_like(y_true)
    pred[order[:k]] = 1

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    effective_threshold = float(scores[order[k - 1]])
    cost_at_top_k = (
        tp * cm.true_positive
        + fp * cm.false_positive
        + fn * cm.false_negative
        + tn * cm.true_negative
    )

    return {
        "k": k,
        "effective_threshold": effective_threshold,
        "cost_at_top_k": float(cost_at_top_k),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def sensitivity_analysis(
    scores: np.ndarray,
    y_true: np.ndarray,
    fn_fp_ratios: list[float] | None = None,
) -> list[dict]:
    """How does the optimal threshold shift as the FN/FP cost ratio varies?

    Useful when stakeholders aren't sure of the exact cost numbers — shows
    the threshold's sensitivity to the assumption.
    """
    if fn_fp_ratios is None:
        fn_fp_ratios = [2, 5, 10, 20, 50, 100]
    results = []
    for ratio in fn_fp_ratios:
        cm = CostMatrix(false_positive=1.0, false_negative=float(ratio))
        result = optimal_threshold(scores, y_true, cm)
        results.append(
            {
                "fn_fp_ratio": ratio,
                "best_threshold": result["best_threshold"],
                "best_cost": result["best_cost"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            }
        )
    return results
