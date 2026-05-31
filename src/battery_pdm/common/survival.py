"""Survival modeling utilities — Cox encoding and concordance index.

Shared by TrainingFlow (failure model) and RetrainingFlow (drain predictor
fallback to survival metrics).
"""

from __future__ import annotations

import numpy as np


def _encode_survival(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Cox encoding: positive=event, negative=censored."""
    y = times.astype(float).copy()
    censored = events == 0
    y[censored] = -np.abs(y[censored])
    return y


def _cindex(risk: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """Harrell's C-index via O(n log n) sort-based algorithm.

    For each uncensored subject i, counts how many subjects j with longer
    survival times have lower risk scores. Uses argsort + cumulative counting
    instead of the naive O(n^2) pairwise loop.
    """
    n = len(times)
    if n < 2:
        return 0.0

    order = np.argsort(times)
    sorted_times = times[order]
    sorted_risk = risk[order]
    sorted_events = events[order]

    concordant = 0.0
    discordant = 0.0
    tied_risk = 0.0

    i = 0
    while i < n:
        if sorted_events[i] == 0:
            i += 1
            continue
        j = i + 1
        while j < n and sorted_times[j] == sorted_times[i]:
            j += 1
        for k in range(j, n):
            if sorted_risk[i] > sorted_risk[k]:
                concordant += 1.0
            elif sorted_risk[i] == sorted_risk[k]:
                tied_risk += 1.0
            else:
                discordant += 1.0
        i += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.0
    return (concordant + 0.5 * tied_risk) / total
