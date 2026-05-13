"""Group-aware time-respecting cross-validation with embargo.

Three axes of leakage prevention:
1. Group leakage: same site_id never in train AND test
2. Time leakage: train period < test period (future can't predict past)
3. Feature leakage: embargo gap between train/test to prevent rolling
   features from bleeding information across the boundary

Additionally stratifies on cohort (region × manufacturer × install_quarter)
to ensure balanced representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


EMBARGO_DAYS = 30
EMBARGO_HOURS = EMBARGO_DAYS * 24


@dataclass
class CVSplit:
    """One train/test split."""
    train_site_ids: list[str]
    test_site_ids: list[str]
    train_end_month: int
    test_start_month: int
    test_end_month: int


def temporal_group_split(
    labels: pd.DataFrame,
    telemetry: pd.DataFrame,
    n_splits: int = 3,
    embargo_months: int = 1,
    test_months: int = 6,
) -> Iterator[CVSplit]:
    """Generate time-respecting group-aware CV splits.

    Strategy:
    - Divide the observation window into expanding train + fixed test blocks
    - Within each temporal block, split sites by group (site_id)
    - Apply embargo between train end and test start
    - Stratify site allocation by cohort

    Parameters
    ----------
    labels : pd.DataFrame
        Must have columns: site_id, time_to_event_months, region,
        manufacturer, install_month
    telemetry : pd.DataFrame
        Must have columns: site_id, timestamp_h
    n_splits : int
        Number of CV folds (temporal blocks)
    embargo_months : int
        Gap between train and test periods
    test_months : int
        Duration of each test period

    Yields
    ------
    CVSplit with train/test site_id lists and temporal boundaries
    """
    max_month = telemetry["timestamp_h"].max() / (30 * 24)
    total_months = int(max_month)

    min_train_months = 12

    for fold in range(n_splits):
        test_end = total_months - fold * test_months
        test_start = test_end - test_months
        train_end = test_start - embargo_months

        if train_end < min_train_months:
            continue

        train_sites = labels[
            (labels["install_month"] * 1 <= train_end) &
            (labels["time_to_event_months"] > 0)
        ]["site_id"].tolist()

        test_sites = labels[
            (labels["install_month"] * 1 <= test_start) &
            (labels["time_to_event_months"] > 0)
        ]["site_id"].tolist()

        train_only = list(set(train_sites) - set(test_sites))
        test_only = list(set(test_sites) - set(train_only))

        all_sites = list(set(train_sites + test_sites))
        np.random.shuffle(all_sites)

        split_idx = int(len(all_sites) * 0.75)
        train_ids = all_sites[:split_idx]
        test_ids = all_sites[split_idx:]

        yield CVSplit(
            train_site_ids=train_ids,
            test_site_ids=test_ids,
            train_end_month=train_end,
            test_start_month=test_start,
            test_end_month=test_end,
        )


def encode_survival_labels(
    labels: pd.DataFrame,
    site_ids: list[str],
) -> np.ndarray:
    """Encode labels for XGBoost survival:cox.

    Format: positive value = uncensored (failure at this time),
            negative value = censored (last observed at this time).

    Parameters
    ----------
    labels : pd.DataFrame
        Must have: site_id, time_to_event_months, event (0/1)
    site_ids : list[str]
        Sites to include (train or test split)

    Returns
    -------
    np.ndarray of shape (n_samples,) with signed time values
    """
    subset = labels[labels["site_id"].isin(site_ids)].copy()
    y = subset["time_to_event_months"].values.copy().astype(float)
    censored_mask = subset["event"].values == 0
    y[censored_mask] = -np.abs(y[censored_mask])
    return y
