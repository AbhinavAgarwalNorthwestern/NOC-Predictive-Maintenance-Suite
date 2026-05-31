"""Shared pytest fixtures for battery-pdm tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def small_alarms() -> pd.DataFrame:
    """Tiny synthetic alarm stream for unit tests.

    Two sites, ~30 alarms each, spanning ~3 months. Designed so a few sites
    have AC_MAINS_FAIL events with both LVD-terminated and grid-restored outcomes.
    """
    rows = []
    for site_idx, site in enumerate(["SITE_001", "SITE_002"]):
        base_t = 0.0 + site_idx * 0.5
        # 5 outage cycles
        for cycle in range(5):
            mf_t = base_t + cycle * 100.0
            rows.append(
                {
                    "site_id": site,
                    "timestamp_h": mf_t,
                    "alarm_code": "AC_MAINS_FAIL",
                    "severity": "warning",
                }
            )
            if cycle % 2 == 0:
                # LVD happens
                rows.append(
                    {
                        "site_id": site,
                        "timestamp_h": mf_t + 5.0,
                        "alarm_code": "LOAD_DISCONNECT",
                        "severity": "critical",
                    }
                )
            # Random rectifier fault during recovery
            if cycle % 3 == 0:
                rows.append(
                    {
                        "site_id": site,
                        "timestamp_h": mf_t + 20.0,
                        "alarm_code": "RECTIFIER_FAULT",
                        "severity": "warning",
                    }
                )
        # Add a cell imbalance
        rows.append(
            {
                "site_id": site,
                "timestamp_h": base_t + 150.0,
                "alarm_code": "CELL_IMBALANCE",
                "severity": "warning",
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["site_id", "timestamp_h"])
        .reset_index(drop=True)
    )


@pytest.fixture
def small_site_static() -> pd.DataFrame:
    """Site static config matching small_alarms fixture."""
    return pd.DataFrame(
        [
            {
                "site_id": "SITE_001",
                "region": "lahore",
                "manufacturer": "manufacturer_a",
                "install_month": 0.0,
                "load_A": 10.0,
                "n_cells": 24,
                "nominal_capacity_ah": 100.0,
                "charger_misconfigured": 0,
                "aging_multiplier": 1.0,
            },
            {
                "site_id": "SITE_002",
                "region": "karachi",
                "manufacturer": "manufacturer_b",
                "install_month": 6.0,
                "load_A": 15.0,
                "n_cells": 24,
                "nominal_capacity_ah": 120.0,
                "charger_misconfigured": 0,
                "aging_multiplier": 1.1,
            },
        ]
    )


@pytest.fixture
def small_schedule() -> pd.DataFrame:
    """Minimal load-shedding schedule covering small_alarms time range."""
    rows = []
    for region in ["lahore", "karachi"]:
        for t in range(0, 800):
            rows.append(
                {
                    "region": region,
                    "timestamp_h": t,
                    "hour_of_day": t % 24,
                    "day_of_year": (t // 24) % 365,
                    "is_summer": int(120 < ((t // 24) % 365) < 270),
                    "is_weekend": int(((t // 24) % 7) in (4, 5)),
                    "scheduled_offgrid": int(t % 7 == 0),
                    "severity_score": 0.3 if region == "karachi" else 0.15,
                    "expected_daily_offgrid_hours": 4.0 if region == "karachi" else 2.0,
                }
            )
    return pd.DataFrame(rows)
