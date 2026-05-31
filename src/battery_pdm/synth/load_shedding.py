"""Synthetic load-shedding schedule per region.

In real-world NOC operations, the utility company publishes a load-shedding
calendar by region/feeder. Sites in Lahore are scheduled off-grid 6h/day in
summer, etc. This is INDEPENDENT of the actual physics simulation — it's
metadata operators already have access to.

We generate a calendar at hourly granularity for the full simulation window:
    (region, timestamp_h) -> expected_offgrid (bool), severity_score (0-1)

Schedule structure (realistic for Pakistan-like grids):
    - Daily blocks: morning peak (6-9am) and evening peak (6-10pm)
    - Summer amplification: 1.5-2x daily off-grid hours
    - Region-specific patterns:
        Lahore (urban):     light shedding, mostly evening peaks
        Karachi:            moderate, day-and-night
        Peshawar:           heavy summer shedding
        Quetta:             worst — winter+summer, long blocks
        Islamabad:          mild, mostly summer afternoons
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from battery_pdm.synth.config import Region


# Per region: typical daily off-grid hours in (winter, summer)
# Plus the "peak windows" — hours of day when shedding is concentrated
REGION_SCHEDULE = {
    Region.LAHORE: {
        "winter_hours": 3.0,
        "summer_hours": 6.0,
        "peak_windows": [(7, 10), (18, 22)],  # morning + evening
        "weekend_relief": 0.6,  # weekends slightly less
    },
    Region.KARACHI: {
        "winter_hours": 4.0,
        "summer_hours": 7.0,
        "peak_windows": [(6, 9), (12, 14), (19, 22)],
        "weekend_relief": 0.7,
    },
    Region.PESHAWAR: {
        "winter_hours": 5.0,
        "summer_hours": 10.0,
        "peak_windows": [(6, 11), (17, 22)],
        "weekend_relief": 0.8,
    },
    Region.QUETTA: {
        "winter_hours": 7.0,
        "summer_hours": 12.0,
        "peak_windows": [(5, 10), (15, 19), (21, 24)],
        "weekend_relief": 0.9,
    },
    Region.ISLAMABAD: {
        "winter_hours": 2.0,
        "summer_hours": 5.0,
        "peak_windows": [(13, 16), (19, 22)],
        "weekend_relief": 0.5,
    },
}


def is_summer(day_of_year: int) -> bool:
    """May-Sep is peak summer for Pakistan grids."""
    return 120 < (day_of_year % 365) < 270


def is_weekend(hour: int) -> bool:
    """Fri-Sat are weekend in Pakistan."""
    day_of_week = (hour // 24) % 7
    return day_of_week in (4, 5)  # Fri (4), Sat (5)


def hour_in_peak_window(hour_of_day: int, peak_windows: list[tuple[int, int]]) -> bool:
    return any(start <= hour_of_day < end for start, end in peak_windows)


def expected_offgrid_hours_per_day(
    region: Region, day_of_year: int, weekend: bool
) -> float:
    sched = REGION_SCHEDULE[region]
    base = sched["summer_hours"] if is_summer(day_of_year) else sched["winter_hours"]
    if weekend:
        base *= sched["weekend_relief"]
    return base


def build_load_shedding_schedule(
    n_months: int = 36,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate hourly load-shedding schedule for all regions.

    Returns DataFrame with columns:
        region, timestamp_h, scheduled_offgrid (bool),
        severity_score (0-1: utility's published "intensity" for this hour),
        expected_daily_offgrid_hours
    """
    rng = np.random.default_rng(seed)
    total_hours = n_months * 30 * 24

    records = []
    for region in Region:
        sched = REGION_SCHEDULE[region]
        peak_windows = sched["peak_windows"]

        for hour in range(0, total_hours, 1):
            day_of_year = (hour // 24) % 365
            hour_of_day = hour % 24
            weekend = is_weekend(hour)
            summer = is_summer(day_of_year)

            expected_daily = expected_offgrid_hours_per_day(
                region, day_of_year, weekend
            )

            # Schedule says: peak hours are more likely off-grid
            in_peak = hour_in_peak_window(hour_of_day, peak_windows)
            if in_peak:
                # During peak window, probability of off-grid is high
                prob = min(
                    0.85,
                    expected_daily / sum(end - start for start, end in peak_windows),
                )
            else:
                # Outside peak: rare unplanned outages
                prob = 0.05 if summer else 0.02

            severity = prob  # used as a continuous signal

            records.append(
                {
                    "region": region.value,
                    "timestamp_h": hour,
                    "hour_of_day": hour_of_day,
                    "day_of_year": day_of_year,
                    "is_summer": int(summer),
                    "is_weekend": int(weekend),
                    "scheduled_offgrid": int(
                        in_peak and (summer or rng.random() < 0.7)
                    ),
                    "severity_score": float(severity),
                    "expected_daily_offgrid_hours": float(expected_daily),
                }
            )

    df = pd.DataFrame(records)
    return df


def aggregate_schedule_window(
    schedule: pd.DataFrame,
    region: str,
    ref_time_h: float,
    lookback_h: float = 30 * 24,
    lookahead_h: float = 24,
) -> dict:
    """Compute aggregate schedule features around a reference time.

    Returns dict of features computable from the schedule alone (no battery state).
    """
    region_df = schedule[schedule["region"] == region]

    past = region_df[
        (region_df["timestamp_h"] < ref_time_h)
        & (region_df["timestamp_h"] >= ref_time_h - lookback_h)
    ]
    future = region_df[
        (region_df["timestamp_h"] >= ref_time_h)
        & (region_df["timestamp_h"] < ref_time_h + lookahead_h)
    ]

    if past.empty or future.empty:
        return {
            "schedule_offgrid_hours_past_30d": 0.0,
            "schedule_severity_mean_past_30d": 0.0,
            "schedule_offgrid_hours_next_24h": 0.0,
            "schedule_severity_max_next_24h": 0.0,
            "currently_in_peak_window": 0,
            "expected_daily_offgrid_now": 0.0,
        }

    now_row = region_df[region_df["timestamp_h"] == int(ref_time_h)]
    in_peak_now = int(now_row["scheduled_offgrid"].iloc[0]) if not now_row.empty else 0
    expected_now = (
        float(now_row["expected_daily_offgrid_hours"].iloc[0])
        if not now_row.empty
        else 0.0
    )

    return {
        "schedule_offgrid_hours_past_30d": float(past["scheduled_offgrid"].sum()),
        "schedule_severity_mean_past_30d": float(past["severity_score"].mean()),
        "schedule_offgrid_hours_next_24h": float(future["scheduled_offgrid"].sum()),
        "schedule_severity_max_next_24h": float(future["severity_score"].max()),
        "currently_in_peak_window": in_peak_now,
        "expected_daily_offgrid_now": expected_now,
    }


if __name__ == "__main__":
    from pathlib import Path

    print("Generating load-shedding schedule (36 months, 5 regions)...")
    sched = build_load_shedding_schedule(n_months=36)
    print(f"  Rows: {len(sched):,}")
    print("  Per-region summary:")
    print(sched.groupby("region")["scheduled_offgrid"].agg(["sum", "mean"]).to_string())

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    sched.to_parquet(out_dir / "load_shedding_schedule.parquet", index=False)
    print("\nSaved to outputs/load_shedding_schedule.parquet")
