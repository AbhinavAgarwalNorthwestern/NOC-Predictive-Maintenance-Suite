"""Feature computation functions — five families.

YOU IMPLEMENT the core math functions. The registry wiring and DataFrame
plumbing is scaffolded.

Each feature function receives:
    - telemetry: full telemetry DataFrame (site_id, timestamp_h, voltage,
      current, temperature_c, soc, health, on_grid)
    - labels: label DataFrame (for site_id list and metadata)

Returns:
    - pd.Series indexed by site_id with the feature value

Families:
    1. per_discharge — computed from discharge events
    2. per_recharge — computed from recharge events
    3. rolling_cumulative — windowed aggregates
    4. inter_event — time between events
    5. trend — slopes over time
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from battery_pdm.features.registry import register_feature


WINDOW_DAYS = 30
WINDOW_HOURS = WINDOW_DAYS * 24


# ---------------------------------------------------------------------------
# Helper: identify discharge events
# ---------------------------------------------------------------------------


def extract_discharge_events(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Identify contiguous discharge periods per site.

    Returns DataFrame with columns:
        site_id, start_h, end_h, duration_h, voltage_start, voltage_end,
        mean_current, dv_dt
    """
    events = []
    for site_id, group in telemetry.groupby("site_id"):
        group = group.sort_values("timestamp_h")
        is_discharge = (~group["on_grid"]).astype(int)
        transitions = is_discharge.diff().fillna(0)

        in_event = False
        event_rows: list[pd.Series] = []

        for idx, row in group.iterrows():
            if not in_event and not row["on_grid"]:
                in_event = True
                event_rows = [row]
            elif in_event and not row["on_grid"]:
                event_rows.append(row)
            elif in_event and row["on_grid"]:
                in_event = False
                if len(event_rows) >= 2:
                    start_row = event_rows[0]
                    end_row = event_rows[-1]
                    duration = end_row["timestamp_h"] - start_row["timestamp_h"]
                    if duration > 0:
                        dv_dt = (end_row["voltage"] - start_row["voltage"]) / duration
                        events.append({
                            "site_id": site_id,
                            "start_h": start_row["timestamp_h"],
                            "end_h": end_row["timestamp_h"],
                            "duration_h": duration,
                            "voltage_start": start_row["voltage"],
                            "voltage_end": end_row["voltage"],
                            "mean_current": np.mean([r["current"] for r in event_rows]),
                            "dv_dt": dv_dt,
                        })

    return pd.DataFrame(events) if events else pd.DataFrame(
        columns=["site_id", "start_h", "end_h", "duration_h",
                 "voltage_start", "voltage_end", "mean_current", "dv_dt"]
    )


# ---------------------------------------------------------------------------
# Family 1: Per-Discharge Features
# ---------------------------------------------------------------------------


@register_feature("discharge_slope_mean", family="per_discharge",
                  description="Mean dV/dt across discharge events (last 30d)")
def compute_discharge_slope_mean(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Mean voltage slope during discharge events in the last 30-day window.

    This is the primary capacity/resistance probe. Steeper negative slope
    = higher internal resistance = more degraded battery.

    TODO (USER): implement the core dV/dt calculation.
    The scaffolding extracts events and windows them — you compute the slope.
    """
    raise NotImplementedError("USER: compute discharge slope mean")


@register_feature("discharge_depth_mean", family="per_discharge",
                  description="Mean depth of discharge (voltage drop) in last 30d")
def compute_discharge_depth_mean(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Mean voltage drop across discharge events.

    TODO (USER): implement.
    """
    raise NotImplementedError("USER: compute discharge depth mean")


@register_feature("days_since_last_discharge", family="per_discharge",
                  description="Staleness indicator for discharge features")
def compute_days_since_last_discharge(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Days since last discharge event per site. Staleness companion feature.

    TODO (USER): implement.
    """
    raise NotImplementedError("USER: days since last discharge")


# ---------------------------------------------------------------------------
# Family 2: Per-Recharge Features
# ---------------------------------------------------------------------------


@register_feature("recharge_time_to_float_mean", family="per_recharge",
                  description="Mean time to reach float voltage after grid return")
def compute_recharge_time_to_float(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Mean hours from grid-return to reaching float voltage.

    Degraded batteries take longer due to suppressed charge acceptance.

    TODO (USER): implement.
    """
    raise NotImplementedError("USER: recharge time to float")


# ---------------------------------------------------------------------------
# Family 3: Rolling Cumulative Features
# ---------------------------------------------------------------------------


@register_feature("arrhenius_weighted_age", family="rolling_cumulative",
                  description="Cumulative thermal stress (Arrhenius-integrated hours)")
def compute_arrhenius_weighted_age(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Arrhenius-weighted cumulative age per site.

    A(t) = Σ temperature_acceleration_factor(T_i) × dt

    A 2-year battery at 45°C has higher Arrhenius age than a 5-year battery
    at 20°C. This captures nonlinear temperature-aging interaction that raw
    mean_temperature would miss.

    TODO (USER): implement using temperature_acceleration_factor from physics.
    """
    raise NotImplementedError("USER: arrhenius weighted age")


@register_feature("psoc_fraction_30d", family="rolling_cumulative",
                  description="Fraction of last 30 days spent below float (PSoC)")
def compute_psoc_fraction_30d(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Fraction of time in last 30 days where SoC < 0.95.

    The single best 'exposure' feature. Batteries chronically in PSoC
    accumulate sulfation regardless of temperature.

    TODO (USER): implement.
    """
    raise NotImplementedError("USER: psoc fraction 30d")


@register_feature("coulomb_throughput_30d", family="rolling_cumulative",
                  description="Total |Ah| cycled in last 30 days")
def compute_coulomb_throughput_30d(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Total absolute current * time in last 30 days. Cycle aging proxy.

    TODO (USER): implement.
    """
    raise NotImplementedError("USER: coulomb throughput 30d")


@register_feature("lvd_count_30d", family="rolling_cumulative",
                  description="Number of LVD trips in last 30 days")
def compute_lvd_count_30d(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Count of low-voltage-disconnect events in last 30 days.

    LVD threshold: voltage < V_LVD_PER_CELL * n_cells = 1.75 * 24 = 42V.

    TODO (USER): implement.
    """
    raise NotImplementedError("USER: LVD count 30d")


@register_feature("deep_discharge_count_30d", family="rolling_cumulative",
                  description="Number of deep discharge events (DoD>50%) in last 30d")
def compute_deep_discharge_count_30d(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Count of discharge events where voltage dropped below 46V in last 30d.

    TODO (USER): implement.
    """
    raise NotImplementedError("USER: deep discharge count 30d")


# ---------------------------------------------------------------------------
# Family 4: Inter-Event Features
# ---------------------------------------------------------------------------


@register_feature("mean_time_between_outages", family="inter_event",
                  description="Mean hours between grid outages (MTBO)")
def compute_mtbo(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Mean time between outage starts. Decreasing MTBO = grid deteriorating.

    TODO (USER): implement.
    """
    raise NotImplementedError("USER: MTBO")


@register_feature("time_since_full_recharge", family="inter_event",
                  description="Hours since SoC last reached >0.95")
def compute_time_since_full_recharge(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Hours since battery last reached full charge. Longer = more sulfation risk.

    TODO (USER): implement.
    """
    raise NotImplementedError("USER: time since full recharge")


# ---------------------------------------------------------------------------
# Family 5: Trend Features
# ---------------------------------------------------------------------------


@register_feature("trend_discharge_slope_30d", family="trend",
                  description="Slope of dV/dt trend over last 30 days (Theil-Sen)")
def compute_trend_discharge_slope(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Trend in discharge voltage slope over 30-day window.

    Uses Theil-Sen estimator (median of pairwise slopes) for robustness.
    Negative trend = discharge behavior is worsening = degradation accelerating.

    TODO (USER): implement.
    """
    raise NotImplementedError("USER: trend discharge slope")


@register_feature("trend_float_voltage_30d", family="trend",
                  description="Slope of float voltage over last 30 days")
def compute_trend_float_voltage(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Trend in float voltage over 30-day window.

    Declining float voltage = sulfation / micro-shorts developing.

    TODO (USER): implement.
    """
    raise NotImplementedError("USER: trend float voltage")


# ---------------------------------------------------------------------------
# Static / metadata features (scaffolded — no TODO, these are trivial)
# ---------------------------------------------------------------------------


@register_feature("battery_age_months", family="static",
                  description="Age in months at observation time")
def compute_battery_age(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    site_ids = labels["site_id"]
    max_h = telemetry.groupby("site_id")["timestamp_h"].max()
    min_h = telemetry.groupby("site_id")["timestamp_h"].min()
    age_months = (max_h - min_h) / (30 * 24)
    return age_months.reindex(site_ids).fillna(0)


@register_feature("region_encoded", family="static",
                  description="Region as integer code")
def compute_region_encoded(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    return labels.set_index("site_id")["region"].astype("category").cat.codes


@register_feature("manufacturer_encoded", family="static",
                  description="Manufacturer as integer code")
def compute_manufacturer_encoded(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    return labels.set_index("site_id")["manufacturer"].astype("category").cat.codes
