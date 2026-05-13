"""Fleet-level battery simulator.

Generates synthetic telemetry and labels for a fleet of batteries across
multiple sites and regions. Calls the physics module at each timestep.

Output tables:
- telemetry: (site_id, timestamp, voltage, current, temperature, soc, health)
- labels: (site_id, time_to_event_months, event, label_source, cohort)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from battery_pdm.synth.config import (
    BatteryBank,
    ClimateProfile,
    LoadShedProfile,
    Manufacturer,
    Region,
    Site,
    SimulationConfig,
    DEFAULT_CLIMATE,
    DEFAULT_LOAD_SHED,
)
from battery_pdm.synth.physics import (
    CellState,
    PSOC_THRESHOLD_SOC,
    V_LVD_PER_CELL,
    charge_acceptance_current,
    coulomb_counting_step,
    float_voltage,
    shepherd_discharge_voltage,
    temperature_acceleration_factor,
    update_health,
)


FAILURE_HEALTH_THRESHOLD = 0.65
PREVENTIVE_REPLACEMENT_THRESHOLD = 0.70
BASE_DECAY_PER_YEAR = 0.05
HOURS_PER_STEP = 1.0
N_CELLS = 24
NOMINAL_CAPACITY_AH = 200.0
CHARGER_CURRENT_A = 15.0
SENSOR_NOISE_V = 0.1
SENSOR_NOISE_TEMP_C = 0.5


@dataclass
class SiteSimState:
    site_id: str
    region: Region
    manufacturer: Manufacturer
    install_month: int
    cell_state: CellState
    on_grid: bool
    hours_since_install: float
    discharge_current_a: float
    outage_state: OutageState = None

    def __post_init__(self):
        if self.outage_state is None:
            self.outage_state = OutageState()


def _ambient_temperature(region: Region, hour_of_year: float) -> float:
    """Seasonal + diurnal temperature model for Pakistan regions."""
    climate = DEFAULT_CLIMATE[region]
    day_of_year = (hour_of_year % 8760) / 24.0
    hour_of_day = hour_of_year % 24

    seasonal = climate.annual_mean_temp_c + \
               climate.annual_amplitude_c * \
               np.sin(2 * np.pi * (day_of_year - climate.summer_peak_doy + 90) / 365.0)
    diurnal = climate.daily_amplitude_c / 2.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)
    return float(seasonal + diurnal)


@dataclass
class OutageState:
    """Track outage duration to model extended load-shedding."""
    in_outage: bool = False
    remaining_hours: float = 0.0


def _generate_outage_duration(region: Region, rng: np.random.Generator) -> float:
    """Sample outage duration. Pakistan worst-case: 4-16 hours."""
    profile = DEFAULT_LOAD_SHED[region]
    mean_hours = profile.avg_outage_duration_min / 60.0
    duration = rng.lognormal(np.log(mean_hours), 0.5)
    return min(duration, 20.0)


def _should_start_outage(region: Region, hour_of_year: float, rng: np.random.Generator) -> bool:
    """Probability of a NEW outage starting this hour."""
    profile = DEFAULT_LOAD_SHED[region]
    day_of_year = (hour_of_year % 8760) / 24.0
    is_summer = 120 < day_of_year < 270
    amplification = profile.summer_amplification if is_summer else 1.0
    prob_per_hour = profile.avg_outages_per_day * amplification / 24.0
    return bool(rng.random() < prob_per_hour)


def _site_load_current(region: Region, rng: np.random.Generator) -> float:
    """Typical discharge current when on battery (site-dependent)."""
    base = rng.uniform(8.0, 25.0)
    return base


def simulate_fleet(
    n_sites: int = 500,
    n_months: int = 36,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fleet simulation and return (telemetry_df, labels_df).

    Parameters
    ----------
    n_sites : int
        Number of battery sites to simulate.
    n_months : int
        Observation window in months.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    telemetry_df : pd.DataFrame
        Columns: site_id, timestamp_h, voltage, current, temperature_c,
                 soc, health, on_grid
    labels_df : pd.DataFrame
        Columns: site_id, time_to_event_months, event (1=failure, 0=censored),
                 label_source, region, manufacturer, install_month
    """
    rng = np.random.default_rng(seed)
    total_hours = n_months * 30 * 24

    regions = list(Region)
    manufacturers = list(Manufacturer)

    sites: list[SiteSimState] = []
    for i in range(n_sites):
        region = regions[i % len(regions)]
        mfr = manufacturers[i % len(manufacturers)]
        install_month = int(rng.integers(0, max(1, n_months // 3)))
        discharge_current = _site_load_current(region, rng)

        pre_age_years = float(rng.uniform(0, 3))
        initial_health = max(0.88, 1.0 - pre_age_years * rng.uniform(0.02, 0.04))
        initial_sulfation = pre_age_years * rng.uniform(0.05, 0.2)
        initial_throughput = pre_age_years * rng.uniform(300, 1000)

        initial_state = CellState(
            soc=rng.uniform(0.7, 1.0),
            health=initial_health,
            cumulative_throughput_ah=initial_throughput,
            time_in_psoc_hours=pre_age_years * rng.uniform(500, 3000),
            arrhenius_age_factor=pre_age_years * rng.uniform(5000, 15000),
            sulfation_index=initial_sulfation,
        )

        sites.append(SiteSimState(
            site_id=f"SITE_{i:04d}",
            region=region,
            manufacturer=mfr,
            install_month=install_month,
            cell_state=initial_state,
            on_grid=True,
            hours_since_install=0.0,
            discharge_current_a=discharge_current,
        ))

    telemetry_records: list[dict] = []
    labels: list[dict] = []
    failed_sites: set[str] = set()
    replaced_sites: set[str] = set()

    sample_interval = 12

    for hour in range(0, total_hours, int(HOURS_PER_STEP)):
        month = hour // (30 * 24)
        hour_of_day = hour % 24
        hour_of_year = hour % 8760

        for site in sites:
            if site.site_id in failed_sites or site.site_id in replaced_sites:
                continue

            start_hour = site.install_month * 30 * 24
            if hour < start_hour:
                continue

            site.hours_since_install += HOURS_PER_STEP
            temp_c = _ambient_temperature(site.region, hour_of_year)
            temp_c += rng.normal(0, SENSOR_NOISE_TEMP_C)

            was_on_grid = site.on_grid
            if site.outage_state.in_outage:
                site.outage_state.remaining_hours -= HOURS_PER_STEP
                if site.outage_state.remaining_hours <= 0:
                    site.outage_state.in_outage = False
                    site.on_grid = True
                else:
                    site.on_grid = False
            else:
                if _should_start_outage(site.region, hour_of_year, rng):
                    site.outage_state.in_outage = True
                    site.outage_state.remaining_hours = _generate_outage_duration(site.region, rng)
                    site.on_grid = False
                else:
                    site.on_grid = True

            if site.on_grid:
                accepted = charge_acceptance_current(
                    site.cell_state, CHARGER_CURRENT_A, NOMINAL_CAPACITY_AH
                )
                current_a = -accepted
            else:
                if site.cell_state.soc <= 0.05:
                    current_a = 0.0
                else:
                    current_a = site.discharge_current_a

            site.cell_state = coulomb_counting_step(
                site.cell_state, current_a, HOURS_PER_STEP, NOMINAL_CAPACITY_AH
            )
            site.cell_state = update_health(
                site.cell_state, HOURS_PER_STEP, temp_c, current_a, NOMINAL_CAPACITY_AH
            )

            if site.on_grid:
                voltage = float_voltage(temp_c, site.cell_state, N_CELLS)
            else:
                voltage = shepherd_discharge_voltage(
                    site.cell_state, current_a, NOMINAL_CAPACITY_AH, N_CELLS
                )

            voltage += rng.normal(0, SENSOR_NOISE_V)

            if hour % sample_interval == 0:
                telemetry_records.append({
                    "site_id": site.site_id,
                    "timestamp_h": hour,
                    "voltage": round(voltage, 3),
                    "current": round(current_a, 3),
                    "temperature_c": round(temp_c, 2),
                    "soc": round(site.cell_state.soc, 4),
                    "health": round(site.cell_state.health, 4),
                    "on_grid": site.on_grid,
                })

            if site.cell_state.health <= FAILURE_HEALTH_THRESHOLD:
                time_months = site.hours_since_install / (30 * 24)
                labels.append({
                    "site_id": site.site_id,
                    "time_to_event_months": round(time_months, 2),
                    "event": 1,
                    "label_source": "observed_failure",
                    "region": site.region.value,
                    "manufacturer": site.manufacturer.value,
                    "install_month": site.install_month,
                })
                failed_sites.add(site.site_id)

            elif (site.cell_state.health <= PREVENTIVE_REPLACEMENT_THRESHOLD
                  and rng.random() < 0.0005):
                time_months = site.hours_since_install / (30 * 24)
                labels.append({
                    "site_id": site.site_id,
                    "time_to_event_months": round(time_months, 2),
                    "event": 1,
                    "label_source": "preventive_replacement",
                    "region": site.region.value,
                    "manufacturer": site.manufacturer.value,
                    "install_month": site.install_month,
                })
                replaced_sites.add(site.site_id)

    for site in sites:
        if site.site_id not in failed_sites and site.site_id not in replaced_sites:
            time_months = site.hours_since_install / (30 * 24)
            if time_months > 0:
                labels.append({
                    "site_id": site.site_id,
                    "time_to_event_months": round(time_months, 2),
                    "event": 0,
                    "label_source": "admin_censored",
                    "region": site.region.value,
                    "manufacturer": site.manufacturer.value,
                    "install_month": site.install_month,
                })

    telemetry_df = pd.DataFrame(telemetry_records)
    labels_df = pd.DataFrame(labels)

    return telemetry_df, labels_df


if __name__ == "__main__":
    print("Simulating fleet...")
    telem, labels = simulate_fleet(n_sites=100, n_months=36, seed=42)
    print(f"Telemetry: {len(telem):,} rows, {telem['site_id'].nunique()} sites")
    print(f"Labels: {len(labels)} rows")
    print(f"  Failures: {(labels['event']==1).sum()}")
    print(f"  Censored: {(labels['event']==0).sum()}")
    print(f"  Label sources: {labels['label_source'].value_counts().to_dict()}")
    print(f"\nSample telemetry:\n{telem.head(10)}")
    print(f"\nSample labels:\n{labels.head(10)}")
