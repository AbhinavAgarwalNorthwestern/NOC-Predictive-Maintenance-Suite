"""Fleet-level battery simulator.

Generates synthetic telemetry, alarms, and labels for a fleet of batteries
across multiple sites and regions. Calls the physics module at each timestep.

Output tables:
- telemetry: (site_id, timestamp_h, voltage, current, temperature_c, on_grid)
- alarms: (site_id, timestamp_h, alarm_code, severity)
- labels: (site_id, time_to_event_months, event, event_hour, label_source, region, manufacturer, install_month)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from battery_pdm.synth.config import (
    Manufacturer,
    Region,
    DEFAULT_CLIMATE,
    DEFAULT_LOAD_SHED,
)
from battery_pdm.synth.physics import (
    CellState,
    charge_acceptance_current,
    coulomb_counting_step,
    float_voltage,
    shepherd_discharge_voltage,
    update_health,
)


FAILURE_HEALTH_THRESHOLD = 0.65
PREVENTIVE_REPLACEMENT_THRESHOLD = 0.70
HOURS_PER_STEP = 1.0
N_CELLS = 24
NOMINAL_CAPACITY_AH = 200.0
CHARGER_CURRENT_A = 15.0
SENSOR_NOISE_V = 1.2  # realistic: cheap probes on 48V bus, ~2.5% relative noise
SENSOR_NOISE_TEMP_C = 3.0  # common for telecom outdoor enclosures
SENSOR_DRIFT_V_PER_MONTH = 0.4  # probe corrosion, wiring degradation
TELEMETRY_GAP_PROB = 0.08  # 8% gap rate — comms outages, controller resets
SAMPLE_INTERVAL_H = 12
REPLACEMENT_DELAY_H = 7 * 24  # crew arrives ~1 week after event

# Alarm codes — threshold-based alarms (coherent with telemetry)
# Plus non-observable signals that justify ML over pure physics
ALARM_UNDERVOLTAGE_V = 44.0
ALARM_HIGH_TEMP_C = 40.0
ALARM_LVD_V = 42.0

# Non-observable: rectifier fault (grid on, but charger dead — invisible in V/I for hours)
RECTIFIER_FAULT_BASE_PROB = 0.0003  # per hour
RECTIFIER_FAULT_DURATION = (24, 120)  # hours
RECTIFIER_FAULT_AGING_MULT = 1.05  # subtle — damage accumulates over months, not days

# Non-observable: BMS cell imbalance (bank voltage masks weak cell)
CELL_IMBALANCE_PROB = 0.0008  # per hour when sulfation > 0.3
CELL_IMBALANCE_AGING_MULT = 1.03  # subtle

# Non-observable: site-level issues (installation quality, repeat failures)
CHARGER_MISCONFIGURED_PROB = 0.08  # per site at creation

# Sudden failure: accumulated rectifier faults weaken plates internally.
# After enough faults, next fault can trigger plate short / thermal event.
# Telemetry looks NORMAL right up until catastrophe — only alarm history predicts it.
SUDDEN_FAILURE_RECTIFIER_THRESHOLD = 6  # cumulative faults before risk kicks in
SUDDEN_FAILURE_PROB_PER_FAULT = (
    0.008  # probability per additional fault beyond threshold
)


@dataclass
class OutageState:
    """Track outage duration to model extended load-shedding."""

    in_outage: bool = False
    remaining_hours: float = 0.0


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
    aging_multiplier: float = 1.0  # manufacturing quality — NOT observable in telemetry
    rectifier_faulty: bool = False
    rectifier_fault_hour: int = -1
    charger_misconfigured: bool = False  # permanent site defect — accelerates aging
    replacement_count: int = 0  # how many batteries this site has burned through
    lifecycle_start_h: int = 0  # hour when current battery was installed
    cumulative_rectifier_faults: int = (
        0  # lifetime fault count (persists across replacements)
    )

    def __post_init__(self):
        if self.outage_state is None:
            self.outage_state = OutageState()


def _ambient_temperature(region: Region, hour_of_year: float) -> float:
    """Seasonal + diurnal temperature model for Pakistan regions."""
    climate = DEFAULT_CLIMATE[region]
    day_of_year = (hour_of_year % 8760) / 24.0
    hour_of_day = hour_of_year % 24

    seasonal = climate.annual_mean_temp_c + climate.annual_amplitude_c * np.sin(
        2 * np.pi * (day_of_year - climate.summer_peak_doy + 90) / 365.0
    )
    diurnal = (
        climate.daily_amplitude_c / 2.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)
    )
    return float(seasonal + diurnal)


def _generate_outage_duration(region: Region, rng: np.random.Generator) -> float:
    """Sample outage duration. Pakistan worst-case: 4-16 hours."""
    profile = DEFAULT_LOAD_SHED[region]
    mean_hours = profile.avg_outage_duration_min / 60.0
    duration = rng.lognormal(np.log(mean_hours), 0.5)
    return min(duration, 20.0)


def _should_start_outage(
    region: Region, hour_of_year: float, rng: np.random.Generator
) -> bool:
    """Probability of a NEW outage starting this hour."""
    profile = DEFAULT_LOAD_SHED[region]
    day_of_year = (hour_of_year % 8760) / 24.0
    is_summer = 120 < day_of_year < 270
    amplification = profile.summer_amplification if is_summer else 1.0
    prob_per_hour = profile.avg_outages_per_day * amplification / 24.0
    return bool(rng.random() < prob_per_hour)


def _site_load_current(region: Region, rng: np.random.Generator) -> float:
    """Typical discharge current when on battery (site-dependent)."""
    return rng.uniform(8.0, 25.0)


def simulate_fleet(
    n_sites: int = 500,
    n_months: int = 36,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Run fleet simulation and return a dict of DataFrames.

    Keys: 'telemetry', 'alarms', 'labels', 'site_static'.
    Returning a dict (instead of a tuple) keeps the API extensible — new
    data streams (e.g., maintenance_tickets, weather) can be added without
    breaking callers that destructure by position.

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
        Columns: site_id, timestamp_h, voltage, current, temperature_c, on_grid
    alarms_df : pd.DataFrame
        Columns: site_id, timestamp_h, alarm_code, severity
    labels_df : pd.DataFrame
        Columns: site_id, time_to_event_months, event (1=failure, 0=censored),
                 event_hour, label_source, region, manufacturer, install_month
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

        # Manufacturing quality: log-normal so most are ~1x but some are 2-3x worse
        # NOT observable in voltage/current/temperature — only manifests as faster aging
        # Sites with worse quality also trigger more cell imbalance alarms
        aging_mult = float(rng.lognormal(mean=0.0, sigma=0.4))
        aging_mult = max(0.5, min(aging_mult, 3.5))  # clip extremes

        sites.append(
            SiteSimState(
                site_id=f"SITE_{i:04d}",
                region=region,
                manufacturer=mfr,
                install_month=install_month,
                cell_state=initial_state,
                on_grid=True,
                hours_since_install=0.0,
                discharge_current_a=discharge_current,
                aging_multiplier=aging_mult,
                charger_misconfigured=bool(rng.random() < CHARGER_MISCONFIGURED_PROB),
                lifecycle_start_h=install_month * 30 * 24,
            )
        )

    telemetry_records: list[dict] = []
    alarm_records: list[dict] = []
    labels: list[dict] = []
    failed_sites: set[str] = set()
    pending_replacement: dict[str, int] = {}  # site_id -> hour replacement completes
    labeled_sites: set[str] = set()  # prevent duplicate labels
    # Track active alarms to only fire on transitions (onset, not continuously)
    active_alarms: dict[str, set[str]] = {}  # site_id -> set of active alarm codes

    for hour in range(0, total_hours, int(HOURS_PER_STEP)):
        hour_of_year = hour % 8760

        for site in sites:
            # Site is dark during replacement swap
            if site.site_id in pending_replacement:
                if hour >= pending_replacement[site.site_id]:
                    del pending_replacement[site.site_id]
                    failed_sites.discard(site.site_id)
                    labeled_sites.discard(site.site_id)
                    site.cell_state = CellState.fresh()
                    site.hours_since_install = 0.0
                    site.lifecycle_start_h = hour
                    site.replacement_count += 1
                    if site.replacement_count >= 1:
                        alarm_records.append(
                            {
                                "site_id": site.site_id,
                                "timestamp_h": hour,
                                "alarm_code": "REPEAT_FAILURE_FLAG",
                                "severity": "critical",
                            }
                        )
                else:
                    continue

            start_hour = site.install_month * 30 * 24
            if hour < start_hour:
                continue

            site.hours_since_install += HOURS_PER_STEP
            temp_c = _ambient_temperature(site.region, hour_of_year)
            temp_c += rng.normal(
                0, 0.5
            )  # real temperature variation (physics sees this)

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
                    site.outage_state.remaining_hours = _generate_outage_duration(
                        site.region, rng
                    )
                    site.on_grid = False
                else:
                    site.on_grid = True

            # --- Rectifier fault: invisible in telemetry for hours ---
            # Grid is on, voltage looks OK briefly, but charger is dead
            if not site.rectifier_faulty:
                fault_prob = RECTIFIER_FAULT_BASE_PROB
                fault_prob *= 1 + site.hours_since_install / 20000
                if site.charger_misconfigured:
                    fault_prob *= 3.0  # misconfigured sites fail more
                if rng.random() < fault_prob:
                    site.rectifier_faulty = True
                    site.rectifier_fault_hour = hour
                    site.cumulative_rectifier_faults += 1
                    alarm_records.append(
                        {
                            "site_id": site.site_id,
                            "timestamp_h": hour,
                            "alarm_code": "RECTIFIER_FAULT",
                            "severity": "critical",
                        }
                    )
                    # Sudden failure: plate short from accumulated charge abuse
                    # Health is NOT degraded — telemetry looks normal up until this moment
                    excess = (
                        site.cumulative_rectifier_faults
                        - SUDDEN_FAILURE_RECTIFIER_THRESHOLD
                    )
                    if (
                        excess > 0
                        and site.site_id not in failed_sites
                        and site.site_id not in labeled_sites
                        and rng.random() < SUDDEN_FAILURE_PROB_PER_FAULT * excess
                    ):
                        time_months = site.hours_since_install / (30 * 24)
                        labels.append(
                            {
                                "site_id": site.site_id,
                                "time_to_event_months": round(time_months, 2),
                                "event": 1,
                                "event_hour": hour,
                                "label_source": "sudden_plate_short",
                                "region": site.region.value,
                                "manufacturer": site.manufacturer.value,
                                "install_month": site.install_month,
                                "lifecycle_id": site.replacement_count,
                                "lifecycle_start_h": site.lifecycle_start_h,
                            }
                        )
                        failed_sites.add(site.site_id)
                        labeled_sites.add(site.site_id)
                        pending_replacement[site.site_id] = hour + REPLACEMENT_DELAY_H
            elif hour - site.rectifier_fault_hour > rng.integers(
                *RECTIFIER_FAULT_DURATION
            ):
                site.rectifier_faulty = False

            # --- BMS cell imbalance: bank voltage hides weak cell ---
            # Bad manufacturing quality → cells diverge faster → more imbalance alarms
            if (
                site.cell_state.sulfation_index > 0.3
                and rng.random() < CELL_IMBALANCE_PROB * site.aging_multiplier
            ):
                alarm_records.append(
                    {
                        "site_id": site.site_id,
                        "timestamp_h": hour,
                        "alarm_code": "CELL_IMBALANCE",
                        "severity": "major",
                    }
                )

            # Charging logic (rectifier fault prevents charging)
            if site.on_grid and not site.rectifier_faulty:
                charger_a = CHARGER_CURRENT_A
                if site.charger_misconfigured:
                    charger_a *= 0.7  # wrong settings → undercharging
                accepted = charge_acceptance_current(
                    site.cell_state, charger_a, NOMINAL_CAPACITY_AH
                )
                current_a = -accepted
            elif site.on_grid and site.rectifier_faulty:
                current_a = 0.0
            else:
                if site.cell_state.soc <= 0.05:
                    current_a = 0.0
                else:
                    current_a = site.discharge_current_a

            site.cell_state = coulomb_counting_step(
                site.cell_state, current_a, HOURS_PER_STEP, NOMINAL_CAPACITY_AH
            )
            # Accelerated aging from non-observable conditions
            aging_dt = HOURS_PER_STEP * site.aging_multiplier
            if site.rectifier_faulty:
                aging_dt *= RECTIFIER_FAULT_AGING_MULT
            if site.cell_state.sulfation_index > 0.3:
                aging_dt *= CELL_IMBALANCE_AGING_MULT
            site.cell_state = update_health(
                site.cell_state, aging_dt, temp_c, current_a, NOMINAL_CAPACITY_AH
            )

            if site.on_grid:
                voltage = float_voltage(temp_c, site.cell_state, N_CELLS)
            else:
                voltage = shepherd_discharge_voltage(
                    site.cell_state, current_a, NOMINAL_CAPACITY_AH, N_CELLS
                )

            if hour % SAMPLE_INTERVAL_H == 0:
                # Sensor noise + systematic drift (degrades signal quality over time)
                months_active = site.hours_since_install / (30 * 24)
                sensor_drift = (
                    SENSOR_DRIFT_V_PER_MONTH * months_active * rng.choice([-1, 1])
                )
                recorded_v = (
                    voltage + rng.normal(0, SENSOR_NOISE_V) + sensor_drift * 0.1
                )
                recorded_t = temp_c + rng.normal(0, SENSOR_NOISE_TEMP_C)

                # Telemetry gaps (comms failure, controller reset, power issue)
                if rng.random() > TELEMETRY_GAP_PROB:
                    telemetry_records.append(
                        {
                            "site_id": site.site_id,
                            "timestamp_h": hour,
                            "voltage": round(recorded_v, 3),
                            "current": round(current_a, 3),
                            "temperature_c": round(recorded_t, 2),
                            "on_grid": site.on_grid,
                        }
                    )

            # --- Alarm generation (transition-based, on sample interval) ---
            site_alarms = active_alarms.setdefault(site.site_id, set())

            # Track state transitions every hour (for onset detection)
            uv_active = voltage < ALARM_UNDERVOLTAGE_V
            ht_active = temp_c > ALARM_HIGH_TEMP_C
            lvd_active = voltage < ALARM_LVD_V

            # Only record alarms at sample interval (realistic NOC polling)
            if hour % SAMPLE_INTERVAL_H == 0:
                if not site.on_grid and was_on_grid:
                    alarm_records.append(
                        {
                            "site_id": site.site_id,
                            "timestamp_h": hour,
                            "alarm_code": "AC_MAINS_FAIL",
                            "severity": "minor",
                        }
                    )

                if uv_active and "BATT_UNDERVOLTAGE" not in site_alarms:
                    alarm_records.append(
                        {
                            "site_id": site.site_id,
                            "timestamp_h": hour,
                            "alarm_code": "BATT_UNDERVOLTAGE",
                            "severity": "critical",
                        }
                    )

                if ht_active and "BATT_HIGH_TEMP" not in site_alarms:
                    alarm_records.append(
                        {
                            "site_id": site.site_id,
                            "timestamp_h": hour,
                            "alarm_code": "BATT_HIGH_TEMP",
                            "severity": "major",
                        }
                    )

                if lvd_active and "LOAD_DISCONNECT" not in site_alarms:
                    alarm_records.append(
                        {
                            "site_id": site.site_id,
                            "timestamp_h": hour,
                            "alarm_code": "LOAD_DISCONNECT",
                            "severity": "critical",
                        }
                    )

                # Non-observable: tech notices swollen cells
                if site.cell_state.health < 0.78 and rng.random() < 0.001:
                    alarm_records.append(
                        {
                            "site_id": site.site_id,
                            "timestamp_h": hour,
                            "alarm_code": "TICKET_SWOLLEN_CELLS",
                            "severity": "major",
                        }
                    )

                # Non-observable: tech flags charger misconfiguration
                if site.charger_misconfigured and rng.random() < 0.0005:
                    alarm_records.append(
                        {
                            "site_id": site.site_id,
                            "timestamp_h": hour,
                            "alarm_code": "TICKET_CHARGER_MISCONFIGURED",
                            "severity": "major",
                        }
                    )

            # Update active alarm state
            if uv_active:
                site_alarms.add("BATT_UNDERVOLTAGE")
            else:
                site_alarms.discard("BATT_UNDERVOLTAGE")
            if ht_active:
                site_alarms.add("BATT_HIGH_TEMP")
            else:
                site_alarms.discard("BATT_HIGH_TEMP")
            if lvd_active:
                site_alarms.add("LOAD_DISCONNECT")
            else:
                site_alarms.discard("LOAD_DISCONNECT")

            # Record failure label once, telemetry continues until replacement
            if (
                site.cell_state.health <= FAILURE_HEALTH_THRESHOLD
                and site.site_id not in failed_sites
                and site.site_id not in labeled_sites
            ):
                time_months = site.hours_since_install / (30 * 24)
                labels.append(
                    {
                        "site_id": site.site_id,
                        "time_to_event_months": round(time_months, 2),
                        "event": 1,
                        "event_hour": hour,
                        "label_source": "observed_failure",
                        "region": site.region.value,
                        "manufacturer": site.manufacturer.value,
                        "install_month": site.install_month,
                        "lifecycle_id": site.replacement_count,
                        "lifecycle_start_h": site.lifecycle_start_h,
                    }
                )
                failed_sites.add(site.site_id)
                labeled_sites.add(site.site_id)
                # Schedule replacement after delay
                pending_replacement[site.site_id] = hour + REPLACEMENT_DELAY_H

            elif (
                site.cell_state.health <= PREVENTIVE_REPLACEMENT_THRESHOLD
                and site.site_id not in failed_sites
                and site.site_id not in labeled_sites
                and site.site_id not in pending_replacement
                and rng.random() < 0.0005
            ):
                time_months = site.hours_since_install / (30 * 24)
                labels.append(
                    {
                        "site_id": site.site_id,
                        "time_to_event_months": round(time_months, 2),
                        "event": 1,
                        "event_hour": hour,
                        "label_source": "preventive_replacement",
                        "region": site.region.value,
                        "manufacturer": site.manufacturer.value,
                        "install_month": site.install_month,
                        "lifecycle_id": site.replacement_count,
                        "lifecycle_start_h": site.lifecycle_start_h,
                    }
                )
                labeled_sites.add(site.site_id)
                pending_replacement[site.site_id] = hour + REPLACEMENT_DELAY_H

    for site in sites:
        if site.site_id not in labeled_sites:
            time_months = site.hours_since_install / (30 * 24)
            if time_months > 0:
                labels.append(
                    {
                        "site_id": site.site_id,
                        "time_to_event_months": round(time_months, 2),
                        "event": 0,
                        "event_hour": total_hours,
                        "label_source": "admin_censored",
                        "region": site.region.value,
                        "manufacturer": site.manufacturer.value,
                        "install_month": site.install_month,
                        "lifecycle_id": site.replacement_count,
                        "lifecycle_start_h": site.lifecycle_start_h,
                    }
                )

    telemetry_df = pd.DataFrame(telemetry_records)
    alarms_df = (
        pd.DataFrame(alarm_records)
        if alarm_records
        else pd.DataFrame(columns=["site_id", "timestamp_h", "alarm_code", "severity"])
    )
    labels_df = pd.DataFrame(labels)

    site_static_df = pd.DataFrame(
        [
            {
                "site_id": s.site_id,
                "region": s.region.value,
                "manufacturer": s.manufacturer.value,
                "install_month": s.install_month,
                "load_A": s.discharge_current_a,
                "n_cells": N_CELLS,
                "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
                "charger_misconfigured": int(s.charger_misconfigured),
                "aging_multiplier": s.aging_multiplier,
            }
            for s in sites
        ]
    )

    return {
        "telemetry": telemetry_df,
        "alarms": alarms_df,
        "labels": labels_df,
        "site_static": site_static_df,
    }


if __name__ == "__main__":
    print("Simulating fleet...")
    data = simulate_fleet(n_sites=500, n_months=36, seed=42)
    telem, alarms, labels, site_static = (
        data["telemetry"],
        data["alarms"],
        data["labels"],
        data["site_static"],
    )
    print(f"Telemetry: {len(telem):,} rows, {telem['site_id'].nunique()} sites")
    print(f"Alarms: {len(alarms):,} rows")
    if not alarms.empty:
        print(f"  Alarm breakdown: {alarms['alarm_code'].value_counts().to_dict()}")
    print(f"Labels: {len(labels)} rows")
    print(f"  Failures: {(labels['event'] == 1).sum()}")
    print(f"  Censored: {(labels['event'] == 0).sum()}")
    print(f"  Label sources: {labels['label_source'].value_counts().to_dict()}")
    print(f"Site static: {len(site_static)} rows")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    telem.to_parquet(output_dir / "telemetry.parquet", index=False)
    alarms.to_parquet(output_dir / "alarms.parquet", index=False)
    labels.to_parquet(output_dir / "labels.parquet", index=False)
    site_static.to_parquet(output_dir / "site_static.parquet", index=False)
    print(f"\nSaved to {output_dir}/")
