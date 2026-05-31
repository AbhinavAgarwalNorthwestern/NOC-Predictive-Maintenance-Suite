"""Configuration types for the synthetic data generator.

Defines the data structures that parameterize a simulation run: sites,
manufacturers, regions, climate, load-shed regimes, and global settings.

Intent: the runner consumes these. A YAML/JSON config or a Python script
populates them. Defaults at the bottom provide sensible Pakistan-region
profiles as a starting point — they are not gospel, tune them once you
see the simulation output.

This file is scaffold-complete. No TODOs here; you will tune defaults
once the generator runs end-to-end and you see whether the produced
failure rates and feature distributions look right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Manufacturer(str, Enum):
    """Battery manufacturers. Each gets its own aging behavior in physics.py.

    The 'manufacturer effect' is one of the cohort axes for stratified drift
    detection, so making this a first-class field matters for downstream CV
    and monitoring code.
    """

    EXIDE = "exide"
    AMARA_RAJA = "amara_raja"
    HBL = "hbl"
    UNKNOWN = "unknown"


class Region(str, Enum):
    """Operating regions, each with characteristic climate and load-shed regime.

    Pakistan-flavored defaults below; extend or rename for other markets.
    Region drives both ambient temperature dynamics and outage frequency,
    which jointly determine PSoC stress.
    """

    LAHORE = "lahore"
    KARACHI = "karachi"
    ISLAMABAD = "islamabad"
    QUETTA = "quetta"
    PESHAWAR = "peshawar"


@dataclass(frozen=True)
class ClimateProfile:
    """Annual climate parameters for a region.

    Used to drive an ambient-temperature trace. Temperature feeds Arrhenius
    aging in physics.update_health and shows up as a model feature.
    """

    annual_mean_temp_c: float
    annual_amplitude_c: float  # peak-summer to peak-winter half-range
    daily_amplitude_c: float  # daily diurnal swing
    summer_peak_doy: int = 196  # day-of-year of peak heat (~mid-July, NH)
    humidity_factor: float = 1.0  # multiplier on thermal stress (humid coastal > dry)


@dataclass(frozen=True)
class LoadShedProfile:
    """Load-shedding regime for a region.

    Drives outage event arrivals. Outages → discharge events → PSoC stress
    when recovery is incomplete. The summer_amplification reflects that
    Pakistan grid stress peaks in summer due to AC demand.
    """

    avg_outages_per_day: float  # mean outage count per day per site
    avg_outage_duration_min: float  # mean discharge duration
    duration_std_min: float  # std for log-normal duration sampling
    summer_amplification: float = 1.5  # multiplier on rate during summer


@dataclass
class BatteryBank:
    """A battery bank installed at a site."""

    install_date: datetime
    manufacturer: Manufacturer
    nominal_capacity_ah: float = 200.0
    n_strings: int = 1  # parallel strings
    n_cells_per_string: int = 24  # -48V VRLA: 24 cells × 2V nominal
    initial_health: float = 1.0  # 1.0 = nameplate at install


@dataclass
class Site:
    """A cell site.

    `site_load_a` is the average DC current the BTS draws from the bank
    during a grid outage. Real sites vary 20-50A depending on load mix;
    we'll let scenarios override per-site if needed.
    """

    site_id: str
    region: Region
    has_genset: bool
    battery_bank: BatteryBank
    site_load_a: float = 30.0


@dataclass
class SimulationConfig:
    """Top-level simulation parameters."""

    start_date: datetime
    end_date: datetime
    telemetry_interval_min: int = 5  # 5-min telemetry sampling cadence
    seed: int = 42
    climate_by_region: dict[Region, ClimateProfile] = field(default_factory=dict)
    load_shed_by_region: dict[Region, LoadShedProfile] = field(default_factory=dict)


@dataclass
class ScenarioOverlay:
    """A scenario perturbation injected onto the base simulation.

    Scenarios drive the cohort/regional/cohort-correlated failure patterns
    that the model needs to learn. Day 2-3 we'll exercise:
        - heatwave: regional ambient_temp_c spike for a 2-week window
        - lot_defect: a manufacturer's batches in a date range age 3x faster
        - theft: rural sites have a probability of sudden discontinuity
    """

    name: str
    affected_regions: list[Region] = field(default_factory=list)
    affected_manufacturer: Manufacturer | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    severity: float = 1.0


# ---------------------------------------------------------------------------
# Default profiles — Pakistan-flavored. Tune after first simulation run.
# ---------------------------------------------------------------------------

DEFAULT_CLIMATE: dict[Region, ClimateProfile] = {
    Region.LAHORE: ClimateProfile(
        annual_mean_temp_c=24.0,
        annual_amplitude_c=14.0,
        daily_amplitude_c=10.0,
        humidity_factor=1.0,
    ),
    Region.KARACHI: ClimateProfile(
        annual_mean_temp_c=26.0,
        annual_amplitude_c=8.0,
        daily_amplitude_c=7.0,
        humidity_factor=1.2,  # coastal humidity worsens thermal stress
    ),
    Region.ISLAMABAD: ClimateProfile(
        annual_mean_temp_c=21.0,
        annual_amplitude_c=12.0,
        daily_amplitude_c=10.0,
        humidity_factor=0.9,
    ),
    Region.QUETTA: ClimateProfile(
        annual_mean_temp_c=15.0,
        annual_amplitude_c=15.0,
        daily_amplitude_c=12.0,
        humidity_factor=0.8,
    ),
    Region.PESHAWAR: ClimateProfile(
        annual_mean_temp_c=23.0,
        annual_amplitude_c=14.0,
        daily_amplitude_c=11.0,
        humidity_factor=1.0,
    ),
}


DEFAULT_LOAD_SHED: dict[Region, LoadShedProfile] = {
    # Severe load-shedding: 3-5 outages/day, ~2hr each. Heavy PSoC stress.
    Region.LAHORE: LoadShedProfile(
        avg_outages_per_day=4.0,
        avg_outage_duration_min=120.0,
        duration_std_min=45.0,
        summer_amplification=1.8,
    ),
    # Moderate: better infrastructure, 1-2 outages/day.
    Region.KARACHI: LoadShedProfile(
        avg_outages_per_day=2.5,
        avg_outage_duration_min=90.0,
        duration_std_min=30.0,
        summer_amplification=1.5,
    ),
    # Light: capital city, prioritized supply.
    Region.ISLAMABAD: LoadShedProfile(
        avg_outages_per_day=1.0,
        avg_outage_duration_min=60.0,
        duration_std_min=20.0,
        summer_amplification=1.3,
    ),
    Region.QUETTA: LoadShedProfile(
        avg_outages_per_day=3.0,
        avg_outage_duration_min=100.0,
        duration_std_min=40.0,
        summer_amplification=1.4,
    ),
    Region.PESHAWAR: LoadShedProfile(
        avg_outages_per_day=4.5,
        avg_outage_duration_min=130.0,
        duration_std_min=50.0,
        summer_amplification=1.7,
    ),
}
