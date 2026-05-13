"""Synthetic data + alarm-stream generator for telecom battery PdM.

Produces physically-plausible telemetry (voltage, current, temperature),
NMS-style alarm event streams, and ground-truth failure labels for
training XGBoost survival models and Cox PH models.

Design intent:
- Aging dynamics encode Arrhenius (thermal acceleration) and PSoC
  sulfation (the dominant VRLA aging mechanism in load-shed regimes).
- Alarm cascades follow realistic telecom NMS patterns (rectifier fail
  → low-voltage → battery-low → outage), not sanitized one-event-per-failure.
- Site/region/manufacturer heterogeneity creates the cohort structure
  needed for stratified drift detection and group-aware CV.

The generator must be physics-faithful enough that PSoC features and
trend features are genuinely predictive in the synthetic data.
Otherwise Day 2 models will look deceptively good and be uninformative.
"""

from battery_pdm.synth.config import (
    Manufacturer,
    Region,
    ClimateProfile,
    LoadShedProfile,
    BatteryBank,
    Site,
    SimulationConfig,
    ScenarioOverlay,
    DEFAULT_CLIMATE,
    DEFAULT_LOAD_SHED,
)

__all__ = [
    "Manufacturer",
    "Region",
    "ClimateProfile",
    "LoadShedProfile",
    "BatteryBank",
    "Site",
    "SimulationConfig",
    "ScenarioOverlay",
    "DEFAULT_CLIMATE",
    "DEFAULT_LOAD_SHED",
]
