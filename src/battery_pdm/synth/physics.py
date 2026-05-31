"""Battery physics — aging, discharge, charge dynamics.

This is the intellectually load-bearing module of the synthetic data
generator. You implement the math; the runner calls it. Functions here
are pure: state in, new state out. No side effects, no I/O.

What you implement here directly determines whether the synthetic data
has the structure your Day 2-4 models need to learn:

    1. Arrhenius temperature acceleration of calendar aging.
    2. PSoC sulfation accelerator (the load-shed → incomplete-recharge
       → sulfation chain that dominates VRLA aging in Pakistan).
    3. Shepherd-equation discharge voltage dynamics.
    4. CV/CC charge acceptance.
    5. Health update integrating the above.

If the synthetic batteries age purely on calendar time, your PSoC
features on Day 2 will look uninformative and your model will look
deceptively bad (or your trend features will look deceptively good
because there's nothing to predict). The physics here has to bite.

Conventions
-----------
- Voltages are positive magnitudes in volts. The -48V system sign is a
  display detail handled in telemetry.py, not here.
- Currents in amps. Discharge current is positive (current leaving battery).
- Times in hours unless suffixed otherwise.
- soc ∈ [0, 1].
- health ∈ [0, 1] where 1.0 = nameplate capacity. Health monotonically
  decreases (no self-repair).
- Per-cell quantities throughout. Bank-level aggregation lives in site.py.

References (worth a skim before implementing)
---------------------------------------------
- Schiffer et al. (2007), "Model prediction for ranking lead-acid
  batteries..." — VRLA aging factor decomposition, including PSoC.
- Shepherd (1965), "Design of primary and secondary cells II" —
  the classic discharge voltage equation.
- Arrhenius equation — standard, appears in any battery aging text.
- Pinson & Bazant (2013), "Theory of SEI formation..." — Li-ion analog
  if you later extend to Li-ion banks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


# ---------------------------------------------------------------------------
# Physics constants — adjust if you have specific manufacturer data.
# ---------------------------------------------------------------------------

R_GAS_J_PER_MOL_K: float = 8.314  # ideal gas constant
T_REF_K: float = 298.15  # 25°C reference

# VRLA activation energy. Literature values 40-70 kJ/mol depending on
# failure mode (positive grid corrosion ~50, negative active mass ~40).
# 50 is a defensible single-number choice.
ACTIVATION_ENERGY_J_PER_MOL: float = 50_000.0

# Nominal cell voltages
V_FLOAT_PER_CELL: float = 2.27  # float voltage at 25°C
V_OPEN_CIRCUIT_PER_CELL: float = 2.10  # OCV at full SoC
V_LVD_PER_CELL: float = 1.75  # low-voltage disconnect threshold
V_TEMP_COMPENSATION_PER_C: float = -0.003  # float voltage drops with temp

# PSoC parameters — calibrate so that pure PSoC operation roughly halves
# expected lifetime vs proper recharge regimes.
PSOC_THRESHOLD_SOC: float = 0.95  # below this, sulfation accumulates
PSOC_AGING_COEFFICIENT: float = 0.8  # calibrated: PSoC-heavy sites fail in 2-4 yrs


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class CellState:
    """State of a single cell at a moment in time.

    All quantities at the cell level. Bank-level state (parallel/series
    aggregation) is composed in site.py.

    Attributes
    ----------
    soc:
        State of charge ∈ [0, 1]. 1.0 = fully charged.
    health:
        Capacity ratio ∈ [0, 1]. 1.0 = nameplate at install.
        Monotonically non-increasing.
    cumulative_throughput_ah:
        Total charge that has moved through the cell (|integral of |I|dt|).
        Cycle-aging proxy.
    time_in_psoc_hours:
        Cumulative hours spent below PSOC_THRESHOLD_SOC. Drives sulfation.
    arrhenius_age_factor:
        Cumulative thermal stress (∫ acceleration_factor dt). Effective age.
    sulfation_index:
        Latent damage variable ∈ [0, ∞). Rises in PSoC, doesn't reverse.
        Affects discharge voltage curve shape (degraded cells fall faster).
    """

    soc: float
    health: float
    cumulative_throughput_ah: float
    time_in_psoc_hours: float
    arrhenius_age_factor: float
    sulfation_index: float

    @classmethod
    def fresh(cls) -> "CellState":
        """A brand-new cell at install."""
        return cls(
            soc=1.0,
            health=1.0,
            cumulative_throughput_ah=0.0,
            time_in_psoc_hours=0.0,
            arrhenius_age_factor=0.0,
            sulfation_index=0.0,
        )


# ---------------------------------------------------------------------------
# Aging dynamics — YOU IMPLEMENT
# ---------------------------------------------------------------------------


def temperature_acceleration_factor(
    temp_c: float,
    activation_energy_j_per_mol: float = ACTIVATION_ENERGY_J_PER_MOL,
) -> float:
    """Arrhenius acceleration factor for aging at temperature `temp_c`.

    Returns the multiplier on calendar aging rate vs the 25°C reference.
    Value of 2.0 means the cell ages twice as fast at this temperature.

    The Arrhenius form:

        rate(T) / rate(T_ref) = exp( Ea / R * (1/T_ref - 1/T) )

    where T is in Kelvin (= temp_c + 273.15).

    Sanity checks (write these as tests):
        - At 25°C, factor == 1.0 exactly.
        - At 35°C, factor ~ 1.8-2.5 (rule of thumb: doubling per ~10°C).
        - At 45°C, factor ~ 3-6.
        - At 5°C, factor < 1 (cold slows aging).
        - Function is monotonically increasing in temp_c.

    TODO (USER): implement.
    """
    t_k = temp_c + 273.15
    t_ref_k = 298.15
    exponent = (activation_energy_j_per_mol / R_GAS_J_PER_MOL_K) * (
        1.0 / t_ref_k - 1.0 / t_k
    )
    return math.exp(exponent)


def psoc_aging_multiplier(
    soc: float,
    dt_hours: float,
    psoc_threshold: float = PSOC_THRESHOLD_SOC,
    coefficient: float = PSOC_AGING_COEFFICIENT,
) -> float:
    """Aging acceleration from operating below full SoC.

    For VRLA, sulfation accumulates whenever the cell is not allowed to
    return to fully charged. The longer the cell sits below threshold,
    the more aggressive the sulfation rate per unit time.

    A reasonable form:

        if soc >= psoc_threshold:
            return 1.0
        else:
            depth_factor = (psoc_threshold - soc) / psoc_threshold
            return 1.0 + coefficient * dt_hours * depth_factor

    Tune `coefficient` so a battery permanently held at SoC=0.7 has its
    expected lifetime roughly halved versus one held >0.95.

    Sanity checks:
        - At soc >= psoc_threshold, returns 1.0 (no penalty).
        - At soc < psoc_threshold, returns > 1.0.
        - Larger dt_hours → larger multiplier (compounds).
        - Lower soc → larger multiplier (deeper PSoC = worse).

    TODO (USER): implement.
    """
    if soc >= psoc_threshold:
        return 1.0
    else:
        depth_factor = (psoc_threshold - soc) / psoc_threshold
        return 1.0 + coefficient * depth_factor


def update_health(
    state: CellState,
    dt_hours: float,
    ambient_temp_c: float,
    current_a: float,
    nominal_capacity_ah: float,
) -> CellState:
    """Advance cell state forward by `dt_hours`.

    Combines:
        - Arrhenius temperature acceleration (from temperature_acceleration_factor)
        - PSoC penalty (from psoc_aging_multiplier)
        - Cycle aging from charge throughput

    Suggested form:

        base_health_decay_per_hour = HEALTH_DECAY_PER_YEAR / (365 * 24)
        decay = (
            base_health_decay_per_hour
            * temp_factor
            * psoc_factor
            * (1 + cycle_factor)
            * dt_hours
        )
        new_health = max(0.0, state.health - decay)

    Where HEALTH_DECAY_PER_YEAR ≈ 0.05 gives roughly 20-year nameplate
    life under ideal conditions, dropping to <10 years under PSoC + heat.
    Tune by running the simulator and inspecting failure rates.

    Also update:
        - cumulative_throughput_ah += abs(current_a) * dt_hours
        - time_in_psoc_hours += dt_hours if soc < threshold else 0
        - arrhenius_age_factor += temp_factor * dt_hours
        - sulfation_index: rises in PSoC, irreversible

    Note: SoC update is NOT done here; that lives in coulomb_counting_step
    below because it depends on charge/discharge direction and the
    runner's current load context.

    TODO (USER): implement aging dynamics.
    """
    temp_factor = temperature_acceleration_factor(ambient_temp_c)
    psoc_factor = psoc_aging_multiplier(state.soc, dt_hours)
    cycle_factor = abs(current_a) * dt_hours / nominal_capacity_ah
    HEALTH_DECAY_PER_YEAR = 0.05

    base_health_decay_per_hour = HEALTH_DECAY_PER_YEAR / (365 * 24)
    decay = (
        base_health_decay_per_hour
        * temp_factor
        * psoc_factor
        * (1 + cycle_factor)
        * dt_hours
    )
    new_health = max(0.0, state.health - decay)

    new_cumulative_throughput_ah = (
        state.cumulative_throughput_ah + abs(current_a) * dt_hours
    )
    if state.soc < PSOC_THRESHOLD_SOC:
        new_time_in_psoc_hours = state.time_in_psoc_hours + dt_hours
    else:
        new_time_in_psoc_hours = state.time_in_psoc_hours

    new_arrhenius_age_factor = state.arrhenius_age_factor + temp_factor * dt_hours
    sulfation_rate = 5e-4
    sulfation_delta = sulfation_rate * dt_hours * max(0, PSOC_THRESHOLD_SOC - state.soc)
    new_sulfation_index = min(state.sulfation_index + sulfation_delta, 10.0)

    return replace(
        state,
        health=new_health,
        cumulative_throughput_ah=new_cumulative_throughput_ah,
        time_in_psoc_hours=new_time_in_psoc_hours,
        arrhenius_age_factor=new_arrhenius_age_factor,
        sulfation_index=new_sulfation_index,
    )


def coulomb_counting_step(
    state: CellState,
    current_a: float,
    dt_hours: float,
    nominal_capacity_ah: float,
) -> CellState:
    """Update SoC via coulomb counting, honoring current health (capacity fade).

    Effective capacity = nominal_capacity_ah * state.health.
    Charge added (signed): -current_a * dt_hours  (discharge current positive
    by convention, so discharge reduces SoC).

    Clip soc to [0, 1].

    TODO (USER): implement. Short — should be ~5 lines.
    """
    effective_capacity = max(nominal_capacity_ah * state.health, 0.01)
    delta_soc = -(current_a * dt_hours) / effective_capacity
    new_soc = max(0.0, min(1.0, state.soc + delta_soc))
    return replace(state, soc=new_soc)


# ---------------------------------------------------------------------------
# Discharge / charge voltage dynamics — YOU IMPLEMENT
# ---------------------------------------------------------------------------


def shepherd_discharge_voltage(
    state: CellState,
    current_a: float,
    nominal_capacity_ah: float,
    n_cells: int = 24,
) -> float:
    """Terminal voltage of the bank under discharge (positive magnitude, volts).

    Shepherd-equation form (per cell):

        V = E0 - K * Q / (Q - it) - R * i + A * exp(-B * it)

    where:
        E0    = ~2.10 V (open-circuit reference)
        K     = polarization constant (~0.01)
        Q     = effective capacity in Ah = nominal_capacity_ah * state.health
        it    = Ah removed = (1 - state.soc) * Q
        R     = internal resistance (rises with sulfation_index!)
        i     = discharge current (amps)
        A, B  = exponential zone parameters (~0.3, ~3.5)

    Bank voltage = per-cell voltage * n_cells.

    Crucially, R should depend on state.sulfation_index so degraded cells
    show steeper voltage drop under load. This is the physical basis for
    the dV/dt-under-load feature you'll engineer on Day 2.

    Sanity checks:
        - At soc=1.0, current=0, healthy: V_bank ~ V_OPEN_CIRCUIT * n_cells.
        - At soc near 0: V drops sharply (the "knee" of the curve).
        - Higher current → lower V (IR drop).
        - Higher sulfation_index → steeper drop (worse R).
        - Returned voltage is positive (we're returning magnitude).

    TODO (USER): implement.
    """
    E0 = V_OPEN_CIRCUIT_PER_CELL
    K = 0.01
    A = 0.3
    B = 3.5
    Q = nominal_capacity_ah * state.health
    it = (1.0 - state.soc) * Q
    R = 0.002 * (1.0 + state.sulfation_index)
    denom = max(Q - it, 0.01 * Q)
    v_cell = E0 - K * Q / denom - R * current_a + A * (math.exp(-B * it / Q) - 1.0)
    return max(0.0, v_cell * n_cells)


def float_voltage(
    ambient_temp_c: float,
    state: CellState,
    n_cells: int = 24,
) -> float:
    """Float voltage of the bank when on rectifier (positive magnitude, volts).

    Healthy cells hold V_FLOAT_PER_CELL with temperature compensation
    (V_TEMP_COMPENSATION_PER_C per °C off 25°C). Degraded cells hold
    slightly lower because of internal leakage / micro-shorts — encode
    this as a small downward offset proportional to sulfation_index.

    Add a small Gaussian noise term for sensor realism in telemetry.py
    rather than here.

    Sanity:
        - Healthy at 25°C: returns ~ V_FLOAT_PER_CELL * n_cells (~54.5V for 24 cells).
        - Higher ambient temp → slightly lower float (the negative
          temp coefficient).
        - Higher sulfation_index → slightly lower float (cell can't
          maintain voltage under standing load).

    TODO (USER): implement.
    """
    temp_offset = V_TEMP_COMPENSATION_PER_C * (ambient_temp_c - 25.0)
    sulfation_offset = -0.005 * state.sulfation_index
    v_per_cell = V_FLOAT_PER_CELL + temp_offset + sulfation_offset
    return v_per_cell * n_cells


def charge_acceptance_current(
    state: CellState,
    available_charger_current_a: float,
    nominal_capacity_ah: float,
) -> float:
    """How much current the cell accepts during recharge.

    CV/CC pattern:
        - Low SoC: cell accepts the full available_charger_current_a (CC phase).
        - High SoC (above ~0.85): current tapers exponentially toward zero
          (CV phase).
        - Degraded cells (low health, high sulfation) accept less even
          at low SoC — internal resistance limits how fast they can
          take charge.

    A reasonable form:

        if soc < 0.85:
            return available_charger_current_a * (1 - sulfation_index * 0.1)
        else:
            taper = exp(-(soc - 0.85) / 0.05)
            return available_charger_current_a * taper * (1 - sulfation_index * 0.1)

    The sulfation suppression is what makes degraded batteries take
    longer to charge — your "recharge time trend" feature on Day 2
    falls directly out of this.

    TODO (USER): implement.
    """
    suppression = max(0.0, 1.0 - state.sulfation_index * 0.1)
    if state.soc < 0.85:
        return available_charger_current_a * suppression
    else:
        taper = math.exp(-(state.soc - 0.85) / 0.05)
        return available_charger_current_a * taper * suppression
