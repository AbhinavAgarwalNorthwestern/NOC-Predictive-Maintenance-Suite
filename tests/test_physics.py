"""Sanity tests for battery_pdm.synth.physics.

These tests pin the physical correctness properties — not exact numerical
values, which depend on parameter choices — so the user can implement
freely and still have something to verify against.

Run with: `pytest tests/test_physics.py -v`

Each test corresponds to a property a senior reviewer would check at the
implementation review. If a test fails, your physics is non-physical.
"""

from __future__ import annotations


import pytest

from battery_pdm.synth.physics import (
    CellState,
    V_OPEN_CIRCUIT_PER_CELL,
    charge_acceptance_current,
    coulomb_counting_step,
    float_voltage,
    psoc_aging_multiplier,
    shepherd_discharge_voltage,
    temperature_acceleration_factor,
    update_health,
)


# ---------------------------------------------------------------------------
# Arrhenius
# ---------------------------------------------------------------------------


def test_arrhenius_at_reference_temp_is_one() -> None:
    assert temperature_acceleration_factor(25.0) == pytest.approx(1.0, abs=1e-9)


def test_arrhenius_higher_temp_accelerates() -> None:
    assert temperature_acceleration_factor(35.0) > 1.0
    assert temperature_acceleration_factor(45.0) > temperature_acceleration_factor(35.0)


def test_arrhenius_doubling_rule_of_thumb() -> None:
    """A 10C rise should roughly double aging rate (industry rule of thumb)."""
    factor_25 = temperature_acceleration_factor(25.0)
    factor_35 = temperature_acceleration_factor(35.0)
    assert 1.5 < factor_35 / factor_25 < 3.0


def test_arrhenius_cold_decelerates() -> None:
    assert temperature_acceleration_factor(5.0) < 1.0


def test_arrhenius_monotonic() -> None:
    temps = [-5, 5, 15, 25, 35, 45, 55]
    factors = [temperature_acceleration_factor(t) for t in temps]
    assert all(b > a for a, b in zip(factors, factors[1:]))


# ---------------------------------------------------------------------------
# PSoC
# ---------------------------------------------------------------------------


def test_psoc_no_penalty_above_threshold() -> None:
    assert psoc_aging_multiplier(soc=0.98, dt_hours=1.0) == pytest.approx(1.0)
    assert psoc_aging_multiplier(soc=1.00, dt_hours=10.0) == pytest.approx(1.0)


def test_psoc_penalty_below_threshold() -> None:
    assert psoc_aging_multiplier(soc=0.70, dt_hours=1.0) > 1.0


def test_psoc_deeper_is_worse() -> None:
    shallow = psoc_aging_multiplier(soc=0.85, dt_hours=1.0)
    deep = psoc_aging_multiplier(soc=0.40, dt_hours=1.0)
    assert deep > shallow


def test_psoc_compounds_with_time() -> None:
    """PSoC effect compounds via update_health over longer periods."""
    state = CellState(
        soc=0.7,
        health=1.0,
        cumulative_throughput_ah=0,
        time_in_psoc_hours=0,
        arrhenius_age_factor=0,
        sulfation_index=0,
    )
    short = update_health(
        state, dt_hours=24, ambient_temp_c=25, current_a=0, nominal_capacity_ah=200
    )
    long = update_health(
        state, dt_hours=24 * 30, ambient_temp_c=25, current_a=0, nominal_capacity_ah=200
    )
    assert long.health < short.health


# ---------------------------------------------------------------------------
# Coulomb counting
# ---------------------------------------------------------------------------


def test_coulomb_discharge_lowers_soc() -> None:
    state = CellState.fresh()
    new = coulomb_counting_step(
        state, current_a=10.0, dt_hours=1.0, nominal_capacity_ah=200.0
    )
    assert new.soc < state.soc


def test_coulomb_charge_raises_soc() -> None:
    state = CellState.fresh()
    state.soc = 0.5
    new = coulomb_counting_step(
        state, current_a=-10.0, dt_hours=1.0, nominal_capacity_ah=200.0
    )
    assert new.soc > 0.5


def test_coulomb_clips_to_unit_interval() -> None:
    state = CellState.fresh()
    state.soc = 0.99
    new = coulomb_counting_step(
        state, current_a=-100.0, dt_hours=10.0, nominal_capacity_ah=200.0
    )
    assert new.soc <= 1.0
    new2 = coulomb_counting_step(
        state, current_a=100.0, dt_hours=100.0, nominal_capacity_ah=200.0
    )
    assert new2.soc >= 0.0


def test_coulomb_capacity_fade_reduces_effective_capacity() -> None:
    """Degraded cell should reach SoC=0 faster than healthy under same load."""
    healthy = CellState.fresh()
    degraded = CellState(
        soc=1.0,
        health=0.5,
        cumulative_throughput_ah=0,
        time_in_psoc_hours=0,
        arrhenius_age_factor=0,
        sulfation_index=0,
    )
    h_after = coulomb_counting_step(
        healthy, current_a=20, dt_hours=5, nominal_capacity_ah=200.0
    )
    d_after = coulomb_counting_step(
        degraded, current_a=20, dt_hours=5, nominal_capacity_ah=200.0
    )
    assert d_after.soc < h_after.soc


# ---------------------------------------------------------------------------
# Health update
# ---------------------------------------------------------------------------


def test_health_monotonically_decreases() -> None:
    state = CellState.fresh()
    new = update_health(
        state, dt_hours=24, ambient_temp_c=30, current_a=0, nominal_capacity_ah=200.0
    )
    assert new.health <= state.health


def test_health_decays_faster_at_high_temp() -> None:
    state = CellState.fresh()
    cool = update_health(
        state,
        dt_hours=24 * 30,
        ambient_temp_c=20,
        current_a=0,
        nominal_capacity_ah=200.0,
    )
    hot = update_health(
        state,
        dt_hours=24 * 30,
        ambient_temp_c=45,
        current_a=0,
        nominal_capacity_ah=200.0,
    )
    assert hot.health < cool.health


def test_health_decays_faster_in_psoc() -> None:
    full = CellState.fresh()
    psoc = CellState(
        soc=0.7,
        health=1.0,
        cumulative_throughput_ah=0,
        time_in_psoc_hours=0,
        arrhenius_age_factor=0,
        sulfation_index=0,
    )
    full_after = update_health(
        full,
        dt_hours=24 * 30,
        ambient_temp_c=25,
        current_a=0,
        nominal_capacity_ah=200.0,
    )
    psoc_after = update_health(
        psoc,
        dt_hours=24 * 30,
        ambient_temp_c=25,
        current_a=0,
        nominal_capacity_ah=200.0,
    )
    assert psoc_after.health < full_after.health


# ---------------------------------------------------------------------------
# Shepherd discharge
# ---------------------------------------------------------------------------


def test_open_circuit_voltage_full_charge() -> None:
    state = CellState.fresh()
    v = shepherd_discharge_voltage(
        state, current_a=0, nominal_capacity_ah=200, n_cells=24
    )
    expected = V_OPEN_CIRCUIT_PER_CELL * 24
    assert v == pytest.approx(expected, rel=0.05)


def test_voltage_drops_under_load() -> None:
    state = CellState.fresh()
    v_idle = shepherd_discharge_voltage(
        state, current_a=0, nominal_capacity_ah=200, n_cells=24
    )
    v_load = shepherd_discharge_voltage(
        state, current_a=30, nominal_capacity_ah=200, n_cells=24
    )
    assert v_load < v_idle


def test_voltage_falls_with_soc() -> None:
    high_soc = CellState.fresh()
    low_soc = CellState(
        soc=0.2,
        health=1.0,
        cumulative_throughput_ah=0,
        time_in_psoc_hours=0,
        arrhenius_age_factor=0,
        sulfation_index=0,
    )
    v_high = shepherd_discharge_voltage(
        high_soc, current_a=20, nominal_capacity_ah=200, n_cells=24
    )
    v_low = shepherd_discharge_voltage(
        low_soc, current_a=20, nominal_capacity_ah=200, n_cells=24
    )
    assert v_low < v_high


def test_sulfation_steepens_drop() -> None:
    """A sulfated cell should drop more voltage at the same load."""
    healthy = CellState.fresh()
    sulfated = CellState(
        soc=1.0,
        health=0.9,
        cumulative_throughput_ah=0,
        time_in_psoc_hours=500,
        arrhenius_age_factor=0,
        sulfation_index=2.0,
    )
    v_h = shepherd_discharge_voltage(
        healthy, current_a=30, nominal_capacity_ah=200, n_cells=24
    )
    v_s = shepherd_discharge_voltage(
        sulfated, current_a=30, nominal_capacity_ah=200, n_cells=24
    )
    assert v_s < v_h


# ---------------------------------------------------------------------------
# Float voltage
# ---------------------------------------------------------------------------


def test_float_voltage_at_25c_is_nominal() -> None:
    state = CellState.fresh()
    v = float_voltage(ambient_temp_c=25.0, state=state, n_cells=24)
    # ~54.5V for 24-cell VRLA at 25C
    assert 53.5 < v < 55.5


def test_float_voltage_negative_temp_coefficient() -> None:
    state = CellState.fresh()
    v_cold = float_voltage(ambient_temp_c=10.0, state=state, n_cells=24)
    v_hot = float_voltage(ambient_temp_c=40.0, state=state, n_cells=24)
    assert v_cold > v_hot


# ---------------------------------------------------------------------------
# Charge acceptance
# ---------------------------------------------------------------------------


def test_charge_taper_near_full() -> None:
    """At high SoC, accepted current should be < available."""
    state = CellState.fresh()
    state.soc = 0.95
    accepted = charge_acceptance_current(
        state, available_charger_current_a=20, nominal_capacity_ah=200
    )
    assert accepted < 20.0


def test_low_soc_accepts_full() -> None:
    state = CellState(
        soc=0.5,
        health=1.0,
        cumulative_throughput_ah=0,
        time_in_psoc_hours=0,
        arrhenius_age_factor=0,
        sulfation_index=0,
    )
    accepted = charge_acceptance_current(
        state, available_charger_current_a=20, nominal_capacity_ah=200
    )
    assert accepted == pytest.approx(20.0, rel=0.05)


def test_degraded_battery_accepts_less() -> None:
    """High sulfation_index should suppress charge acceptance."""
    healthy = CellState(
        soc=0.5,
        health=1.0,
        cumulative_throughput_ah=0,
        time_in_psoc_hours=0,
        arrhenius_age_factor=0,
        sulfation_index=0,
    )
    degraded = CellState(
        soc=0.5,
        health=0.7,
        cumulative_throughput_ah=0,
        time_in_psoc_hours=500,
        arrhenius_age_factor=0,
        sulfation_index=2.0,
    )
    a_h = charge_acceptance_current(
        healthy, available_charger_current_a=20, nominal_capacity_ah=200
    )
    a_d = charge_acceptance_current(
        degraded, available_charger_current_a=20, nominal_capacity_ah=200
    )
    assert a_d < a_h
