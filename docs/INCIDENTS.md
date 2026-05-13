# INCIDENTS — Simulated Production Scenarios

## INC-001: ZeroDivisionError when battery health reaches zero

**Detected:** Edge-case test — `health=0.0`, `coulomb_counting_step` called with 10A discharge  
**Root cause:** `effective_capacity = nominal_capacity_ah * state.health` = 0. Division by zero on `delta_soc = -(current_a * dt_hours) / effective_capacity`.  
**Impact:** Simulator crash mid-run. In production, any battery that degrades to health=0 (or very near it) would crash the scoring pipeline if the feature computation shares this code path.  
**Fix:** Guard effective capacity with a floor: `max(effective_capacity, 0.01)` or return state unchanged when health ≤ 0.  
**Lesson:** Any function that divides by a derived quantity must guard against zero. In ML systems, the scoring path sees inputs the training path never did — extreme degradation states that only appear after months of production drift.

---

## INC-002: PSoC aging multiplier was timestep-dependent

**Detected:** Comparison test — single 8760h step gave PSoC multiplier of 4.46 vs 1.0004 for a 1h step. Total decay over one year differed by orders of magnitude depending on simulator step size.  
**Root cause:** `psoc_aging_multiplier` multiplied by `dt_hours` internally, AND the caller (`update_health`) multiplied by `dt_hours` again. Double-counting time made results step-size-dependent.  
**Impact:** Simulator produces different failure distributions depending on temporal resolution. Models trained on 1h-step data would not transfer to 5-min-step data. Silent correctness bug — no crash, just wrong numbers.  
**Fix:** Removed `dt_hours` from inside `psoc_aging_multiplier`. It now returns a pure instantaneous rate multiplier. Time accumulation happens once, in `update_health`.  
**Lesson:** When composing multipliers in a decay equation, each physical quantity should appear exactly once. Dimensional analysis catches this: `[decay] = [rate/hour] * [multiplier] * [hours]`. If the multiplier already has `[hours]` in it, the units don't balance. In production ML systems, this class of bug manifests as "model works on hourly data but not on 5-min data" — a train/serve skew that's invisible until you change ingestion frequency.
