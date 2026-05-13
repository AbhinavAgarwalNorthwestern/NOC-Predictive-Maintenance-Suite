# Day 1 Handoff — Telecom Battery PdM Project

**Date:** 2026-05-09 (Day 1 of 30)
**Project:** `project_01` — Predictive maintenance for telecom battery infrastructure
**Defends bullet:** Etisalat / PTCL
**Status:** Mid-afternoon. Theory + Q&A complete. Skeleton delivered. Implementation not started.

---

## How to bootstrap the next thread

In your new chat, paste:
1. Your master prompt (the 30-day plan / positioning / approach rules document).
2. **This handoff doc.**
3. Tell the new thread: "I'm continuing Day 1. I've completed everything in the handoff up to and including the seven questions and adversarial review. I'm ready to implement `physics.py`. Start by confirming you have all four skeleton files, then walk me through implementing `temperature_acceleration_factor` first."

That's enough context for the new thread to pick up cleanly without relitigating decisions.

---

## Locked decisions (do not relitigate in next thread)

### Project scope: Option A
- 5-day timeline preserved.
- Days 1–3: data + offline modeling.
- Day 4: real-time feature pipeline + endpoint deploy.
- Day 5: monitoring + retraining + recorded demo.
- Streaming layer ships as "production-ready architecture, single-site demo with synthetic stream."

### Real-time inference IN, online learning OUT
- **Real-time inference + streaming features:** YES (Kinesis + feature store + SageMaker endpoint).
- **Online learning / continuous weight updates:** NO. Reasons:
  - Label latency (failure observed weeks-to-months after scoring).
  - Rare-event regime (~1% failure rate; gradient SNR is brutal).
  - Cox partial likelihood doesn't online-update cleanly.
  - Auditability — ops needs reproducible decisions; mutating models break that.
- **Senior framing:** "real-time" applies to inference and feature compute, not to learning. Online inference + scheduled-and-triggered batch retraining is the correct production pattern for rare-event PdM.

### Production-grade retraining loop (Day 5 deliverable)
1. Shadow-mode challengers (1–2 weeks parallel scoring before promotion).
2. Stratified drift detection (PSI/KS per cohort: manufacturer × region × install-year).
3. Calibration-triggered retrain (realized Brier / calibration slope, not just feature drift).
4. Performance attribution (when C-index drops, log which feature distributions moved most).
5. Cost-sensitive promotion criterion ($/site/year, not C-index alone).
6. Automated rollback path encoded in Step Functions.

### Architectural framing
- Hybrid: XGBoost (`survival:cox` objective) for short-horizon triage + Cox PH for planning horizon, stacked.
- Five feature families: per-discharge, per-recharge, rolling cumulative, inter-event, trend.
- Physics-informed feature engineering, not pure ML and not pure physics.
- Group-aware time-respecting CV with embargo.

---

## Theory framework summary

### Domain context (Pakistan)
- Tower sites run **-48V DC plants**. -48V is **nominal**, not a failure value.
- Voltage hierarchy: float ~54.0–54.5V → nominal -48V → low-voltage alarm ~-45/-46V → battery-low ~-44V → LVD ~-42V.
- Heat (45°C+ summers), severe load-shedding (4–12 hrs/day in regions), genset backup at many sites, **theft as a real failure mode** in rural areas.
- VRLA chemistry dominates legacy networks; Li-ion increasing.

### Why hybrid XGBoost + Cox PH
- Cox handles censoring and gives interpretable hazard ratios but is linear-in-x in log hazard.
- XGBoost captures non-linear interactions but ignores censoring naively → use `survival:cox` objective.
- Stacking: XGBoost score becomes a covariate in Cox.
- Alternatives (RSF, DeepSurv, Weibull AFT) documented in "Models Considered" but not shipped — interpretability and ops-trust constraints.

### Cox PH essentials (be ready to whiteboard)
- Hazard: `h(t|x) = h_0(t) · exp(β^T x)`.
- Partial likelihood — baseline hazard left unspecified, β estimated via risk-set ranking.
- Censored subjects contribute through risk sets.
- PH testable via Schoenfeld residuals.
- Survival metrics: C-index (AUC analog), Brier/IBS, IPCW corrections.

### Physics-informed features (going into Day 2)
- **Arrhenius-weighted age:** `Σ exp(-Eₐ/RT(τ))` — cumulative thermal stress.
- **Coulomb-counted throughput:** `Σ |I(τ)| dτ`.
- **Peukert-corrected discharge events.**
- **dV/dt under known load** — direct capacity proxy from Shepherd-equation behavior.
- **Float voltage stability** — sulfation indicator.
- **PSoC indicators:** especially "fraction of last 30 days spent below float voltage" (the senior single-feature pick).

### Why physics models alone don't work
- Parameter ID at scale is infeasible (P2D needs 30–50 params per cell, drift with age).
- Physics predicts cell state, not operational failure.
- Failure modes outside physics (theft, rectifier faults, corrosion, install errors).
- Data resolution mismatch (we have 5-min averages, not Hz-scale).
- Model risk — engineers over-trust principled-looking models.
- Engineering economics — lab characterization doesn't pay back.
- **Position:** physics-informed features fed into ML models. Physics for state estimation in BMS layer below us; ML for fleet-level failure prediction.

---

## The seven questions — your answers + adversarial review

### Q1: Censoring framing
**Your answer:** different events; battery lifetime + cycles as features.

**Review:** Right verdict (different), wrong reasoning track. Cycles and lifetime are *features*, not censoring encoding — orthogonal questions.

**Senior framing:**
- End-of-study at month 36 = **administrative / non-informative** censoring. Encode `(T=36, δ=0)`.
- Preemptive replacement at month 18 = potentially **informative** censoring. If the regional sweep was risk-correlated, treating both the same biases β downward.
- Three options: sensitivity analysis, competing-risks framework, IPTW.

### Q2: Operating point
**Your answer:** `p*(truck roll + battery cost) < avoidable outage` for optimal p; `expected cost = p*(truck roll + battery cost)`.

**Review:** Right intuition, incomplete.

**Senior framing:**
- Costs: dispatch $1,000 (deterministic). No-dispatch + failure $3,500. No-dispatch + survive $0.
- Breakeven: `$1,000 = p* × $3,500` → **p* ≈ 0.286**.
- Budget constraint: real telecom ops rank by hazard, dispatch top-K. Breakeven tells you if budget is right-sized.
- Calibration is load-bearing — bad calibration + great AUC = wrong threshold.

### Q3: PH violation on temperature
**Your answer:** initially skipped; clarified after explanation.

**Senior answer:** **Time-interaction term**, specifically `β · log(age) · temp_exposure`, keeping `temp_exposure` as a main effect.
- Why for deployed system: continuous feature (stratification needs ad-hoc binning), single interpretable formula for ops, physics-grounded (older batteries have less thermal margin), drift-monitorable.
- When stratification IS right: categorical features with operational meaning as strata — **manufacturer** is the canonical case.
- Validation discipline: re-run Schoenfeld on the augmented model.

### Q4: CV strategy
**Your answer:** future shouldn't predict past in time series.

**Review:** One leg of three.

**Senior framing — three axes of leakage:**
1. Time leakage (you got this).
2. **Group leakage** — same battery in train and test fold lets memorization look like generalization.
3. **Cluster leakage** — sites in same region share regional shocks (heatwaves).

**Right scheme:**
- Group on `site_id` (sometimes `region`).
- Time-respecting within each group split (train months 1–18, test 19–24).
- ~30-day embargo period to prevent rolling-feature leakage.
- Stratify on cohort (manufacturer × region × install-year).

### Q5: Why not LSTM
**Your answer:** multivariate problem with exogenous features.

**Review:** True but the weakest of three points. Interviewer will say "OK, multivariate LSTM" and you've conceded.

**Stronger three-point answer:**
1. **Sample efficiency under rare events** — failures ~1%; LSTMs need many positives. Tree models dominate small-n high-imbalance.
2. **Native censoring handling** — LSTMs don't natively output survival distributions; XGBoost has `survival:cox` built in.
3. **Interpretability and ops trust** — field ops needs hazard ratios / SHAP for dispatch decisions; black-box LSTM doesn't survive audit.

### Q6: Alarm correlation
**Your answer:** same event most likely; identify mean inter-alarm time; features include "did rectifier fail" and time-from-low-voltage-to-outage (~5 min).

**Review:** Right direction; the 5-min feature you named is **the battery margin feature**, top-3 predictor. Senior-grade derivation.

**Sharpening needed:**
- **Principled grouping rule:** plot histogram of inter-alarm gaps — typically bimodal (within-event <15 min vs between-event hours/days). Pick the trough, document the cutoff.
- **Causal hierarchy:** encode root cause + cascade as separate features, not just per-alarm flags.
- **Storm de-dup:** same alarm code firing 10× in 2 min is one observation.
- **Active vs cleared:** alarm duration is a feature (auto-cleared in 3 min ≠ active for 6 hrs).

### Q7: Real-time / batch / static feature boundary
**Your answer:**
- Streaming: avg voltage/current/temp.
- Static: make, n_cells, batteries per NOC element.
- Batch-precomputed: age, n_cycles, n_outages.

**Review:** Two of three buckets wrong. **Cycles and outage counts are STREAMING, not batch.**

**Senior reframe:**
> Streaming-vs-batch isn't "what data" — it's "how fresh does this need to be at scoring time, and how does it stay consistent between training and serving?"

- **Streaming** = changes meaningfully between batch refreshes AND staleness affects prediction → outage counts, cycle counts, time-since-X, rolling aggregates.
- **Batch-precomputed** = slow-moving aggregates → per-cohort base hazard rates, climate baselines, manufacturer priors, age-in-months.
- **Static** = unchanging → site metadata, install date, n_cells, manufacturer.

**Train/serve skew kicker:** the same code path that computes "outage count in last 30 days" online from the stream must produce the same value when computing it offline from S3 history for training. Single feature definition that runs in both modes = the resume bullet.

---

## Files delivered today

Four Python files dropped in `/mnt/user-data/outputs/` with flattened names because the outputs dir doesn't allow nested directories. Place them as:

| Downloaded as | Drop into |
|---|---|
| `synth_physics.py` | `project_01/src/battery_pdm/synth/physics.py` |
| `synth_config.py` | `project_01/src/battery_pdm/synth/config.py` |
| `synth__init__.py` | `project_01/src/battery_pdm/synth/__init__.py` |
| `tests_test_physics.py` | `project_01/tests/test_physics.py` |

Also touch empty `__init__.py` files at:
- `project_01/src/battery_pdm/__init__.py`
- `project_01/tests/__init__.py`

### File contents summary

**`config.py` (scaffold-complete, no TODOs):**
- Enums: `Manufacturer`, `Region`.
- Frozen dataclasses: `ClimateProfile`, `LoadShedProfile`.
- Dataclasses: `BatteryBank`, `Site`, `SimulationConfig`, `ScenarioOverlay`.
- Pakistan-flavored defaults: `DEFAULT_CLIMATE`, `DEFAULT_LOAD_SHED` for 5 regions (Lahore, Karachi, Islamabad, Quetta, Peshawar).

**`physics.py` (skeleton — 7 functions, all `NotImplementedError`):**
1. `temperature_acceleration_factor` — Arrhenius. Easy warmup, ~5 lines.
2. `psoc_aging_multiplier` — PSoC sulfation accelerator.
3. `coulomb_counting_step` — SoC bookkeeping.
4. `update_health` — composes Arrhenius + PSoC + cycle aging.
5. `float_voltage` — linear with temp coefficient and sulfation offset.
6. `charge_acceptance_current` — CV/CC pattern, sulfation-suppressed.
7. `shepherd_discharge_voltage` — Shepherd equation; the hard one, save for last.

Constants defined at top of file: `R_GAS_J_PER_MOL_K`, `T_REF_K`, `ACTIVATION_ENERGY_J_PER_MOL=50_000`, `V_FLOAT_PER_CELL=2.27`, `V_OPEN_CIRCUIT_PER_CELL=2.10`, `V_LVD_PER_CELL=1.75`, `V_TEMP_COMPENSATION_PER_C=-0.003`, `PSOC_THRESHOLD_SOC=0.95`, `PSOC_AGING_COEFFICIENT=1.5e-3`.

`CellState` dataclass: `soc`, `health`, `cumulative_throughput_ah`, `time_in_psoc_hours`, `arrhenius_age_factor`, `sulfation_index`. Has `CellState.fresh()` classmethod.

**`test_physics.py` (22 sanity tests):**
- Arrhenius: ref temp = 1.0, monotonic, doubling rule of thumb, cold decelerates.
- PSoC: no penalty above threshold, deeper is worse, compounds with time.
- Coulomb counting: discharge lowers SoC, charge raises, clipping, capacity fade.
- Health update: monotonic decrease, faster at high temp, faster in PSoC.
- Shepherd: open-circuit at full charge, voltage drops under load, falls with SoC, sulfation steepens drop.
- Float voltage: ~54.5V at 25°C for 24-cell bank, negative temp coefficient.
- Charge acceptance: tapers near full, full at low SoC, suppressed for degraded.

**Theory briefs (in earlier outputs):**
- `day01_theory_brief.md` — v1 (pre-real-time framing).
- `day01_theory_brief_v2.md` — v2 (with alarm streams, real-time arch, retraining sophistications). **Use v2 as `docs/01_problem_framing.md`.**

---

## What's done ✅ vs what's pending ❌

### Done
- ✅ Theory framework locked (v2 brief).
- ✅ Scope decisions: Option A, real-time inference, online-learning-out, four retraining sophistications.
- ✅ Pakistan domain context integrated (load-shed, heat, theft, voltage hierarchy).
- ✅ Physics-vs-ML positioning articulated.
- ✅ Five feature families taxonomized.
- ✅ Seven questions answered + adversarially reviewed (Q3 explained after initial skip).
- ✅ Q3 senior answer captured (time-interaction term, not stratification).
- ✅ Skeleton files delivered: `config.py` (complete), `physics.py` (TODOs), `test_physics.py` (sanity tests), `__init__.py`.

### Pending
- ❌ Repo not yet scaffolded on disk in `C:\Users\abhin\projects\project_01\`.
- ❌ Files not yet placed at correct paths.
- ❌ `physics.py` functions not yet implemented (all 7 still `NotImplementedError`).
- ❌ Tests not yet run (no green/red status).
- ❌ End-of-day edge case grilling.
- ❌ End-of-day ML system design question.
- ❌ Update master prompt doc with Day 1 status.
- ❌ INCIDENTS.md not started (no production scenarios documented yet).

---

## Next session immediate actions (in order)

1. **Scaffold the repo** under `C:\Users\abhin\projects\project_01\`:
   ```
   project_01/
   ├── README.md
   ├── docs/
   │   ├── 01_problem_framing.md   ← copy v2 brief
   │   ├── 02_data_dictionary.md   ← write today
   │   ├── 03_eda_findings.md
   │   ├── 04_methodology.md       ← Models Considered table preview
   │   ├── 05_validation_report.md
   │   ├── 06_deployment_design.md
   │   ├── 07_monitoring_plan.md
   │   └── INCIDENTS.md
   ├── src/battery_pdm/
   │   ├── __init__.py             ← empty
   │   └── synth/
   │       ├── __init__.py         ← from delivered file
   │       ├── config.py           ← from delivered file
   │       └── physics.py          ← from delivered file (skeleton)
   ├── tests/
   │   ├── __init__.py             ← empty
   │   └── test_physics.py         ← from delivered file
   ├── deployment/
   ├── monitoring/
   ├── outputs/
   └── pyproject.toml              ← uv-managed
   ```

2. **Implement `physics.py`** in this order:
   1. `temperature_acceleration_factor` (Arrhenius — easiest)
   2. `psoc_aging_multiplier`
   3. `coulomb_counting_step`
   4. `update_health`
   5. `float_voltage`
   6. `charge_acceptance_current`
   7. `shepherd_discharge_voltage` (hardest — save for last)

   After each function: run `pytest tests/test_physics.py::test_<relevant_test_name> -v`. All 22 should pass before moving to the next.

3. **30-min stuck rule.** If stuck longer than 30 min on any function, paste your attempt for adversarial review.

4. **End-of-day:**
   - Edge-case grilling (5–7 cases)
   - One ML system design question
   - Update master prompt doc with Day 1 done
   - Write `INCIDENTS.md` entry if any physics implementation surprised you

---

## Days 2–5 preview (do not work ahead)

- **Day 2:** Feature engineering using physics-informed approach. Implement `telemetry.py` and `alarms.py` skeletons. Begin XGBoost with `survival:cox` objective. Five feature families operationalized.
- **Day 3:** Cox PH model. Schoenfeld testing. Time-interaction term for temperature (per Q3). Stacking XGBoost score into Cox. Group-aware time-respecting CV with embargo.
- **Day 4:** Real-time feature pipeline (Kinesis Data Analytics or Lambda). SageMaker endpoint. Train/serve skew prevention via single feature definition module. AWS deployment with Terraform.
- **Day 5:** Monitoring (stratified drift, calibration drift, performance attribution). Step Functions retraining loop. Shadow-mode champion-challenger. Cost-sensitive promotion. Recorded demo. Cost estimate documented.

---

## Open questions to resolve in next thread

1. Final cohort definition for stratified drift: manufacturer × region × install-year-quartile? Confirm bin sizes once first sim runs.
2. Theft labeling: how to separate theft events from genuine capacity failures in synthetic data? — affects label table construction.
3. Genset interaction: does genset availability count as an exogenous covariate in Cox, or as a competing risk for the "outage" event? Lean toward covariate; revisit on Day 3.
4. Calibration choice: Platt vs isotonic — decide on Day 3 after seeing first-pass C-index.

---

## Approach rules reminder (from master prompt)

- User writes intellectually load-bearing code (math, model logic, evaluation).
- Claude scaffolds boilerplate (data loading, plotting, config).
- Hard math gets read-and-reimplement: working reference, then re-implement from memory while explaining each line.
- AWS: Terraform-first, locally-tested, deploy once for recording, teardown.
- Every project's methodology has a "Models Considered" table.
- Cost estimate documented per project.
- DS hacks to demonstrate across projects: calibration (Platt, isotonic, ALE), ensembling, stacking with meta-models, monotonicity constraints, conformal prediction, class imbalance (incl. why SMOTE often hurts), cost-sensitive learning, target encoding with leakage prevention, Bayesian hyperparameter tuning, time-series CV vs group CV vs stratified, drift detection beyond PSI (KS, MMD, Wasserstein).

---

*End of Day 1 handoff. Resume in new thread by pasting master prompt + this doc.*
