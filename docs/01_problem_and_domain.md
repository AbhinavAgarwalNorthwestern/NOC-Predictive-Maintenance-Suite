# 01. The problem and the domain

## TL;DR

Telecom base stations need backup batteries during grid outages. In markets
with daily load-shedding (Pakistan, Nigeria, parts of India), these batteries
cycle 30-50× more than spec sheets predict, fail 3-5× faster than warranties
assume, and routinely cause **call drops, SLA breaches, and emergency dispatches**.

The Network Operations Center (NOC) needs three different ML answers, on three
different latency cadences:

1. **"Will this site survive the *current* outage?"** — event-driven (seconds)
2. **"Will this site drain in the *next 48 hours*?"** — daily batch
3. **"Should this battery be *replaced*?"** — weekly batch

This project builds all three, sharing a single feature pipeline.

## The setting

### Telecom infrastructure 101

A typical telecom site has:

```
                    ┌──────────────────────────┐
                    │  Radio Base Station      │
                    │  (BTS / BSC / eNodeB)    │
                    │  + air conditioning      │
                    │  + lighting / aux loads  │
                    │  TOTAL LOAD: 5-25A DC    │
                    └──────────────────────────┘
                          │
                          │ DC bus (-48V)
                          ▼
            ┌─────────────┴──────────────┐
            │  Rectifier (AC → DC)       │ ← from grid when available
            │  Float charger 13-15A     │
            └─────────────┬──────────────┘
                          │
                          ▼
            ┌──────────────────────────────┐
            │  Battery bank (VRLA / Li)    │
            │  100-200 Ah, 24 cells        │
            │  Backup when grid fails      │
            └──────────────────────────────┘
```

When the grid fails (`AC_MAINS_FAIL` alarm), the rectifier stops charging. The
battery takes over powering the load, slowly discharging. Several things can
happen:

| Event | Alarm code | What it means |
|-------|------------|---------------|
| Grid restored within 1-4h | (no LVD) | Battery recharges. Normal cycle. |
| Battery discharged past safe SoC | `BATT_UNDERVOLTAGE` | Warning — operator should investigate |
| Battery so low that load is cut | `LOAD_DISCONNECT` (LVD) | **Site goes dark. Call drops. SLA violation.** |
| Charger malfunction | `RECTIFIER_FAULT` | Battery not getting full recharge |
| Cell drift in the bank | `CELL_IMBALANCE` | Bank degrading; capacity reduced |

The LVD event is what we ultimately want to prevent — that's when the customer
actually feels the failure.

### Why this is a hard problem

Three factors compound the difficulty:

#### 1. The cycling is brutal

In a Pakistani urban site, the battery may experience:
- 4-8 hours of off-grid time per day in summer
- 5-10 outages per day, each lasting 30 min - 3 hours
- Recharge between outages is often incomplete (rectifier underpowered, hot
  battery accepts less charge)

A battery rated for 1,500 deep cycles burns through them in 1-2 years instead of 7-10.

#### 2. The sensors are unreliable

Most telecom NOC platforms have:
- ✅ The **alarm stream** from the NMS (reliable, low-latency)
- ⚠️ Voltage telemetry only at coarse polls (every 5-15 minutes; gaps common)
- ❌ Temperature, internal resistance, SOC sensors: present at a fraction of sites
  and the data is often stale or wrong

So **production-realistic ML must work without good telemetry.** The most reliable
signal is the alarm stream. That's the constraint this project embraces.

#### 3. Causation is muddied

A battery's lifetime drains depend on factors that compound non-linearly:
- Site-specific load (5A vs 25A — 5× different discharge rate)
- Battery type, age, manufacturer (3-year-old VRLA degrades differently from a
  Li-Fe-Po4 install)
- Regional grid quality (Peshawar vs Islamabad: 6 vs 1 hour daily off-grid)
- Ambient temperature (Karachi summer accelerates aging)
- Manufacturing heterogeneity (two batteries from the same SKU differ ±30% in
  expected lifetime)

A simple model that says "high load → fast failure" misses 90% of the signal.

## The three ML questions

### Q1: Will this site survive the *current* outage?

**When asked:** the moment `AC_MAINS_FAIL` fires for a site.
**Latency budget:** seconds-to-minutes (operator has 4-12h to dispatch a generator).
**Output:** `hours_to_LVD` estimate + risk ranking.
**Decision:** dispatch generator? Truck roll? Or let it ride?

This is a **survival regression** problem. The label is `hours_to_LVD` with
right-censoring (some outages end before LVD because grid is restored). Cox
proportional hazards is the textbook model. C-index measures how well we rank
sites by who will fail soonest.

We call this the **autonomy model**.

### Q2: Will this site drain in the *next 48 hours*?

**When asked:** daily, in a batch run that screens all sites.
**Latency budget:** 24 hours (operator has 1-2 days lead time to plan).
**Output:** `P(LVD within 48h)` per site.
**Decision:** schedule a proactive generator dispatch, or rotate which sites get
shed first during peak load.

This is a **binary classification** problem. Label = "did LVD happen in the
[ref_time, ref_time + 48h] window?" We pick observations weekly per site so the
dataset is balanced enough to learn from. AUC measures ranking quality;
calibrated Brier measures whether the probabilities are operationally usable.

We call this the **drain predictor**.

### Q3: Should this battery be *replaced*?

**When asked:** weekly, in a batch run that ranks the entire fleet.
**Latency budget:** a week (replacement scheduling happens in monthly cycles).
**Output:** Cox risk score per battery lifecycle.
**Decision:** which 50-200 batteries to replace next month, given a fixed budget.

This is a **survival regression** at the lifecycle granularity. Each battery
gets one observation per lifecycle (install → failure or right-censored at
observation cutoff). C-index measures ranking quality across lifecycles.

We call this the **failure model**.

## Why these three are different

| | Autonomy | Drain predictor | Failure model |
|--|----------|-----------------|---------------|
| **Triggered by** | `AC_MAINS_FAIL` alarm | Cron (daily) | Cron (weekly) |
| **Granularity** | Per outage event | Per (site, day) | Per battery lifecycle |
| **Time horizon** | Next 0-12h | Next 48h | Next 6-24 months |
| **Algorithm** | XGBoost survival:cox | XGBoost binary:logistic + isotonic | XGBoost survival:cox |
| **Headline metric** | C-index | AUC + Brier | C-index |
| **Business decision** | Generator dispatch now | Schedule generator tomorrow | Replacement order next month |
| **Cost of FP** | ~$100 (unneeded gen run) | ~$50 (scheduled gen) | ~$200 (early replacement) |
| **Cost of FN** | $1000+ (SLA violation, churn) | ~$500 (emergency dispatch) | ~$3000 (catastrophic failure, outage) |

The asymmetry in FP/FN costs is huge — missed predictions are 10-20× more
expensive than false alarms. This is why we deliberately accept lower
precision in exchange for higher recall in operational thresholds.

## Business KPIs the system optimizes

| KPI | How the system helps | Baseline | With ML |
|-----|----------------------|--------:|--------:|
| SLA breaches per month | Drain predictor flags at-risk sites 48h ahead | ~50/region | -60% target |
| Emergency truck rolls per month | Replace reactive dispatch with proactive | ~120 | -50% target |
| Generator fuel cost per month | Only dispatch where actually needed | ~$15k | -25% target |
| Battery early-replacement cost | Risk-rank replacements instead of round-robin | ~$80k/yr | -30% target |
| Mean time between failures (MTBF) | Replace before failure, not after | 2.5 yr | 4 yr target |

The numbers above are illustrative for the project's narrative; in a real
deployment you'd back them out from the customer's actual SLA tariff sheet and
operational cost ledger.

## Why "alarms only" is the architectural choice, not a constraint

A reviewer might ask: "Why not use telemetry features when available?"

Three reasons:

1. **Telemetry coverage is low and uneven.** Modeling it as "available everywhere"
   produces a system that quietly degrades wherever a feed goes down.
2. **The alarms-only model already gets to C-index 0.90 on failure prediction.**
   The marginal value of adding telemetry is small.
3. **Operations leaders care about uniform reliability** more than peak accuracy.
   A model that works the same in Quetta as in Karachi is more valuable than one
   that's 2% better in Karachi but unusable in Quetta.

This is why the feature pipeline is designed to accept telemetry feature groups
later (via the extensible `@register_feature_group` pattern in
[`src/battery_pdm/streaming/autonomy/features.py`](../src/battery_pdm/streaming/autonomy/features.py))
but doesn't depend on them.

## What's specifically NOT in scope

- Real-time scoring (sub-second). The autonomy model could be wrapped behind a
  SageMaker endpoint if needed, but the use case doesn't actually demand it —
  4-12h grace before LVD is plenty for minute-level cron.
- Replacement *vendor selection* — that's a separate optimization problem.
- Network capacity planning — different system, different model.
- Detection of physical sabotage (cable theft is real and a major issue in some
  markets, but it's an anomaly-detection problem with different signals).

These are documented in the README "what's missing" section as potential extensions.

## Domain references

- **Load-shedding background:** Pakistan / Nigeria / India grid documentation
  (the schedule features we use are based on these public utility calendars).
- **Battery degradation modeling:** Doyle-Fuller-Newman (DFN), Shepherd
  equation, Arrhenius temperature factor — all encoded in our synthetic
  simulator at [`src/battery_pdm/synth/physics.py`](../src/battery_pdm/synth/physics.py).
- **Telecom NMS alarm taxonomies:** Ericsson, Nokia, Huawei NMS docs are the
  reference for the alarm codes we use.

## Reading next

- [02_DATA_AND_FEATURES.md](02_DATA_AND_FEATURES.md) — how the data is built and
  what features are computed
- [03_MODELS_AND_CHOICES.md](03_MODELS_AND_CHOICES.md) — why each algorithm was
  chosen and what alternatives were rejected
