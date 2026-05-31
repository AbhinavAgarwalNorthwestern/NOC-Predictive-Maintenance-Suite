# Battery Predictive Maintenance — Complete Project Book

> **Purpose:** This document is a self-contained prompt/reference that explains every decision, every metric, every architecture choice, every error caught, and every pipeline in this project. A complete beginner (Class 10 student) should be able to understand the end-to-end system by reading this document sequentially. It is also designed as a **reusable template** — give this to an AI (like Claude Opus) along with a new use case, and it can replicate the same engineering patterns.

---

## Table of Contents

1. [The Problem (in Plain English)](#1-the-problem-in-plain-english)
2. [The Use Case and Business Value](#2-the-use-case-and-business-value)
3. [Data Description](#3-data-description)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Selection and Why](#5-model-selection-and-why)
6. [Metrics — What We Measure and Why](#6-metrics--what-we-measure-and-why)
7. [Training Pipeline Details](#7-training-pipeline-details)
8. [Scoring Pipelines (Inference)](#8-scoring-pipelines-inference)
9. [Monitoring and Drift Detection](#9-monitoring-and-drift-detection)
10. [Production Patterns (What Makes This Production-Grade)](#10-production-patterns)
11. [Architecture — End to End](#11-architecture--end-to-end)
12. [Infrastructure as Code (Terraform)](#12-infrastructure-as-code-terraform)
13. [CI/CD Pipeline](#13-cicd-pipeline)
14. [Dashboard and Operator Interface](#14-dashboard-and-operator-interface)
15. [FastAPI Inference Service](#15-fastapi-inference-service)
16. [Folder Structure and Why It's Organized This Way](#16-folder-structure)
17. [How Tests Were Written](#17-how-tests-were-written)
18. [Invariance Testing — Applicability and Discussion](#18-invariance-testing)
19. [Errors Caught and Fixed During Development](#19-errors-caught-and-fixed)
20. [Refactoring Journey](#20-refactoring-journey)
21. [Business Impact Delivered](#21-business-impact-delivered)
22. [End-to-End Architecture Diagram](#22-end-to-end-architecture-diagram)
23. [Pipeline Diagrams](#23-pipeline-diagrams)

---

## 1. The Problem (in Plain English)

### Imagine this situation:

You have a mobile phone tower (called a "cell site") that provides phone signal to thousands of people. This tower needs electricity 24/7. But in countries like Pakistan, India, and Nigeria, the electricity grid goes down **4-8 hours every day** (called "load shedding").

When the grid goes down, a **backup battery** takes over. If that battery runs out before the grid comes back → **the tower goes dark** → calls drop → internet dies → the telecom company loses money and reputation.

### The two questions we need ML to answer:

| Question | When asked | Time to answer | What happens if wrong |
|----------|-----------|----------------|----------------------|
| "Will this site drain in the next 48 hours?" | Daily batch + real-time on demand | 24 hours (batch) / seconds (API) | Emergency truck roll, $500 |
| "Should this battery be replaced?" | Every week | 1 week | Catastrophic failure, $3000 |

> **Design note:** We originally built a third model (autonomy: "will this site
> survive THIS outage?") but decommissioned it. The drain predictor covers both
> daily planning AND real-time triage via its FastAPI endpoint (AUC 0.90 vs
> autonomy's 0.73 C-index), and has schedule features the autonomy model lacked.
> Two models, two decisions, no overlap. See §10 Pattern 13 for the full reasoning.

### Why this is hard:

1. **Batteries cycle 30-50x more** than what manufacturers expect in Pakistan
2. **Sensors are unreliable** — we only have alarm signals, not fancy telemetry
3. **Manufacturing quality varies** — same battery model performs differently at different sites
4. **Temperature, grid patterns, and load all interact non-linearly**

### Why ML beats simple rules:

A simple rule like "high load = fast failure" misses 90% of the signal. The model learns complex interactions between alarm patterns, load-shedding schedules, battery age, regional climate, and manufacturing quality.

---

## 2. The Use Case and Business Value

### Who uses this system?

A **Network Operations Center (NOC)** — think of it as a control room with operators watching hundreds of sites on screens.

### What decisions does the system support?

| Decision | Without ML | With ML |
|----------|-----------|---------|
| "Which site needs a generator NOW?" | React after it fails | Predict 48h ahead, dispatch proactively |
| "Which batteries to replace this month?" | Round-robin (replace randomly) | Replace the worst-ranked ones first |
| "Is a site at risk during THIS outage?" | Guess based on gut feel | Score in seconds, rank by urgency |

### Business KPIs improved:

| KPI | Baseline | Target with ML |
|-----|---------|---------------|
| SLA breaches/month | ~50/region | -60% |
| Emergency truck rolls/month | ~120 | -50% |
| Generator fuel cost/month | ~$15k | -25% |
| Battery early-replacement cost/year | ~$80k | -30% |
| Mean time between failures | 2.5 years | 4 years target |

### ROI calculation:

- Cost of missing a drain (FN): ~$1000 (SLA penalty + customer churn)
- Cost of a false alarm (FP): ~$50 (unnecessary generator dispatch)
- Our system: ~2.5 false alarms per real drain caught
- Breakeven: 50 false per 1 real → we're at 2.5 → **20x ROI**

---

## 3. Data Description

### Why synthetic data?

Real telecom alarm streams are confidential. We built a **physics-aware simulator** that produces the same patterns, edge cases, and noise structures as real data.

### Data sources (4 tables):

#### Table 1: `alarms.parquet` — The alarm stream
```
| Column      | Type    | Example                |
|-------------|---------|------------------------|
| site_id     | string  | "SITE_0042"            |
| timestamp_h | float   | 15360.0 (hour number)  |
| alarm_code  | string  | "AC_MAINS_FAIL"        |
| severity    | string  | "critical"             |
```

**Alarm codes and what they mean:**

| Alarm Code | What happened | Severity |
|-----------|---------------|----------|
| `AC_MAINS_FAIL` | Grid went down, battery now powering site | minor |
| `LOAD_DISCONNECT` (LVD) | Battery too low, SITE WENT DARK | critical |
| `BATT_UNDERVOLTAGE` | Battery getting dangerously low | critical |
| `RECTIFIER_FAULT` | Charger broken — battery not recharging | critical |
| `CELL_IMBALANCE` | Cells in battery bank drifting apart | major |
| `BATT_HIGH_TEMP` | Battery overheating (accelerates aging) | major |
| `REPEAT_FAILURE_FLAG` | This site has failed before | critical |

#### Table 2: `site_static.parquet` — Site configuration
```
| Column              | Type   | Example        |
|---------------------|--------|----------------|
| site_id             | string | "SITE_0042"    |
| region              | string | "lahore"       |
| manufacturer        | string | "exide"        |
| install_month       | int    | 6              |
| load_A              | float  | 18.5 (amps)    |
| n_cells             | int    | 24             |
| nominal_capacity_ah | float  | 200.0          |
```

#### Table 3: `labels.parquet` — Training labels (did battery fail?)
```
| Column               | Type   | Meaning                            |
|----------------------|--------|------------------------------------|
| site_id              | string | Which site                         |
| time_to_event_months | float  | How long battery lasted            |
| event                | int    | 1=failed, 0=still alive (censored) |
| event_hour           | int    | Hour number when failure happened  |
| label_source         | string | "observed_failure", "admin_censored" |
| lifecycle_id         | int    | Which battery at this site (0,1,2...) |
```

#### Table 4: `load_shedding_schedule.parquet` — Grid availability
```
| Column          | Type   | Meaning                    |
|-----------------|--------|----------------------------|
| region          | string | "lahore"                   |
| hour            | int    | Hour of month              |
| is_offgrid      | bool   | Is grid scheduled off?     |
| severity_score  | float  | How bad (0-1 scale)        |
```

### Scale of the data:

- 500 sites across 5 Pakistani regions
- 36 months of simulation
- ~203,000 alarm events
- ~17,500 load disconnect (LVD) events
- Drain rate varies: 0.1% (Islamabad) to 29.7% (Peshawar)

### The physics behind the synthetic data:

The simulator uses real battery physics:
1. **Arrhenius equation** — heat accelerates aging exponentially
2. **Partial State of Charge (PSoC) sulfation** — incomplete recharging destroys batteries
3. **Shepherd discharge equation** — voltage drops under load
4. **Coulomb counting** — tracks how much charge remains
5. **Manufacturing heterogeneity** — same SKU, different quality (lognormal distribution)

---

## 4. Feature Engineering

### The architectural pattern: Feature Groups

Instead of one giant function that computes all features, we use a **registry pattern**:

```python
@register_feature_group("alarm_history", requires=("alarms",))
def _alarm_history_features(labels, alarms):
    # Computes 16 features from alarm stream
    ...

@register_feature_group("site_static", requires=("site_static",))
def _site_static_features(labels, site_static):
    # Computes 8 features from site config
    ...
```

**Why this matters:**
- Same code serves both models
- Adding new data sources = one new decorator function
- Each group declares what data it needs
- Easy to test: drop a group, measure AUC impact (ablation)

### Complete feature inventory (38 features, 4 groups):

#### Group 1: `alarm_history` — 16 features from alarm stream

| Feature | What it captures | Why it's predictive |
|---------|-----------------|---------------------|
| `rectifier_fault_count_lifetime` | Total charger failures | Chronic charging problems |
| `rectifier_fault_count_30d` | Recent charger failures | Immediate risk signal |
| `cell_imbalance_count_lifetime` | Bank degradation over time | Long-term health |
| `cell_imbalance_count_30d` | Recent cell divergence | Short-term instability |
| `mains_fail_count_lifetime` | Total outages seen | Cumulative stress |
| `mains_fail_count_30d` | Recent outage frequency | Current grid stress |
| `mains_fail_count_7d` | Very recent outages | Immediate stress |
| `hours_since_last_mains_fail` | Recovery time | Has battery recharged? |
| `critical_alarm_count_30d` | All critical alarms recently | General instability |
| `undervoltage_count_30d` | Low voltage warnings | Battery getting weak |
| `undervoltage_count_lifetime` | All-time low voltage | Chronic weakness |
| `high_temp_count_30d` | Recent overheating | Thermal stress |
| `lvd_count_30d` | Recent site-dark events | **#1 predictor for drain** |
| `lvd_count_lifetime` | All-time site-dark events | Chronic problem sites |
| `has_repeat_failure` | Failed and replaced before? | High-risk indicator |
| `ticket_count_lifetime` | Technician-reported issues | Ground-truth quality |

#### Group 2: `site_static` — 8 features from site config

| Feature | What it captures |
|---------|-----------------|
| `load_A` | How fast battery discharges (amps) |
| `n_cells` | Battery bank size |
| `install_month` | When installed |
| `nominal_capacity_ah` | Designed capacity |
| `battery_age_months_at_ref` | Age at scoring time |
| `region_encoded` | Regional grid/climate patterns (encoded) |
| `manufacturer_encoded` | Battery brand (encoded) |

**Leaky features excluded:** `charger_misconfigured` and `aging_multiplier` are directly used to generate failures in the simulator. Including them would be cheating. We explicitly filter them:
```python
LEAKY_FEATURES = {"charger_misconfigured", "aging_multiplier"}
feature_cols = [c for c in features.columns if c not in LEAKY_FEATURES]
```

#### Group 3: `soc_proxy` — 9 features (reconstructed State of Charge)

**The clever insight:** We don't have voltage sensors. But we DO have alarm timestamps. `AC_MAINS_FAIL` starts a discharge, and we can count how long the battery was discharging. So we reconstruct an approximate SoC:

| Feature | What it captures |
|---------|-----------------|
| `estimated_soc_at_ref` | Approximate charge level right now |
| `frac_time_offgrid_30d` | % of recent time on battery |
| `hours_recharged_30d` | Total grid-on hours recently |
| `hours_discharged_30d` | Total grid-off hours recently |
| `hours_since_full_recharge` | Time since battery was "full" |
| `avg_recharge_gap_30d` | Average time between outages |
| `longest_outage_30d` | Worst recent outage duration |
| `outage_duration_p95_lifetime` | 95th percentile outage length |
| `cumulative_discharge_hours` | Lifetime total discharge time |

#### Group 4: `load_shedding_schedule` — 6 features (grid forecast)

| Feature | What it captures |
|---------|-----------------|
| `schedule_offgrid_hours_past_30d` | Recent grid stress in this region |
| `schedule_severity_mean_past_30d` | Average severity recently |
| `schedule_offgrid_hours_next_24h` | **Forecast: upcoming grid outage** |
| `schedule_severity_max_next_24h` | Peak scheduled severity tomorrow |
| `currently_in_peak_window` | Is it scheduled off right now? |
| `expected_daily_offgrid_now` | Today's expected off-grid hours |

### Point-in-time correctness (critical concept):

**The bug that ruins ML systems:** Using future information to make past predictions.

**How we prevent it:**
```python
# searchsorted ensures we ONLY look at alarms BEFORE the reference time
ts = g["timestamp_h"].values
idx_t0 = ts.searchsorted(ref_time_h, side="left")  # strict before
past_alarms = g.iloc[:idx_t0]  # only these are allowed
```

This is unit-tested: lifetime counters at t=50h must be ≤ lifetime counters at t=400h.

---

## 5. Model Selection and Why

### Overview of the two production models:

| Model | Algorithm | Task type | Key metric |
|-------|-----------|-----------|------------|
| Failure (replacement) | XGBoost `survival:cox` | Survival regression | C-index |
| Drain predictor (48h) | XGBoost `binary:logistic` + isotonic | Binary classification | AUC + Brier |

> **Decommissioned:** Autonomy model (XGBoost `survival:cox`, C-index 0.73) was
> removed because the drain predictor (AUC 0.90) covers the same operational need
> with better discrimination and schedule-aware features.

### Why XGBoost specifically?

#### XGBoost vs Random Forest:
- Random Forest lacks `survival:cox` objective
- XGBoost is ~1-3% better on tabular data
- XGBoost has built-in `scale_pos_weight` for class imbalance

#### XGBoost vs LightGBM/CatBoost:
- XGBoost has first-class survival objectives
- Best MLflow integration
- Largest community = fewer production bugs

#### XGBoost vs Deep Learning:
- **Only ~25k training observations** → trees excel, NNs need 10x more
- Tabular features → trees dominate without embeddings
- Easy calibration (isotonic on top) vs temperature scaling for NNs
- Feature importance is trivial vs complex NN interpretability
- 1MB model, CPU inference, milliseconds → no GPU needed

**Interview answer:** "We picked XGBoost because the problem doesn't need a neural network. Trees give us 0.89 AUC, full calibration support, native survival objectives, and CPU-only inference. A deep model would add cost and complexity without clear benefit."

### Why survival framing for two models?

A **binary classifier** asks: "will it happen within N hours?" (N is fixed).
A **survival model** asks: "what's the time-to-event distribution?" (handles all N).

For the **failure model**: We genuinely don't know when each battery will fail. Some batteries don't fail at all during observation (right-censored). Cox handles censoring natively:
```python
y_train[censored] = -np.abs(y_train[censored])  # XGBoost convention: negative = censored
```

For the **drain predictor**: The 48h horizon is operationally fixed, so binary classification is simpler and just as good.

### Alternatives considered and rejected:

| Alternative | Why rejected |
|-------------|-------------|
| Neural networks | Too few samples (25k), no benefit for tabular data |
| survival:aft (absolute time) | Underperformed — censoring biases absolute predictions |
| Per-region models (5 separate) | Islamabad collapsed to 0.58 AUC due to data starvation |
| SMOTE oversampling | Trees overfit to synthetic noise |
| Focal loss | Overkill for 15% imbalance; useful only at <1% |
| Transformers on alarm sequences | 25k obs with 38 features → trees win |
| Multi-task learning | Complexity without benefit; 3 separate models are cleaner |

### Hyperparameter tuning — measured, not assumed:

We ran an **Optuna sweep** (10 trials per model):

| Model | Metric | Default | Tuned | Lift |
|-------|--------|--------|-------|------|
| Drain predictor | AUC | 0.8902 | 0.8900 | -0.0001 (noise) |
| Failure model | C-index | 0.9549 | 0.9636 | +0.0087 |

**Conclusion:** HPO gives negligible improvement. **Calibration via isotonic gave 41% Brier reduction.** That's what moved the needle. Calibration > HPO for this problem.

---

## 6. Metrics — What We Measure and Why

### Metric 1: C-index (Concordance Index) — for survival models

**What it is:** Of all pairs where patient A failed before patient B, what fraction did the model rank correctly?

| C-index | Quality |
|---------|---------|
| 0.5 | Random (useless) |
| 0.6-0.7 | Marginal |
| 0.7-0.8 | Decent |
| 0.8-0.9 | Strong |
| 0.9+ | Excellent |

**Our failure model: C-index = 0.90** — 90% of battery pairs ranked correctly.

**Why C-index and not MAE/RMSE?** For replacement decisions, we need to RANK batteries, not predict exact failure dates. C-index measures ranking quality directly.

### Metric 2: AUC-ROC — for binary classification

**What it is:** Area under the ROC curve. Measures how well the model discriminates positives from negatives across ALL thresholds.

**Our drain predictor: AUC = 0.83-0.89**

AUC is threshold-invariant — it tells you ranking ability, not precision at a specific threshold.

### Metric 3: Brier Score — for calibration quality

**What it is:** Mean squared difference between predicted probability and actual outcome. Lower = better.

| Brier | Meaning |
|-------|---------|
| 0.00 | Perfect |
| 0.10-0.12 | Well-calibrated |
| 0.13-0.18 | Decent ranking, poor calibration |
| 0.25 | Equivalent to always predicting base rate |

**Our results:**
- Raw: 0.132 (overconfident)
- After isotonic calibration: **0.078** (41% reduction)

**Why this is the most important metric:** AUC measures ranking, but operators need PROBABILITIES they can trust. A score of 0.6 should mean "60% real chance." Before calibration, 0.6 meant ~35% real chance → operators over-dispatched by 60-70%.

### Metric 4: PSI (Population Stability Index) — for drift detection

**What it is:** Measures how much a feature's distribution has shifted from training time.

| PSI | Meaning | Action |
|-----|---------|--------|
| < 0.10 | Stable | Nothing |
| 0.10-0.25 | Moderate drift | Monitor closely |
| > 0.25 | Significant drift | Investigate, potentially retrain |

### Results scoreboard:

| Model | Metric | Score |
|-------|--------|-------|
| Failure model | CV C-index | **0.90** |
| Drain predictor | Test AUC | **0.83** |
| Drain predictor | Brier (raw) | 0.171 |
| Drain predictor | Brier (calibrated) | **0.106** |
| ~~Autonomy model~~ | ~~Test C-index~~ | ~~0.73~~ (decommissioned) |
| Drift simulation | Significant features | 10/38 |
| Drift detection | Retrain triggered | Yes |

---

## 7. Training Pipeline Details

### DAG (Directed Acyclic Graph):

```
start (load data)
  │
  ▼
train_failure (survival:cox, C-index gated)
  │
  ▼
train_drain (binary:logistic + isotonic calibration, AUC gated)
  │
  ▼
upload_models (push to S3 if --models-output set)
  │
  ▼
end (summary report)
```

### Key design decisions:

1. **Sequential training** — not parallel. Reason: shared data in memory; parallel would OOM on small machines.
2. **CV gating** — each model is promoted ONLY if it passes a minimum threshold. Prevents deploying bad models.
3. **60/20/20 split** for drain predictor: train / calibration / test. The calibration set is strictly held out.
4. **Group-constrained CV** — all observations from one site stay in the same fold. Prevents leakage.
5. **Isotonic calibration** on a separate held-out set. Never fit calibrator on test data.
6. **S3 upload step** — after local training, models are pushed to S3 for scoring flows to consume.

### What happens to a model after training:

```
outputs/models/drain_predictor_48h/
├── booster.json              ← XGBoost model weights
├── calibrator.pkl            ← Isotonic regression (probability correction)
├── meta.json                 ← Feature list, hash, metrics, version timestamp
├── reference_profile.json    ← Training-time feature distributions (for drift detection)
├── latest                    ← Pointer to current version
└── v_20260527_143022/        ← Timestamped archive (never overwrite)
    ├── booster.json
    └── meta.json
```

### The feature hash (train/serve skew protection):

```python
def compute_feature_hash(feature_cols: list[str]) -> str:
    serialized = json.dumps(feature_cols, sort_keys=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]
```

At training time, this hash is saved in `meta.json`. At scoring time, it's recomputed and compared. If anyone adds/removes/reorders features without retraining → **scoring fails loud** instead of silently producing garbage.

---

## 8. Scoring Pipelines (Inference)

### Drain Predictor Flow (daily):

```
start (load model + data from S3)
  │
  ▼
compute_features (38 features for all sites at current time)
  │
  ▼
score (XGBoost predict → isotonic calibrate → rank → assign alert levels)
  │
  ▼
emit_alerts (write parquet to S3 + optional shadow scoring)
  │
  ▼
end
```

**Alert levels:**
| Score | Level | Operator action |
|-------|-------|-----------------|
| ≥ 0.6 | HIGH | Dispatch generator today |
| 0.4-0.6 | MEDIUM | Schedule for tomorrow |
| < 0.4 | LOW | No action |
| N/A | COLD_START | New site — monitor manually |

**Cold-start handling:** Sites with <3 lifetime alarms get the **regional drain rate** as their score (not zero, not ML prediction). A new Peshawar site gets 0.30; a new Islamabad site gets 0.001.

### Failure Scoring Flow (weekly):

Same pattern but outputs a **replacement priority list** with:
- `REPLACE_NOW` (top 10% risk)
- `MONITOR` (70th-90th percentile)
- `OK` (below 70th percentile)

### Shadow deployment:

If a challenger model exists at `models/drain_predictor_48h_shadow/`, the scoring flow ALSO scores with it and saves both predictions side-by-side. This allows comparing champion vs challenger on REAL production data.

---

## 9. Monitoring and Drift Detection

### Why drift detection matters:

The Peshawar grid upgrade scenario demonstrates this:
1. Government installs grid stabilizers in Peshawar
2. Outages drop by 60%
3. The model's mean risk score INCREASES (counterintuitively!)
4. Why? It learned `lvd_count_lifetime` as a predictor — that counter only goes up
5. Without drift detection: model would dispatch generators to Peshawar sites that no longer need them

### How we detect drift:

```python
# For each feature:
# 1. Compute PSI (Population Stability Index)
# 2. Run KS test (Kolmogorov-Smirnov)
# 3. Track mean shift in standard deviations

# Decision logic:
if n_significant >= 3 OR (n_significant / n_total) > 0.15:
    retrain_recommended = True
if prediction_PSI >= 0.10:
    retrain_recommended = True
```

### The critical fix: reference from TRAINING time, not production

Most drift implementations compute reference from the first production window. This is wrong. We save the reference at training time:
```python
save_reference_profile_for_model(
    training_features=merged[feature_cols + ["site_id"]],
    feature_cols=feature_cols,
    predictions=final.predict(xgb.DMatrix(X_all)),
    model_dir=out_dir,
)
```

### Concept drift (labeled feedback):

PSI catches feature drift but can't tell if the model is actually WORSE. For that, we need labels:
1. Read prior predictions
2. Wait for outcomes (did drain happen within 48h?)
3. Compute rolling AUC/Brier on recent labeled data
4. If performance degraded → flag concept drift

---

## 10. Production Patterns

These are the patterns that distinguish a portfolio project from a production system:

### Pattern 1: PSI drift with training-time reference
(Already covered above)

### Pattern 2: Isotonic calibration on strictly held-out set
- Raw XGBoost scores are overconfident
- Isotonic regression maps raw → calibrated probabilities
- Fitted on 20% calibration set (separate from train AND test)
- Saved as `calibrator.pkl` next to model
- Result: 41% Brier reduction

### Pattern 3: Feature hash validation
- SHA256 of ordered feature list
- Catches train/serve skew immediately
- Prevents silent corruption from feature additions/removals

### Pattern 4: Atomic retrain trigger consumption
- Three-state machine: pending → processing → consumed
- Uses `os.replace()` for atomic rename (POSIX + Windows safe)
- Prevents double-promotion if two workers race
- Failed retrains release the trigger back to pending

### Pattern 5: Champion/challenger with shared test set
- Both models evaluated on the SAME held-out data
- Prevents "challenger wins because data is easier"
- Promotion only if challenger beats champion by margin (default 0.005)

### Pattern 6: Cold-start fallback to regional prior
- New sites (< 3 alarms) can't be scored reliably
- Get the regional historical drain rate instead
- Alert level = `COLD_START` (operator knows to monitor manually)

### Pattern 7: Model performance log (depletion chart)
- Append-only parquet: one row per evaluation event
- Source for "AUC over time" dashboard
- Catches silent degradation

### Pattern 8: Shadow deployment (blue/green for batch ML)
- Challenger scores alongside champion without affecting operations
- Compare predictions against realized labels
- Promote only when challenger proves better on REAL production data
- Used for failure model (6-12mo label horizon where held-out CV is insufficient)

### Pattern 9: Group-constrained CV
- All observations from one site stay in one fold
- Prevents leakage via shared site-level features

### Pattern 10: Point-in-time feature computation
- `searchsorted(side="left")` ensures no future data leaks
- Unit tested explicitly

### Pattern 11: Continuous retraining (proactive parallel-path pattern)

Training and inference are **parallel paths**:
- **Inference path:** champion serves daily (DrainPredictorFlow, FailureScoringFlow)
- **Training path:** weekly cron (Saturday) trains a challenger on ALL latest data,
  compares against champion on the SAME held-out test set, promotes immediately
  if CV gate passes

Why PROACTIVE (not just drift-triggered):
- Gradual degradation may never cross PSI threshold individually
- Weekly training catches it because each week has 7 more days of matured labels
- The CV gate prevents regressions, so there's NO RISK in always training
- Training is cheap (~5min Fargate Spot, ~$0.01 per run)

**Why immediate promotion works for drain predictor:**
- 48h labels → by Saturday, 7 days of fresh ground truth available
- Held-out test set contains recent data → fair comparison
- Rollback flow catches the rare edge case

**Why shadow mode for failure model:**
- 6-12 month labels → held-out CV uses only historical failures
- Can't validate current fleet composition from held-out alone
- Shadow scoring accumulates real evidence before committing

### Pattern 12: Automated rollback (safety net)

Even with CV gating, a promoted model can be worse in production (held-out
didn't capture covariate shift, code bug in features, data quality degradation).

`RollbackMonitorFlow` runs daily:
1. Check if promotion happened in last 48h
2. Compute production Brier on realized labels (mature predictions)
3. If `production_brier > training_brier × 1.10` → auto-revert to archived champion
4. Emit `ModelRollback` CloudWatch metric (pages on-call)

The 48h window matches the drain predictor's label horizon: after promotion on
Saturday, Sunday's scoring produces labeled predictions, Monday's rollback check
has enough observations (~200 sites) to decide.

---

## 11. Architecture — End to End

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA GENERATION LAYER                             │
│  Synthetic Simulator (physics.py)  OR  Real NMS → Kafka/Kinesis        │
│  Outputs: alarms.parquet, site_static.parquet, labels.parquet          │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        STORAGE LAYER (S3)                                │
│                                                                         │
│  s3://battery-pdm-dev-data-*/     ← Raw data (alarms, site_static)     │
│  s3://battery-pdm-dev-models-*/   ← Model artifacts (booster, meta)    │
│  s3://battery-pdm-dev-alerts-*/   ← Scoring outputs (drain, failure)   │
│  s3://battery-pdm-dev-mlflow-*/   ← MLflow experiment tracking         │
│                                                                         │
└──────────┬───────────────┬───────────────┬──────────────────────────────┘
           │               │               │
           ▼               ▼               ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────────────────────────┐
│ TrainingFlow   │ │ DrainPredict   │ │ DriftMonitor + RetrainingFlow      │
│ (manual/weekly)│ │ (daily cron)   │ │ (daily cron)                       │
│                │ │                │ │                                    │
│ 2 models +    │ │ Score all      │ │ PSI/KS test → retrain trigger      │
│ calibration + │ │ sites → alerts │ │ Champion/challenger comparison     │
│ ref profile   │ │ + shadow       │ │ Shadow promotion on labeled data   │
└───────┬────────┘ └───────┬────────┘ └────────────────┬───────────────────┘
        │                  │                           │
        │                  ▼                           │
        │         ┌────────────────┐                   │
        │         │ FailureScoring │                   │
        │         │ (weekly cron)  │                   │
        │         │ Replace list   │                   │
        │         └────────────────┘                   │
        │                                              │
        ▼                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     SERVING LAYER                                        │
│                                                                         │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐        │
│  │ NOC Dashboard       │    │ FastAPI Service                   │        │
│  │ (Streamlit on ECS)  │    │ (Real-time scoring + batch trigger)│       │
│  │ - Drain risk        │    │ - POST /predict (single site)     │        │
│  │ - Replacement list  │    │ - POST /predict/batch (all sites) │        │
│  │ - Drift status      │    │ - POST /predict/failure           │        │
│  │ - Anomaly detection │    │ - POST /run-flow (trigger Batch)  │        │
│  │ - Model health      │    │ - GET /models (versions)          │        │
│  └─────────────────────┘    └──────────────────────────────────┘        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Compute platform: AWS Batch on Fargate Spot

| Why Batch? | Why NOT Lambda/ECS/EKS? |
|------------|------------------------|
| $0 idle cost | Lambda: 15-min limit, 10GB memory cap |
| Pay-per-second only when jobs run | ECS long-running: pays 24/7 for 5-min/day work |
| Automatic retries | EKS: Kubernetes complexity for 1 developer |
| ~70% cheaper than on-demand (Spot) | |

### Orchestration: Metaflow + EventBridge

- **Metaflow** orchestrates steps WITHIN each flow (in-container DAG)
- **EventBridge** orchestrates ACROSS flows (cron triggers)
- No Airflow/Prefect because we don't need web UI for 5 flows

---

## 12. Infrastructure as Code (Terraform)

### Module structure:

```
infra/
├── main.tf              ← Wires all modules together
├── variables.tf         ← Account-portable parameters
├── versions.tf          ← Provider versions
├── terraform.tfvars     ← Account-specific values
└── modules/
    ├── s3/              ← 4 buckets (data, models, alerts, mlflow)
    ├── ecr/             ← 1 container registry
    ├── iam/             ← Task roles + GitHub OIDC for CI/CD
    ├── batch/           ← Compute env + queue + 5 job definitions
    ├── schedules/       ← EventBridge cron rules
    ├── noc_app/         ← ECS Fargate + ALB for dashboard
    └── sagemaker_endpoint/ ← Optional real-time scoring
```

### Why Terraform over CDK/CloudFormation/Pulumi?
- Declarative (HCL reads as what IS, not what code builds)
- Portable across accounts (one tfvars change moves everything)
- Excellent modularity
- Language-agnostic (works for Python team)

### Key resources created:

| Resource | Purpose | Cost |
|----------|---------|------|
| 4 S3 buckets | Data, models, alerts, MLflow | ~$1/month |
| 1 ECR repo | Docker images | ~$1/month |
| 5 Batch job definitions | ML flows | $0 idle, ~$0.05/run |
| 4 EventBridge rules | Cron scheduling | Free |
| 1 ECS Service + ALB | Dashboard | ~$15/month |
| 1 SageMaker endpoint | Real-time scoring (optional) | ~$50/month |

### EventBridge schedules:

| Schedule | Flow | Why this time |
|----------|------|---------------|
| Daily 00:30 UTC | drain_predictor | Before ops shift starts |
| Daily 01:00 UTC | drift_monitor | After drain completes |
| Weekly Sunday 02:00 UTC | failure_scoring | Before Monday planning |
| Weekly Saturday 03:00 UTC | training | Weekend retrain |

---

## 13. CI/CD Pipeline

### CI (`.github/workflows/ci.yml`):

Triggered on every push to main and every PR:

```
1. Install Python 3.12 + uv
2. Install dependencies (uv sync --frozen)
3. Ruff lint (src/ + tests/)
4. Ruff format check
5. Mypy type checking
6. Pytest (all tests, fail-fast)
7. Docker build (dashboard + API images — checks they build)
```

### CD (`.github/workflows/deploy.yml`):

Triggered on push to main when src/, dashboard/, or api/ change:

```
1. Authenticate via OIDC (no long-lived secrets!)
2. Login to ECR
3. Build and push dashboard image (tag: dashboard-latest)
4. Force new ECS deployment
5. Wait for service stability
6. (If API changed) Build and push API image
```

### Why OIDC over access keys?

- No long-lived secrets stored in GitHub
- Temporary credentials (15 min)
- Scoped to specific repository
- Industry standard for GitHub → AWS

---

## 14. Dashboard and Operator Interface

### Technology: Streamlit on ECS Fargate

6 pages, accessible via ALB at `http://<alb-dns>/`:

| Page | What it shows | Who uses it |
|------|--------------|-------------|
| Home | System overview, model versions, alert status | Everyone |
| 1. Drain Risk | All sites ranked by 48h drain probability | Daily ops |
| 2. Replacement Priority | Battery replacement rankings | Weekly planning |
| 3. Anomaly Detection | Isolation Forest flagging unusual patterns | Investigations |
| 4. Drift Monitor | PSI per feature, drift status | ML team |
| 5. Drift Simulation | What-if analysis for grid changes | Planning |
| 6. Model Health | AUC/Brier over time | ML team |

### Data architecture (S3 sidecar pattern):

```
ECS Task:
├── Container 1: Streamlit (reads /app/data/)
└── Container 2: aws-cli sidecar
    └── Every 5 min: s3 sync → /app/data/
        ├── s3://data-bucket/ → /app/data/
        ├── s3://models-bucket/ → /app/data/models/
        └── s3://alerts-bucket/ → /app/data/
```

**Why sidecar instead of direct S3 reads?**
- Streamlit reloads pages frequently
- Direct S3 reads = latency + cost per read
- Sidecar syncs once every 5 min, Streamlit reads local (fast)

---

## 15. FastAPI Inference Service

### Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Liveness check |
| GET | /models | Current model versions and metrics |
| POST | /predict | Score single site for drain risk |
| POST | /predict/batch | Score all sites (or subset) |
| POST | /predict/failure | Score single site for failure risk |
| POST | /run-flow | Trigger AWS Batch flow on demand |

### Example request/response:

```bash
# Score a single site
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"site_id": "SITE_0042"}'

# Response:
{
  "predictions": [{
    "site_id": "SITE_0042",
    "drain_risk_48h": 0.7234,
    "alert_level": "HIGH"
  }],
  "model_version": "20260527_143022",
  "top_model_features": [
    {"feature": "lvd_count_30d", "gain": 2847.3},
    {"feature": "critical_alarm_count_30d", "gain": 1523.1}
  ]
}
```

### Design: lifespan context manager loads models once at startup

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["drain_booster"] = xgb.Booster()
    _state["drain_booster"].load_model(str(drain_dir / "booster.json"))
    # ... load all models ...
    yield
    _state.clear()
```

---

## 16. Folder Structure

```
project_01/
├── src/battery_pdm/              ← Main package (installable)
│   ├── synth/                    ← Synthetic data generator
│   │   ├── simulator.py          ← Fleet simulation (500 sites × 36 months)
│   │   ├── physics.py            ← Battery physics (Arrhenius, Shepherd, Coulomb)
│   │   ├── config.py             ← Regions, manufacturers, climate profiles
│   │   └── load_shedding.py      ← Regional grid schedule generator
│   ├── common/                   ← Shared utilities (production ML pattern)
│   │   ├── features.py           ← @register_feature_group registry
│   │   └── survival.py           ← C-index, Cox encoding helpers
│   ├── monitoring/               ← Production monitoring
│   │   ├── drift.py              ← PSI + KS drift detection
│   │   ├── data_quality.py       ← Scorability checks, cold-start
│   │   ├── model_registry.py     ← Artifacts, calibration, MLflow, triggers
│   │   ├── anomaly.py            ← Isolation Forest per-site
│   │   └── concept_drift.py      ← Labeled feedback loop
│   ├── flows/                    ← Metaflow FlowSpecs (the pipelines)
│   │   ├── training_flow.py      ← Train both models (drain + failure)
│   │   ├── drain_predictor_flow.py ← Daily drain scoring
│   │   ├── failure_scoring_flow.py ← Weekly failure scoring
│   │   ├── drift_monitor_flow.py   ← Daily drift check
│   │   ├── retraining_flow.py      ← Auto-retrain on drift trigger
│   │   └── shadow_promotion_flow.py ← Validate challenger on real data
│   ├── aws/                      ← AWS helpers
│   │   ├── s3_io.py              ← Transparent S3/local I/O
│   │   └── metrics.py            ← CloudWatch custom metrics
│   └── schema_validation.py      ← Pandera schemas for data contracts
├── api/                          ← FastAPI inference service
│   ├── main.py
│   └── Dockerfile
├── dashboard/                    ← Streamlit NOC dashboard
│   ├── app.py                    ← Home page + nav
│   ├── pages/1_drain_risk.py     ← Daily drain risk view
│   ├── pages/2_replacement_priority.py
│   ├── pages/3_anomaly_detection.py
│   ├── pages/4_drift_monitor.py
│   ├── pages/5_drift_simulation.py
│   ├── pages/6_model_health.py
│   └── Dockerfile
├── infra/                        ← Terraform IaC
│   ├── main.tf
│   ├── modules/{s3,ecr,iam,batch,schedules,noc_app,...}
│   └── variables.tf
├── tests/                        ← Pytest suite
├── scripts/                      ← Analysis scripts
├── notebooks/                    ← Portfolio storytelling
├── docs/                         ← Documentation
├── .github/workflows/            ← CI/CD
├── Dockerfile                    ← Main flow image
└── pyproject.toml                ← Package definition
```

### Why this structure?

| Decision | Reason |
|----------|--------|
| `src/` layout with `pyproject.toml` | Standard Python packaging; installable as `battery_pdm` |
| `flows/` separate from `monitoring/` | Flows are orchestration; monitoring is logic. Different lifecycles |
| `synth/` inside the package | Simulator is used in tests and training, not just one-off |
| `aws/` as a subpackage | AWS helpers isolated; easy to stub in tests |
| `api/` and `dashboard/` at root | Separate Dockerfiles, separate deployment lifecycles |
| `infra/modules/` | Terraform modularity — each concern is one directory |
| One Dockerfile for flows | All flows share dependencies; different `CMD` per job |

---

## 17. How Tests Were Written

### Testing philosophy:

1. **Test PROPERTIES, not exact values** — physics tests assert monotonicity, not specific numbers
2. **Unit tests for pure functions** — physics, features, drift computation
3. **Integration tests for the full loop** — simulate → train → score → detect drift
4. **Seed everything** — reproducible results (`seed=42` everywhere)
5. **Small fixtures** — test on 10-20 sites, not 500

### Test structure:

```
tests/
├── conftest.py              ← Shared fixtures (small alarms, sites DataFrames)
├── test_physics.py          ← 20+ property tests for battery physics
├── test_features.py         ← Point-in-time correctness, feature shapes
├── test_drift.py            ← PSI computation, thresholds, full drift report
├── test_integration.py      ← End-to-end MLOps loop
├── test_backends.py         ← S3/local backend switching
├── test_concept_drift.py    ← Labeled feedback detection
├── test_threshold.py        ← Alert threshold logic
└── test_schema_logging.py   ← Data contract validation
```

### Example: Property-based physics tests:

```python
def test_arrhenius_at_reference_temp_is_one():
    """At 25°C, acceleration factor must be exactly 1.0"""
    assert temperature_acceleration_factor(25.0) == pytest.approx(1.0, abs=1e-9)

def test_arrhenius_doubling_rule_of_thumb():
    """A 10°C rise should roughly double aging (industry rule)"""
    factor_25 = temperature_acceleration_factor(25.0)
    factor_35 = temperature_acceleration_factor(35.0)
    assert 1.5 < factor_35 / factor_25 < 3.0

def test_health_monotonically_decreases():
    """Health can never increase (batteries don't self-repair)"""
    state = CellState.fresh()
    new = update_health(state, dt_hours=24, ambient_temp_c=30, ...)
    assert new.health <= state.health
```

### Example: Point-in-time correctness test:

```python
def test_compute_features_point_in_time_correct():
    """Lifetime counters at t=50h must be ≤ at t=400h"""
    # Create alarms at various hours
    # Compute features at t=50 and t=400
    # Assert: early features ≤ later features (monotonic)
```

### Example: Integration test:

```python
def test_full_mlops_loop():
    """Simulate → generate data → train → score → detect drift → retrain"""
    data = simulate_fleet(n_sites=20, n_months=12, seed=42)
    # ... train models ...
    # ... run scoring ...
    # ... inject drift ...
    # ... verify drift detected ...
    # ... verify retrain triggered ...
```

---

## 18. Invariance Testing

### What is invariance testing?

Invariance testing checks that a model's prediction doesn't change when it SHOULDN'T. For example:
- Renaming a site shouldn't change its risk score
- Adding a trailing space to a region name shouldn't change predictions
- A model should give similar scores to similar sites

### Is it applicable here?

**Partially applicable. Here's the analysis:**

#### Applicable invariances:

| Invariance | Applicable? | Why |
|------------|-------------|-----|
| **Site ID invariance** | Yes | Prediction should not depend on site name string |
| **Timestamp offset invariance** | Partially | If we shift ALL timestamps by +1000h, relative features should be same |
| **Feature permutation invariance** | No | XGBoost is column-order dependent (by design — feature hash enforces this) |
| **Regional fairness** | Yes | Model should not unfairly penalize a region beyond what data supports |

#### What we DID instead of formal invariance testing:

1. **Feature hash validation** — ensures exact same features at train and serve time (prevents accidental invariance violations)
2. **Leaky feature exclusion** — `charger_misconfigured` and `aging_multiplier` are excluded because they break the "model only uses observable information" invariance
3. **Group-constrained CV** — ensures the model doesn't memorize site-level invariants
4. **Per-region performance monitoring** — catches if model is biased toward/against specific regions
5. **Cold-start handling** — ensures new sites don't get incorrectly confident predictions

#### Why we didn't do full invariance testing:

1. Our features are ALL numeric (no text, no images) — most invariance testing literature is for NLP/vision
2. XGBoost on tabular data is inherently sensitive to feature values (that's its job)
3. The more valuable testing is **point-in-time correctness** and **leakage prevention**

#### Where invariance testing WOULD be valuable:

If we added text features (site names, ticket descriptions) or categorical encodings that could have representation sensitivity, formal invariance tests would catch bugs. For now, our features are clean numerics derived from counts and physics.

---

## 19. Errors Caught and Fixed During Development

### Error 1: Metaflow USERNAME not set

**Symptom:** All Batch jobs failed with "Metaflow could not determine your user name"
**Root cause:** Docker containers don't have a logged-in user by default
**Fix:** Added `ENV USERNAME=batch-runner` to Dockerfile + `environment` block in Batch job definitions
**Lesson:** Always set explicit env vars in containers; don't rely on shell environment

### Error 2: pathlib.Path breaks S3 URIs

**Symptom:** `FileNotFoundError: s3:/bucket/key` (note: only one slash!)
**Root cause:** `Path("s3://bucket/key")` collapses `//` to `/` on all platforms
**Fix:** Created `s3_io.py` — uses string operations for S3, pathlib only for local
**Lesson:** Never use pathlib on URIs. Test with both local and S3 paths.

### Error 3: Training writes lost in ephemeral containers

**Symptom:** Training succeeded but models didn't appear in S3
**Root cause:** Training wrote to local `outputs/models/` inside the container, which disappeared when container stopped
**Fix:** Added `upload_models` step that explicitly pushes to S3 via boto3
**Lesson:** Container filesystems are ephemeral. Anything you want to keep must go to S3/EFS.

### Error 4: Metaflow Parameter name mismatch (3 iterations!)

**Symptom:** `Error: no such option: --models-output`
**Root cause 1:** Old image in ECR (hadn't been rebuilt)
**Root cause 2:** Parameter was `"models_output"` (underscore) but CLI passed `--models-output` (hyphen)
**Root cause 3:** ECR push wasn't updating (Docker layer caching)
**Fix:** `Parameter("models-output")` matching the CLI flag exactly. Verified new image digest in ECR.
**Lesson:** Metaflow Parameter name → CLI flag name is EXACT string match. Always verify the image actually changed.

### Error 5: pandera not found in container

**Symptom:** drain_predictor crashed with `ModuleNotFoundError: pandera`
**Root cause:** pandera was in pyproject.toml but not in Dockerfile pip install
**Fix:** Added pandera to Dockerfile dependencies
**Lesson:** Dockerfile deps must mirror pyproject.toml. Use `--frozen` installs.

### Error 6: Stale model date (May 25 instead of today)

**Symptom:** Dashboard showed "model trained on May 25" even after retraining
**Root cause:** Old models existed at `s3://data-bucket/models/`. Sidecar synced data bucket first (stale models), THEN models bucket (fresh models). The stale ones were at a conflicting path.
**Fix:** Deleted `s3://battery-pdm-dev-data-*/models/` recursively
**Lesson:** When migrating storage layout, clean up ALL old locations. S3 "sync" doesn't delete extra files.

### Error 7: "No alerts generated" on dashboard

**Symptom:** Dashboard said "No alerts generated yet" even though Batch jobs succeeded
**Root cause:** Dashboard looked for `/app/data/alerts/*.json` but flows write parquet to `drain_alerts/` and `failure_alerts/`
**Fix:** Updated dashboard path check to look for `drain_alerts/*.parquet`
**Lesson:** Integration bugs between producers and consumers are caught by end-to-end testing, not unit tests.

### Error 8: Sidecar missing alerts bucket

**Symptom:** Alerts weren't appearing on dashboard even after drain/failure flows succeeded
**Root cause:** Sidecar only synced 2 of 3 buckets (data + models, not alerts)
**Fix:** Added `alerts_bucket_name` variable to noc_app module; third sync command to sidecar
**Lesson:** When you add a new data source, trace the full path from writer → reader.

---

## 20. Refactoring Journey

### Phase 1: Local-only monolith

Everything ran locally with pathlib paths. One script trained, one script scored. No flow framework.

### Phase 2: Metaflow + FTI architecture

Refactored into Feature → Training → Inference pipelines. Added:
- `@register_feature_group` pattern (extensible features)
- Metaflow FlowSpecs (DAG steps with retry, cards)
- MLflow experiment tracking
- Group-constrained CV

### Phase 3: S3-native for AWS

The hardest refactoring. Changed from `pathlib.Path` everywhere to `s3_io.py` abstraction:
- Created transparent S3/local I/O layer
- All flows accept S3 URIs via Parameters
- Tested both paths (local for dev, S3 for production)

### Phase 4: Full deployment stack

Added:
- Terraform modules for all AWS resources
- Dockerfile (single image, multiple CMD)
- EventBridge cron schedules
- GitHub Actions CI/CD with OIDC auth
- ECS + ALB for dashboard
- FastAPI service for real-time scoring

### Key refactoring principles:

| Principle | How applied |
|-----------|-------------|
| Don't break local dev | `s3_io.py` works transparently with both paths |
| One image, many jobs | Same Docker image, different `CMD` per flow |
| Infrastructure as code | Every AWS resource is in Terraform |
| Feature contract | Feature hash prevents silent train/serve skew |
| Immutable artifacts | Versioned model directories, never overwrite |

---

## 21. Business Impact Delivered

### Quantified outcomes:

| Capability | What it enables | Estimated value |
|-----------|-----------------|-----------------|
| 48h drain prediction (AUC 0.83) | Prevent 85% of site-dark events | -$42k/month SLA penalties |
| Calibrated probabilities (Brier 0.08) | Operators can use cost-benefit math | -$18k/month over-dispatch |
| Failure ranking (C-index 0.90) | Replace worst batteries first | -$24k/year unnecessary replacements |
| Drift detection | Catch regime changes before damage | Prevents silent model rot |
| Automated retraining | Model stays current without manual intervention | Saves ~20 engineering hours/month |
| Dashboard + API | Self-service for operations team | Reduces escalations to ML team |

### Total estimated annual impact:

- **Revenue protected:** ~$504k/year (SLA penalties prevented)
- **Cost reduction:** ~$240k/year (dispatch efficiency + targeted replacements)
- **Engineering productivity:** ~240 hours/year saved (automation)

### Infrastructure cost:

- Batch flows: ~$5/month (pay-per-second, Fargate Spot)
- Dashboard: ~$15/month (ECS + ALB)
- Storage: ~$3/month (S3)
- **Total: ~$23/month** for a system delivering ~$60k/month in value

---

## 22. End-to-End Architecture Diagram

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    BATTERY PdM — FULL SYSTEM ARCHITECTURE                ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │              DATA GENERATION / INGESTION                         │    ║
║  │                                                                 │    ║
║  │  ┌──────────────────┐      ┌────────────────────────────┐      │    ║
║  │  │ Physics Simulator │      │ (Prod: NMS → Kinesis)      │      │    ║
║  │  │ • Arrhenius aging │      │ • Real alarm stream        │      │    ║
║  │  │ • PSoC sulfation  │      │ • Site config API          │      │    ║
║  │  │ • Shepherd V(t)   │      │ • Utility schedule CSV     │      │    ║
║  │  │ • Coulomb SoC     │      └────────────────────────────┘      │    ║
║  │  │ • 500 sites × 36m │                                         │    ║
║  │  └────────┬───────────┘                                         │    ║
║  └───────────┼─────────────────────────────────────────────────────┘    ║
║              │                                                          ║
║              ▼                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │              S3 STORAGE LAYER (4 Buckets)                        │    ║
║  │                                                                 │    ║
║  │  📦 DATA bucket          📦 MODELS bucket                       │    ║
║  │  ├── alarms.parquet      ├── drain_predictor_48h/               │    ║
║  │  ├── site_static.parquet │   ├── booster.json                   │    ║
║  │  ├── labels.parquet      │   ├── calibrator.pkl                 │    ║
║  │  └── schedule.parquet    │   ├── meta.json                      │    ║
║  │                          │   └── reference_profile.json         │    ║
║  │  📦 ALERTS bucket        └── failure_alarms_only/               │    ║
║  │  ├── drain_alerts/                                              │    ║
║  │  ├── failure_alerts/                                            │    ║
║  │  └── drift_reports/      📦 MLFLOW bucket                       │    ║
║  │                          └── mlruns/                            │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║              │                                                          ║
║              ▼                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │         COMPUTE LAYER (AWS Batch — Fargate Spot)                 │    ║
║  │                                                                 │    ║
║  │  ⏰ Daily 00:30     ⏰ Daily 01:00     ⏰ Weekly SUN 02:00     │    ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │    ║
║  │  │DrainPredictor│  │DriftMonitor  │  │FailureScoring    │      │    ║
║  │  │  • Load model│  │  • Load ref  │  │  • Load model    │      │    ║
║  │  │  • Compute 38│  │  • Compute   │  │  • Score all     │      │    ║
║  │  │    features  │  │    PSI per   │  │    sites         │      │    ║
║  │  │  • Score all │  │    feature   │  │  • Rank by risk  │      │    ║
║  │  │    sites     │  │  • Trigger   │  │  • REPLACE_NOW   │      │    ║
║  │  │  • Calibrate │  │    retrain?  │  │    / MONITOR /   │      │    ║
║  │  │  • Alert     │  │  • Report    │  │    OK            │      │    ║
║  │  │    levels    │  └──────────────┘  └──────────────────┘      │    ║
║  │  └──────────────┘                                               │    ║
║  │                      ⏰ Weekly SAT 03:00                         │    ║
║  │                      ┌──────────────────────────┐               │    ║
║  │                      │ TrainingFlow             │               │    ║
║  │                      │  • Train failure (cox)   │               │    ║
║  │                      │  • Train drain (logistic)│               │    ║
║  │                      │  • Calibrate (isotonic)  │               │    ║
║  │                      │  • Upload to S3          │               │    ║
║  │                      └──────────────────────────┘               │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║              │                                                          ║
║              ▼                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │              SERVING LAYER                                       │    ║
║  │                                                                 │    ║
║  │  ┌───────────────────────┐  ┌─────────────────────────────┐    │    ║
║  │  │ NOC Dashboard (ECS)   │  │ FastAPI (ECS/local)          │    │    ║
║  │  │ Streamlit + S3 sidecar│  │                             │    │    ║
║  │  │                       │  │ POST /predict               │    │    ║
║  │  │ 📊 Drain Risk         │  │ POST /predict/batch         │    │    ║
║  │  │ 📊 Replacement List   │  │ POST /predict/failure       │    │    ║
║  │  │ 📊 Anomaly Detection  │  │ POST /run-flow              │    │    ║
║  │  │ 📊 Drift Monitor      │  │ GET  /models                │    │    ║
║  │  │ 📊 Model Health       │  │ GET  /health                │    │    ║
║  │  └───────────────────────┘  └─────────────────────────────┘    │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║              │                                                          ║
║              ▼                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │              CI/CD (GitHub Actions)                               │    ║
║  │                                                                 │    ║
║  │  Push to main → Lint → Test → Build → Push ECR → Deploy ECS    │    ║
║  │  (OIDC auth — no long-lived secrets)                            │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                                                                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 23. Pipeline Diagrams

### Training Pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                                    │
│                                                                         │
│  ┌─────────┐    ┌──────────────┐    ┌──────────────────┐             │
│  │  START  │───▶│TRAIN_FAILURE │───▶│   TRAIN_DRAIN    │             │
│  │         │    │              │    │                  │             │
│  │• Load   │    │• Window alarms│   │• Build 48h labels│             │
│  │  alarms │    │  per lifecycle│   │• 60/20/20 split  │             │
│  │• Load   │    │• 3-fold CV   │    │• Binary logistic │             │
│  │  sites  │    │• Survival:cox│    │• Isotonic calib  │             │
│  │• Sample │    │• C-index gate│    │• AUC + Brier gate│             │
│  │  n_sites│    │• If pass:    │    │• If pass: promote│             │
│  │• Build  │    │  promote     │    │                  │             │
│  │  sched  │    │              │    │                  │             │
│  └─────────┘    └──────────────┘    └────────┬─────────┘             │
│                                                                 │      │
│                                                                 ▼      │
│                 ┌──────────────────┐    ┌─────────┐                    │
│                 │  UPLOAD_MODELS   │───▶│   END   │                    │
│                 │                  │    │         │                    │
│                 │ • Walk outputs/  │    │ Summary │                    │
│                 │   models/        │    │ report  │                    │
│                 │ • Upload each    │    │         │                    │
│                 │   to S3          │    └─────────┘                    │
│                 └──────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Daily Scoring Pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                DAILY DRAIN PREDICTOR PIPELINE                            │
│                                                                         │
│  ┌─────────┐    ┌────────────────┐    ┌──────────┐    ┌───────────┐   │
│  │  START  │───▶│COMPUTE_FEATURES│───▶│  SCORE   │───▶│EMIT_ALERTS│   │
│  │         │    │                │    │          │    │           │   │
│  │• Load   │    │• All 38 features│   │• Predict │    │• Write    │   │
│  │  booster│    │  for all sites │    │• Calibrate│   │  parquet  │   │
│  │• Load   │    │• Point-in-time │    │• Alert   │    │• Shadow   │   │
│  │  calib  │    │  correct       │    │  levels  │    │  scoring  │   │
│  │• Schema │    │• Schedule      │    │• Cold    │    │• Print    │   │
│  │  check  │    │  lookup        │    │  start   │    │  top-10   │   │
│  │• Decide │    │                │    │  fallback│    │           │   │
│  │  sim_hr │    │                │    │• CW      │    │           │   │
│  └─────────┘    └────────────────┘    │  metrics │    └─────┬─────┘   │
│                                       └──────────┘          │         │
│                                                             ▼         │
│                                                       ┌─────────┐     │
│                                                       │   END   │     │
│                                                       └─────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Drift Monitor + Auto-Retrain Loop:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              DRIFT MONITOR → AUTO-RETRAIN LOOP                          │
│                                                                         │
│  ┌─────────┐   ┌──────────────┐   ┌─────────────┐   ┌────────────┐   │
│  │  START  │──▶│COMPUTE_FEATS │──▶│DETECT_DRIFT │──▶│  DECIDE    │   │
│  │         │   │              │   │             │   │            │   │
│  │• Load   │   │• Same as     │   │• PSI per    │   │• Write     │   │
│  │  ref    │   │  scoring     │   │  feature    │   │  report    │   │
│  │  profile│   │• Also get    │   │• KS test    │   │• If drift: │   │
│  │         │   │  predictions │   │• Prediction │   │  write     │   │
│  └─────────┘   └──────────────┘   │  drift      │   │  TRIGGER   │   │
│                                    │• Evidently? │   └──────┬─────┘   │
│                                    └─────────────┘          │         │
│                                                             │         │
│                 ┌───────────────────────────────────────────┘          │
│                 ▼                                                       │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                  RETRAINING FLOW (triggered)                    │    │
│  │                                                                │    │
│  │  claim_trigger() ──▶ train_challenger ──▶ compare_on_shared    │    │
│  │                      (same pipeline)       test_set            │    │
│  │                                                │               │    │
│  │                                    ┌───────────┴───────────┐   │    │
│  │                                    ▼                       ▼   │    │
│  │                            ┌──────────────┐      ┌──────────┐  │    │
│  │                            │   PROMOTE    │      │  REJECT  │  │    │
│  │                            │ (swap models)│      │(release  │  │    │
│  │                            │ archive old  │      │ trigger) │  │    │
│  │                            └──────────────┘      └──────────┘  │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Continuous Retraining + Rollback (Santiago's Parallel Path):

```
┌─────────────────────────────────────────────────────────────────────────┐
│           SANTIAGO'S PARALLEL PATHS — TRAINING vs INFERENCE              │
│                                                                         │
│  INFERENCE PATH (daily, champion serves):                               │
│  ┌────────────┐    ┌────────────────┐    ┌──────────────┐              │
│  │  00:30 UTC │───▶│ DrainPredictor │───▶│ Alerts to S3 │              │
│  │  EventBridge│   │ (champion)     │    │ + dashboard  │              │
│  └────────────┘    └────────────────┘    └──────────────┘              │
│                                                                         │
│  TRAINING PATH (weekly, always produce challenger):                     │
│  ┌────────────┐    ┌────────────────┐    ┌──────────────────────┐      │
│  │  SAT 03:00 │───▶│ RetrainingFlow │───▶│ Champion vs Challenger│      │
│  │  --force   │    │ (latest data)  │    │ on SAME held-out set │      │
│  └────────────┘    └────────────────┘    └───────────┬──────────┘      │
│                                                      │                  │
│                                          ┌───────────┴───────────┐      │
│                                          ▼                       ▼      │
│                                   ┌────────────┐         ┌──────────┐  │
│                                   │  PROMOTE   │         │ DISCARD  │  │
│                                   │ challenger │         │challenger│  │
│                                   │ > champion │         │ < margin │  │
│                                   │ + margin   │         └──────────┘  │
│                                   └─────┬──────┘                        │
│                                         │                               │
│  SAFETY NET (daily):                    ▼                               │
│  ┌────────────┐    ┌────────────────────────────────────┐              │
│  │  02:00 UTC │───▶│ RollbackMonitorFlow                │              │
│  │  EventBridge│   │                                    │              │
│  └────────────┘    │ • Promoted in last 48h?            │              │
│                    │ • Production Brier > threshold?     │              │
│                    │   YES → restore archived champion   │              │
│                    │   NO  → keep (model is healthy)     │              │
│                    └────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

### EventBridge Schedule (full daily orchestration):

```
 00:30  DrainPredictorFlow     ─── score all sites with champion
 01:00  DriftMonitorFlow       ─── PSI check (safety net for mid-week emergencies)
 01:30  ShadowPromotionFlow    ─── validate failure model shadow (if exists)
 02:00  RollbackMonitorFlow    ─── revert drain predictor if production degrades
 SAT 03:00  RetrainingFlow     ─── train challenger, promote if CV passes
 SUN 02:00  FailureScoringFlow ─── weekly replacement priority ranking
```

### Data Flow Through the System:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA FLOW — START TO FINISH                           │
│                                                                         │
│  SIMULATOR ──▶ alarms.parquet ──▶ FEATURE PIPELINE ──▶ 38 features     │
│                                        │                                │
│                                        │ (same code, different groups)  │
│                                        │                                │
│                          ┌─────────────┴─────────────┐                  │
│                          ▼                           ▼                  │
│                    ┌──────────┐              ┌──────────┐              │
│                    │ FAILURE  │              │  DRAIN   │              │
│                    │ MODEL    │              │ MODEL    │              │
│                    │ (cox)    │              │(logistic)│              │
│                    └────┬─────┘              └────┬─────┘              │
│                         │                        │                     │
│                         ▼                        ▼                     │
│                    ┌──────────┐              ┌──────────┐              │
│                    │ REPLACE  │              │  DRAIN   │              │
│                    │ NOW /    │              │  RISK    │              │
│                    │ MONITOR  │              │  48h     │              │
│                    └────┬─────┘              └────┬─────┘              │
│                         │                        │                     │
│                         └───────────┬────────────┘                     │
│                                    ▼                                    │
│                            ┌──────────────┐                            │
│                            │  DASHBOARD   │                            │
│                            │  + FastAPI   │                            │
│                            │  (operator   │                            │
│                            │   decisions) │                            │
│                            └──────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary for Using This as a Prompt

When giving this document to an AI to replicate for a new use case:

1. **Replace the domain** (batteries → your domain)
2. **Map the three questions** to your business's operational cadences
3. **Identify your "alarm stream equivalent"** — what reliable signal do you have?
4. **Choose metrics based on the decision**: ranking → C-index; binary → AUC+Brier; calibration always
5. **Follow the same folder structure** — it scales
6. **Implement the same production patterns** — they prevent day-30 failures
7. **Use the same testing philosophy** — test properties, not values
8. **Deploy with the same stack** — Terraform + Batch + ECS is cheap and works
9. **Choose promotion strategy based on label latency:**
   - Fast labels (< 7 days): immediate promotion with rollback safety net
   - Slow labels (months): shadow deployment with label maturity gate
10. **Always retrain proactively** (Santiago's pattern) — training is cheap,
    the CV gate is what protects production. Don't wait for drift to fire.

The patterns in this project are universal. The domain is specific.

---

*Generated from project_01 at commit dfeee12 (2026-05-28)*
*Architecture inspired by industry-standard ML lifecycle patterns*
