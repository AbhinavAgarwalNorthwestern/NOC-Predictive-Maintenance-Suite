# 02. Data and features

## TL;DR

We deliberately constrain ourselves to **alarms + static site config + the
regional load-shedding schedule** — no telemetry. This is closer to what most
telecom NOCs actually have access to. The feature pipeline produces 38 features
spanning four groups, all computed point-in-time correctly so models never
see the future.

## What data exists

### Sources

| Source | What it is | Latency | Reliability |
|--------|------------|--------:|------------:|
| **Alarm stream** | Events: `AC_MAINS_FAIL`, `LOAD_DISCONNECT`, `RECTIFIER_FAULT`, `BATT_UNDERVOLTAGE`, `CELL_IMBALANCE`, `BATT_HIGH_TEMP`, `REPEAT_FAILURE_FLAG`, `TICKET_*` | seconds | ✅ high |
| **Site static config** | Region, manufacturer, install_month, load_A, n_cells, nominal_capacity_ah | static (snapshot) | ✅ high |
| **Load-shedding schedule** | Hour-level region calendar from utility: `(region, hour) → is_offgrid, severity_score` | daily | ✅ medium (depends on utility publishing) |
| **Telemetry** (not used) | Voltage, current, temperature, internal resistance | minutes (when present) | ⚠️ uneven coverage |

The system reads only the first three. The pipeline could ingest telemetry by
registering a new feature group, but does not require it.

### Synthetic data simulator

Because real telecom data is proprietary, we generate it via a physics-aware
simulator at [`src/battery_pdm/synth/simulator.py`](../src/battery_pdm/synth/simulator.py).

The simulator generates 36 months of data across 500 sites distributed across
5 regions (Lahore, Karachi, Peshawar, Quetta, Islamabad), with realistic:

- **Outage patterns** driven by the load-shedding schedule
- **Battery degradation** via Arrhenius temperature acceleration and
  Coulomb-counting state of charge
- **Sudden-failure mode** triggered by accumulated rectifier faults (we built
  this in deliberately — pure deterministic physics produced models that were
  trivially predictable; sudden failures add irreducible noise that mirrors
  reality)
- **Manufacturing heterogeneity** via per-site `aging_multiplier` (lognormal),
  representing the same battery SKU performing differently across sites
- **Sensor noise + telemetry gaps** (simulated even though we don't use
  telemetry — keeps the option open and lets future feature groups demonstrate
  robustness)

#### What the simulator output looks like

```python
result = simulate_fleet(n_sites=500, n_months=36)
# Returns a dict (extensible API):
result["alarms"]        # DataFrame: site_id, timestamp_h, alarm_code, severity
result["site_static"]   # DataFrame: site_id, region, manufacturer, install_month, load_A, ...
result["labels"]        # DataFrame: per-lifecycle event/censor labels for failure model
result["telemetry"]     # DataFrame: optional, retained for future feature groups
```

For the realized run on which all metrics in this doc are reported:
- 203k alarms across 500 sites
- 17.5k load disconnect events
- 5 regions, 36 months
- Drain rate per outage ranges from 0.1% (Islamabad) to 29.7% (Peshawar)

### Why not real data?

Real telecom alarm streams are confidential. Even when anonymized, the data
sharing agreements typically don't permit posting to GitHub. We document this
clearly in the README — the simulator gives the same shape, the same physics
constraints, and the same kinds of edge cases (sudden failures, manufacturing
variance, regional bias) that real data would have.

The flip side: an interviewer who wants to test you on real data should be able
to point your trained model at their dataset and have it work, because the
simulator was constructed to match the constraints they'd impose.

## Feature engineering

### Architectural pattern: extensible feature groups

The feature pipeline uses a registry pattern in
[`src/battery_pdm/common/features.py (shared feature pipeline)`](../src/battery_pdm/common/features.py (shared feature pipeline)):

```python
@register_feature_group("alarm_history", requires=("alarms",))
def _alarm_history_features(labels, alarms):
    ...

@register_feature_group("site_static", requires=("site_static",))
def _site_static_features(labels, site_static):
    ...

@register_feature_group("soc_proxy", requires=("alarms", "site_static"))
def _soc_proxy_features(labels, alarms, site_static):
    ...

@register_feature_group("load_shedding_schedule", requires=("schedule", "site_static"))
def _load_shedding_features(labels, schedule, site_static):
    ...
```

Then any caller composes which groups they want:

```python
features = compute_features(
    labels=labels, groups=["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"],
    inputs={"alarms": ..., "site_static": ..., "schedule": ...},
    ref_time_col="mains_fail_h",  # or "event_hour" for the failure model
)
```

**Why this pattern matters:**
- Same code serves both active models (failure, drain predictor)
- Adding telemetry later = one new `@register_feature_group("telemetry_stats", requires=("telemetry",))` function
- Each group declares its required inputs, so missing data fails loud at function
  entry instead of producing silent NaN cascades
- Easy to ablation-test (drop a group, see how AUC moves)

### Feature inventory (current — 38 features)

#### Group 1: `alarm_history` (16 features) — counters from the alarm stream

Computed via `searchsorted`-based point-in-time queries (so we never include
alarms with `timestamp_h >= ref_time_h`).

| Feature | What it captures |
|---------|------------------|
| `rectifier_fault_count_lifetime` | Cumulative rectifier issues — flags chronic charging problems |
| `rectifier_fault_count_30d` | Recent rectifier issues |
| `cell_imbalance_count_lifetime` | Battery bank degradation accumulator |
| `cell_imbalance_count_30d` | Recent imbalance signals |
| `mains_fail_count_lifetime` | How many outages this site has seen |
| `mains_fail_count_30d` | Recent outage frequency (proxies regional shedding) |
| `mains_fail_count_7d` | Very recent outage frequency |
| `hours_since_last_mains_fail` | Recovery time since last outage |
| `critical_alarm_count_30d` | Recent critical alarms (any severity=critical) |
| `undervoltage_count_30d` | Recent low-voltage warnings (battery health proxy) |
| `undervoltage_count_lifetime` | Lifetime low-voltage warnings |
| `high_temp_count_30d` | Recent high-temp events (accelerated aging) |
| `lvd_count_30d` | Recent load disconnects (**dominant feature for drain prediction**) |
| `lvd_count_lifetime` | Lifetime load disconnects (chronic at-risk indicator) |
| `has_repeat_failure` | REPEAT_FAILURE_FLAG ever fired? |
| `ticket_count_lifetime` | Tech tickets — separate from machine alarms |

#### Group 2: `site_static` (8 features) — per-site config

| Feature | What it captures |
|---------|------------------|
| `load_A` | DC load amperage — directly proportional to discharge rate |
| `n_cells` | Battery bank size |
| `install_month` | Calendar month of install |
| `nominal_capacity_ah` | Nameplate capacity |
| `battery_age_months_at_ref` | Computed: (ref_time_h / 720) - install_month |
| `region_encoded` | Categorical region as int |
| `manufacturer_encoded` | Categorical manufacturer as int |
| `charger_misconfigured` | ⚠️ **Excluded from feature set as leaky** — this is directly used to generate failures in the simulator |

`aging_multiplier` is also leaky and excluded. Both are part of the simulator
ground truth, not something an operator would observe. The leaky-feature
exclusion is enforced explicitly:

```python
LEAKY_FEATURES = {"charger_misconfigured", "aging_multiplier"}
feature_cols = [c for c in features.columns if c not in LEAKY_FEATURES]
```

#### Group 3: `soc_proxy` (9 features) — coulomb-counted state of charge

The clever part. We never see voltage telemetry, but we DO see the alarm
timestamps: `AC_MAINS_FAIL` starts a discharge, the alarm sequence implies
recharge windows. So we can reconstruct an approximate SoC trajectory:

```python
def _compute_soc_ledger(site_alarms, t0, load_A, capacity_Ah, charger_A=15.0,
                       lifecycle_start_h=0.0):
    # Walk forward from lifecycle_start through each alarm event:
    #   AC_MAINS_FAIL → start discharge at load_A
    #   next alarm or end-of-window → end discharge, start recharge at charger_A
    # Sum cumulative discharge hours, hours-since-full-recharge, etc.
```

| Feature | What it captures |
|---------|------------------|
| `estimated_soc_at_ref` | Coulomb-counted SoC fraction at ref time |
| `frac_time_offgrid_30d` | How much of recent past was on battery |
| `hours_recharged_30d` | Total grid-on hours in last 30 days |
| `hours_discharged_30d` | Total grid-off hours in last 30 days |
| `hours_since_full_recharge` | Time since SoC ≥ 0.98 (proxy for partial-SoC stress) |
| `avg_recharge_gap_30d` | Average time between outages (regional pattern proxy) |
| `longest_outage_30d` | Worst recent outage duration |
| `outage_duration_p95_lifetime` | 95th percentile outage duration across lifetime |
| `cumulative_discharge_hours` | Lifetime discharged hours (degradation proxy) |

**Why this matters:** even without telemetry, alarm timing alone gives a
useful (if noisy) estimate of battery state. The SoC proxy features rank top-10
by gain in both the drain predictor and failure model.

#### Group 4: `load_shedding_schedule` (6 features) — external grid data

For each (region, ref_time) pair, we look up the utility's published schedule:

| Feature | What it captures |
|---------|------------------|
| `schedule_offgrid_hours_past_30d` | Recent grid stress regionally |
| `schedule_severity_mean_past_30d` | Smoothed regional severity |
| `schedule_offgrid_hours_next_24h` | **Forecast of upcoming grid availability** |
| `schedule_severity_max_next_24h` | Peak severity in next 24h |
| `currently_in_peak_window` | Is the scoring time inside a published shed window? |
| `expected_daily_offgrid_now` | The day's expected offgrid hours |

**Why schedule features dominate the failure model:** they encode the
*deterministic* part of grid behavior. A battery in Quetta will degrade faster
than the same battery in Islamabad because Quetta's schedule says so, regardless
of any specific alarm pattern.

In the failure model's feature importance:
1. `schedule_severity_max_next_24h`
2. `battery_age_months_at_ref`
3. `schedule_offgrid_hours_past_30d`

The top three are all schedule + age — not alarm-derived.

## Point-in-time correctness — the bug class that ruins ML systems

The single biggest correctness risk in production ML is using future data to
predict the past. Our pipeline avoids it via three patterns:

### Pattern 1: `searchsorted` on sorted timestamps

Every alarm group is pre-sorted by `(site_id, timestamp_h)`. For each label's
`ref_time_h`, we slice with `searchsorted(side="left")` so the slice contains
ALARMS STRICTLY BEFORE `ref_time_h`:

```python
ts = g["timestamp_h"].values
idx_t0 = ts.searchsorted(t0, side="left")  # leftmost insertion point
idx_30 = ts.searchsorted(t0 - 30 * 24, side="left")

past = g.iloc[:idx_t0]
recent_30d = g.iloc[idx_30:idx_t0]
```

This is unit-tested at
[`tests/test_features.py::test_compute_features_point_in_time_correct`](../tests/test_features.py).
The test asserts that lifetime counters at t=50h are ≤ lifetime counters at t=400h.

### Pattern 2: Lifecycle windowing for the failure model

When a battery is replaced mid-history, the new lifecycle gets a new
`site_id = "SITE_001_L1"`. The training code windows alarms per lifecycle so
the L1 features only see L1's alarms:

```python
for _, row in labels_raw.iterrows():
    original_sid = row["original_site_id"]
    start_h = row["lifecycle_start_h"]
    end_h = row["event_hour"]
    windowed = alarms[
        (alarms["site_id"] == original_sid)
        & (alarms["timestamp_h"] >= start_h)
        & (alarms["timestamp_h"] <= end_h)
    ].copy()
    windowed["site_id"] = row["site_id"]  # rename to L0/L1/L2 lifecycle id
```

This was a subtle bug we caught in code review and fixed: without lifecycle
windowing, the L1 model saw L0 failure events as feature history.

### Pattern 3: Group-constrained train/test split

If we randomly split observations into train/test, the model could see
*future* observations of the *same site* in test that share latent properties
with training observations (load, region, manufacturer). This leaks via the
shared site-level features.

The fix is to split by **site**, not observation:

```python
unique_sites = features["site_id"].unique()
rng.shuffle(unique_sites)
n_total = len(unique_sites)
train_sites = set(unique_sites[: int(n_total * 0.60)])
cal_sites   = set(unique_sites[int(n_total * 0.60) : int(n_total * 0.80)])
test_sites  = set(unique_sites[int(n_total * 0.80) :])
```

All observations of one site stay in one fold. This is implemented in every
training script and CV routine in the repo.

## Missing data handling

[`src/battery_pdm/monitoring/data_quality.py`](../src/battery_pdm/monitoring/data_quality.py)
defines three strategies:

### Strategy 1: Refuse to score insufficient sites

A site needs ≥3 lifetime alarms before we'll score it. Sites that don't qualify
get tagged with alert_level `COLD_START` and use the regional drain rate as
their default risk score:

```python
if not r["scorable"]:
    risk = regional_prior.get(region, 0.15)
    alert_level = "COLD_START"
```

This avoids the "confidently wrong" failure mode for newly-installed batteries.

### Strategy 2: Track null rates as a feature

`data_completeness_score` is computed per row as the fraction of features that
are both non-null AND non-zero. A low score means the row was mostly
default-imputed values — the model can learn to be more uncertain there.

### Strategy 3: Warn loudly on schedule gaps

If a site's region isn't covered by the schedule data, `load_shedding_schedule`
emits a Python warning that says exactly which regions are missing. This caught
a bug in deployment where the schedule file was uploaded with one region typo'd.

## Class imbalance

The drain predictor has ~15% positive class rate. We handle this with:

1. **`scale_pos_weight`** in XGBoost = (negatives / positives) ≈ 5.7. Positives
   are weighted ~6× more in the loss function.
2. **`aucpr` as eval metric** (area under precision-recall curve) instead of
   plain `auc`. AUC-PR is more sensitive to minority-class performance.
3. **Early stopping on `aucpr`** so we stop boosting when minority performance plateaus.

Why we DIDN'T use SMOTE or oversampling:
- Tree models are robust to imbalance via `scale_pos_weight`; SMOTE introduces
  synthetic noise that XGBoost overfits.
- Undersampling discards information — bad for our 15% rate where we have data
  to learn from, just need to weight it.

This is documented in [03_MODELS_AND_CHOICES.md](03_MODELS_AND_CHOICES.md) too.

## Drift in the data

Once deployed, the data distribution will drift. We treat the schedule as
**ingested data** (uploaded to S3 by an external process) rather than
regenerated on the fly — this means schedule changes from the utility
company will be visible to the model and to the drift monitor. See
[06_PRODUCTION_PATTERNS.md](06_PRODUCTION_PATTERNS.md) for the drift detection
design.

## Reading next

- [03_MODELS_AND_CHOICES.md](03_MODELS_AND_CHOICES.md) — model selection
  rationale + alternatives we rejected
- [04_RESULTS_AND_METRICS.md](04_RESULTS_AND_METRICS.md) — what each chart and
  metric means
