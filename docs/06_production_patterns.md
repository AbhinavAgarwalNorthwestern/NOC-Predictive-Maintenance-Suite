# 06. Production patterns

This is the doc that distinguishes a portfolio from a production system. Each
pattern below is a specific bug class or operational risk that production ML
systems hit, and how we addressed it.

## Pattern 1: PSI-based drift detection with TRAINING-TIME reference

### What it solves

Data distributions change over time. Models trained on data from 6 months ago
silently degrade. You want to know BEFORE this becomes operationally damaging.

### The naive bug

Most "drift detection" implementations compute a reference profile from the
**first production window** they see. This is wrong because:
1. The first production window might already be drifted
2. The reference becomes whatever production happens to look like, not what
   training assumed
3. Drift then means "different from yesterday" not "different from training"

### How we fixed it

`save_reference_profile_for_model` is called at PROMOTION time inside the
training flow, using TRAINING-time features:

```python
# src/battery_pdm/flows/training_flow.py
meta = save_model_artifacts(...)
save_reference_profile_for_model(
    training_features=merged[feature_cols + ["site_id"]],
    feature_cols=feature_cols,
    predictions=final.predict(xgb.DMatrix(X_all)),
    model_dir=out_dir,
)
```

The reference profile is saved as a JSON next to `booster.json` and
`calibrator.pkl`. The drift monitor loads it and compares against the latest
window:

```python
# src/battery_pdm/monitoring/drift.py
def detect_drift(reference_profile, current_features, ...):
    feature_drift = compute_psi(reference_profile, current_features, feature_cols)
    ...
```

### PSI thresholds

| PSI | Interpretation | Action |
|----:|----------------|--------|
| < 0.10 | Stable | Nothing |
| 0.10-0.25 | Moderate drift | Flag, monitor more closely |
| 0.25+ | Significant drift | Investigate; potentially retrain |

When 3+ features exceed 0.25 OR prediction PSI exceeds 0.10, the drift monitor
writes `retrain_trigger.json`.

### The Peshawar grid upgrade demo

`scripts/simulate_drift.py` and `notebooks/01_drift_detection_demo.ipynb`
demonstrate the full loop:
1. Drop 60% of `AC_MAINS_FAIL` alarms in Peshawar (simulated grid stabilizer
   intervention)
2. Run drift detection on month 30 (6 months after intervention)
3. Observe: 10/38 features SIGNIFICANT drift, prediction PSI 0.548, retrain
   triggered.

**The deeper finding** documented in the notebook: Peshawar's mean risk
*increases* despite the batteries being objectively healthier. This is because
the model learned `lvd_count_lifetime` as a predictor, and that counter only
ever goes up. Retraining alone won't fix this — see notebook 01 for the
discussion of operational response.

## Pattern 2: Isotonic calibration on a strictly held-out set

### What it solves

Raw XGBoost binary:logistic scores are usually miscalibrated. For our drain
predictor, scores in the [0.7, 0.9] range corresponded to actual rates around
0.35-0.45. Operators acting on raw scores would over-dispatch by 60-70%.

### How we fixed it

Three-way split: 60% train / 20% calibration / 20% test. The calibrator is
fit on the strictly-held-out 20% calibration set:

```python
# src/battery_pdm/flows/training_flow.py
raw_cal_scores = booster.predict(xgb.DMatrix(X_cal))
calibrator = train_isotonic_calibrator(raw_cal_scores, y_cal)
```

The calibrator is then **saved alongside the model** as `calibrator.pkl`:

```python
save_calibrator(calibrator, model_dir)
```

At scoring time, the calibrator is loaded and applied:

```python
# src/battery_pdm/flows/drain_predictor_flow.py
self.calibrator = load_calibrator(model_path)  # None if missing — graceful fallback
...
raw_scores = self.booster.predict(dmat)
risk_scores = apply_calibrator(self.calibrator, raw_scores)  # identity if None
```

### Why isotonic specifically

| Method | When |
|--------|------|
| **Platt scaling** | <1000 calibration samples; or known-sigmoidal miscalibration |
| **Isotonic regression** | ≥1000 samples; arbitrary monotonic shape; standard for tree models |

We have ~2000 calibration samples. Trees produce non-sigmoidal miscalibration
patterns (multi-modal score histograms). Isotonic is textbook.

### Why a strictly held-out set

A common mistake: fit the calibrator on the test set. This **deflates** the
apparent improvement because the calibrator overfits to the test distribution
the model is already being evaluated on.

For the **first training**, we accept a small in-sample optimism by refitting
the calibrator on the cal set after the final booster includes it. For the
**RetrainingFlow** (champion vs challenger), the calibration set stays strictly
held out so comparisons are fair.

### The measurable improvement

41% Brier reduction (0.132 → 0.078) for the global model. AUC unchanged at
0.890 because calibration is monotonic and ranking is invariant to monotonic
transforms.

## Pattern 3: Feature hash validation between training and inference

### What it solves

Train/serve skew. If the feature column list at training time differs from
the feature column list at scoring time (in name OR order), predictions are
silently corrupted.

### How we fixed it

A stable SHA256 hash of the ordered feature column list is computed at training
time and saved into `meta.json`:

```python
# src/battery_pdm/monitoring/model_registry.py
def compute_feature_hash(feature_cols: list[str]) -> str:
    serialized = json.dumps(feature_cols, sort_keys=False).encode()
    return hashlib.sha256(serialized).hexdigest()[:16]
```

At scoring time, the same hash is recomputed and compared:

```python
def validate_feature_hash(meta: dict, feature_cols_used: list[str]) -> None:
    expected = meta.get("feature_hash")
    if expected is None:
        return  # legacy model — no enforcement
    actual = compute_feature_hash(feature_cols_used)
    if expected != actual:
        raise ValueError(
            f"Feature hash mismatch: model expects {expected}, "
            f"got {actual}. Feature contract has drifted."
        )
```

If anyone:
- Adds a new feature to the pipeline without retraining
- Reorders columns in the meta.json
- Drops a feature

…the scoring flow fails fast with a loud error.

### Where this matters most

When the team grows beyond one person. Multiple developers can independently
change the feature pipeline; the hash mismatch protects production from
silent skew.

## Pattern 4: Atomic retrain trigger consumption

### What it solves

When the drift monitor decides "retrain needed" and the retraining flow runs,
the trigger should be **consumed exactly once**, even if:
- The retraining flow crashes mid-run
- Two retraining flows run concurrently (race condition)
- Network failure during trigger deletion

The naive approach (`os.remove(trigger_path)` at end of retraining) breaks
under all three.

### How we fixed it

Three-state machine: pending → processing → consumed.

```python
# src/battery_pdm/monitoring/model_registry.py
TRIGGER_PATH = Path("outputs/drift_reports/retrain_trigger.json")
TRIGGER_PROCESSING_PATH = Path("outputs/drift_reports/retrain_trigger.processing.json")
TRIGGER_CONSUMED_DIR = Path("outputs/drift_reports/consumed_triggers")


def claim_retrain_trigger() -> dict | None:
    """Atomically claim the trigger via os.rename (POSIX-atomic)."""
    if not TRIGGER_PATH.exists():
        return None
    if TRIGGER_PROCESSING_PATH.exists():
        return None  # another worker already claimed it
    try:
        os.rename(TRIGGER_PATH, TRIGGER_PROCESSING_PATH)
    except (FileExistsError, OSError):
        return None
    return json.loads(TRIGGER_PROCESSING_PATH.read_text())


def complete_retrain_trigger(promoted: bool, reason: str = "") -> None:
    """Move to consumed/ after successful promotion."""
    ...

def release_retrain_trigger() -> None:
    """If processing fails mid-way, put it back to pending state."""
    if TRIGGER_PROCESSING_PATH.exists() and not TRIGGER_PATH.exists():
        os.rename(TRIGGER_PROCESSING_PATH, TRIGGER_PATH)
```

The retraining flow wraps promotion in try/except:

```python
try:
    ...promote challenger...
    complete_retrain_trigger(promoted=True, reason="challenger_won")
except Exception as exc:
    release_retrain_trigger()  # back to pending for retry
    raise
```

### What this gives you

- ✅ Crash recovery — failed retrains release the trigger; the next run retries
- ✅ No double-promotion — second concurrent worker sees processing file and exits
- ✅ Audit trail — `consumed_triggers/` has timestamped JSON of every
  promotion + reasoning

### S3 considerations

`os.rename` is atomic on POSIX file systems and (mostly) on Windows. On S3 the
equivalent is a `copy_object` + `delete_object` pair, which is NOT atomic. For
true production multi-worker safety on S3 we'd use:
- A DynamoDB lock table (canonical S3 + Terraform pattern), OR
- S3 Strong Consistency Conditional Writes (newer, less universal)

For our current single-Batch-job scenario, file rename via the local
filesystem (which is then synced to S3 by the flow) is sufficient.

## Pattern 5: Champion / challenger with shared held-out test set

### What it solves

When you retrain, you want to compare the new model against the deployed
model. The naive comparison fails: each model was evaluated on its OWN test
set drawn from its own time window, with its own random seed. Apples to oranges.

### How we fixed it

The RetrainingFlow:
1. Trains a challenger model on the current data, holding out a calibration
   set and a test set
2. Loads the champion (currently deployed) model
3. **Re-evaluates the champion on the challenger's exact test set** (since the
   champion's training-time test set is not available — it was a different
   random sample)
4. Compares AUC + Brier on this SHARED held-out set
5. Promotes only if challenger beats champion by `promotion_margin` (default 0.005)

```python
# src/battery_pdm/flows/retraining_flow.py compare step
champion_booster = xgb.Booster(); champion_booster.load_model(...)
challenger_booster = xgb.Booster(); challenger_booster.load_model(...)

X_test = self.test_set_features  # from challenger's split
X_champion = pd.DataFrame(index=X_test.index)
for col in champion_feature_cols:
    X_champion[col] = X_test[col] if col in X_test.columns else 0.0

champion_scores = champion_booster.predict(xgb.DMatrix(X_champion))
champion_val_shared = float(roc_auc_score(self.test_set_labels, champion_scores))

challenger_val = ...
margin = challenger_val - champion_val_shared
self.promote = margin >= self.promotion_margin
```

### What this prevents

- A challenger that "wins" because recent data is easier (not because it's
  better)
- A challenger that beats stale champion metrics but actually scores worse
  on fresh data
- A champion that gets replaced by noise

### Caveat

The shared test set is from the challenger's data window, not a true
"point-in-time" champion test. If we wanted ultra-rigorous comparison, we'd
sample a time-stratified test set and evaluate BOTH models on it (with
appropriate point-in-time feature computation). For our scale this is
overengineering; the shared-test approach gives ~95% of the benefit at
minimal complexity.

## Pattern 6: Cold-start fallback to regional prior

### What it solves

A brand-new battery (just installed) has zero alarm history. All
alarm-derived features are zero. The model sees this and (depending on
training-time distribution) might predict LOW risk — wrong, because we
genuinely don't know.

### How we fixed it

Three layers, in `src/battery_pdm/monitoring/data_quality.py` +
`src/battery_pdm/flows/drain_predictor_flow.py`:

#### Layer A: scorability check

A site needs ≥3 lifetime alarms OR be in a region we have coverage for, before
we'll score it with the ML model:

```python
def assess_scoring_inputs(site_ids, alarms, site_static, schedule, ref_time_h, history_days=30):
    ...
    for s in site_ids:
        site_lifetime_count = int((alarms["site_id"] == s).sum())
        if site_lifetime_count < MIN_ALARMS_PER_SITE:  # 3
            insufficient.append(s)
        else:
            scorable.add(s)
```

#### Layer B: regional prior fallback

Insufficient sites get the regional drain rate as their score:

```python
regional_prior = compute_regional_priors(alarms_pit, self.site_static, horizon_h=48)
site_to_region = dict(zip(self.site_static["site_id"], self.site_static["region"]))
cold_mask = ~self.alerts["scorable"]
for idx in self.alerts.index[cold_mask]:
    site = self.alerts.at[idx, "site_id"]
    region = site_to_region.get(site, "unknown")
    self.alerts.at[idx, "drain_risk_48h"] = regional_prior.get(region, 0.15)
```

E.g., a new Peshawar site gets 0.30 (the regional drain rate); a new Islamabad
site gets 0.001.

#### Layer C: alert level `COLD_START`

The alert level is set to `COLD_START` (not `INSUFFICIENT_DATA`) so operators
know "use the regional default for now, watch this manually for 30 days":

```python
self.alerts["alert_level"] = self.alerts.apply(
    lambda r: "COLD_START" if not r["scorable"]
    else ("HIGH" if r["drain_risk_48h"] >= 0.6 else ...),
    axis=1,
)
```

### What this prevents

The "model said 0.05 risk but battery actually died" failure mode. With this
in place:
- High-risk regions (Peshawar) → new site auto-flagged as 0.30 → operator monitors
- Low-risk regions (Islamabad) → new site auto-deprioritized → operator
  doesn't waste cycles

## Pattern 7: Model performance history log (the depletion chart's source)

### What it solves

The dashboard wants to show "AUC over time" — a line chart showing whether
the model's quality is degrading. You need a time-series log of every
evaluation event.

### How we fixed it

`model_performance_log.parquet` — append-only, one row per evaluation event:

```python
# src/battery_pdm/monitoring/model_registry.py
def append_performance_log(model_name, model_version, metric_name, metric_value,
                            n_observations, feature_hash, extras=None,
                            path=PERFORMANCE_LOG_PATH) -> None:
    new_row = {
        "logged_at": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "n_observations": int(n_observations),
        "feature_hash": feature_hash,
        **(extras or {}),
    }
    df = (pd.read_parquet(path) if path.exists() else pd.DataFrame())
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_parquet(path, index=False)
```

Called from:
- `TrainingFlow` after each promotion → AUC / C-index at training time
- `DriftMonitorFlow` after each drift report → prediction PSI
- `RetrainingFlow` after each promotion → AUC of the challenger

The CloudWatch dashboard reads this from S3 and plots "AUC over time" with
points colored by event type (training, drift_monitor, promotion).

## Pattern 8: Schedule as ingested data, not generated

### What it solves

The load-shedding schedule is **external data published by the utility**. In
production it would arrive via API or CSV upload. Treating it as "regenerate
on the fly with a fixed seed" is convenient but misses three production
concerns:

1. Schedule changes (utility updates calendar) need to flow through
2. Schedule drift (utility changes the SHAPE of the schedule) needs detection
3. Schedule outages (we don't get an updated calendar) need explicit failure
   modes

### How we fixed it

The schedule is generated once (locally) and uploaded to S3:

```bash
aws s3 cp outputs/load_shedding_schedule.parquet s3://battery-pdm-dev-data-<account>/load_shedding_schedule.parquet
```

The drain predictor flow accepts a `--schedule-path` parameter:

```python
schedule_path = Parameter("schedule-path", default="", help="S3 or local path...")
```

If the parameter is set (AWS deployment), it reads from S3 via
`battery_pdm.aws.s3_io.read_parquet`. If not (local dev), it falls back to
the generator. Two branches, same downstream code.

### Future extension

If we wanted to ingest real schedules, we'd add a Lambda function that polls
the utility API daily and writes a fresh parquet to S3. The drain predictor's
schedule_path stays the same; only the upstream changes.

## Pattern 9: Group-constrained CV

Already covered in [02_DATA_AND_FEATURES.md](02_DATA_AND_FEATURES.md). The
short version: random splits leak via site-level features. Always split by
site (the "group"), not by observation, when observations of the same site
share covariates.

## Pattern 10: Point-in-time feature computation

Already covered in [02_DATA_AND_FEATURES.md](02_DATA_AND_FEATURES.md). Use
`searchsorted(side="left")` + `iloc[:idx]` to slice alarms strictly before
ref_time_h. Unit-tested at `tests/test_features.py::test_compute_features_point_in_time_correct`.

## Pattern 11: Concept drift detection (the labeled-feedback layer)

### What it solves

PSI catches feature drift but cannot tell you whether the model is actually
worse — only that inputs changed. To know "is my model actually degrading?"
you need **labels** for recent predictions.

### How we implemented it

[`monitoring/concept_drift.py`](../src/battery_pdm/monitoring/concept_drift.py) +
[`flows/concept_drift_monitor_flow.py`](../src/battery_pdm/flows/concept_drift_monitor_flow.py)

Daily pipeline:
1. Read prior predictions from `outputs/drain_alerts/*.parquet`
2. For each prediction, look up if the actual event occurred within horizon
3. Compute rolling AUC/Brier over recent labeled windows
4. Compare to baseline AUC (from `meta.json`) — if degradation > threshold, flag concept drift
5. Log to MLflow + emit CloudWatch metrics
6. If concept drift detected → augment retrain trigger with the reason

### Why this matters

A feature-drift-only system retrains on any input shift — including benign
ones. Adding labeled feedback means retraining fires ONLY when the model
actually got worse, not when the input distribution merely shifted.

For **short-horizon models** (drain at 48h, autonomy at 12h): labels mature
quickly, this loop runs within a week.

For **long-horizon models** (failure at 6-12 months): labels mature slowly;
concept drift detection is the only honest answer to "is the new technology
actually being predicted correctly?"

---

## Pattern 12: Shadow deployment + label-aware promotion (blue/green for batch ML)

### What it solves

The naive flow ("drift detected → train challenger → CV gate → promote") has
a subtle flaw: the CV gate compares challenger vs champion on data we ALREADY
HAVE, not on the new regime that triggered the drift. A challenger might
"win" CV by fitting to the historical distribution that's no longer relevant.

### How we implemented it

[`flows/shadow_promotion_flow.py`](../src/battery_pdm/flows/shadow_promotion_flow.py) +
shadow-aware updates to [`drain_predictor_flow`](../src/battery_pdm/flows/drain_predictor_flow.py) +
[`retraining_flow`](../src/battery_pdm/flows/retraining_flow.py).

The pattern:

```
Day 0 (drift detected):
  RetrainingFlow runs with --shadow-mode true
  → trains challenger, saves to outputs/models/<name>_shadow/
  → does NOT promote
  → champion stays live

Day 0-N (every scoring run):
  DrainPredictorFlow scores with champion (production scores)
  IF outputs/models/<name>_shadow/ exists:
    → ALSO scores with shadow
    → writes shadow_comparisons/shadow_h<ts>.parquet
       (logs both predictions side-by-side per site)
  → MLflow run logs the comparison

Day N+1 onward (daily):
  ShadowPromotionFlow runs
  → loads shadow_comparisons/
  → fetches realized labels from alarm stream
  → computes AUC/Brier of BOTH champion + shadow on shared labeled data
  → if shadow.AUC > champion.AUC + margin AND label_maturity >= min_days:
       → PROMOTE shadow → champion (atomic swap)
       → archive old champion
       → log to MLflow
  → else: keep shadow scoring in parallel another day
```

### What this enables

| | Without shadow | With shadow |
|--|---------------|-------------|
| Validate on real production data | ❌ only on historical CV | ✅ on actual mature labels |
| Detect challenger that overfits to drift period | ❌ | ✅ |
| Operator visibility into "challenger waiting" | ❌ | ✅ via MLflow run history |
| Easy rollback if shadow fails | manual | automatic ("discard shadow" decision) |
| Cost | low | ~2× scoring compute (acceptable) |

### Comparison to other deployment patterns

| Pattern | Where used | What we have |
|---------|------------|--------------|
| **Blue/green** | Standard for batch ML | ✅ champion/shadow → swap when validated |
| **Canary** (gradual traffic) | Real-time inference | ❌ N/A for batch |
| **Shadow** | Validation before promotion | ✅ exactly this |
| **A/B test** | Online services | ❌ N/A for batch |

---

## Pattern 13: Label-maturity gate (the "don't promote too early" guard)

### What it solves

For slow-feedback models (e.g., failure model with 6-12 month labels), even
a successful challenger should wait until enough realized outcomes prove the
improvement. Without this gate, you can promote on noise.

### How we implemented it

`check_label_maturity()` in [`monitoring/concept_drift.py`](../src/battery_pdm/monitoring/concept_drift.py)
wired into [`retraining_flow.py`](../src/battery_pdm/flows/retraining_flow.py) via
`--min-label-maturity-days` parameter.

```python
# In RetrainingFlow.compare step:
margin_check = (challenger_auc - champion_auc) >= promotion_margin
maturity_check = check_label_maturity(...).ready_to_promote
self.promote = margin_check and maturity_check
```

### Defaults per model

| Model | Suggested `min_label_maturity_days` | Why |
|-------|------------------------------------:|-----|
| Drain predictor (48h) | 7-14 | One-two weeks of post-drift labels |
| Autonomy (12h) | 3-7 | Labels mature quickly |
| Failure (6-12mo) | 180+ | The honest answer |
| Local dev | 0 (disabled) | Allows iteration |

---

## Pattern 14: Evidently AI as alternative drift implementation

### Why a swap option?

Our hand-rolled PSI in `monitoring/drift.py` works fine for our scale. But:
- Industry standard tooling (Evidently is the OSS leader)
- Battle-tested implementations of PSI, KS, JS, Wasserstein, MMD
- Standard reports interviewers recognize
- Easy to use richer drift metrics later

### How we implemented it

[`monitoring/evidently_drift.py`](../src/battery_pdm/monitoring/evidently_drift.py)
exposes the same `detect_drift()` interface as our PSI implementation.
Drop-in replacement.

```python
# Swap implementations without changing DriftMonitorFlow:
from battery_pdm.monitoring.drift import detect_drift                    # our PSI
from battery_pdm.monitoring.evidently_drift import detect_drift_evidently # alternative

# Same return shape, same downstream code
```

Dependency: `evidently>=0.4,<0.5` (added via `uv add`, tracked in pyproject.toml).

---

## How these patterns relate

| Pattern group | What it solves |
|---------------|----------------|
| 1-3 (drift, calibration, feature hash) | **Silent model degradation** — detect input/output/contract changes |
| 4-5 (atomic triggers, shared test sets) | **Safe deployment** — no half-promoted state, fair comparisons |
| 6-8 (cold-start, performance log, schedule ingestion) | **Operational robustness** — handle edge cases gracefully |
| 9-10 (group CV, point-in-time) | **Correctness at training time** — no leakage |
| **11-12 (concept drift, shadow promotion)** | **Labeled feedback loop** — validate retraining decisions empirically |
| **13 (label maturity)** | **Don't promote too early** — the long-feedback gate |
| **14 (Evidently)** | **Industry-standard tooling** — drop-in alternative |

Without ANY of these, the system could appear to "work" on day 1 and then
silently corrupt itself by day 30. The whole point of MLOps is making the
day-30 behavior trustworthy.

## Pattern 15: Continuous retraining — proactive parallel-path pattern

### What it solves

Reactive retraining (only when drift fires) has a blind spot: gradual
degradation that never crosses the PSI threshold individually but erodes model
quality over weeks. By the time drift fires, the model has been subtly worse
for days.

### The pattern (from ml.school)

Training and inference are PARALLEL PATHS:
- **Inference path:** champion serves predictions daily (unchanged, reliable)
- **Training path:** runs weekly on ALL latest data (including newly matured
  labels), trains a challenger, compares against champion on the SAME held-out
  test set, promotes immediately if CV gate passes

```
Weekly cron (Saturday 03:00 UTC):
    RetrainingFlow --force true
    ├── Train challenger on latest data
    ├── Evaluate champion AND challenger on identical held-out set
    ├── Challenger AUC > Champion AUC + margin?
    │   ├── YES → atomic promotion (archive old, copy new)
    │   └── NO  → discard challenger, champion stays
    └── RollbackMonitorFlow (next day) verifies production performance
```

### Why it works for drain predictor (but not failure model)

Labels mature in 48h. By Saturday, 7 days of new labeled data is available.
The held-out test set contains recent ground truth → fair, unbiased comparison.
Immediate promotion is safe because the rollback flow catches the rare case
where held-out CV was misleading.

For the failure model (6-12mo labels), we use **shadow mode** instead — deploy
challenger alongside champion, wait for realized labels, promote only when
production validates improvement.

### Key insight

Training is cheap (~5min Fargate Spot). The CV gate prevents regressions.
There's no risk in always training — only upside.

```python
# infra/modules/batch/main.tf — retraining job passes --force true
retraining = {
    command = ["python", "-m", "battery_pdm.flows.retraining_flow", "run",
               "--force", "true", ...]
}
```

### How this differs from drift-triggered retraining

| | Continuous (weekly) | Drift-triggered (reactive) |
|--|--|--|
| When | Every Saturday | When PSI > threshold |
| Why | Opportunity to improve | Emergency — model is provably degrading |
| Promotion | Immediate (CV-gated) | Immediate (CV-gated) |
| Safety net | RollbackMonitorFlow | RollbackMonitorFlow |

Both paths use the same `RetrainingFlow`. The weekly cron passes `--force true`
(train regardless of trigger). The drift monitor writes `retrain_trigger.json`
for mid-week emergencies.


## Pattern 16: Automated rollback

### What it solves

Even with CV gating, a promoted model can be worse in production if:
1. The held-out set didn't capture a covariate shift happening RIGHT NOW
2. A code bug in feature computation affects production but not the test set
3. Data quality degrades between training and serving

### The pattern

`RollbackMonitorFlow` runs daily (after scoring). If a model was promoted
within the last 48h and its production Brier score exceeds the training-time
Brier by more than a tolerance:

```
Production Brier > Training Brier × (1 + tolerance)
→ ROLLBACK: restore archived champion
```

### Implementation

```python
# src/battery_pdm/flows/rollback_monitor_flow.py
threshold = self.training_brier * (1 + self.brier_tolerance)  # 10% tolerance
self.should_rollback = self.production_brier > threshold
```

On rollback:
1. Save current (bad) model to `archive/{model}_rolled_back_{timestamp}/`
2. Restore previous champion from `archive/{model}_{timestamp}/`
3. Emit CloudWatch `ModelRollback` metric (pages on-call)
4. Log to performance log for post-mortem

### Why 48h window

- Drain predictor labels mature in 48h
- After promotion on Saturday, Sunday's scoring produces labeled predictions
- Monday's rollback check has enough observations to decide

### Schedule (EventBridge)

```
00:30 UTC  DrainPredictorFlow (score with current champion)
01:00 UTC  DriftMonitorFlow (safety net)
01:30 UTC  ShadowPromotionFlow (for failure model)
02:00 UTC  RollbackMonitorFlow (for drain predictor)
03:00 SAT  RetrainingFlow --force (continuous training)
```


## Platform-agnostic via MLflow

All monitoring events log to **MLflow** (file backend locally, S3-backed in
AWS). The system is fully observable WITHOUT AWS — you can run everything
locally and inspect via the MLflow UI.

| Event | MLflow experiment |
|-------|-------------------|
| Training | `battery-pdm-drain`, `battery-pdm-failure`, `battery-pdm-autonomy` |
| Drift detection | `battery-pdm-drift-{model}` |
| Concept drift | `battery-pdm-concept-drift-{model}` |
| Shadow scoring | `battery-pdm-shadow` |
| Shadow promotion decisions | `battery-pdm-shadow-promotion-{model}` |

To browse locally: `uv run mlflow ui --backend-store-uri file:./mlruns`.
In AWS: set `MLFLOW_TRACKING_URI=s3://your-bucket/mlflow` env var.

## Reading next

- [07_INTERVIEW_QA.md](07_INTERVIEW_QA.md) — questions someone might ask + how
  to answer
- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) — the AWS deployment
  architecture
