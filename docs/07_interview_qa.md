# 07. Interview Q&A — anticipated questions with full answers

Questions are grouped by topic. Each has a concise answer (interview-ready)
plus the deeper reasoning if pressed.

---

## Topic A: Problem framing

### Q1. Why did you build three models instead of one?

**Short answer:** Different operational decisions have different latency
budgets and time horizons.

- **Autonomy model:** triggered by `AC_MAINS_FAIL` events, scores hours-to-LVD
  for *this specific outage*. Need it within seconds.
- **Drain predictor:** screens all sites daily, scores P(drain in next 48h).
  Operator plans for 1-2 days out.
- **Failure model:** scores entire fleet weekly, ranks for monthly replacement
  procurement.

One model would either (a) be at the lowest common denominator latency
(weeks) — useless for real-time dispatch, or (b) be a complex multi-task
network trying to predict three different things simultaneously — harder to
maintain and explain.

### Q2. Why alarms-only? Why not use voltage telemetry?

Three reasons:
1. **Production reality:** most telecom NOCs lack good telemetry. Voltage
   sensors are unreliable, polls are coarse (5-15 min), gaps are common.
2. **Performance:** alarms-only failure model still hits C-index 0.90.
   Marginal value of adding telemetry is small.
3. **Operational uniformity:** a model that works the same across all sites
   is more valuable to operations than one that's slightly better where
   telemetry is good.

The feature pipeline IS extensible — we could add a `telemetry_stats` feature
group via `@register_feature_group` — but it's not required.

### Q3. Why XGBoost and not a neural network?

Sample size is small (~25k training observations after group splitting).
Tabular features with 38 columns. Tree models excel at this scale; neural
nets typically need 10× more data for tabular tasks.

Plus:
- XGBoost has first-class `survival:cox` objective (lifelines + custom
  gradient boosting would be fragile)
- Calibration via isotonic is straightforward on top of probabilities
- Feature importance + SHAP for stakeholder explainability
- CPU-only inference, ~1MB model size

A neural net might get +1-2% AUC at the cost of significant complexity. Not
worth it.

### Q4. What's the ROI of this system to the business?

Estimates based on the precision/recall achieved (at threshold 0.5 calibrated):
- **~85-90% recall** on drain events → prevent 85-90% of LVD incidents at
  flagged sites
- **~30-40% precision** → 2-3 false dispatches per real drain prevented
- Net effect: 4-5× efficiency vs reactive baseline dispatch

If you assume ~$1000 cost per SLA-breaching LVD event and ~$50 cost per
false dispatch, the breakeven is at 50 false dispatches per 1 true drain
prevented. We're operating at ~2.5 false per real → ~20× ROI.

Replacement decisions: C-index 0.90 means the top-X% list for replacement
is reliable. Compared to round-robin replacement, this prevents ~30% of
catastrophic failures by replacing the right batteries first.

---

## Topic B: Model selection and metrics

### Q5. Why C-index instead of MAE or RMSE for the failure model?

C-index measures *ranking* accuracy with censoring. For replacement
decisions, ranking is what matters — we want to pick the X riskiest batteries
to replace next month, not predict their exact failure date.

MAE/RMSE on absolute failure time would be a different objective (AFT — the
`survival:aft` XGBoost variant). We tried this and it underperformed because
right-censoring biases the absolute-time estimates strongly. The censored
batteries (didn't fail in the window) pull the model toward over-predicting
survival time. Cox + ranking is more robust.

If a stakeholder asked "when will this battery fail?" we'd return a
conditional median time-to-event with a confidence interval, derived from the
Cox hazard curve. We don't currently expose this because the operational use
case (procurement) doesn't need it.

### Q6. AUC 0.89 sounds good — but you said precision is only 30%. Reconcile.

AUC measures *ranking ability* across all thresholds. Precision measures
*operational accuracy* at a SPECIFIC threshold (e.g., 0.5).

Our model ranks well (AUC 0.89) but the precision at any reasonable
threshold is limited by:
1. **Base rate:** only 15% of observations are positive. Even a perfect AUC
   model would have limited precision because there are 5× more negatives.
2. **Operational tradeoff:** we deliberately tune for high recall (85-90%)
   because missing a drain costs ~10× more than a false alarm.

If we wanted higher precision, we'd raise the threshold to 0.7 → precision
0.42, recall 0.55, F1 0.48. The choice is operational.

### Q7. What's a Brier score and why is it more important than AUC for production?

Brier = mean squared error between predicted probabilities and actual
outcomes. Lower is better; 0.25 = random (predicting base rate uniformly),
0 = perfect.

It captures BOTH ranking AND calibration. AUC only captures ranking.

For production: we want operators to be able to use the predicted probabilities
in cost-benefit math. "Expected dispatch cost = p × $100. Expected miss cost
= (1-p) × $1000. Dispatch if expected miss > expected dispatch." For this
to work, p has to be a real probability.

Our raw model had Brier 0.132 (AUC 0.89). After isotonic calibration: Brier
0.078 (AUC still 0.89 because calibration is monotonic). The 41% Brier
improvement means operators can now trust the probabilities for thresholding
decisions.

### Q8. Why isotonic instead of Platt scaling?

| Method | Fits | When to use |
|--------|------|-------------|
| Platt | A sigmoid: `1 / (1 + exp(A*raw + B))` | <1000 cal samples, or known-sigmoidal miscalibration |
| Isotonic | A monotonic step function | ≥1000 samples; arbitrary monotonic shape (typical for tree models) |

We have ~2000 calibration samples. Trees produce non-sigmoidal miscalibration
because the score histogram is multi-modal. Isotonic is the textbook choice
for gradient-boosted trees with enough cal data.

### Q9. How do you handle class imbalance?

For our 15% positive rate, we use XGBoost's `scale_pos_weight = neg/pos ≈ 5.7`
and `aucpr` as eval metric.

We did NOT use SMOTE/oversampling because:
- Tree models overfit to synthetic noise
- We have enough data, just need to weight it
- `scale_pos_weight` is the standard textbook choice for moderate imbalance

We did NOT use focal loss because it's overkill at 15% imbalance — focal is
typically used at 1% or extreme imbalance.

### Q10. What's the per-region experiment and what did it tell you?

We ran an ablation: global model (1 model, region as feature) vs per-region
(5 models, one per region) vs hybrid (global + region prior offset).

| Strategy | AUC |
|----------|----:|
| Global | 0.890 |
| Per-region | 0.873 |
| Hybrid | 0.890 |

**Global won.** Per-region failed for two reasons:
1. Region is already a feature in the global model (XGBoost learns
   region-specific splits)
2. Per-region models suffer data starvation (Islamabad's per-region model
   had only 6 training positives → AUC collapsed to 0.58)

**The point of running this:** evidence-based architectural decision.
Implementing the popular thing (per-region) without testing would have made
the system worse.

---

## Topic C: Architecture

### Q11. Why Metaflow and not Airflow?

| | Metaflow | Airflow |
|--|----------|---------|
| Python authoring | FlowSpec class | DAG as config/code |
| AWS Batch integration | @batch decorator | Custom operator |
| Local-cloud parity | Excellent — same `python -m flow run` | Painful |
| State persistence | S3 datastore (native) | XCom (limited) |
| Cron | EventBridge external | Native |
| UI | Optional service | Required webserver |

For our use case (5 flows, daily/weekly cadence, AWS-only): Metaflow's
@batch decorator and S3 datastore are excellent. We don't need Airflow's
web UI because we're not running hundreds of flows.

If we had 50+ flows or cross-team workflow dependencies, Airflow would be the
right choice.

### Q12. Why AWS Batch and not SageMaker / Lambda / ECS?

- **Lambda:** 15-min limit, 10GB memory cap. Retraining runs ~10 min and
  uses 8GB. Cutting it too close, and any future heavier flow breaks.
- **ECS long-running service:** Pay 24/7 for compute that runs 5 min/day.
  Wasteful.
- **SageMaker Processing Jobs:** ~2× more expensive than Fargate Spot.
  Worth it if you need SageMaker's MLflow + model registry integration; we
  don't.
- **AWS Batch on Fargate Spot:** $0 idle, pay-per-second when jobs run,
  built-in retries, serverless. Right fit for cron-style ML batch.

### Q13. Why isn't there a real-time inference endpoint?

The use case doesn't need it. Operators have 4-12 hours from the moment
`AC_MAINS_FAIL` fires to LVD; a minute-level cron is plenty.

If a stakeholder needed sub-second scoring (e.g., real-time generator
dispatch automation), we'd wrap the model in a SageMaker endpoint or use
AWS Lambda + S3 model loading. ~50 lines of code; we just don't have the
use case.

### Q14. How would you scale to 100k sites?

Currently we handle 500 sites in synthetic data; the architecture supports
~10-50k sites without changes (Batch + S3 scale linearly).

At 100k+ sites:
1. **Feature computation** becomes the bottleneck (it iterates per-label).
   Refactor to vectorized DataFrame operations or move to a Spark job.
2. **Drift detection** stays cheap (PSI is O(features × bins), not
   O(observations)).
3. **Alert volume:** 100k sites × 0.15 drain rate × 5 alerts/site/day = 75k
   alerts/day. Need a "smart fan-out" layer that aggregates by region or
   severity before paging operators.
4. **Cost:** 100k sites in Batch Fargate Spot scoring is still <$10/day.
   The real cost driver is alerts processing downstream.

### Q15. Why didn't you deploy Metaflow Service?

Cost. The Metaflow Service requires:
- RDS Postgres (~$13/mo)
- ECS Fargate task (~$9/mo)
- ALB (~$22/mo)
- Total: ~$45/mo idle

For a single-developer system that runs 5 minutes per day, the cost wasn't
justified. We documented the upgrade path in
[`docs/PRODUCTION_DEPLOYMENT.md`](PRODUCTION_DEPLOYMENT.md) — it's a
Terraform-only change, not a code rewrite.

When you'd actually deploy it: team size >1, need centralized run history,
compliance requires audit trail of every training run.

---

## Topic D: Data and features

### Q16. Walk me through your feature pipeline.

Four feature groups, registered via a decorator:

```python
@register_feature_group("alarm_history", requires=("alarms",))
def _alarm_history_features(labels, alarms):
    # 16 counters from the alarm stream, point-in-time correct
    ...

@register_feature_group("site_static", requires=("site_static",))
def _site_static_features(labels, site_static):
    # 8 static features: load, region, manufacturer, age, etc.
    ...

@register_feature_group("soc_proxy", requires=("alarms", "site_static"))
def _soc_proxy_features(labels, alarms, site_static):
    # 9 features from a Coulomb-counted SoC ledger replay
    ...

@register_feature_group("load_shedding_schedule", requires=("schedule", "site_static"))
def _load_shedding_features(labels, schedule, site_static):
    # 6 features from the utility's published shed calendar
    ...
```

The caller composes which groups they want:

```python
features = compute_features(
    labels=labels,
    groups=["alarm_history", "site_static", "soc_proxy", "load_shedding_schedule"],
    inputs={"alarms": ..., "site_static": ..., "schedule": ...},
    ref_time_col="mains_fail_h",
)
```

Three architectural properties:
1. **Same code for all 3 models** — only `groups` arg differs
2. **Required inputs are declared** — missing data fails loud, not silent
3. **Point-in-time correct** — searchsorted slicing on sorted alarms

### Q17. How do you ensure point-in-time correctness?

Three patterns:

1. **searchsorted on sorted timestamps:**
   ```python
   ts = g["timestamp_h"].values
   idx_t0 = ts.searchsorted(t0, side="left")
   past = g.iloc[:idx_t0]  # strictly t < t0
   ```
   `side="left"` ensures alarms with `t == t0` are excluded.

2. **Lifecycle windowing for the failure model:**
   When a battery is replaced, the new lifecycle gets a new site_id. Features
   for the new lifecycle only see alarms from after the replacement.

3. **Group-constrained CV:**
   Train/test split by SITE not by OBSERVATION, so features don't leak
   from a site's future observations into training.

Unit-tested at `tests/test_features.py::test_compute_features_point_in_time_correct`.

### Q18. What about leaky features?

`charger_misconfigured` and `aging_multiplier` are leaky — both are used in
the simulator to generate failures. Explicit exclusion:

```python
LEAKY_FEATURES = {"charger_misconfigured", "aging_multiplier"}
feature_cols = [c for c in features.columns if c not in LEAKY_FEATURES]
```

We initially included these by mistake and got C-index 0.99+ on the failure
model — too good to be true. Code review caught it. With them excluded, we
get C-index 0.90 (strong, plausible for synthetic data).

### Q19. How do you handle missing data?

Three strategies:

1. **Refuse to score (cold-start):** sites with <3 lifetime alarms get
   `COLD_START` alert level + regional prior as risk score
2. **Track null rates as a feature:** `data_completeness_score` per row
   (fraction of features that are non-null non-zero)
3. **Warn loudly on schedule gaps:** Python warnings if a region is missing
   from the schedule data

We do NOT silently impute zeros. Insufficient data flows through to the
operator as an explicit signal.

---

## Topic E: Production patterns

### Q20. How does drift detection work?

PSI (Population Stability Index) per feature, comparing the current scoring
window against a **training-time reference profile** (saved when the model
was promoted).

- PSI < 0.1: stable
- 0.1-0.25: moderate drift
- 0.25+: significant drift

When 3+ features hit significant OR prediction PSI > 0.10, the drift monitor
writes `retrain_trigger.json`.

The key bug we avoided: making the reference profile from the FIRST
production window (which itself might be drifted). We save it at training
time, from training-time features.

### Q21. What's the Peshawar story?

In the drift simulation: we drop 60% of `AC_MAINS_FAIL` events in Peshawar
post month 24 (simulating a grid stabilizer upgrade). The drift monitor
catches it: 10/38 features at significant PSI.

But the COUNTERINTUITIVE finding: Peshawar's mean risk score goes *up*
(0.50 → 0.58) even though batteries are objectively healthier. Why?
- The model's #1 feature is `lvd_count_lifetime` (cumulative drain count)
- This counter only ever increases or stays flat
- We dropped only 25% of LVDs (not 60%) because worn batteries still drain
  during the remaining outages
- So lifetime counters grow → model predicts higher risk

**The lesson:** the model learned downstream symptoms (LVD counts), not the
root cause (outage frequency). Retraining alone won't fix this because the
intervention is transient (in 3 months the batteries get replaced and the
counters stabilize). The right operational response is **investigate, then
override manually for 30-90 days**, then retrain with new ground truth.

### Q22. How does champion/challenger work?

- The drift monitor writes `retrain_trigger.json` when significant drift is
  detected
- The `RetrainingFlow` runs daily; if a trigger exists, it claims it atomically
  (rename to `.processing.json`)
- Trains a challenger model with the new data (60/20/20 split, including a
  held-out calibration set)
- Loads the current champion and **re-evaluates it on the challenger's exact
  test set** (apples-to-apples comparison)
- Promotes only if `challenger_AUC - champion_AUC ≥ 0.005` (margin)
- On promotion: archives the previous champion to `outputs/models/archive/`,
  writes the new model artifacts including refreshed `reference_profile.json`
- Marks the trigger consumed (moves to `consumed_triggers/`)

If anything fails mid-flight: `release_retrain_trigger()` puts the trigger
back to pending so the next run retries.

### Q23. Feature hash validation — what does it protect against?

A SHA256 hash of the ordered feature column list is saved in `meta.json` at
training time. At scoring time, the same hash is recomputed and compared. If
the lists differ (in name OR order), scoring fails fast:

```python
def validate_feature_hash(meta, feature_cols_used):
    expected = meta.get("feature_hash")
    if expected is None: return  # legacy model
    actual = compute_feature_hash(feature_cols_used)
    if expected != actual:
        raise ValueError("Feature contract violation: ...")
```

This protects against train/serve skew: if someone adds a new feature to
the pipeline without retraining, or reorders meta.json, scoring breaks
loudly instead of silently producing garbage.

---

## Topic F: MLOps and operations

### Q24. How does retraining know it should NOT retrain?

The drift monitor only writes a trigger if:
1. ≥3 features show SIGNIFICANT drift (PSI > 0.25), OR
2. ≥30% of features show MODERATE+ drift, OR
3. Prediction PSI > 0.10

Without a trigger, the RetrainingFlow exits at the start step.

You can also force retraining manually via `--force` parameter (useful for
operator-initiated retraining after intervention investigation).

### Q25. How would you handle a 100GB alarm dataset?

Currently we load all alarms into memory. At 100GB:
1. **Partition by date/site in S3** (we already do date partitioning for
   incoming streams)
2. **Read only the relevant slice** — for daily scoring, we only need the
   last 30 days; for the failure model, we need the relevant lifecycle window
3. **Switch to Dask or Spark** for feature computation if the relevant slice
   is still GB-scale
4. **Materialize feature tables** instead of recomputing each run — a
   feature store would help here (Feast, SageMaker FS)

The feature pipeline itself is structured around per-label iteration, which
makes it amenable to chunked processing without code rewrites.

### Q26. What's missing for true production?

Honest gaps:
1. **Real-time scoring endpoint** — only if the use case demanded sub-minute
   latency (it doesn't)
2. **Metaflow Service** — for centralized run history across team. Not
   deployed because $45/mo cost wasn't justified.
3. **Per-region performance alerts** — we track per-region AUC but don't
   alert on individual regions yet
4. **Feature store** — useful if multiple models share features at scale
5. **Online A/B testing** — for real-time scoring; our champion/challenger
   is offline-only
6. **Multi-region failover** — single region (ap-south-1) — disaster recovery
   would need cross-region replication
7. **PagerDuty / Slack alerting** — currently logs only

Each of these is a known gap, with the upgrade path documented.

---

## Topic G: Engineering rigor

### Q27. How do you test ML code?

50 tests, four classes:

1. **Unit tests** for pure functions (PSI math, brier score, isotonic
   calibrator, lifecycle windowing). Fast, isolated.
2. **Feature pipeline tests** for compose-correctness, missing-input handling,
   point-in-time correctness assertions.
3. **Integration test** for the full MLOps loop — train → save artifacts →
   validate hash → detect drift → atomic trigger. Runs in <10 seconds.
4. **Physics tests** for the simulator (Arrhenius monotonicity, Coulomb
   counting closure, etc.).

Tests are under `tests/`, runnable via `uv run pytest`. CI integration is
done via GitHub Actions (see `docs/PRODUCTION_DEPLOYMENT.md`).

### Q28. How do you guard against silent failures?

Five mechanisms:

1. **Feature hash validation** at scoring time (Pattern 3 in
   [06_PRODUCTION_PATTERNS.md](06_PRODUCTION_PATTERNS.md))
2. **Drift detection** with training-time reference (Pattern 1)
3. **Data quality assessment** (cold-start, schedule region coverage,
   minimum alarms)
4. **Calibration evaluation** during training (Brier before/after — caught
   if calibration would make it worse)
5. **Atomic trigger consumption** (no double-processing, no lost triggers)

Plus the integration test runs the full pipeline end-to-end before any
deployment — if anything in the loop is broken, the test fails.

### Q29. How portable is this across AWS accounts?

Fully. The portability story:

1. `infra/terraform.tfvars` has all account-specific values (account_id,
   region, project_prefix)
2. `infra/bootstrap/main.tf` creates the tfstate bucket — run once per new
   account
3. `infra/main.tf` and modules are account-agnostic
4. The Docker image is the same across accounts; only the ECR URL differs
5. The flows use `s3://` paths configured via parameters, not hardcoded
   bucket names

To move to a new AWS account:
```bash
# 1. Update one file
vi infra/terraform.tfvars  # change aws_account_id

# 2. Run bootstrap
cd infra/bootstrap && terraform apply -var="aws_account_id=NEW" -var="aws_region=NEW"

# 3. Run main apply
cd .. && terraform init -reconfigure && terraform apply

# 4. Build + push image to the new ECR
aws ecr get-login-password ... | docker login ...
docker build ...
docker push ...

# 5. Upload data
aws s3 cp outputs/*.parquet s3://<new-data-bucket>/
```

Total time: ~30 minutes for someone familiar with the codebase.

---

## Topic H: ML decision trade-offs

### Q30. You over-confidence wasn't underconfidence. What was your earlier mistake?

In an earlier draft of notebook 02, I described the calibration plot as
"underconfident" when the curve was BELOW the diagonal. That's actually
overconfident (predictions exceed reality).

The cleaner framing:
- Curve above diagonal = predictions LOWER than reality = underconfident
- Curve below diagonal = predictions HIGHER than reality = overconfident

I caught this when reviewing the fresh calibration plots. The notebook is
updated. The operational implication shifts too: overconfident means
operators dispatch more often than needed; underconfident would mean missing
real drains.

### Q31. What would change if your raw model was UNDERconfident?

Then calibration would BOOST predictions in the high-score range. Operators
would dispatch MORE often, not less. The Brier improvement direction would
be the same (closer to truth = lower Brier), but the operational
implications flip.

This is why "calibration improves Brier 41%" isn't enough — you should know
WHICH DIRECTION the miscalibration is, because that determines the
operational implication.

### Q32. Why did you split 60/20/20 and not 70/15/15?

We tested both. 60/20/20 gave us 20% in the cal set and 20% in the test
set, which is enough for both:
- Isotonic calibration on 20% works (~1000-2000 samples — enough for the
  monotonic step function)
- Test set with 20% gives a stable AUC estimate

70/15/15 would have given less stable isotonic fit (~750 samples in cal).
80/10/10 would have made test AUC too noisy. 60/20/20 is the sweet spot for
our dataset size.

### Q33. Why not use sklearn's CalibratedClassifierCV?

CalibratedClassifierCV wraps a sklearn-compatible classifier and applies
calibration via cross-validation OR a held-out set. It works, but for
XGBoost we'd need a thin wrapper class to satisfy the sklearn estimator
API.

We chose direct use of `sklearn.isotonic.IsotonicRegression` because:
- No sklearn-API wrapping needed (XGBoost is C++, not a sklearn estimator)
- `iso.fit(raw_scores, y_true)` is one line; `iso.predict(raw_scores)` is
  one line
- Easier to save with pickle and load in the scoring flow

Functionally identical. Less ceremony.

---

## Topic I: General

### Q34. What would you do differently if you started over?

1. **Notebooks from day 1.** I built up scripts/ first and notebooks at the
   end. Notebooks would have made the iterative storytelling easier.
2. **CI/CD before AWS.** I built Terraform manually then planned GitHub
   Actions. The reverse order would have caught some IAM permission issues
   earlier.
3. **MLflow integration earlier.** I added it late; it would have made the
   model versioning story clearer from the start.
4. **Real-time scoring path.** I'd build at least a minimal SageMaker
   endpoint to demonstrate the pattern, even if the use case doesn't strictly
   require it.

### Q35. What's the single thing you're most proud of in this project?

The **honest finding sequence**:
1. Per-region modeling experiment that DISPROVED my hypothesis (global beats
   per-region)
2. Drift simulation showing Peshawar risk goes UP even though batteries are
   healthier (the "model learns symptoms not causes" finding)
3. Calibration story: raw model was over-confident, isotonic fixes Brier 41%

These aren't "I trained a model and got X AUC." They're evidence that I
**ran experiments**, **changed my mind based on data**, and **caught my own
mistakes**. That's senior-engineer behavior.

### Q36. What if I asked you to add SageMaker serving right now?

~3 hours of work. The plan:
1. Write a `serve.py` handler that loads the booster + calibrator + meta
   from S3, validates feature hash, applies calibrator, returns calibrated
   probability
2. Build a Docker image with SageMaker's required `/ping` and `/invocations`
   endpoints
3. Push to ECR
4. Add `infra/modules/sagemaker_endpoint/` Terraform creating:
   - SageMaker Model from the ECR image
   - EndpointConfig (instance type, etc.)
   - Endpoint
5. Wire a Lambda or API Gateway in front to translate from JSON request to
   SageMaker invocation

The reason I didn't do it: the daily-cadence use case doesn't need it.
Adding it would be a portfolio decoration, not a real architectural need.

### Q37. Walk me through the system end-to-end as if I were the operator.

```
00:00 UTC — daily cron fires DrainPredictorFlow
  → Reads alarms.parquet + site_static.parquet + load_shedding_schedule.parquet from S3
  → Computes features for all sites at current time (point-in-time correct)
  → Validates feature hash against meta.json
  → Assesses data quality — flags cold-start sites with regional prior
  → Loads booster.json + calibrator.pkl
  → Predicts: raw scores → isotonic calibration → calibrated probabilities
  → Tags alert levels (HIGH / MEDIUM / LOW / COLD_START)
  → Writes alerts.parquet to S3

01:00 UTC — daily cron fires DriftMonitorFlow
  → Computes the same features
  → Loads reference_profile.json (training-time distribution)
  → Computes PSI per feature, KS p-value, prediction drift
  → If retrain recommended: writes retrain_trigger.json to S3

02:00 UTC — daily cron fires RetrainingFlow
  → Atomically claims retrain_trigger.json if present
  → Trains a challenger model (60/20/20 split, with calibration)
  → Loads champion, re-evaluates it on challenger's test set
  → If challenger beats champion by 0.005+ on shared test set:
    → Archives champion to outputs/models/archive/
    → Promotes challenger
    → Refreshes reference_profile.json from challenger's training data
  → Marks trigger as consumed

Throughout — flows emit custom CloudWatch metrics:
  → ModelAUC, ModelCIndex, DriftSignificantFeatures, AlertsEmitted, etc.

CloudWatch dashboard shows live time-series of all of the above.

Operator opens the alerts.parquet daily:
  → Filters for HIGH or COLD_START
  → Cross-references with regional priorities
  → Dispatches generators or schedules truck rolls
```

That's the whole loop. No human in the training loop unless drift triggers
override review.

---

## Topic J: Things you might get wrong

These are the "gotcha" questions where a less-prepared candidate would
stumble. The answers are blunt.

### Q38. Why is AUC unchanged by calibration?

Because AUC measures *ranking*. Calibration is monotonic — it doesn't change
the order of scores, just remaps the values. A monotonic transform preserves
ranking. So AUC is invariant.

Brier and log-loss DO change because they measure absolute probability
quality, not ranking.

### Q39. Why is your failure model C-index 0.90?

This is plausible for synthetic data where the physics are known. We checked
for leakage:
1. **Leaky features.** Checked — `charger_misconfigured` and
   `aging_multiplier` are explicitly excluded.
2. **Random split leakage.** Checked — group-constrained CV by site.
3. **Synthetic data ceiling.** The simulator generates failures from a
   deterministic physics model. On real data we'd expect 0.80-0.88.

For an interview answer: "C-index 0.90 on synthetic data where the physics
are deterministic. Real data would likely give 0.80-0.88 due to unmeasured
environmental factors. The architecture and approach (alarms-only, Cox with
XGBoost, group-constrained CV, feature engineering with schedule) would
transfer; the absolute number wouldn't."

### Q40. Your drift detection just runs PSI on features — what about concept drift?

**Critical distinction:**

- **Feature drift** = input distribution shifts (P(X) changes). PSI catches this.
- **Concept drift** = relationship between X and Y changes (P(Y|X) changes).
  PSI alone CAN'T catch this — you need ground truth labels.

Our PSI on PREDICTIONS catches a proxy for concept drift: if predictions
shift but features haven't, that means the relationship has changed. But it
requires the features to actually move.

For true concept drift detection, you'd need:
1. A labeled feedback stream (which outage actually drained?)
2. A rolling AUC / Brier computation on recent labels
3. Trigger on rolling-metric degradation

This is the next step in the production hardening — we have all the pieces
(alerts get tagged with outcomes when LVD occurs; we can append to a labeled
log), just haven't built the feedback monitor yet. Documented as a gap.

### Q41. What if AWS Batch Spot interrupts your retraining flow mid-run?

Two safeguards:

1. **Atomic trigger consumption.** The flow's `start` step renames the
   trigger to `.processing.json`. If the flow crashes, `release_retrain_trigger()`
   in the exception handler renames it back. Next run picks it up.
2. **AWS Batch retries.** Batch job definitions have built-in retry on
   transient failures. The interrupted job is requeued automatically.
3. **No persistent in-memory state.** All artifacts written between steps
   go through S3. If a step is retried, it reads its inputs fresh from S3.

The flow is idempotent: running it twice produces the same result.

### Q42. Why retrain weekly instead of only when drift is detected?

**Short answer:** proactive parallel-path pattern. Training is cheap
(~5min Fargate Spot, ~$0.01). The CV gate prevents regressions. There's only
upside in always training.

**Deeper:** Reactive-only retraining has a blind spot: gradual degradation that
never crosses any single PSI threshold but erodes quality over weeks. Weekly
retraining catches it because each week has 7 more days of matured labels in
the training set. The drift monitor becomes a SAFETY NET for sudden mid-week
shifts, not the primary retraining trigger.

### Q43. How do you decide whether to promote immediately vs shadow-test?

**Decision rule:** depends on label feedback latency.

- **Drain predictor (48h labels):** promote immediately. By Saturday, held-out
  test data contains recent ground truth. Rollback flow as safety net.
- **Failure model (6-12mo labels):** shadow mode. Held-out CV uses only
  historical failures, which may not reflect current fleet composition. Need
  realized production labels before committing.

### Q44. What happens if a promoted model is actually worse in production?

**RollbackMonitorFlow** (daily, 02:00 UTC):
1. Checks if promotion happened in last 48h
2. Computes production Brier on realized labels
3. If `production_brier > training_brier × 1.10` → auto-revert to archived champion
4. Emits `ModelRollback` CloudWatch metric

This catches the case where held-out CV was misleading (e.g., covariate shift
happening between training and serving that the test set didn't reflect).

### Q45. How do the autonomy and drain predictor models compare?

They answer different operational questions:
- **Autonomy:** "Which of these 5 currently-failing sites gets the generator?"
  (real-time triage, relative ranking)
- **Drain predictor:** "Which sites will fail TOMORROW?" (proactive planning,
  calibrated probability, incorporates schedule features)

On the shared task of "predict LVD within 48h of AC_MAINS_FAIL," the drain
predictor typically wins on AUC because it has load-shedding schedule features
and SoC proxy — information the autonomy model doesn't use. But during active
outages, autonomy's hours-to-LVD ranking is more operationally useful for
dispatch decisions.

They're **complementary, not competing.**

---

## How to use this doc

For an interview prep day:
1. **Hour 1:** read 01_PROBLEM, 02_DATA, 03_MODELS
2. **Hour 2:** read 04_RESULTS, focus on the chart explanations
3. **Hour 3:** read 05_ARCHITECTURE, 06_PRODUCTION_PATTERNS
4. **Hour 4:** read this doc, write your own answers in your voice

If you can answer Q1-Q15 confidently in your own words, you're set. Q16-Q45
are deeper / nice-to-have.
