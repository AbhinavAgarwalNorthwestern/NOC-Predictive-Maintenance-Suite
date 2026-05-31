# 03. Models and the choices behind them

## TL;DR

Three models, all XGBoost, deliberately chosen for different reasons:

| Model | Algorithm | Why this and not something else |
|-------|-----------|--------------------------------|
| Failure (long-term) | `xgboost.survival:cox` | Right-censored survival data with non-linear feature interactions. Cox handles censoring; XGB handles interactions. |
| Drain predictor (48h) | `xgboost.binary:logistic` + isotonic | Binary classification with class imbalance. Isotonic for calibrated probabilities. |
| Autonomy (event-triggered) | `xgboost.survival:cox` | Time-to-event with censoring (some outages end before LVD). |

Below: why XGBoost over alternatives, why survival vs binary in each case, why
isotonic vs Platt for calibration, and why we DIDN'T pick deep learning, GBMs
other than XGBoost, or hand-crafted physics models.

## Why XGBoost specifically

### XGBoost vs Random Forest

| | Random Forest | XGBoost |
|--|--------------|---------|
| **Hyperparameter sensitivity** | Low — works out of the box | Medium — needs tuning, but defaults are fine for our scale |
| **Training speed** | Fast | Comparable with `tree_method=hist` |
| **Prediction quality on tabular** | Solid | Generally 1-3% better |
| **Built-in survival objective** | ❌ | ✅ `survival:cox`, `survival:aft` |
| **Built-in class-imbalance** | Manual `class_weight` | `scale_pos_weight` |
| **Production ecosystem** | sklearn — fine | XGBoost has C++ runtime, ONNX export, GPU |

For our case the `survival:cox` objective was decisive — without it we'd need
to wrap lifelines/scikit-survival with manual gradient boosting, fragile.

### XGBoost vs LightGBM / CatBoost

LightGBM and CatBoost are both solid alternatives. We picked XGBoost because:

1. **Survival objectives:** XGBoost has first-class `survival:cox` and `survival:aft`.
   LightGBM's survival support is community-only. CatBoost has it via Ranking.
2. **MLflow integration:** Both LightGBM and CatBoost work with MLflow, but
   XGBoost is the most-supported by far — fewer edge cases.
3. **Familiarity (anti-mistake):** XGBoost has the largest install base, most
   StackOverflow answers, most StackOverflow-fixed bugs. Lower operational risk.

Performance-wise, these three are within 1% on most benchmarks. The
infrastructure ecosystem matters more.

### XGBoost vs deep learning

We did NOT use a neural network. Reasons:

1. **Sample size is small.** We have ~25k training observations after group
   splitting. Tree models excel at this scale; NNs typically need 10× more.
2. **Tabular features.** Trees handle heterogeneous tabular data (counters,
   categoricals, continuous) better than NNs without significant feature
   engineering or embeddings.
3. **Calibration.** XGBoost binary:logistic outputs are easy to post-calibrate
   with isotonic. Deep nets are notoriously badly calibrated and require
   temperature scaling or full retraining with focal loss.
4. **Interpretability.** Feature importance (gain), partial dependence plots,
   SHAP — all out of the box for tree models. NN interpretability is much
   harder for stakeholders.
5. **Operational simplicity.** XGBoost models are small (~1MB) and run on CPU
   in milliseconds. NN models would need GPU or quantization for cost-parity.

In an interview, the right phrasing is: **"We picked XGBoost because the
problem doesn't actually need a neural network. Trees give us 0.89 AUC, full
calibration support, native survival objectives, and CPU-only inference. A deep
model would add cost and complexity without clear benefit."**

## Why the survival framing for two of the three

### What "survival modeling" actually buys you

A binary classifier asks: "will the event happen within the next N hours?" with
N fixed. You have to pick N at training time.

A survival model asks: "given the features, what's the *distribution* of
time-to-event?" One model handles all N. The risk score is monotonic with the
predicted hazard.

For the **failure model**, we genuinely don't know when each battery will fail.
Some don't fail at all during the observation window — they're
**right-censored**. Cox handles this natively:

```python
y_train[censored] = -np.abs(y_train[censored])  # negative time = censored
```

A binary classifier can't represent "this battery survived to observation end
but might fail next week." Cox can.

For the **autonomy model**, same logic: some outages end (grid restored) before
LVD, leaving us with a censored observation. The Cox formulation captures this.

For the **drain predictor**, the 48h horizon is operationally fixed (operators
plan a day ahead), so a binary classifier with a precise threshold is more
useful. We could have done a Cox model and computed P(LVD by 48h) from the
hazard curve, but it's overkill — a binary objective is simpler and just as good.

### Cox proportional hazards in 30 seconds

The model learns a hazard function `h(t | x) = h_0(t) * exp(β · x)`, where:
- `h_0(t)` is the baseline hazard (everyone shares it)
- `β · x` is the linear combination of features
- Risk score = `exp(β · x)`, used for ranking

XGBoost's `survival:cox` extends this to non-linear feature interactions via
trees while preserving the partial-likelihood loss for censored observations.

### C-index — what it is and isn't

The concordance index measures **ranking accuracy** with censoring:

> Of all pairs (i, j) where i is observed to fail before j, what fraction does
> the model rank correctly (higher risk for i)?

- C-index = 0.5: random ranking
- C-index = 1.0: perfect ranking
- C-index = 0.90 (our failure model): strong

**What C-index doesn't tell you:**
- It doesn't tell you about *absolute time* prediction accuracy. A model with
  C-index 0.90 might predict "60 days" when the truth is "365 days" — but it
  correctly ranked this battery as failing before a higher-risk one.
- For replacement decisions, C-index is what matters (you rank by risk).
- For absolute time predictions, you'd need an AFT (accelerated failure time)
  model and report MAE.

We tried AFT (`survival:aft`) and it underperformed — the censoring induces
strong bias when many batteries don't fail in the window. Cox + ranking is more
robust.

## Why binary:logistic + isotonic for the drain predictor

### The decision pipeline

1. **Binary objective:** for fixed 48h horizon, just classify.
2. **`scale_pos_weight` for class imbalance:** see below.
3. **Isotonic post-hoc calibration:** so emitted probabilities mean what they
   say.

### Class imbalance handling — the full discussion

Our positive class rate is ~15%. Standard techniques:

| Technique | We used? | Why / why not |
|-----------|---------|---------------|
| `scale_pos_weight = neg/pos ≈ 5.7` | ✅ yes | Built into XGBoost. Reweights positives in the loss. Standard for tree models with moderate imbalance. |
| `aucpr` eval metric | ✅ yes | Area under precision-recall is more sensitive to minority class than ROC AUC. |
| Stratified sampling | ✅ implicitly (group split preserves regional ratios) | Manual stratification within group split would over-engineer. |
| SMOTE / oversampling | ❌ no | Trees overfit to synthetic noise. Generally not recommended for XGBoost. |
| Undersample majority | ❌ no | Throws away data. Only useful for extreme imbalance (1% or less). |
| Focal loss | ❌ no | Overkill for 15%; typically used at 1% or less. |
| Calibrated decision threshold | ✅ yes (operationally — operators choose threshold based on calibrated probabilities) | Better than picking 0.5 blindly. |

**Result:** `scale_pos_weight` + `aucpr` reaches AUC 0.89 with no further tricks.

### Isotonic vs Platt scaling — the calibration choice

Raw XGBoost binary:logistic outputs are **systematically overconfident** for our
problem. After fitting on the training distribution, scores in the [0.7, 0.9]
range over-predict reality (calibration curve sits below the diagonal).

Two standard fixes:

| Method | Fits | When to use |
|--------|------|-------------|
| **Platt scaling** | A sigmoid: `1 / (1 + exp(A*raw + B))` | <1000 calibration samples, or known-sigmoidal miscalibration |
| **Isotonic regression** | A monotonic step function — arbitrary shape | ≥1000 calibration samples; tree-model outputs commonly have non-sigmoidal miscalibration |

We have ~2000 observations in our calibration set, more than enough for
isotonic. Trees often have non-monotonic-sigmoidal miscalibration (because the
score histogram is multi-modal), so isotonic is the textbook choice for
gradient-boosted trees.

We use `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`. The clip
ensures inference-time scores outside [0,1] don't crash (rare but possible at
boundary cases).

### Why a strict held-out calibration set

A common mistake: fit calibrator on the test set the model was already
evaluated on. This *deflates* the apparent improvement because the calibrator
overfits to the test distribution.

We split:
- **60% train** — for the booster
- **20% calibration** — strictly held out from training; used to fit isotonic
- **20% test** — strictly held out from both; the honest performance estimate

For the *final* deployment model, we retrain on train+calibration combined,
then refit the calibrator using the final booster's predictions on the same
calibration set. There's a small in-sample optimism here (the calibrator's
calibration set was now seen by the booster), but it's bounded and acceptable
for single-shot training.

In the `RetrainingFlow`, the calibration set stays strictly held out so
champion/challenger comparisons are fair.

### Brier score — what it is and what to read

Brier = mean((p_predicted - y_true)²). Lower is better.

| Brier | Interpretation |
|------:|----------------|
| 0.00 | Perfect predictor |
| 0.10-0.12 | Well-calibrated decent model |
| 0.13-0.18 | Decent ranking, poor calibration |
| 0.20+ | Problematic — either poor ranking, miscalibration, or both |
| 0.25 | Equivalent to predicting class base rate uniformly |

Our drain predictor: **0.132 raw → 0.078 calibrated** — a 41% reduction.

**Key insight:** AUC is **invariant to monotonic calibration** (it only measures
ranking). So AUC stays at 0.890 whether we calibrate or not. But Brier captures
both ranking AND probability accuracy. Calibration improves Brier without
changing AUC. This is exactly the story you want to tell about isotonic
post-processing: "AUC unchanged; Brier 41% better; operator decisions based on
probability are now trustworthy."

## Cross-validation strategy

For both classification and survival, we use **group-constrained k-fold CV**:

```python
def temporal_group_cv_splits(labels, n_splits=3, test_fraction=0.25, seed=42):
    """All lifecycles of one physical site stay together in train OR test."""
    rng = np.random.default_rng(seed)
    groups = labels["original_site_id"].values
    unique_groups = np.array(sorted(set(groups)))
    for fold in range(n_splits):
        rng.shuffle(unique_groups)
        split_idx = int(len(unique_groups) * (1 - test_fraction))
        train_groups = set(unique_groups[:split_idx])
        test_groups = set(unique_groups[split_idx:])
        train_mask = np.array([g in train_groups for g in groups])
        test_mask = np.array([g in test_groups for g in groups])
        yield fold, train_mask, test_mask
```

This prevents leakage from shared site-level features (load, region, etc.).
Without it the failure model's C-index would inflate to ~0.99 because the model
would essentially memorize "site X is high-risk" from training to test.

## Hyperparameters and why these specific values

Drain predictor:

```python
{
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",        # fast histogram-based splits
    "max_depth": 5,               # depth-5 trees — captures interactions, low overfitting
    "learning_rate": 0.05,        # small step → smoother fits
    "min_child_weight": 10,       # require ≥10 hess sum in leaf — prevents overfitting on minority class
    "subsample": 0.8,             # row subsampling per tree
    "colsample_bytree": 0.8,      # feature subsampling per tree
    "scale_pos_weight": ~5.7,     # class imbalance reweight
    "num_boost_round": 300,
    "early_stopping_rounds": 30,
}
```

### Hyperparameter tuning — measured, not assumed

We **ran** an Optuna sweep (Bayesian/TPE, 10 trials per model) to test
whether defaults were leaving meaningful performance on the table. Notebook:
[`notebooks/03_hyperparameter_tuning.ipynb`](../../notebooks/03_hyperparameter_tuning.ipynb).

Results on the held-out test set:

| Model | Metric | Default | Tuned | Lift | HPO time |
|-------|--------|--------:|------:|-----:|---------:|
| Drain predictor | AUC | 0.8902 | 0.8900 | **-0.0001** | 1.9 s |
| Failure model | C-index | 0.9549 | 0.9636 | **+0.0087** | 0.8 s |

Reading:
1. **Drain predictor:** tuning is statistical noise. Defaults are
   near-optimal at this scale (25k obs). At our threshold (0.5), an extra
   0.0001 AUC doesn't change any operator decision.
2. **Failure model:** +0.0087 C-index (+0.91% relative) is a real but small
   gain at C-index already ≥ 0.95. Operationally, the top-X% replacement
   list barely changes.
3. **Compute cost is negligible** (~2 seconds at our scale). So leaving HPO
   out of the production training flow is a choice about pipeline
   complexity, not compute budget.

**Conclusion:** keep defaults in production. Add HPO as an opt-in
`--with-hpo` parameter in `TrainingFlow` for major retrains where every
fraction of a percent matters. This was an explicit experiment, not an
assumption — the kind of measurement that distinguishes "thought about
tuning" from "ran tuning and verified the ROI."

For comparison: **calibration via isotonic regression reduced Brier by 41%**.
That's the change that actually moved the needle on operational quality.
HPO was 4-5 orders of magnitude smaller. **Calibration > HPO** for this
problem.

## What we explicitly didn't model

1. **Real-time scoring endpoint.** Could be built with SageMaker; the use case
   doesn't justify the cost.
2. **Multi-task learning** (one network predicting all three outputs). Possible
   with a shared encoder; gain is small and operational complexity is high.
3. **Probabilistic ensembles** (quantile predictions). Useful for cost-aware
   optimization downstream; out of scope for v1.
4. **Causal inference** (do X% better grids actually cause Y% fewer drains?).
   Different problem; needs randomized intervention data.
5. **Transformers / sequence models on the raw alarm timeline.** A `TabPFN`-style
   model could learn from sequences, but with our 25k observations and 38 features,
   trees win.

## Reading next

- [04_RESULTS_AND_METRICS.md](04_RESULTS_AND_METRICS.md) — every chart and what
  it means, with business implications
- [06_PRODUCTION_PATTERNS.md](06_PRODUCTION_PATTERNS.md) — calibration in the
  production loop
