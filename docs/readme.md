# Battery PdM — Documentation index

End-to-end documentation of the battery predictive-maintenance system. Each doc
is designed to be self-contained but the order below reads as a natural progression
for anyone reviewing the project.

> **HPO finding (March 2026):** an Optuna sweep over both the drain predictor
> and the failure model showed only sub-1% lift from hyperparameter tuning
> (drain predictor: AUC -0.0001; failure model: C-index +0.0087). The dominant
> improvements came from feature engineering and calibration (Brier -41%), not
> from tuning XGBoost defaults. Full experiment in
> [`notebooks/03_hyperparameter_tuning.ipynb`](../notebooks/03_hyperparameter_tuning.ipynb).

| # | File | What it covers | Time to read |
|---|------|----------------|-------------:|
| 1 | [01_PROBLEM_AND_DOMAIN.md](01_PROBLEM_AND_DOMAIN.md) | The telecom NOC battery problem, why ML matters, business KPIs | 10 min |
| 2 | [02_DATA_AND_FEATURES.md](02_DATA_AND_FEATURES.md) | Synthetic data design, alarms-only architecture, feature engineering, point-in-time correctness | 15 min |
| 3 | [03_MODELS_AND_CHOICES.md](03_MODELS_AND_CHOICES.md) | The three models, why XGBoost survival vs binary, alternatives considered, class imbalance, calibration | 15 min |
| 4 | [04_RESULTS_AND_METRICS.md](04_RESULTS_AND_METRICS.md) | Every result chart, how to read each metric, business implications, what's good and what's a gap | 20 min |
| 5 | [05_ARCHITECTURE.md](05_ARCHITECTURE.md) | System architecture, Metaflow, AWS Batch, S3 layout, why each choice | 15 min |
| 6 | [06_PRODUCTION_PATTERNS.md](06_PRODUCTION_PATTERNS.md) | Drift detection, calibration, champion/challenger, atomic triggers, cold-start, feature hash | 15 min |
| 7 | [07_INTERVIEW_QA.md](07_INTERVIEW_QA.md) | Anticipated interview questions with full answers — both ML and engineering | 20 min |
| 8 | [08_SAGEMAKER_VS_BATCH.md](08_SAGEMAKER_VS_BATCH.md) | Why AWS Batch over SageMaker, when SageMaker WOULD be right, phased upgrade path | 10 min |
| 9 | [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) | The full AWS architecture target + cost breakdown + migration path | 10 min |

**Total: ~2 hours to read every doc end-to-end.**

## Quick navigation by question

| You're wondering about... | Read |
|----------------------------|------|
| "What's the business problem?" | 01 |
| "Why alarms-only? Why not telemetry?" | 02 |
| "Why XGBoost? Why survival modeling for the failure case?" | 03 |
| "What do the AUC / Brier / C-index numbers actually mean here?" | 04 |
| "Why this exact stack of technologies?" | 05 |
| "How does drift detection avoid silent decay?" | 06 |
| "What questions would an interviewer ask?" | 07 |
| "Why didn't you use SageMaker?" | 08 |
| "How would I deploy this to a different AWS account?" | 09 |

## At-a-glance results

| Model | Question | Metric | Score |
|-------|----------|--------|------:|
| Failure (long-term) | "Replace this battery?" | C-index | 0.90 |
| Drain predictor (48h) | "Will it drain in 2 days?" | AUC | 0.83 |
| Drain predictor (calibrated Brier) | "Are probabilities trustworthy?" | Brier | 0.106 |
| Autonomy (event-triggered) | "How long once outage starts?" | C-index | 0.73 |

Detailed walkthrough of every result is in [04_RESULTS_AND_METRICS.md](04_RESULTS_AND_METRICS.md).
