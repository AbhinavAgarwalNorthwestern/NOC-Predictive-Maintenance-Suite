# NOC Predictive Maintenance Suite

End-to-end production ML system for predicting telecom backup-battery failures and drain events. Built around an **alarms-only architecture** (no telemetry required), deployed to AWS via Terraform, with proactive champion/challenger retraining, shadow promotion on realized labels, and automated rollback.

[![CI](https://github.com/AbhinavAgarwalNorthwestern/NOC-Predictive-Maintenance-Suite/actions/workflows/ci.yml/badge.svg)](https://github.com/AbhinavAgarwalNorthwestern/NOC-Predictive-Maintenance-Suite/actions)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org)
[![license](https://img.shields.io/badge/license-portfolio-lightgrey.svg)](#license)

---

## What this is

A real-world ML system, not a notebook. The full lifecycle runs on AWS Fargate Spot ($0 idle):

- **Daily scoring** of 500 sites for 48h drain risk
- **PSI drift detection** against training-time reference profile
- **Weekly champion/challenger retraining** with CV-gated promotion (proactive parallel-path pattern)
- **Shadow validation** on realized production labels before promoting
- **Automated rollback** within 48h if Brier degrades
- **Step Functions orchestration** with EventBridge cron triggers
- **GitHub Actions CI/CD** with 6-stage pipeline, branch protection, CODEOWNERS

```
        Champion              Drift Monitor              Retraining (weekly)
        scores daily   <----  catches shifts             trains challenger
        (drain + failure)     (safety net)               compares on shared test set
              |                                                  |
              v                                                  v
        drain_alerts/                                    Margin >= 0.005?
        failure_alerts/                                  +-- YES -> shadow
        (S3)                                             +-- NO  -> reject
                                                                  |
                                                                  v
                                                          Shadow Promotion (daily)
                                                          validates on realized labels
                                                          promotes when proven better
                                                                  |
                                                                  v
                                                          Rollback Monitor (daily)
                                                          auto-reverts within 48h
                                                          if production Brier degrades
```

## Headline results

| Model | Question | Metric | Score |
|-------|----------|--------|------:|
| Drain predictor (48h) | "Will it drain in the next 48 hours?" | AUC | **0.88** |
| Drain (calibrated) | "Is the predicted probability accurate?" | Brier | **0.074** (42% reduction vs raw) |
| Failure (survival) | "Should this battery be replaced?" | C-index | **0.90** |

Per-region drain predictor AUC ranges 0.76 (Peshawar) to 0.87 (Islamabad) — captures the regional physics differences.

## Architecture

| Layer | Components |
|-------|-----------|
| **Storage** | 4 S3 buckets (data, models, alerts, mlflow) |
| **Container registry** | ECR (Docker image with Python 3.12, uv, XGBoost) |
| **Compute** | AWS Batch on Fargate Spot, 8 job definitions |
| **Orchestration** | Step Functions (6 state machines) + Metaflow (DAG within each) |
| **Scheduling** | EventBridge (7 cron rules: daily scoring, weekly retraining) |
| **Dashboard** | Streamlit on ECS Fargate + ALB |
| **CI/CD** | GitHub Actions OIDC (no static keys), 6-stage pipeline |
| **IaC** | Terraform (all infra in code, portable across accounts) |

## Quick start

### Local (5 minutes)

```bash
git clone https://github.com/AbhinavAgarwalNorthwestern/NOC-Predictive-Maintenance-Suite.git
cd NOC-Predictive-Maintenance-Suite
uv sync
uv run pytest              # 93 tests pass
uv run python scripts/run_end_to_end.py   # full pipeline locally
```

### AWS deployment (30 minutes)

See [SETUP.md](SETUP.md) for the complete reproducible deployment guide.

```bash
cd infra
terraform init && terraform apply        # provisions all AWS resources
cd ..
docker build -t battery-pdm:latest .     # build ML image
# (push to ECR, then:)
uv run python scripts/register_step_functions.py   # creates 6 state machines
aws stepfunctions start-execution --state-machine-arn <arn> --name demo
```

## Champion / Challenger Lifecycle

The core ML pattern (proactive parallel path — not reactive):

1. **Champion serves** daily scoring + monitoring
2. **Drift monitor** runs daily as a safety net (not the trigger)
3. **Challenger trains weekly** regardless of drift (always tries to improve)
4. **CV gate** — challenger must beat champion by >= 0.005 AUC **on the same held-out test set** (apples to apples)
5. **Shadow mode** — challenger scores in parallel with champion
6. **Shadow promotion** — promoted only when realized production labels prove improvement
7. **Rollback** — if production Brier degrades > 10% within 48h, auto-reverts to archived champion

**Why this is safer than naive "drift -> retrain -> deploy":**
- Drift-triggered retraining can fit to a transient regime
- Weekly proactive retraining catches gradual degradation before drift threshold
- CV gate on shared test prevents regressions
- Shadow + label maturity ensures real-world validation, not just held-out CV
- Rollback catches the rare case where CV was misleading

## Scheduled jobs

| Time (UTC) | Job | Description |
|------------|-----|-------------|
| Daily 00:30 | drain_predictor | Score 500 sites for 48h drain risk |
| Daily 01:00 | drift_monitor | PSI feature + prediction drift detection |
| Daily 01:30 | shadow_promotion | Validate shadow on realized labels |
| Daily 02:00 | rollback_monitor | Auto-revert if production Brier degrades |
| Weekly Sun 02:00 | failure_scoring | Score failure risk (survival model) |
| Weekly Sat 03:00 | retraining (drain) | Champion/challenger CV-gated |
| Weekly Sat 03:30 | retraining (failure) | Shadow mode (6-12mo label horizon) |

## CI/CD pipeline

6-stage GitHub Actions workflow (`ci.yml`):

1. **Fast checks** — ruff lint + format + mypy
2. **Tests + coverage** — pytest with `--cov-fail-under=60`
3. **Security scan** — bandit
4. **Terraform validate** — `terraform fmt + validate`
5. **Docker build** — ML, dashboard, API images
6. **CI Success summary** — single required check for branch protection

Branch protection enforces: PR-only merges, CODEOWNERS review, status checks pass, conversation resolution. See `.github/BRANCH_PROTECTION.md`.

Local pre-push hook runs lint + tests before allowing `git push`.

## Project structure

```
src/battery_pdm/
    common/           shared model definitions, features, survival
    flows/            10 Metaflow flows (full ML lifecycle)
    monitoring/       drift detection, model registry, threshold
    aws/              S3 I/O, CloudWatch metrics
    synth/            synthetic data generation

infra/                Terraform (S3, ECR, IAM, Batch, ECS, EventBridge, SFN)
scripts/              register_step_functions.py, batch_entrypoint.sh, simulate_drift.py
notebooks/            5 analysis notebooks (storytelling)
tests/                93 passing tests
.github/workflows/    CI/CD (ci.yml, deploy.yml, register-flows.yml)
.devcontainer/        Linux-native dev environment
docs/                 Detailed architecture + interview Q&A
```

## Documentation

| Doc | Topic |
|-----|-------|
| [SETUP.md](SETUP.md) | Full deployment guide (local + AWS in 30 min) |
| [docs/01_problem_and_domain.md](docs/01_problem_and_domain.md) | Telecom NOC battery problem, business KPIs |
| [docs/02_data_and_features.md](docs/02_data_and_features.md) | Alarms-only architecture, feature engineering |
| [docs/03_models_and_choices.md](docs/03_models_and_choices.md) | Why XGBoost survival, why isotonic calibration |
| [docs/04_results_and_metrics.md](docs/04_results_and_metrics.md) | Every chart explained |
| [docs/05_architecture.md](docs/05_architecture.md) | System architecture, infra decisions |
| [docs/06_production_patterns.md](docs/06_production_patterns.md) | Champion/challenger, shadow, rollback, drift |
| [docs/07_interview_qa.md](docs/07_interview_qa.md) | 40+ interview questions with answers |
| [docs/noc_runbook.md](docs/noc_runbook.md) | Operator runbook (oncall procedures) |
| [.github/BRANCH_PROTECTION.md](.github/BRANCH_PROTECTION.md) | FAANG-style branch protection config |

## Notebooks (storytelling)

| Notebook | What you see |
|----------|--------------|
| [00_system_overview](notebooks/00_system_overview.ipynb) | System at a glance, headline metrics |
| [01_drift_detection_demo](notebooks/01_drift_detection_demo.ipynb) | Peshawar grid upgrade story — drift caught, why naive retraining fails |
| [02_model_evaluation_per_region](notebooks/02_model_evaluation_per_region.ipynb) | Calibration analysis, per-region AUC |
| [03_hyperparameter_tuning](notebooks/03_hyperparameter_tuning.ipynb) | Optuna HPO — defaults near-optimal, calibration matters more |
| [04_autonomy_vs_drain_comparison](notebooks/04_autonomy_vs_drain_comparison.ipynb) | Historical: analysis that led to decommissioning the autonomy model |

## Costs

| Resource | Monthly cost |
|----------|--------------|
| Fargate Spot (all batch jobs) | ~$5-10 |
| S3 storage | ~$1 |
| NOC dashboard (ECS Fargate) | ~$15 |
| ALB | ~$20 |
| **Total** | **~$40-45/mo** |

## License

Portfolio project. Not for production deployment without review.
