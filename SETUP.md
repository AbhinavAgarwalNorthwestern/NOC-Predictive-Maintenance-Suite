# Battery PdM — Setup Guide

Predictive maintenance for telecom backup batteries. Two ML models (drain predictor + failure scoring), automated retraining with champion/challenger CV gate, shadow promotion on realized labels, automated rollback.

## Prerequisites

- Python 3.12+, [uv](https://docs.astral.sh/uv/) for package management
- Docker Desktop
- AWS CLI configured (`aws configure`)
- Terraform (`winget install Hashicorp.Terraform`)
- [just](https://github.com/casey/just) task runner (`scoop install just` or `winget install Casey.Just`)

## Quick Start (Local)

```bash
git clone https://github.com/AbhinavAgarwalNorthwestern/NOC-Predictive-Maintenance-Suite.git
cd battery-pdm
uv sync
just test          # run all tests
just run-e2e       # full pipeline: train -> score -> drift -> retrain
just lint          # ruff check
just fmt-check     # ruff format check
```

## AWS Deployment (30 minutes)

### 1. Infrastructure

```bash
cd infra
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

This creates:
- 4 S3 buckets (data, models, alerts, mlflow)
- ECR repository
- IAM roles (Batch task, SFN execution, GitHub Actions OIDC, EventBridge scheduler)
- Batch compute environment (Fargate Spot, $0 idle)
- Job queue + 8 job definitions (drain, drift, retraining, retraining_failure, failure_scoring, shadow_promotion, rollback_monitor, training)
- EventBridge schedules (7 cron rules)
- Step Functions IAM roles
- ECS service for NOC dashboard

### 2. Build and Push Docker Image

```bash
just docker-build
just docker-push
```

Or manually:
```bash
docker build -t battery-pdm:latest .
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.ap-south-1.amazonaws.com
docker tag battery-pdm:latest <ACCOUNT>.dkr.ecr.ap-south-1.amazonaws.com/battery-pdm-dev:latest
docker push <ACCOUNT>.dkr.ecr.ap-south-1.amazonaws.com/battery-pdm-dev:latest
```

### 3. Seed Data to S3

```bash
aws s3 cp outputs/alarms.parquet s3://battery-pdm-dev-data-<ACCOUNT>/alarms.parquet
aws s3 cp outputs/site_static.parquet s3://battery-pdm-dev-data-<ACCOUNT>/site_static.parquet
aws s3 cp outputs/labels.parquet s3://battery-pdm-dev-data-<ACCOUNT>/labels.parquet
aws s3 cp outputs/load_shedding_schedule.parquet s3://battery-pdm-dev-data-<ACCOUNT>/load_shedding_schedule.parquet
aws s3 sync outputs/models/ s3://battery-pdm-dev-models-<ACCOUNT>/
```

### 4. Register Step Functions State Machines

```bash
uv run python scripts/register_step_functions.py
```

This creates 7 Step Functions state machines, one per flow. Each submits a Batch job on Fargate Spot with S3 sync entrypoint.

### 5. Trigger the Full Lifecycle

```bash
# Score all sites
aws stepfunctions start-execution --state-machine-arn arn:aws:states:ap-south-1:<ACCOUNT>:stateMachine:battery-pdm-drain-predictor --name demo-drain

# Detect drift
aws stepfunctions start-execution --state-machine-arn arn:aws:states:ap-south-1:<ACCOUNT>:stateMachine:battery-pdm-drift-monitor --name demo-drift

# Retrain (champion vs challenger, CV-gated)
aws stepfunctions start-execution --state-machine-arn arn:aws:states:ap-south-1:<ACCOUNT>:stateMachine:battery-pdm-retraining --name demo-retrain

# Validate shadow on realized labels
aws stepfunctions start-execution --state-machine-arn arn:aws:states:ap-south-1:<ACCOUNT>:stateMachine:battery-pdm-shadow-promotion --name demo-shadow

# Auto-rollback if promoted model degrades
aws stepfunctions start-execution --state-machine-arn arn:aws:states:ap-south-1:<ACCOUNT>:stateMachine:battery-pdm-rollback-monitor --name demo-rollback
```

### 6. View Logs

```bash
just logs   # tail CloudWatch logs
```

Or from the AWS Console: CloudWatch > Log Groups > `/aws/batch/battery-pdm-dev`

## Architecture

```
alarms.parquet (S3)
    |
    v
DrainPredictorFlow -----> drain_alerts/ (S3)      scored daily at 00:30
    |                         |
    v                         v
DriftMonitorFlow -------> drift_reports/ (S3)      checked daily at 01:00
    |                     retrain_trigger.json
    v
RetrainingFlow ---------> champion vs challenger   weekly Saturday 03:00
    |                     on shared held-out set
    |                     margin >= 0.005 AUC?
    |
    +-- YES (drain) ----> shadow/ for parallel scoring
    +-- YES (failure) --> shadow/ (6-12mo label horizon)
    +-- NO -------------> challenger discarded
    |
    v
ShadowPromotionFlow ----> validates shadow on       daily at 01:30
    |                     realized production labels
    |                     promotes when proven better
    v
RollbackMonitorFlow ----> auto-reverts within 48h   daily at 02:00
                          if production Brier degrades
```

## Scheduled Jobs

| Time (UTC) | Job | Description |
|------------|-----|-------------|
| Daily 00:30 | drain_predictor | Score all sites for 48h drain risk |
| Daily 01:00 | drift_monitor | PSI-based feature + prediction drift detection |
| Daily 01:30 | shadow_promotion | Validate shadow model on realized labels |
| Daily 02:00 | rollback_monitor | Auto-revert if promoted model degrades within 48h |
| Weekly Sun 02:00 | failure_scoring | Score failure risk (survival model) |
| Weekly Sat 03:00 | retraining | Drain model champion/challenger (CV-gated) |
| Weekly Sat 03:30 | retraining_failure | Failure model champion/challenger (shadow mode) |

## CI/CD (GitHub Actions)

### Three workflows

1. **`ci.yml`** — runs on every PR + push to `main`. Six-stage pipeline:
   - **Stage 1 (fast-checks)** — ruff lint + format + mypy (gradual typing)
   - **Stage 2 (test)** — pytest with `--cov-fail-under=60` coverage gate
   - **Stage 3 (security)** — bandit security scan (advisory)
   - **Stage 4 (terraform-validate)** — `terraform fmt + validate` for infra
   - **Stage 5 (build)** — docker build for ML, dashboard, API images
   - **Stage 6 (ci-success)** — single summary check for branch protection

2. **`deploy.yml`** — on push to main: ECR push + ECS force-deploy (dashboard)

3. **`register-flows.yml`** — on push to main: re-registers Metaflow flows as
   per-step Step Functions state machines (`step-functions create --with batch`)

### Branch protection (FAANG-style)

Required configuration (see `.github/BRANCH_PROTECTION.md`):
- No direct push to `main` — every change goes through a PR
- Required status check: `CI Success` (the Stage 6 summary)
- Required PR review from CODEOWNERS (`.github/CODEOWNERS`)
- Up-to-date branches required before merge
- Conversation resolution required

### Local pre-push hook

`git push` runs `.git/hooks/pre-push` which executes lint + format + pytest
before allowing the push. Bypass with `git push --no-verify`.

### Pull request template

`.github/pull_request_template.md` enforces:
- What/Why/How structure
- Test plan checklist (lint, format, tests, terraform plan if infra)
- Risk assessment (reversibility, blast radius)

## Project Structure

```
src/battery_pdm/
    common/           # shared model definitions, features, survival utils
    flows/            # 10 Metaflow flows (the ML lifecycle)
    monitoring/       # drift detection, model registry, concept drift
    aws/              # S3 I/O, CloudWatch metrics, SageMaker packaging
    synth/            # synthetic data generation

infra/                # Terraform (S3, ECR, IAM, Batch, ECS, EventBridge, SFN)
scripts/              # register_step_functions.py, batch_entrypoint.sh, simulate_drift.py
notebooks/            # 5 analysis notebooks
tests/                # pytest suite
.github/workflows/    # CI/CD
.devcontainer/        # devcontainer for Linux-native Metaflow
```

## Champion / Challenger Lifecycle

The core ML lifecycle pattern (proactive parallel path):

```
Week N:                     Week N+1:                  Week N+2:
  Champion serves daily       Challenger trains          Shadow validates on
  scoring + monitoring        on latest data             realized production labels
       |                           |                           |
       v                           v                           v
  Drift monitor             Champion re-evaluated         Margin met?
  catches shifts            on challenger's test set      Maturity met?
  (safety net)              (apples to apples)            |
                                  |                       +-- YES -> promote shadow -> champion
                                  v                       +-- NO + maturity -> discard
                            Margin >= 0.005?              +-- NO + no maturity -> wait
                                  |
                            +-- YES (drain) -> shadow mode
                            +-- YES (failure) -> shadow mode (6-12mo label horizon)
                            +-- NO -> challenger discarded
```

**Why this is safer than reactive retraining:**
- Drift-triggered retraining can fit to a transient regime (e.g., temporary outage spike)
- Weekly proactive retraining catches gradual degradation before drift threshold
- CV gate on shared test set prevents regressions
- Shadow mode + label maturity ensures real-world validation, not just held-out CV
- 48h rollback window catches the rare case where CV was misleading

## Key Design Decisions

1. **Proactive retraining** (proactive retraining pattern) — always train a challenger weekly. Drift monitor is a safety net, not the trigger.
2. **Shadow mode** — challenger scores in parallel with champion. Promoted only when realized labels prove improvement.
3. **CV gate** — challenger must beat champion by >= 0.005 AUC on the same held-out test set. Prevents premature promotion.
4. **Label maturity** — failure model uses 6-12 month labels. Shadow stays until enough labels accumulate.
5. **Automated rollback** — if production Brier degrades > 10% within 48h of promotion, auto-reverts to archived champion.
6. **No feature store** — batch system, daily cadence, single feature pipeline. Feature hash validation prevents train/serve skew.
7. **Fargate Spot** — $0 idle cost. Pay per-second only when jobs run. ~70% cheaper than on-demand.

## Costs

| Resource | Monthly Cost |
|----------|-------------|
| Fargate Spot (all jobs) | ~$5-10 |
| S3 storage | ~$1 |
| NOC dashboard (ECS) | ~$15 |
| ALB | ~$20 |
| **Total** | **~$40-45/mo** |
