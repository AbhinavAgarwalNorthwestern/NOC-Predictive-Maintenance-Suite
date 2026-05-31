# Battery PdM — task automation. Run `just <command>`.
# Install just: scoop install just  OR  winget install Casey.Just

default:
    @just --list

# Run all tests
test:
    uv run pytest tests/ -q

# Run tests with verbose output
test-v:
    uv run pytest tests/ -v --tb=short

# Run lint
lint:
    uv run ruff check src tests scripts

# Run format check
fmt-check:
    uv run ruff format --check src tests

# Format code
fmt:
    uv run ruff format src tests

# Run end-to-end pipeline locally
run-e2e:
    uv run python scripts/run_end_to_end.py

# Re-execute all notebooks
notebooks:
    uv run python notebooks/_build_notebooks.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# Build Docker image
docker-build:
    docker build -t battery-pdm:latest .

# Push to ECR (assumes AWS CLI configured)
docker-push:
    cmd /c "aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 998716768706.dkr.ecr.ap-south-1.amazonaws.com"
    docker tag battery-pdm:latest 998716768706.dkr.ecr.ap-south-1.amazonaws.com/battery-pdm-dev:latest
    docker push 998716768706.dkr.ecr.ap-south-1.amazonaws.com/battery-pdm-dev:latest

# Build + push in one shot
deploy-image: docker-build docker-push

# Terraform plan
tf-plan:
    cd infra && terraform plan -out=tfplan

# Terraform apply (after plan)
tf-apply:
    cd infra && terraform apply tfplan

# ---------------------------------------------------------------------------
# AWS — Metaflow Step Functions (Santiago pattern)
# ---------------------------------------------------------------------------
# Each flow DAG becomes an AWS Step Functions state machine.
# `sfn-create` registers it; `sfn-trigger` fires it.
# Steps run on AWS Batch via --with batch. No custom Lambda needed.

# Train both models on AWS Batch (one-shot, not scheduled)
aws-train:
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.training_flow --with batch --with retry run

# Create Step Functions state machines for all flows (run once after infra)
# Each @step becomes a separate Batch job with S3 data passing between steps.
sfn-create-all:
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.drain_predictor_flow --with batch --with retry step-functions create
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.drift_monitor_flow --with batch --with retry step-functions create
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.retraining_flow --with batch --with retry step-functions create
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.failure_scoring_flow --with batch --with retry step-functions create
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.shadow_promotion_flow --with batch --with retry step-functions create
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.rollback_monitor_flow --with batch --with retry step-functions create
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.training_flow --with batch --with retry step-functions create

# Trigger individual flows via Step Functions
sfn-trigger-train:
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.training_flow step-functions trigger
sfn-trigger-drain:
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.drain_predictor_flow step-functions trigger
sfn-trigger-drift:
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.drift_monitor_flow step-functions trigger
sfn-trigger-deploy:
    METAFLOW_PROFILE=production uv run python -m battery_pdm.flows.deployment_flow step-functions trigger

# ---------------------------------------------------------------------------
# AWS — Legacy Batch direct submit (still works, sfn-* preferred)
# ---------------------------------------------------------------------------

# Submit a drain predictor Batch job
submit-drain:
    aws batch submit-job --job-name drain-{{`date +%Y%m%d%H%M%S`}} --job-queue battery-pdm-dev-queue --job-definition battery-pdm-dev-drain_predictor

# Submit a drift monitor Batch job
submit-drift:
    aws batch submit-job --job-name drift-{{`date +%Y%m%d%H%M%S`}} --job-queue battery-pdm-dev-queue --job-definition battery-pdm-dev-drift_monitor

# Trigger the demo drift scenario
demo-drift:
    powershell .\scripts\demo_drift.ps1

# View dashboard URL
dashboard-url:
    cd infra && terraform output -raw dashboard_url

# Tail recent Batch logs
logs:
    aws logs tail /aws/batch/battery-pdm-dev --follow --since 30m

# Recent alerts in S3
recent-alerts:
    aws s3 ls s3://battery-pdm-dev-alerts-998716768706/drain_alerts/ --recursive | sort | tail -10

# Deploy model (local validation only — no SageMaker calls)
deploy-local:
    uv run python -m battery_pdm.flows.deployment_flow run --target local

# Deploy model to SageMaker (uploads model.tar.gz + smoke test)
deploy-sagemaker:
    uv run python -m battery_pdm.flows.deployment_flow run --target sagemaker --models-bucket battery-pdm-dev-models-998716768706

# View Metaflow cards (drift reports, deployment logs)
card-server:
    uv run python -m battery_pdm.flows.drift_monitor_flow card server

# MLflow UI (local; for S3-backed runs set MLFLOW_TRACKING_URI first)
mlflow-ui:
    uv run mlflow ui --backend-store-uri file:./mlruns

# Teardown EventBridge + Lambda (stop auto-running things, keep storage)
pause-auto:
    cd infra && terraform destroy -target=module.eventbridge -target=module.lambda_simulator -auto-approve

# Full teardown (deletes everything including S3 buckets — careful!)
teardown:
    cd infra && terraform destroy
