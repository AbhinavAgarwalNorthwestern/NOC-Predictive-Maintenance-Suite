#!/usr/bin/env bash
set -euo pipefail

# Deploy the NOC dashboard to AWS ECS Fargate.
# Prerequisites: AWS CLI configured, Docker running, Terraform initialized.

REGION="${AWS_REGION:-ap-south-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-998716768706}"
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/battery-pdm"
IMAGE_TAG="dashboard-latest"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Battery PdM NOC Dashboard Deploy ==="
echo "Region:  ${REGION}"
echo "Account: ${ACCOUNT_ID}"
echo "ECR:     ${ECR_REPO}:${IMAGE_TAG}"
echo ""

# Step 1: Build Docker image
echo "[1/5] Building Docker image..."
docker build -t battery-pdm-noc:latest \
    -f "${PROJECT_ROOT}/dashboard/Dockerfile" \
    "${PROJECT_ROOT}"

# Step 2: Tag and push to ECR
echo "[2/5] Pushing to ECR..."
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker tag battery-pdm-noc:latest "${ECR_REPO}:${IMAGE_TAG}"
docker push "${ECR_REPO}:${IMAGE_TAG}"

# Step 3: Sync data to S3 (so dashboard has data to read)
echo "[3/5] Syncing outputs to S3..."
DATA_BUCKET="battery-pdm-dev-data-${ACCOUNT_ID}"
MODELS_BUCKET="battery-pdm-dev-models-${ACCOUNT_ID}"

aws s3 sync "${PROJECT_ROOT}/outputs/" "s3://${DATA_BUCKET}/" \
    --exclude "*.pyc" --exclude "__pycache__/*"
aws s3 sync "${PROJECT_ROOT}/outputs/models/" "s3://${MODELS_BUCKET}/" \
    --exclude "*.pyc" --exclude "__pycache__/*"

# Step 4: Terraform apply
echo "[4/5] Applying Terraform (noc_app module)..."
cd "${PROJECT_ROOT}/infra"
terraform apply -target=module.noc_app -auto-approve

# Step 5: Get dashboard URL
echo "[5/5] Dashboard deployed!"
DASHBOARD_URL=$(terraform output -raw noc_app_dashboard_url 2>/dev/null || echo "Check terraform output")
echo ""
echo "========================================="
echo "NOC Dashboard URL: ${DASHBOARD_URL}"
echo "========================================="
echo ""
echo "Wait 2-3 minutes for ECS task to start and S3 sync to complete."
echo "Health check: curl ${DASHBOARD_URL}/_stcore/health"
