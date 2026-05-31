# main.tf — wires together the modules to deploy the full battery PdM stack.
# Read top-to-bottom. Each module call is one logical chunk of infra.

# ─────────────────────────────────────────────────────────────────────
# 1. Storage: 4 S3 buckets (data, models, alerts, mlflow)
# ─────────────────────────────────────────────────────────────────────
module "s3" {
  source         = "./modules/s3"
  name_prefix    = local.name_prefix
  aws_account_id = var.aws_account_id
}

# ─────────────────────────────────────────────────────────────────────
# 2. Container registry: 1 ECR repo to hold our Docker image
# ─────────────────────────────────────────────────────────────────────
module "ecr" {
  source = "./modules/ecr"
  name   = local.ecr_repo_name
}

# ─────────────────────────────────────────────────────────────────────
# 3. IAM: roles for Batch tasks + GitHub Actions OIDC
#    Note how this depends on S3 (uses module.s3.bucket_arns)
# ─────────────────────────────────────────────────────────────────────
module "iam" {
  source            = "./modules/iam"
  name_prefix       = local.name_prefix
  data_bucket_arn   = module.s3.bucket_arns["data"]
  models_bucket_arn = module.s3.bucket_arns["models"]
  alerts_bucket_arn = module.s3.bucket_arns["alerts"]
  mlflow_bucket_arn = module.s3.bucket_arns["mlflow"]
  github_repo       = var.github_repo
}

# ─────────────────────────────────────────────────────────────────────
# Outputs: visible after `terraform apply`. Used in next steps.
# ─────────────────────────────────────────────────────────────────────
output "ecr_repo_url" {
  description = "Push your Docker image here. Used by build + push step."
  value       = module.ecr.repository_url
}

output "data_bucket" {
  value = module.s3.bucket_names["data"]
}

output "models_bucket" {
  value = module.s3.bucket_names["models"]
}

output "alerts_bucket" {
  value = module.s3.bucket_names["alerts"]
}

output "mlflow_bucket" {
  value = module.s3.bucket_names["mlflow"]
}

output "task_role_arn" {
  value = module.iam.task_role_arn
}

output "github_actions_role_arn" {
  description = "Paste this into your GitHub repo secrets as AWS_DEPLOY_ROLE_ARN"
  value       = module.iam.github_actions_role_arn
}

# ─────────────────────────────────────────────────────────────────────
# 4. AWS Batch (Fargate Spot) — serverless containers for ML jobs
#    Creates: 1 compute environment, 1 job queue, 5 job definitions
#    (one per flow), 1 security group, 1 CloudWatch log group.
#    Cost idle: $0 (pay per-second only when a job runs).
# ─────────────────────────────────────────────────────────────────────
locals {
  container_image_uri = "${module.ecr.repository_url}:${var.container_image_tag}"
}

module "batch" {
  source          = "./modules/batch"
  name_prefix     = local.name_prefix
  task_role_arn   = module.iam.task_role_arn
  container_image = local.container_image_uri
  data_bucket     = module.s3.bucket_names["data"]
  models_bucket   = module.s3.bucket_names["models"]
  alerts_bucket   = module.s3.bucket_names["alerts"]
}

output "batch_job_queue" {
  description = "Submit jobs against this queue"
  value       = module.batch.job_queue_name
}

output "batch_job_definitions" {
  description = "Per-flow job definition names — use these in 'aws batch submit-job'"
  value       = module.batch.job_definitions
}

# ─────────────────────────────────────────────────────────────────────
# 9. SageMaker autonomy endpoint — DECOMMISSIONED
#    The autonomy model was consolidated into the drain predictor.
#    Real-time scoring is served by FastAPI on ECS (/predict endpoint).
#    Savings: ~$50/mo (ml.t3.medium)
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# 10. NOC Dashboard — Streamlit on ECS Fargate + ALB
#     Operator-facing UI: drain risk, replacement priority, anomaly
#     detection, drift monitoring, drift simulation.
#     Cost: ~$15/mo (0.5 vCPU + 1GB Fargate + ALB)
# ─────────────────────────────────────────────────────────────────────
module "noc_app" {
  source              = "./modules/noc_app"
  name_prefix         = local.name_prefix
  region              = var.aws_region
  account_id          = var.aws_account_id
  ecr_repository_url  = module.ecr.repository_url
  image_tag           = "dashboard-latest"
  data_bucket_name    = module.s3.bucket_names["data"]
  models_bucket_name  = module.s3.bucket_names["models"]
  alerts_bucket_name  = module.s3.bucket_names["alerts"]
}

module "schedules" {
  source          = "./modules/schedules"
  name_prefix     = local.name_prefix
  job_queue_arn   = module.batch.job_queue_arn
  job_definitions = module.batch.job_definitions
}

output "noc_app_dashboard_url" {
  description = "NOC Dashboard URL (Streamlit on ECS)"
  value       = module.noc_app.dashboard_url
}
