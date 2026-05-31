# All account-specific values live here. To move to a new AWS account:
#   1. Update terraform.tfvars with new account_id + region + project_prefix
#   2. Run `terraform init -reconfigure` (if backend changed)
#   3. Run `terraform apply`
# No code changes needed.

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "Target AWS account ID (12-digit string)"
  type        = string
}

variable "project_prefix" {
  description = "Prefix for all resource names. Change this per environment/account."
  type        = string
  default     = "battery-pdm"
}

variable "environment" {
  description = "Environment name: dev, staging, prod"
  type        = string
  default     = "dev"
}

variable "container_image_tag" {
  description = "Container image tag to deploy (set by CI/CD on each push)"
  type        = string
  default     = "latest"
}

variable "grafana_admin_password" {
  description = "Admin password for the managed Grafana instance"
  type        = string
  sensitive   = true
  default     = ""
}

variable "alarm_simulator_schedule" {
  description = "How often the Lambda alarm simulator runs (EventBridge rate)"
  type        = string
  default     = "rate(5 minutes)"
}

variable "drain_predictor_schedule" {
  description = "How often the drain predictor scoring runs"
  type        = string
  default     = "cron(0 0 * * ? *)" # daily at midnight UTC
}

variable "drift_monitor_schedule" {
  description = "How often the drift monitor runs"
  type        = string
  default     = "cron(0 1 * * ? *)" # daily at 1am UTC, after scoring
}

variable "retraining_check_schedule" {
  description = "How often to check for retrain trigger"
  type        = string
  default     = "cron(0 2 * * ? *)" # daily at 2am UTC, after drift check
}

variable "github_repo" {
  description = "GitHub repo in owner/name format (for OIDC trust policy)"
  type        = string
  default     = ""
}

locals {
  # Derived names — never edit these directly, edit project_prefix + environment
  name_prefix      = "${var.project_prefix}-${var.environment}"
  data_bucket_name = "${local.name_prefix}-data-${var.aws_account_id}"
  models_bucket_name = "${local.name_prefix}-models-${var.aws_account_id}"
  alerts_bucket_name = "${local.name_prefix}-alerts-${var.aws_account_id}"
  ecr_repo_name    = local.name_prefix
}
