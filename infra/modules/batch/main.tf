# AWS Batch on Fargate Spot — serverless containers for ML jobs.
# Pay per second only when jobs run. ~70% cheaper than on-demand Fargate.

variable "name_prefix" {
  type = string
}

variable "task_role_arn" {
  type = string
}

variable "container_image" {
  description = "Full ECR image URI with tag, e.g. 123.dkr.ecr.../battery-pdm-dev:latest"
  type        = string
}

variable "data_bucket" {
  type = string
}

variable "models_bucket" {
  type = string
}

variable "alerts_bucket" {
  type = string
}

# Use default VPC + subnets (free, no NAT gateway cost)
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Outbound-only security group (Batch tasks pull ECR + write S3 over the internet)
resource "aws_security_group" "batch" {
  name        = "${var.name_prefix}-batch-sg"
  description = "Outbound only for ${var.name_prefix} Batch tasks"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Fargate Spot compute environment — pay per second
resource "aws_batch_compute_environment" "fargate_spot" {
  compute_environment_name = "${var.name_prefix}-fargate-spot"
  type                     = "MANAGED"
  state                    = "ENABLED"

  compute_resources {
    type               = "FARGATE_SPOT"
    max_vcpus          = 16
    subnets            = data.aws_subnets.default.ids
    security_group_ids = [aws_security_group.batch.id]
  }
}

resource "aws_batch_job_queue" "main" {
  name     = "${var.name_prefix}-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.fargate_spot.arn
  }
}

data "aws_region" "current" {}

# One job definition per flow. Each is invoked with a different command.
locals {
  # Entrypoint syncs S3 → /app/outputs before flow, syncs results back after.
  # Flows use their default local paths (outputs/alarms.parquet, outputs/models/, etc.)
  flows = {
    drain_predictor = {
      command = ["python", "-m", "battery_pdm.flows.drain_predictor_flow", "run"]
      memory  = 4096
      vcpu    = 2
    }
    failure_scoring = {
      command = ["python", "-m", "battery_pdm.flows.failure_scoring_flow", "run"]
      memory  = 4096
      vcpu    = 2
    }
    drift_monitor = {
      command = ["python", "-m", "battery_pdm.flows.drift_monitor_flow", "run"]
      memory  = 2048
      vcpu    = 1
    }
    retraining = {
      command = ["python", "-m", "battery_pdm.flows.retraining_flow", "run", "--force", "true"]
      memory  = 8192
      vcpu    = 4
    }
    retraining_failure = {
      command = ["python", "-m", "battery_pdm.flows.retraining_flow", "run", "--force", "true", "--model-name", "failure_alarms_only", "--shadow-mode", "true"]
      memory  = 8192
      vcpu    = 4
    }
    training = {
      command = ["python", "-m", "battery_pdm.flows.training_flow", "run", "--n-sites", "100"]
      memory  = 8192
      vcpu    = 4
    }
    shadow_promotion = {
      command = ["python", "-m", "battery_pdm.flows.shadow_promotion_flow", "run"]
      memory  = 2048
      vcpu    = 1
    }
    rollback_monitor = {
      command = ["python", "-m", "battery_pdm.flows.rollback_monitor_flow", "run"]
      memory  = 2048
      vcpu    = 1
    }
  }
}

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/${var.name_prefix}"
  retention_in_days = 30
}

resource "aws_batch_job_definition" "this" {
  for_each              = local.flows
  name                  = "${var.name_prefix}-${each.key}"
  type                  = "container"
  platform_capabilities = ["FARGATE"]

  container_properties = jsonencode({
    image            = var.container_image
    command          = each.value.command
    jobRoleArn       = var.task_role_arn
    executionRoleArn = var.task_role_arn
    resourceRequirements = [
      { type = "VCPU", value = tostring(each.value.vcpu) },
      { type = "MEMORY", value = tostring(each.value.memory) },
    ]
    environment = [
      { name = "USERNAME", value = "batch-runner" },
      { name = "PYTHONPATH", value = "/app/src" },
      { name = "DATA_BUCKET", value = var.data_bucket },
      { name = "MODELS_BUCKET", value = var.models_bucket },
      { name = "ALERTS_BUCKET", value = var.alerts_bucket },
      { name = "METAFLOW_DEFAULT_DATASTORE", value = "s3" },
      { name = "METAFLOW_DATASTORE_SYSROOT_S3", value = "s3://${var.data_bucket}/metaflow" },
      { name = "METAFLOW_DEFAULT_METADATA", value = "local" },
      { name = "METAFLOW_CARD_S3ROOT", value = "s3://${var.data_bucket}/metaflow/cards" },
    ]
    networkConfiguration = {
      assignPublicIp = "ENABLED"
    }
    fargatePlatformConfiguration = {
      platformVersion = "LATEST"
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch.name
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = each.key
      }
    }
  })
}

output "job_queue_arn" {
  value = aws_batch_job_queue.main.arn
}

output "job_queue_name" {
  value = aws_batch_job_queue.main.name
}

output "job_definitions" {
  value = { for k, v in aws_batch_job_definition.this : k => v.name }
}
