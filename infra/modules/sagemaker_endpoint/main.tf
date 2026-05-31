variable "name_prefix" { type = string }
variable "model_name" { type = string }
variable "models_bucket" { type = string }
variable "ecr_image_uri" { type = string }
variable "instance_type" {
  type    = string
  default = "ml.t3.medium"
}
variable "data_capture_bucket" {
  description = "S3 bucket for SageMaker Data Capture (inference traffic logging)"
  type        = string
  default     = ""
}
variable "enable_data_capture" {
  description = "Enable SageMaker Data Capture for monitoring inference traffic"
  type        = bool
  default     = true
}

data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_exec" {
  name               = "${var.name_prefix}-sagemaker-${var.model_name}"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "s3" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

resource "aws_sagemaker_model" "this" {
  name               = "${var.name_prefix}-${var.model_name}"
  execution_role_arn = aws_iam_role.sagemaker_exec.arn
  primary_container {
    image          = var.ecr_image_uri
    model_data_url = "s3://${var.models_bucket}/${var.model_name}/model.tar.gz"
    environment = {
      SAGEMAKER_PROGRAM = "serve.py"
      SAGEMAKER_SUBMIT_DIRECTORY = "/opt/ml/model"
    }
  }
}

locals {
  capture_bucket = var.data_capture_bucket != "" ? var.data_capture_bucket : var.models_bucket
}

resource "aws_sagemaker_endpoint_configuration" "this" {
  name = "${var.name_prefix}-${var.model_name}-config"
  production_variants {
    variant_name           = "default"
    model_name             = aws_sagemaker_model.this.name
    initial_instance_count = 1
    instance_type          = var.instance_type
  }

  dynamic "data_capture_config" {
    for_each = var.enable_data_capture ? [1] : []
    content {
      enable_capture              = true
      initial_sampling_percentage = 100
      destination_s3_uri          = "s3://${local.capture_bucket}/data-capture/${var.model_name}"

      capture_options {
        capture_mode = "Input"
      }
      capture_options {
        capture_mode = "Output"
      }

      capture_content_type_header {
        json_content_types = ["application/json"]
      }
    }
  }
}

resource "aws_sagemaker_endpoint" "this" {
  name                 = "${var.model_name}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.this.name
}

output "endpoint_name" { value = aws_sagemaker_endpoint.this.name }
output "data_capture_s3_uri" {
  value = var.enable_data_capture ? "s3://${local.capture_bucket}/data-capture/${var.model_name}" : ""
}
