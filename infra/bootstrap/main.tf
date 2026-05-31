# Bootstrap: creates the S3 bucket + DynamoDB lock table that hold Terraform state.
# Run this ONCE before the main infra/ apply. Once created, all subsequent
# `terraform apply` runs use this remote state.
#
# Run from infra/bootstrap/:
#   terraform init
#   terraform apply -var="aws_account_id=YOUR_ACCOUNT" -var="aws_region=us-east-1"
#
# Then uncomment the backend "s3" block in ../versions.tf, and run
#   cd ..
#   terraform init -reconfigure

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.50" }
  }
}

variable "aws_region" {
  type = string
}

variable "aws_account_id" {
  type = string
}

variable "project_prefix" {
  type    = string
  default = "battery-pdm"
}

provider "aws" {
  region = var.aws_region
}

locals {
  tfstate_bucket   = "${var.project_prefix}-tfstate-${var.aws_account_id}"
  tfstate_lock_tbl = "${var.project_prefix}-tfstate-lock"
}

resource "aws_s3_bucket" "tfstate" {
  bucket = local.tfstate_bucket
  tags = { Purpose = "terraform-state" }
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "tfstate_lock" {
  name         = local.tfstate_lock_tbl
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"
  attribute {
    name = "LockID"
    type = "S"
  }
}

output "tfstate_bucket" { value = aws_s3_bucket.tfstate.bucket }
output "tfstate_lock_table" { value = aws_dynamodb_table.tfstate_lock.name }
output "backend_block" {
  value = <<-EOT

  Add this to ../versions.tf inside the terraform { ... } block:

  backend "s3" {
    bucket         = "${aws_s3_bucket.tfstate.bucket}"
    key            = "infra/terraform.tfstate"
    region         = "${var.aws_region}"
    dynamodb_table = "${aws_dynamodb_table.tfstate_lock.name}"
    encrypt        = true
  }

  EOT
}
