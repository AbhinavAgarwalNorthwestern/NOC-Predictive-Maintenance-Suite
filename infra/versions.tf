terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
  }

  # Remote state stored in S3 with DynamoDB lock — created by infra/bootstrap.
  # The whole point: state file lives in S3 (versioned + encrypted), so anyone
  # else on the team can `terraform apply` against the same infra without
  # local state files. DynamoDB lock prevents two people applying at once.
  backend "s3" {
    bucket         = "battery-pdm-tfstate-998716768706"
    key            = "infra/terraform.tfstate"
    region         = "ap-south-1"
    dynamodb_table = "battery-pdm-tfstate-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "battery-pdm"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
