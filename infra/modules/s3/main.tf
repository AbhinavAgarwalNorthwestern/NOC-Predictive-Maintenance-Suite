variable "name_prefix" { type = string }
variable "aws_account_id" { type = string }

# Three logical buckets, all encrypted, versioned, public-blocked
locals {
  buckets = {
    data    = "${var.name_prefix}-data-${var.aws_account_id}"
    models  = "${var.name_prefix}-models-${var.aws_account_id}"
    alerts  = "${var.name_prefix}-alerts-${var.aws_account_id}"
    mlflow  = "${var.name_prefix}-mlflow-${var.aws_account_id}"
  }
}

resource "aws_s3_bucket" "this" {
  for_each = local.buckets
  bucket   = each.value
  tags     = { Role = each.key }
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each                = aws_s3_bucket.this
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle rule on alerts bucket — archive old alerts to Glacier after 90 days
resource "aws_s3_bucket_lifecycle_configuration" "alerts" {
  bucket = aws_s3_bucket.this["alerts"].id
  rule {
    id     = "archive-old-alerts"
    status = "Enabled"
    filter {}
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    expiration { days = 730 }
  }
}

output "bucket_names" {
  value = { for k, v in aws_s3_bucket.this : k => v.bucket }
}

output "bucket_arns" {
  value = { for k, v in aws_s3_bucket.this : k => v.arn }
}
