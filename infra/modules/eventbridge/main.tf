# EventBridge cron schedules → submit AWS Batch jobs.
# This is what makes the system "live" — jobs run automatically.

variable "name_prefix" {
  type = string
}

variable "job_queue_arn" {
  type = string
}

variable "job_definitions" {
  description = "Map of flow_name -> Batch job definition name"
  type        = map(string)
}

variable "schedules" {
  description = "Map of flow_name -> cron expression"
  type        = map(string)
  default = {
    drain_predictor = "cron(0 0 * * ? *)" # daily 00:00 UTC
    drift_monitor   = "cron(0 1 * * ? *)" # daily 01:00 UTC
    retraining      = "cron(0 2 * * ? *)" # daily 02:00 UTC
    failure_scoring = "cron(0 3 ? * SUN *)" # weekly Sunday 03:00 UTC
  }
}

# IAM role that EventBridge uses to submit jobs to Batch
data "aws_iam_policy_document" "eventbridge_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eventbridge_to_batch" {
  name               = "${var.name_prefix}-eventbridge-to-batch"
  assume_role_policy = data.aws_iam_policy_document.eventbridge_assume.json
}

data "aws_iam_policy_document" "submit_batch" {
  statement {
    actions   = ["batch:SubmitJob"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "eventbridge_to_batch" {
  name   = "${var.name_prefix}-submit-batch"
  role   = aws_iam_role.eventbridge_to_batch.id
  policy = data.aws_iam_policy_document.submit_batch.json
}

# One rule + target per schedule
resource "aws_cloudwatch_event_rule" "this" {
  for_each            = var.schedules
  name                = "${var.name_prefix}-${each.key}"
  schedule_expression = each.value
  description         = "Trigger ${each.key} on schedule ${each.value}"
}

resource "aws_cloudwatch_event_target" "this" {
  for_each = var.schedules
  rule     = aws_cloudwatch_event_rule.this[each.key].name
  arn      = var.job_queue_arn
  role_arn = aws_iam_role.eventbridge_to_batch.arn

  batch_target {
    job_definition = var.job_definitions[each.key]
    job_name       = "${each.key}-scheduled"
  }
}

output "rule_names" {
  value = { for k, v in aws_cloudwatch_event_rule.this : k => v.name }
}
