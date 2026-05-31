# EventBridge rules to trigger Batch jobs on a schedule.
# Daily: drain_predictor (00:30), drift_monitor (01:00), shadow_promotion (01:30), rollback_monitor (02:00)
# Weekly (Sunday 02:00): failure_scoring
# Weekly (Saturday 03:00): retraining drain (Santiago's parallel path)
# Weekly (Saturday 03:30): retraining failure (shadow mode, 6-12mo label horizon)

variable "name_prefix" {
  type = string
}

variable "job_queue_arn" {
  type = string
}

variable "job_definitions" {
  type = map(string)
}

# IAM role for EventBridge to submit Batch jobs
resource "aws_iam_role" "scheduler" {
  name = "${var.name_prefix}-scheduler-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "events.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "scheduler_batch" {
  name = "batch-submit"
  role = aws_iam_role.scheduler.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "batch:SubmitJob"
      Resource = "*"
    }]
  })
}

# Daily at 00:30 UTC — drain predictor
resource "aws_cloudwatch_event_rule" "drain_daily" {
  name                = "${var.name_prefix}-drain-daily"
  schedule_expression = "cron(30 0 * * ? *)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "drain_daily" {
  rule     = aws_cloudwatch_event_rule.drain_daily.name
  arn      = var.job_queue_arn
  role_arn = aws_iam_role.scheduler.arn

  batch_target {
    job_name       = "${var.name_prefix}-drain-scheduled"
    job_definition = var.job_definitions["drain_predictor"]
  }
}

# Daily at 01:00 UTC — drift monitor (after drain completes)
resource "aws_cloudwatch_event_rule" "drift_daily" {
  name                = "${var.name_prefix}-drift-daily"
  schedule_expression = "cron(0 1 * * ? *)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "drift_daily" {
  rule     = aws_cloudwatch_event_rule.drift_daily.name
  arn      = var.job_queue_arn
  role_arn = aws_iam_role.scheduler.arn

  batch_target {
    job_name       = "${var.name_prefix}-drift-scheduled"
    job_definition = var.job_definitions["drift_monitor"]
  }
}

# Weekly Sunday 02:00 UTC — failure scoring
resource "aws_cloudwatch_event_rule" "failure_weekly" {
  name                = "${var.name_prefix}-failure-weekly"
  schedule_expression = "cron(0 2 ? * SUN *)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "failure_weekly" {
  rule     = aws_cloudwatch_event_rule.failure_weekly.name
  arn      = var.job_queue_arn
  role_arn = aws_iam_role.scheduler.arn

  batch_target {
    job_name       = "${var.name_prefix}-failure-scheduled"
    job_definition = var.job_definitions["failure_scoring"]
  }
}

# Weekly Saturday 03:00 UTC — continuous retraining (Santiago's parallel path)
# PROACTIVE: always trains challenger, compares vs champion on same held-out set,
# promotes immediately if CV gate passes. Drift monitor is just a safety net.
resource "aws_cloudwatch_event_rule" "retraining_weekly" {
  name                = "${var.name_prefix}-retraining-weekly"
  schedule_expression = "cron(0 3 ? * SAT *)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "retraining_weekly" {
  rule     = aws_cloudwatch_event_rule.retraining_weekly.name
  arn      = var.job_queue_arn
  role_arn = aws_iam_role.scheduler.arn

  batch_target {
    job_name       = "${var.name_prefix}-retraining-scheduled"
    job_definition = var.job_definitions["retraining"]
  }
}

# Weekly Saturday 03:30 UTC — failure model retraining (shadow mode, 6-12mo label horizon)
resource "aws_cloudwatch_event_rule" "retraining_failure_weekly" {
  name                = "${var.name_prefix}-retraining-failure-weekly"
  schedule_expression = "cron(30 3 ? * SAT *)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "retraining_failure_weekly" {
  rule     = aws_cloudwatch_event_rule.retraining_failure_weekly.name
  arn      = var.job_queue_arn
  role_arn = aws_iam_role.scheduler.arn

  batch_target {
    job_name       = "${var.name_prefix}-retraining-failure-scheduled"
    job_definition = var.job_definitions["retraining_failure"]
  }
}

# Daily at 01:30 UTC — shadow promotion (for failure model only)
# Failure model uses shadow mode (6-12mo label horizon) so it needs
# realized labels from production to validate before promoting.
resource "aws_cloudwatch_event_rule" "shadow_promotion_daily" {
  name                = "${var.name_prefix}-shadow-promotion-daily"
  schedule_expression = "cron(30 1 * * ? *)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "shadow_promotion_daily" {
  rule     = aws_cloudwatch_event_rule.shadow_promotion_daily.name
  arn      = var.job_queue_arn
  role_arn = aws_iam_role.scheduler.arn

  batch_target {
    job_name       = "${var.name_prefix}-shadow-promotion-scheduled"
    job_definition = var.job_definitions["shadow_promotion"]
  }
}

# Daily at 02:00 UTC — rollback monitor (safety net for drain predictor)
# If a recently promoted model's Brier/AUC degrades within 48h of
# promotion, auto-reverts to the archived champion. Catches the rare
# case where held-out CV was misleading about production performance.
resource "aws_cloudwatch_event_rule" "rollback_monitor_daily" {
  name                = "${var.name_prefix}-rollback-monitor-daily"
  schedule_expression = "cron(0 2 * * ? *)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "rollback_monitor_daily" {
  rule     = aws_cloudwatch_event_rule.rollback_monitor_daily.name
  arn      = var.job_queue_arn
  role_arn = aws_iam_role.scheduler.arn

  batch_target {
    job_name       = "${var.name_prefix}-rollback-monitor-scheduled"
    job_definition = var.job_definitions["rollback_monitor"]
  }
}
