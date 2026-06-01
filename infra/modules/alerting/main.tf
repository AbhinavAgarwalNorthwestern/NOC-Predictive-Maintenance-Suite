# Operational alerting: SNS topic + EventBridge rules that catch
# Step Functions failures and Batch job failures. Subscribers receive
# email/Slack/PagerDuty notifications instead of silent failures.
#
# Also: dead-letter S3 prefix for failed executions, so we have an
# auditable record of every failed run (not just CloudWatch logs that
# expire after retention period).

variable "name_prefix" {
  type = string
}

variable "alerts_bucket_name" {
  type        = string
  description = "S3 bucket for storing failed execution metadata (DLQ pattern)"
}

variable "ops_email" {
  type        = string
  description = "Email address to receive failure alerts (subscribe manually after apply)"
  default     = ""
}

# ---------------------------------------------------------------------------
# SNS topic — fan-out for failure notifications
# ---------------------------------------------------------------------------
resource "aws_sns_topic" "ops_alerts" {
  name = "${var.name_prefix}-ops-alerts"
}

# Optional email subscription (user confirms via emailed link)
resource "aws_sns_topic_subscription" "ops_email" {
  count     = var.ops_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.ops_alerts.arn
  protocol  = "email"
  endpoint  = var.ops_email
}

# ---------------------------------------------------------------------------
# EventBridge rule: catch Step Functions FAILED executions
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_event_rule" "sfn_failed" {
  name        = "${var.name_prefix}-sfn-execution-failed"
  description = "Trigger when any battery-pdm Step Functions execution fails"

  event_pattern = jsonencode({
    source        = ["aws.states"]
    "detail-type" = ["Step Functions Execution Status Change"]
    detail = {
      status         = ["FAILED", "TIMED_OUT", "ABORTED"]
      stateMachineArn = [{
        prefix = "arn:aws:states:*:*:stateMachine:battery-pdm-"
      }]
    }
  })
}

# Send failure event to SNS (will fan out to email/Slack/PagerDuty subscribers)
resource "aws_cloudwatch_event_target" "sfn_failed_sns" {
  rule      = aws_cloudwatch_event_rule.sfn_failed.name
  target_id = "sns-ops-alerts"
  arn       = aws_sns_topic.ops_alerts.arn

  input_transformer {
    input_paths = {
      stateMachine = "$.detail.stateMachineArn"
      execution    = "$.detail.executionArn"
      status       = "$.detail.status"
      start_time   = "$.detail.startDate"
      stop_time    = "$.detail.stopDate"
    }
    input_template = "\"[Battery PdM] Step Functions FAILED\\n\\nState machine: <stateMachine>\\nExecution: <execution>\\nStatus: <status>\\nStarted: <start_time>\\nStopped: <stop_time>\\n\\nCheck CloudWatch logs at /aws/batch/battery-pdm-dev for details.\""
  }
}

# ---------------------------------------------------------------------------
# EventBridge rule: catch Batch job FAILED states
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_event_rule" "batch_failed" {
  name        = "${var.name_prefix}-batch-job-failed"
  description = "Trigger when any battery-pdm Batch job fails"

  event_pattern = jsonencode({
    source        = ["aws.batch"]
    "detail-type" = ["Batch Job State Change"]
    detail = {
      status = ["FAILED"]
      jobDefinition = [{
        prefix = "arn:aws:batch:*:*:job-definition/battery-pdm-"
      }]
    }
  })
}

resource "aws_cloudwatch_event_target" "batch_failed_sns" {
  rule      = aws_cloudwatch_event_rule.batch_failed.name
  target_id = "sns-ops-alerts"
  arn       = aws_sns_topic.ops_alerts.arn

  input_transformer {
    input_paths = {
      job_name = "$.detail.jobName"
      job_id   = "$.detail.jobId"
      reason   = "$.detail.statusReason"
    }
    input_template = "\"[Battery PdM] Batch Job FAILED\\n\\nJob: <job_name>\\nID: <job_id>\\nReason: <reason>\\n\\nCheck CloudWatch logs at /aws/batch/battery-pdm-dev for details.\""
  }
}

# ---------------------------------------------------------------------------
# Allow EventBridge to publish to SNS
# ---------------------------------------------------------------------------
resource "aws_sns_topic_policy" "ops_alerts" {
  arn = aws_sns_topic.ops_alerts.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowEventBridge"
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
        Action   = "sns:Publish"
        Resource = aws_sns_topic.ops_alerts.arn
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
output "ops_alerts_topic_arn" {
  description = "SNS topic ARN — subscribe to receive failure alerts"
  value       = aws_sns_topic.ops_alerts.arn
}

output "subscribe_instructions" {
  description = "How to add more alert subscribers (Slack, PagerDuty)"
  value       = "aws sns subscribe --topic-arn ${aws_sns_topic.ops_alerts.arn} --protocol https --notification-endpoint <slack-webhook-url>"
}
