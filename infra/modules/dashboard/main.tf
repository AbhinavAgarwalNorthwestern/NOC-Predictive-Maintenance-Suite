# CloudWatch dashboard for live drift + model performance monitoring.
# Uses CloudWatch metrics emitted by the flows (custom namespace: BatteryPDM).

variable "name_prefix" {
  type = string
}

variable "region" {
  type = string
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.name_prefix}-overview"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Model AUC over time (depletion chart)"
          region = var.region
          metrics = [
            ["BatteryPDM", "ModelAUC", "ModelName", "drain_predictor_48h"],
            [".", "ModelCIndex", ".", "failure_alarms_only"],
          ]
          period = 86400
          stat   = "Maximum"
          view   = "timeSeries"
          stacked = false
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Drift PSI (significant feature count)"
          region = var.region
          metrics = [
            ["BatteryPDM", "DriftSignificantFeatures", "ModelName", "drain_predictor_48h"],
            [".", "DriftPredictionPSI", ".", "."],
          ]
          period = 86400
          stat   = "Maximum"
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 8
        height = 5
        properties = {
          title  = "HIGH-risk alerts emitted per day"
          region = var.region
          metrics = [
            ["BatteryPDM", "AlertsEmitted", "Severity", "HIGH"],
            [".", ".", ".", "MEDIUM"],
          ]
          period = 86400
          stat   = "Sum"
          view   = "timeSeries"
          stacked = true
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 6
        width  = 8
        height = 5
        properties = {
          title  = "Sites scored / insufficient data"
          region = var.region
          metrics = [
            ["BatteryPDM", "SitesScored"],
            [".", "SitesInsufficientData"],
          ]
          period = 86400
          stat   = "Maximum"
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 16
        y      = 6
        width  = 8
        height = 5
        properties = {
          title  = "Batch jobs (succeeded / failed)"
          region = var.region
          metrics = [
            ["AWS/Batch", "SucceededJobs", "JobQueue", "${var.name_prefix}-queue"],
            [".", "FailedJobs", ".", "."],
          ]
          period = 86400
          stat   = "Sum"
          view   = "timeSeries"
          stacked = true
        }
      },
      {
        type   = "log"
        x      = 0
        y      = 11
        width  = 24
        height = 6
        properties = {
          title  = "Recent flow logs"
          region = var.region
          query  = "SOURCE '/aws/batch/${var.name_prefix}' | fields @timestamp, @message | sort @timestamp desc | limit 100"
          view   = "table"
        }
      },
    ]
  })
}

output "dashboard_url" {
  value = "https://${var.region}.console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards:name=${aws_cloudwatch_dashboard.main.dashboard_name}"
}
