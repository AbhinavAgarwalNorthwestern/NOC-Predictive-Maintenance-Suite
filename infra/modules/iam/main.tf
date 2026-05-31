variable "name_prefix" {
  type = string
}

variable "data_bucket_arn" {
  type = string
}

variable "models_bucket_arn" {
  type = string
}

variable "alerts_bucket_arn" {
  type = string
}

variable "mlflow_bucket_arn" {
  type = string
}

variable "github_repo" {
  description = "GitHub repo in owner/name format. Leave empty to skip GitHub OIDC role."
  type        = string
  default     = ""
}

# ─────────────────────────────────────────────────────────────────────────
# Role 1: Execution role for AWS Batch / Lambda — pulls from ECR, writes to S3
# ─────────────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com", "lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task" {
  name               = "${var.name_prefix}-task-role"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
}

data "aws_iam_policy_document" "task_inline" {
  statement {
    actions = [
      "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket",
    ]
    resources = [
      var.data_bucket_arn, "${var.data_bucket_arn}/*",
      var.models_bucket_arn, "${var.models_bucket_arn}/*",
      var.alerts_bucket_arn, "${var.alerts_bucket_arn}/*",
      var.mlflow_bucket_arn, "${var.mlflow_bucket_arn}/*",
    ]
  }
  statement {
    actions = [
      "cloudwatch:PutMetricData",
      "logs:CreateLogStream", "logs:PutLogEvents", "logs:CreateLogGroup",
    ]
    resources = ["*"]
  }
  statement {
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "${var.name_prefix}-task-policy"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_inline.json
}

# ─────────────────────────────────────────────────────────────────────────
# Role 2: GitHub Actions OIDC role — lets CI deploy without long-lived secrets
# ─────────────────────────────────────────────────────────────────────────

resource "aws_iam_openid_connect_provider" "github" {
  count           = var.github_repo == "" ? 0 : 1
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

data "aws_iam_policy_document" "github_assume" {
  count = var.github_repo == "" ? 0 : 1
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github[0].arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  count              = var.github_repo == "" ? 0 : 1
  name               = "${var.name_prefix}-github-actions"
  assume_role_policy = data.aws_iam_policy_document.github_assume[0].json
}

# Wide permissions for CI to deploy. In real prod you'd narrow this.
resource "aws_iam_role_policy_attachment" "github_admin" {
  count      = var.github_repo == "" ? 0 : 1
  role       = aws_iam_role.github_actions[0].name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

# ─────────────────────────────────────────────────────────────────────────
# Role 3: Step Functions execution role — runs Metaflow DAGs as state machines
# ─────────────────────────────────────────────────────────────────────────

resource "aws_iam_role" "sfn" {
  name = "${var.name_prefix}-sfn-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "states.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "sfn" {
  name = "${var.name_prefix}-sfn-policy"
  role = aws_iam_role.sfn.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["batch:SubmitJob", "batch:DescribeJobs", "batch:TerminateJob"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = aws_iam_role.task.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogDelivery", "logs:GetLogDelivery", "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery", "logs:ListLogDeliveries", "logs:PutResourcePolicy",
          "logs:DescribeResourcePolicies", "logs:DescribeLogGroups"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = ["${var.data_bucket_arn}", "${var.data_bucket_arn}/*"]
      }
    ]
  })
}

# ─────────────────────────────────────────────────────────────────────────
# Role 4: EventBridge role for triggering Step Functions executions
# ─────────────────────────────────────────────────────────────────────────

resource "aws_iam_role" "sfn_events" {
  name = "${var.name_prefix}-sfn-events-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "events.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "sfn_events" {
  name = "${var.name_prefix}-sfn-events-policy"
  role = aws_iam_role.sfn_events.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "states:StartExecution"
      Resource = "*"
    }]
  })
}

# ─────────────────────────────────────────────────────────────────────────
# Outputs
# ─────────────────────────────────────────────────────────────────────────

output "task_role_arn" {
  value = aws_iam_role.task.arn
}

output "github_actions_role_arn" {
  value = var.github_repo == "" ? "" : aws_iam_role.github_actions[0].arn
}

output "sfn_role_arn" {
  value = aws_iam_role.sfn.arn
}

output "sfn_events_role_arn" {
  value = aws_iam_role.sfn_events.arn
}
