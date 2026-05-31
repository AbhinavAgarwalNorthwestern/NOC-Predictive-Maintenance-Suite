# NOC Dashboard — Streamlit app on ECS Fargate behind ALB.
# Reads model artifacts + data from S3, serves operator UI.

variable "name_prefix" {
  type = string
}

variable "region" {
  type = string
}

variable "account_id" {
  type = string
}

# Use default VPC (same pattern as batch module — no NAT cost)
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

variable "ecr_repository_url" {
  type        = string
  description = "ECR repo URL for the dashboard image"
}

variable "image_tag" {
  type    = string
  default = "dashboard-latest"
}

variable "data_bucket_name" {
  type        = string
  description = "S3 bucket with alarms/models/drift reports"
}

variable "models_bucket_name" {
  type        = string
  description = "S3 bucket with model artifacts"
}

variable "alerts_bucket_name" {
  type        = string
  description = "S3 bucket with scoring alerts (drain, failure, drift)"
  default     = ""
}

# --- Security Groups ---

resource "aws_security_group" "alb" {
  name_prefix = "${var.name_prefix}-noc-alb-"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name_prefix}-noc-alb-sg" }
}

resource "aws_security_group" "ecs" {
  name_prefix = "${var.name_prefix}-noc-ecs-"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port       = 8501
    to_port         = 8501
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name_prefix}-noc-ecs-sg" }
}

# --- ALB ---

resource "aws_lb" "noc" {
  name               = "${var.name_prefix}-noc-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids

  tags = { Name = "${var.name_prefix}-noc-alb" }
}

resource "aws_lb_target_group" "noc" {
  name        = "${var.name_prefix}-noc-tg"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    path                = "/_stcore/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 10
    interval            = 30
    matcher             = "200"
  }

  tags = { Name = "${var.name_prefix}-noc-tg" }
}

resource "aws_lb_listener" "noc" {
  load_balancer_arn = aws_lb.noc.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.noc.arn
  }
}

# --- IAM ---

resource "aws_iam_role" "noc_task_execution" {
  name = "${var.name_prefix}-noc-task-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "noc_task_exec_policy" {
  role       = aws_iam_role.noc_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "noc_task" {
  name = "${var.name_prefix}-noc-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "noc_s3_read" {
  name = "${var.name_prefix}-noc-s3-read"
  role = aws_iam_role.noc_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        "arn:aws:s3:::${var.data_bucket_name}",
        "arn:aws:s3:::${var.data_bucket_name}/*",
        "arn:aws:s3:::${var.models_bucket_name}",
        "arn:aws:s3:::${var.models_bucket_name}/*",
        "arn:aws:s3:::${var.alerts_bucket_name}",
        "arn:aws:s3:::${var.alerts_bucket_name}/*",
      ]
    }]
  })
}

# --- ECS Cluster + Service ---

resource "aws_ecs_cluster" "noc" {
  name = "${var.name_prefix}-noc"
  tags = { Name = "${var.name_prefix}-noc-cluster" }
}

resource "aws_cloudwatch_log_group" "noc" {
  name              = "/ecs/${var.name_prefix}-noc"
  retention_in_days = 14
}

resource "aws_ecs_task_definition" "noc" {
  family                   = "${var.name_prefix}-noc"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.noc_task_execution.arn
  task_role_arn            = aws_iam_role.noc_task.arn

  container_definitions = jsonencode([
    {
      name      = "noc-dashboard"
      image     = "${var.ecr_repository_url}:${var.image_tag}"
      essential = true
      portMappings = [{
        containerPort = 8501
        hostPort      = 8501
        protocol      = "tcp"
      }]
      mountPoints = [{
        sourceVolume  = "data"
        containerPath = "/app/data"
      }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.noc.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "noc"
        }
      }
      environment = [
        { name = "DATA_DIR", value = "/app/data" },
      ]
    },
    {
      name       = "s3-sync"
      image      = "amazon/aws-cli:latest"
      essential  = false
      entryPoint = ["sh", "-c"]
      command    = [
        "while true; do aws s3 sync s3://${var.data_bucket_name}/ /app/data/ --quiet && aws s3 sync s3://${var.models_bucket_name}/ /app/data/models/ --quiet && aws s3 sync s3://${var.alerts_bucket_name}/ /app/data/ --quiet; sleep 300; done"
      ]
      mountPoints = [{
        sourceVolume  = "data"
        containerPath = "/app/data"
      }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.noc.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "s3sync"
        }
      }
    }
  ])

  volume {
    name = "data"
  }
}

resource "aws_ecs_service" "noc" {
  name            = "${var.name_prefix}-noc"
  cluster         = aws_ecs_cluster.noc.id
  task_definition = aws_ecs_task_definition.noc.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.noc.arn
    container_name   = "noc-dashboard"
    container_port   = 8501
  }

  depends_on = [aws_lb_listener.noc]
}

# --- Outputs ---

output "dashboard_url" {
  value = "http://${aws_lb.noc.dns_name}"
}

output "alb_dns" {
  value = aws_lb.noc.dns_name
}
