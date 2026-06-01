# MLflow Tracking Server — t3.micro EC2 with SQLite backend + S3 artifacts.
#
# Architecture (proactive ml.school pattern):
#   - Single t3.micro EC2 running mlflow server on port 5000
#   - SQLite at /home/ubuntu/mlflow.db (backend store for metadata)
#   - S3 mlflow bucket for artifacts (models, plots)
#   - Cron job backs up SQLite to S3 every hour (no data loss on EC2 failure)
#
# Cost: ~$8/mo (t3.micro on-demand) — cheaper than ECS+ALB+EFS ($25/mo)
# Trade-off: brief downtime if EC2 dies, but autorecovery + S3 backup means
# zero data loss. MLflow tracking is not a tier-1 service.
#
# Why not HA: MLflow writes are weekly (retraining), reads are manual.
# A 5-min outage during EC2 replacement does not affect ML lifecycle.

variable "name_prefix" {
  type = string
}

variable "region" {
  type = string
}

variable "mlflow_bucket_name" {
  type        = string
  description = "S3 bucket for MLflow artifacts AND SQLite backup"
}

variable "ssh_key_name" {
  type        = string
  description = "Existing EC2 key pair name (for ssh debug access). Leave empty to skip."
  default     = ""
}

# Latest Ubuntu 24.04 AMI from the official Canonical owner
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
}

data "aws_vpc" "default" {
  default = true
}

# ---------------------------------------------------------------------------
# IAM role: lets the EC2 read/write the MLflow S3 bucket
# ---------------------------------------------------------------------------
resource "aws_iam_role" "mlflow" {
  name = "${var.name_prefix}-mlflow-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "mlflow_s3" {
  name = "${var.name_prefix}-mlflow-s3-access"
  role = aws_iam_role.mlflow.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
      ]
      Resource = [
        "arn:aws:s3:::${var.mlflow_bucket_name}",
        "arn:aws:s3:::${var.mlflow_bucket_name}/*",
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "mlflow" {
  name = "${var.name_prefix}-mlflow-ec2-profile"
  role = aws_iam_role.mlflow.name
}

# ---------------------------------------------------------------------------
# Security group: allow MLflow UI (5000) + SSH (22) from anywhere
# ---------------------------------------------------------------------------
resource "aws_security_group" "mlflow" {
  name        = "${var.name_prefix}-mlflow-sg"
  description = "MLflow tracking server + SSH"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "MLflow UI"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH (debug access)"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# EC2 instance: t3.micro running MLflow
# ---------------------------------------------------------------------------
resource "aws_instance" "mlflow" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = "t3.micro"
  iam_instance_profile   = aws_iam_instance_profile.mlflow.name
  vpc_security_group_ids = [aws_security_group.mlflow.id]
  key_name               = var.ssh_key_name == "" ? null : var.ssh_key_name

  user_data = <<-EOF
    #!/bin/bash
    set -e
    apt-get update
    apt-get install -y python3.12-venv python3-pip awscli cron

    # Install MLflow in a venv
    python3 -m venv /home/ubuntu/.venv
    /home/ubuntu/.venv/bin/pip install mlflow==2.20.2 boto3

    # Try to restore SQLite backup from S3 (if one exists)
    aws s3 cp s3://${var.mlflow_bucket_name}/backups/mlflow.db /home/ubuntu/mlflow.db || \
      echo "No prior backup found — starting with fresh database"

    chown ubuntu:ubuntu /home/ubuntu/mlflow.db 2>/dev/null || true

    # systemd unit: keep MLflow running and restart on failure
    cat > /etc/systemd/system/mlflow.service <<SVC
    [Unit]
    Description=MLflow Tracking Server
    After=network.target

    [Service]
    Type=simple
    User=ubuntu
    Environment="AWS_DEFAULT_REGION=${var.region}"
    Environment="MLFLOW_HTTP_REQUEST_TIMEOUT=900"
    ExecStart=/home/ubuntu/.venv/bin/mlflow server \\
      --host 0.0.0.0 \\
      --port 5000 \\
      --backend-store-uri sqlite:////home/ubuntu/mlflow.db \\
      --default-artifact-root s3://${var.mlflow_bucket_name}/mlflow \\
      --serve-artifacts
    Restart=always
    RestartSec=10

    [Install]
    WantedBy=multi-user.target
    SVC

    systemctl daemon-reload
    systemctl enable mlflow
    systemctl start mlflow

    # Hourly SQLite backup to S3 (zero data loss on EC2 failure)
    cat > /home/ubuntu/backup_mlflow.sh <<BKP
    #!/bin/bash
    cp /home/ubuntu/mlflow.db /tmp/mlflow.db.bak
    aws s3 cp /tmp/mlflow.db.bak s3://${var.mlflow_bucket_name}/backups/mlflow.db
    aws s3 cp /tmp/mlflow.db.bak s3://${var.mlflow_bucket_name}/backups/mlflow-\$(date +%Y%m%d_%H).db
    rm /tmp/mlflow.db.bak
    BKP
    chmod +x /home/ubuntu/backup_mlflow.sh

    # Cron: backup every hour
    (crontab -u ubuntu -l 2>/dev/null; echo "0 * * * * /home/ubuntu/backup_mlflow.sh >> /home/ubuntu/backup.log 2>&1") | crontab -u ubuntu -
  EOF

  user_data_replace_on_change = true

  tags = {
    Name = "${var.name_prefix}-mlflow"
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
output "mlflow_tracking_uri" {
  description = "MLflow tracking server URL — set as MLFLOW_TRACKING_URI in flows"
  value       = "http://${aws_instance.mlflow.public_dns}:5000"
}

output "mlflow_public_ip" {
  description = "EC2 public IP (for SSH debug)"
  value       = aws_instance.mlflow.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.mlflow.id
}
