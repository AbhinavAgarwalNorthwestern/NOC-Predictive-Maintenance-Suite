# NOC Runbook — Battery PdM Operations

Complete deployment and operations guide. Covers:
1. Prerequisites
2. First-time deployment
3. Daily operations
4. Triggering drift simulation (demo)
5. Responding to alerts
6. Troubleshooting

---

## 1. Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| AWS CLI | v2+ | `aws --version` |
| Docker | 24+ | `docker --version` |
| Terraform | 1.5+ | `terraform --version` |
| Python | 3.12+ | `python --version` |
| WSL2 (Windows) | Ubuntu | `wsl -l -v` |
| AWS account | 998716768706 | `aws sts get-caller-identity` |

**AWS credentials must have permissions for:** ECR, ECS, ALB, S3, CloudWatch, IAM.

---

## 2. First-Time Deployment

### 2.1 Infrastructure (one-time)

```bash
cd infra/

# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Deploy everything (S3, ECR, Batch, SageMaker, NOC Dashboard)
terraform apply
```

This creates:
- 4 S3 buckets (data, models, alerts, mlflow)
- 1 ECR repository
- AWS Batch compute environment + job queue + 5 job definitions
- NOC Dashboard (ECS Fargate + ALB)
- CloudWatch dashboard
- EventBridge schedules

### 2.2 Generate & Upload Data

```bash
# Generate synthetic fleet data (500 sites, 36 months)
cd ..
python -m battery_pdm.synth.run --sites 500 --months 36 --seed 42

# Upload to S3
aws s3 sync outputs/ s3://battery-pdm-dev-data-998716768706/ \
    --exclude "*.pyc" --exclude "__pycache__/*"
```

### 2.3 Run Training Flow (via WSL2)

```bash
# From WSL2 Ubuntu:
cd /mnt/c/Users/abhin/projects/project_01
source .venv/bin/activate  # or create venv: python -m venv .venv && pip install -e .
python -m battery_pdm.flows.training_flow run --seed 42
```

Expected output:
```
Failure model:    C-index=0.8985  PROMOTED
Autonomy model:   C-index=0.7332  PROMOTED
Drain predictor:  AUC=0.8280      PROMOTED
```

### 2.4 Deploy NOC Dashboard

```bash
# Build + push + deploy (from project root)
chmod +x dashboard/deploy.sh
./dashboard/deploy.sh
```

Or step-by-step:

```bash
# 1. Build Docker image
docker build -t battery-pdm-noc:latest -f dashboard/Dockerfile .

# 2. Push to ECR
aws ecr get-login-password --region ap-south-1 | \
    docker login --username AWS --password-stdin 998716768706.dkr.ecr.ap-south-1.amazonaws.com
docker tag battery-pdm-noc:latest 998716768706.dkr.ecr.ap-south-1.amazonaws.com/battery-pdm-dev:dashboard-latest
docker push 998716768706.dkr.ecr.ap-south-1.amazonaws.com/battery-pdm-dev:dashboard-latest

# 3. Sync data to S3
aws s3 sync outputs/ s3://battery-pdm-dev-data-998716768706/
aws s3 sync outputs/models/ s3://battery-pdm-dev-models-998716768706/

# 4. Deploy ECS task
cd infra/
terraform apply -target=module.noc_app

# 5. Get URL
terraform output noc_app_dashboard_url
```

Dashboard will be available at the ALB DNS name (port 80). Allow 2-3 min for startup.

### 2.5 Verify Deployment

```bash
# Health check
curl http://<ALB_DNS>/_stcore/health

# Check ECS task status
aws ecs describe-services --cluster battery-pdm-dev-noc --services battery-pdm-dev-noc --region ap-south-1
```

---

## 3. Daily Operations

### What runs automatically (EventBridge → Batch):

| Time (UTC) | Flow | What it does |
|------------|------|--------------|
| 00:00 | DrainPredictorFlow | Scores all sites, emits HIGH-risk alerts to S3 |
| 00:00 | DriftMonitorFlow | Checks feature + prediction drift, triggers retrain if needed |
| 01:00 | RetrainingFlow | Claims trigger, trains challenger, runs shadow |
| Weekly | FailureScoringFlow | Updates replacement priority list |

### NOC Dashboard Pages:

| Page | What operators use it for |
|------|--------------------------|
| **Fleet Overview** | Morning check — are all systems green? |
| **Drain Risk (48h)** | Which sites need generators today? |
| **Replacement Priority** | Monthly procurement planning |
| **Anomaly Detection** | Sites acting weird that don't yet show high risk |
| **Drift Monitor** | Is the model still trustworthy? (ML engineer view) |
| **Drift Simulation** | Demo/training: what happens when the world changes |
| **Model Health** | Version history, calibration status, performance trends |

### Daily operator workflow:

1. Open **Fleet Overview** — check system status (drift, model freshness, alerts)
2. Open **Drain Risk (48h)** — review HIGH-risk sites above threshold
3. Dispatch generators to top-N sites (threshold or capacity-bound)
4. If anomalies flagged → open **Anomaly Detection** → investigate flagged sites
5. If drift detected → escalate to ML engineer (don't blindly retrain)

---

## 4. Drift Simulation (Demo)

### Via Dashboard:

1. Navigate to **🧪 Drift Simulation** page
2. Set parameters:
   - Grid upgrade month: 24
   - Evaluation month: 30
   - AC_MAINS_FAIL reduction: 60%
   - RECTIFIER_FAULT reduction: 40%
3. Click **Run Drift Simulation**
4. Observe:
   - 🚨 DRIFT DETECTED banner
   - 10+ features with PSI > 0.25
   - Peshawar risk scores go UP (counterintuitively)
   - System recommends retraining

### Via CLI (programmatic):

```bash
# From WSL2:
python -m battery_pdm.flows.drift_monitor_flow run \
    --current-h 21600 \
    --model-name drain_predictor_48h

# Check if trigger was written:
cat outputs/drift_reports/retrain_trigger.json
```

### What happens after drift is detected:

1. `DriftMonitorFlow` writes `retrain_trigger.json`
2. Next `RetrainingFlow` run claims it atomically
3. Trains challenger on recent data
4. Challenger scores in shadow mode
5. After 7+ days of labels, `ShadowPromotionFlow` evaluates
6. If challenger beats champion → atomic promotion
7. All logged to MLflow

---

## 5. Responding to Alerts

### HIGH-risk drain alert:

```
Action: Dispatch generator within 4 hours
Verify: Check site alarm history in Anomaly Detection page
Escalate if: >20 sites in one region (possible fleet-wide issue)
```

### Drift detected:

```
DO NOT: Blindly retrain
DO: Investigate WHY (check Drift Monitor page for which features shifted)
Ask: Was there an external intervention? (grid upgrade, seasonal change, new sites)
Escalate to: ML engineer for causal analysis
Timeline: 1-2 weeks for investigation, not same-day retrain
```

### Anomalous site (not yet high-risk):

```
Action: Schedule proactive inspection within 1 week
Check: Top feature deviations (e.g., rising rectifier faults = charger dying)
Priority: Higher if deviation is in leading indicators (rectifier_fault, cell_imbalance)
```

---

## 6. Troubleshooting

### Dashboard not loading:

```bash
# Check ECS task health
aws ecs describe-tasks --cluster battery-pdm-dev-noc \
    --tasks $(aws ecs list-tasks --cluster battery-pdm-dev-noc --query 'taskArns[0]' --output text) \
    --region ap-south-1

# Check logs
aws logs tail /ecs/battery-pdm-dev-noc --since 30m --region ap-south-1
```

### S3 data not updating:

```bash
# Check sidecar sync
aws logs tail /ecs/battery-pdm-dev-noc --log-stream-name-prefix s3sync --since 30m

# Manual sync
aws s3 sync outputs/ s3://battery-pdm-dev-data-998716768706/
```

### Model not scoring:

```bash
# Verify model artifacts exist
aws s3 ls s3://battery-pdm-dev-models-998716768706/drain_predictor_48h/

# Check feature hash matches
python -c "
import json
meta = json.loads(open('outputs/models/drain_predictor_48h/meta.json').read())
print(f'Feature hash: {meta[\"feature_hash\"]}')
print(f'Features: {len(meta[\"feature_cols\"])}')
"
```

### Retraining not triggering:

```bash
# Check trigger file
ls outputs/drift_reports/retrain_trigger*

# Check consumed triggers
ls outputs/drift_reports/consumed_triggers/

# Force a retrain (skips drift check)
python -m battery_pdm.flows.retraining_flow run --model-name drain_predictor_48h
```

---

## 7. Architecture Quick Reference

```
┌─────────────────────────────────────────────────────┐
│                    AWS (ap-south-1)                  │
│                                                     │
│  ┌──────────┐    ┌──────────────────────────────┐  │
│  │   ALB    │───▶│  ECS Fargate (NOC Dashboard) │  │
│  │  :80     │    │  - Streamlit app              │  │
│  └──────────┘    │  - S3 sidecar sync            │  │
│                  └──────────────────────────────┘  │
│                           │ reads                   │
│                           ▼                         │
│  ┌────────────────────────────────────────────┐    │
│  │  S3 Buckets                                │    │
│  │  ├── data (alarms, schedule, drift reports)│    │
│  │  ├── models (boosters, meta, calibrators)  │    │
│  │  └── alerts (HIGH-risk site lists)         │    │
│  └────────────────────────────────────────────┘    │
│                           ▲ writes                  │
│                           │                         │
│  ┌────────────────────────────────────────────┐    │
│  │  AWS Batch (Fargate Spot)                  │    │
│  │  ├── drain_predictor  (daily)              │    │
│  │  ├── drift_monitor    (daily)              │    │
│  │  ├── retraining       (on trigger)         │    │
│  │  ├── failure_scoring  (weekly)             │    │
│  │  └── training         (manual)             │    │
│  └────────────────────────────────────────────┘    │
│           ▲ triggered by                            │
│           │                                         │
│  ┌────────────────────┐  ┌─────────────────────┐  │
│  │  EventBridge cron  │  │  CloudWatch         │  │
│  │  (schedules)       │  │  (metrics + logs)   │  │
│  └────────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 8. Cost Estimate

| Component | Monthly | Notes |
|-----------|--------:|-------|
| S3 (data + models + alerts) | $2 | ~1GB total |
| ECR | $1 | Image storage |
| Batch (Fargate Spot) | $3 | ~5 min/day of compute |
| NOC Dashboard (ECS Fargate) | $10 | 0.5 vCPU, 1GB, 24/7 |
| ALB | $16 | Base cost + LCUs |
| CloudWatch | $2 | Logs + custom metrics |
| **Total** | **~$34/mo** | |

Scale-to-zero: stop the ECS service when not demoing to save ~$26/mo.

```bash
# Stop dashboard (save cost)
aws ecs update-service --cluster battery-pdm-dev-noc --service battery-pdm-dev-noc --desired-count 0

# Start dashboard
aws ecs update-service --cluster battery-pdm-dev-noc --service battery-pdm-dev-noc --desired-count 1
```
