# Production deployment architecture

This document describes the **production-grade** deployment of the battery-pdm
ML system on AWS. The current `infra/` Terraform deploys a **minimum-viable
working subset** of this — sufficient for a single-developer system, missing
the centralized metadata + UI tier you'd want for a multi-developer team.

Read this if you want to understand:
- What was deployed and why
- What's missing for a true production setup
- The architectural decisions and their tradeoffs
- The exact migration path from "minimum viable" to "production"

---

## Current deployment (minimum viable)

```
                          ┌─────────────────────────────────────────┐
                          │           AWS account                   │
                          │                                         │
   GitHub Actions ─────▶  │  ECR ─── docker image                   │
   (CI/CD)                │   │                                     │
                          │   ▼                                     │
                          │  AWS Batch (Fargate Spot)               │
                          │   ├─ drain_predictor job   (daily)      │
                          │   ├─ drift_monitor   job   (daily)      │
                          │   ├─ retraining      job   (daily)      │
                          │   ├─ failure_scoring job   (weekly)     │
                          │   └─ training        job   (manual)     │
                          │     ▲                                   │
                          │     │ submitted by                      │
                          │     │                                   │
                          │   EventBridge cron schedules            │
                          │                                         │
   Lambda alarm           │  Lambda                                 │
   simulator   ─────────▶ │   (writes alarm batches to S3)          │
                          │                                         │
                          │  S3 buckets                             │
                          │   ├─ data    (alarms, schedule, static) │
                          │   ├─ models  (booster + meta + ref      │
                          │   │           profile + perf log)       │
                          │   ├─ alerts  (HIGH-risk site lists)     │
                          │   └─ mlflow  (experiment tracking)      │
                          │                                         │
                          │  CloudWatch                             │
                          │   ├─ Logs (container stdout)            │
                          │   ├─ Metrics (PSI, AUC, alerts/day)     │
                          │   └─ Dashboard (live drift + perf)      │
                          │                                         │
                          └─────────────────────────────────────────┘
```

### How a single run flows

1. **EventBridge fires** at scheduled time (e.g. `cron(0 0 * * ? *)` for daily midnight UTC)
2. **AWS Batch** receives the SubmitJob call, finds the matching job definition
3. **Fargate Spot** provisions a container task using the ECR image
4. **Container starts**, runs `python -m battery_pdm.flows.drain_predictor_flow run`
5. **Metaflow inside the container** orchestrates the flow steps:
   - All steps run sequentially in the same container
   - State passes between steps via Metaflow's in-memory artifacts
   - At the end, state is *not* persisted (no Metaflow Service)
6. **Container reads** alarms from `s3://...-data-.../`, model from `s3://...-models-.../`
7. **Container emits** alerts to `s3://...-alerts-.../`
8. **Container emits** custom CloudWatch metrics (PSI, AUC, alert counts)
9. **Container exits** — Fargate Spot stops billing

### What's good about this

- **Cost: ~$5/month idle**, ~$0.05/job execution. Cheap.
- **Fully reproducible** via Terraform across AWS accounts (change `terraform.tfvars`).
- **CI/CD is straightforward**: build image → push to ECR → done.
- **Observability** via CloudWatch dashboard (drift + AUC over time).
- **Models versioned** by date-stamped folders + MLflow tracking.
- **Drift detection** with proper PSI thresholds + automatic retraining trigger.

### What's missing for "true production"

- **No central Metaflow Metadata Service** — each container run is isolated
- **No Metaflow UI** — can't browse run history from a web page
- **No per-step resource granularity** — all steps share container resources
- **No per-step retry** — a failed step requires the whole flow to re-run
- **No cross-developer visibility** — your CI's runs and your laptop's runs don't share metadata
- **MLflow file backend** — works, but for a team you'd want a real MLflow server

For a single-developer portfolio system, none of these are blockers. For a
multi-team production system handling 100k+ sites, all of them matter.

---

## Full production architecture (the upgrade target)

```
                         ┌──────────────────────────────────────────┐
                         │              Same AWS account            │
                         │                                          │
   developers ─────────▶ │  Application Load Balancer ($22/mo)      │
   CI runners            │       │                                  │
                         │       ▼                                  │
                         │  Metaflow Service (ECS Fargate, $9/mo)   │
                         │       │                                  │
                         │       ▼                                  │
                         │  RDS Postgres (db.t4g.micro, $13/mo)     │
                         │       └─ run metadata, artifact index    │
                         │                                          │
                         │  Metaflow UI (ECS Fargate, $5/mo)        │
                         │       reads RDS, shows DAG visualizer    │
                         │                                          │
                         │  AWS Batch (per-step execution)          │
                         │       Metaflow's @batch decorator        │
                         │       submits each decorated step as     │
                         │       its own Batch job                  │
                         │                                          │
                         │  S3 datastore (same buckets)             │
                         │  ECR (same)                              │
                         │  CloudWatch (same)                       │
                         │                                          │
                         └──────────────────────────────────────────┘
```

### How the production flow differs

In code, each FlowSpec uses `@batch` to opt-in steps to AWS Batch:

```python
from metaflow import FlowSpec, step, batch, resources

class TrainingFlow(FlowSpec):
    @step
    def start(self):
        # runs wherever you launched `python -m flow run`
        ...

    @batch(cpu=2, memory=4096, queue="battery-pdm-prod-queue",
           image="998...amazonaws.com/battery-pdm:v1.4.2")
    @step
    def train_failure_model(self):
        # this single step runs as its own Fargate task on Batch
        ...

    @batch(cpu=4, memory=16384)  # bigger machine for the heavier step
    @step
    def train_drain_predictor(self):
        ...

    @step
    def end(self):
        # back on the orchestrator after both batch jobs finish
        ...
```

When you run `python -m flow run` from your laptop or CI:
1. The Metaflow client connects to the Metaflow Service via the ALB
2. A new run is registered in RDS
3. Step `start` executes locally
4. Step `train_failure_model` is submitted as a Batch job — Metaflow waits
5. While `train_failure_model` runs in its own Fargate container, its output is
   written to S3 (Metaflow's datastore)
6. When it completes, the next step runs (could be local or another batch job)
7. The Metaflow UI shows all of this as a DAG with per-step status

### Cost breakdown for the upgrade

| Component | Monthly cost | Why this size |
|-----------|-------------:|---------------|
| RDS PostgreSQL db.t4g.micro | $13 | Smallest viable instance; Metaflow needs Postgres specifically (not DynamoDB) |
| ECS Fargate task for Metaflow Service | $9 | 0.25 vCPU, 512 MB, runs 24/7 to handle metadata calls |
| ECS Fargate task for Metaflow UI | $5 | 0.25 vCPU, 512 MB, can scale to zero outside dev hours |
| Application Load Balancer | $22 | Stable HTTPS endpoint; ~$16 base + LCUs |
| CloudWatch logs | $1 | Negligible |
| **Total** | **$45-50/mo** | Plus the existing minimum-viable cost (~$5/mo for S3 etc.) |

### Migration path: minimum-viable → full production

Step-by-step Terraform additions:

#### 1. Add `modules/metaflow_service/` (new Terraform module)

Resources:
- `aws_ecs_cluster` (shared for both metaflow-service and metaflow-ui)
- `aws_rds_subnet_group` + `aws_db_instance` (Postgres)
- `random_password` + `aws_secretsmanager_secret` (DB password)
- `aws_security_group` ×3 (RDS, ECS, ALB)
- `aws_lb` + `aws_lb_target_group` + `aws_lb_listener` (ALB)
- `aws_ecs_task_definition` (image: `outerbounds/metaflow-service:2.4.x`)
- `aws_ecs_service` (1 desired count, autoassign public IP)
- `aws_cloudwatch_log_group` (for the service)
- IAM roles for ECS task execution + secret access

#### 2. Initialize the Metaflow DB

The `metaflow-service` container runs migrations on first startup when the env
var `MF_RUN_MIGRATIONS=true` is set. Just set it in the task definition.

#### 3. Update flow code

Add `@batch` decorators to heavy steps. Light orchestration steps stay un-decorated.

```python
@batch(cpu=2, memory=4096, queue="battery-pdm-dev-queue",
       image="${local.container_image_uri}")
@step
def heavy_step(self):
    ...
```

#### 4. Update Dockerfile

The image needs to have the right Metaflow env vars baked in (or supplied at
runtime). These tell Metaflow client where to find the service + datastore:

```dockerfile
ENV METAFLOW_DEFAULT_DATASTORE=s3
ENV METAFLOW_DATASTORE_SYSROOT_S3=s3://battery-pdm-dev-mlflow-998716768706/metaflow
ENV METAFLOW_DEFAULT_METADATA=service
ENV METAFLOW_SERVICE_URL=http://battery-pdm-metaflow-alb-xxxxx.elb.amazonaws.com
ENV METAFLOW_DEFAULT_AWS_BATCH_JOB_QUEUE=battery-pdm-dev-queue
ENV METAFLOW_ECS_FARGATE_EXECUTION_ROLE=arn:aws:iam::998716768706:role/battery-pdm-dev-task-role
ENV METAFLOW_BATCH_JOB_ROLE=arn:aws:iam::998716768706:role/battery-pdm-dev-task-role
ENV METAFLOW_AWS_REGION=ap-south-1
```

#### 5. (Optional) Deploy Metaflow UI

Same pattern as the service — different container image (`netflix/metaflow-ui`),
points at the same ALB on a different path.

---

## Why we didn't deploy the full stack now

Honest assessment for this portfolio:

1. **$45/mo idle cost** for a system that runs 5 minutes per day is hard to justify
2. **Single-developer use case** — don't need cross-developer run visibility
3. **The interesting ML/engineering parts** (point-in-time correctness, drift detection,
   champion/challenger retraining, calibration analysis, per-region segmentation) are
   already demonstrated and don't require the Service to work
4. **The upgrade path is documented** — adding the Service later is a Terraform-only
   change, not a code rewrite

The **right call for a portfolio** is usually "ship the minimum, document the upgrade."
That's what we did.

---

## ML-specific production patterns implemented

| Pattern | Implementation | Why it matters |
|---------|---------------|----------------|
| **Versioned model artifacts** | Each `save_model_artifacts()` call persists to `v_<timestamp>/` alongside the live model | Prior models never overwritten; audit trail without MLflow dependency |
| **Feature importance tracking** | XGBoost gain-based importance saved in `meta.json` at every training | Detect when important features drift or become irrelevant over retrains |
| **Feature hash validation** | SHA-256 of feature column list + order, validated at inference | Hard fail if train/serve feature contract drifts |
| **Deterministic categorical encoding** | Sorted categories in `pd.Categorical(col, categories=sorted(...))` | Same category codes at training and inference regardless of data ordering |
| **Isotonic calibration on held-out set** | 60/20/20 split: train / calibration / test. Calibrator never sees cal set via the trained model | Calibrated probabilities operators can act on (Brier improvement ~30-40%) |
| **Atomic trigger claim** | `os.replace()` for cross-platform atomicity + processing/consumed lifecycle | Two concurrent retraining workers can't both claim the same trigger |
| **Label maturity gate** | `min_label_maturity_days` parameter; checks labeled feedback exists before promotion | Prevents promoting on stale test metrics when labels are slow (failure model: 6-12mo) |
| **SageMaker Data Capture** | Terraform `data_capture_config` block at 100% sampling to S3 | Full inference audit trail; feeds drift detection in production |
| **DeploymentFlow** | Metaflow pipeline: pull from MLflow Registry → package → deploy → smoke test | Code-reviewable deployments, not manual CLI steps |
| **Evidently HTML cards** | `@card(type="html")` renders interactive drift reports in Metaflow UI | Visual drift monitoring without external tooling |

---

## Other production gaps worth flagging

These aren't ML-system-specific but matter at scale:

| Gap | Severity | Fix when... |
|-----|---------:|-------------|
| No VPC isolation (using default VPC) | low | Multi-team or compliance requirements |
| No KMS encryption (using AES256 default) | low | Data classification ≥ confidential |
| No backup automation (S3 versioning helps) | low | Models valuable enough that loss is unacceptable |
| Single region (ap-south-1) | medium | Need disaster recovery |
| No alerting on flow failures (logs only) | medium | Once flows are critical to operations |
| No PagerDuty/Slack integration | medium | Same |
| Container `IMMUTABLE` not enforced in ECR | low | Want to guarantee a tag never changes (compliance) |
| No secrets rotation | low | Compliance requirement |
| No WAF on ALB | low | Service is public-facing (not the case here) |
| No fine-grained IAM per flow | medium | Multiple teams sharing the account |
| Default network ACLs | low | Strict zero-trust networking required |

---

## Summary

What you can say in an interview:

> "I deployed the minimum-viable production pattern: containerized Metaflow flows
> running on AWS Batch Fargate Spot, S3 for storage, EventBridge for scheduling,
> CloudWatch for observability. Total cost ~$5/mo. For a multi-developer team
> setup I documented the migration to a self-hosted Metaflow Service with RDS
> Postgres and an ALB — that's another ~$45/mo. I chose to ship the minimum-viable
> version because none of the production-only features (centralized run UI,
> per-step retry) were blockers for the actual ML work.
>
> On the ML side: the system has versioned model artifacts with feature importance
> tracking, feature hash validation between train and serve, isotonic calibration
> on a strictly held-out set, atomic champion/challenger retraining with a label
> maturity gate, shadow deployment with blue/green promotion, SageMaker Data
> Capture for inference audit trails, cost-aware threshold optimization with
> sensitivity analysis, and 5-layer drift monitoring (schema, feature, prediction,
> concept, label). 88 tests pass including a full integration test covering the
> train → score → drift → retrain loop."

That's an honest, defensible architectural choice.
