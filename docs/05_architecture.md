# 05. System architecture

## TL;DR

```
                  ┌────────────────────────────────────────────┐
                  │           Synthetic data simulator         │
                  │  (or in production: NMS → Kafka/Kinesis)   │
                  └────────────────────┬───────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
            outputs/alarms.parquet      outputs/site_static.parquet
            outputs/load_shedding_*     outputs/labels.parquet
            (in AWS: s3://...-data-...)

                                       │
       ┌───────────────────────────────┼────────────────────────────────┐
       │                               │                                │
       ▼                               ▼                                ▼
  ┌────────────────┐         ┌──────────────────┐            ┌────────────────────┐
  │ TrainingFlow   │         │ DrainPredFlow    │            │ FailureScoringFlow │
  │ (manual)       │         │ (daily cron)     │            │ (weekly cron)      │
  │ 3 models       │         │                  │            │                    │
  │ + calibration  │         │ alerts/          │            │ alerts/            │
  │ + ref profile  │         └──────────────────┘            └────────────────────┘
  └───────┬────────┘
          │ writes
          ▼
  ┌──────────────────────────────────────────┐
  │ outputs/models/<model_name>/             │
  │   booster.json                           │
  │   calibrator.pkl                         │
  │   meta.json (incl. feature_hash)         │
  │   reference_profile.json                 │
  │ outputs/model_performance_log.parquet    │
  │ MLflow runs                              │
  └──────────────────────────────────────────┘
          │ read by
          ▼
  ┌────────────────────────────┐    ┌────────────────────────────┐
  │ DriftMonitorFlow           │ ─▶ │ RetrainingFlow             │
  │ (daily cron)               │    │ (weekly cron + on-drift)   │
  │ PSI / KS test              │    │ champion/challenger gate   │
  │ writes retrain_trigger.json│    │ shared held-out test set   │
  └────────────────────────────┘    │ immediate or shadow promo  │
                                    └──────────────┬─────────────┘
                                                   │
                                    ┌──────────────▼─────────────┐
                                    │ RollbackMonitorFlow         │
                                    │ (daily, safety net)         │
                                    │ reverts if Brier degrades   │
                                    └────────────────────────────┘
```

## Layers, top-down

### Data layer (S3 or local `outputs/`)

Single source of truth for everything. Designed to be storage-agnostic via
`src/battery_pdm/aws/s3_io.py` which accepts both `s3://...` URIs and local
paths transparently. This is how the flows run unchanged locally vs in AWS.

Bucket layout (when deployed to AWS):

```
s3://battery-pdm-dev-data-<account>/
  alarms.parquet
  site_static.parquet
  labels.parquet
  load_shedding_schedule.parquet
  stream/                      ← Lambda alarm simulator writes here
    date=YYYY-MM-DD/
      hour=HH/
        batch_<ts>.parquet

s3://battery-pdm-dev-models-<account>/
  drain_predictor_48h/
    booster.json
    calibrator.pkl
    meta.json
    reference_profile.json
  failure_alarms_only/
    booster.json
    meta.json
    reference_profile.json
  archive/
    drain_predictor_48h_20260520_214952/
      (snapshot when promoted)

s3://battery-pdm-dev-alerts-<account>/
  drain_alerts/<date>/<hour>/<flow_run_id>.parquet
  failure_alerts/<date>/<flow_run_id>.parquet
  drift_reports/drift_report_h<ref>.json
  retrain_trigger.json          ← created/consumed atomically
  consumed_triggers/            ← audit trail of past trigger consumption

s3://battery-pdm-dev-mlflow-<account>/
  mlflow/                       ← MLflow runs (S3 backend in prod)
  metaflow/                     ← reserved for future Metaflow datastore
```

### Compute layer (AWS Batch Fargate Spot)

8 job definitions, one per flow:
- `drain_predictor` — daily 00:30 UTC, ~5 min, 2 vCPU / 4GB
- `drift_monitor` — daily 01:00 UTC, ~3 min, 1 vCPU / 2GB
- `shadow_promotion` — daily 01:30 UTC, ~2 min, 1 vCPU / 2GB
- `rollback_monitor` — daily 02:00 UTC, ~2 min, 1 vCPU / 2GB
- `failure_scoring` — weekly Sunday 02:00 UTC, ~5 min, 2 vCPU / 4GB
- `retraining` — weekly Saturday 03:00 UTC, ~2 min, 4 vCPU / 8GB (drain model, CV-gated)
- `retraining_failure` — weekly Saturday 03:30 UTC, ~2 min, 4 vCPU / 8GB (failure model, shadow mode, 6-12mo label horizon)
- `training` — manual (initial bootstrap), ~10 min, 4 vCPU / 8GB

All run as the SAME Docker image with an entrypoint that syncs S3 data to local,
runs the Metaflow flow, then syncs results back to S3.

### Orchestration layer (Step Functions + Metaflow + EventBridge)

- **Step Functions** orchestrate each flow as a Batch job submission (with retry + error handling)
- **Metaflow** orchestrates STEPS within each flow (in-container DAG, state
  passed via Metaflow artifacts, cards persisted to S3)
- **EventBridge** triggers Step Functions on cron schedules
- **proactive parallel-path pattern**: training and inference run as independent
  cron schedules — no orchestrator dependency between them. The CV gate and
  rollback flow provide the coordination.

Metaflow config: local metadata + S3 datastore. Cards persist to
`s3://<data-bucket>/metaflow/cards/`. No Metaflow Metadata Service needed.

### Observability layer (CloudWatch)

- **Logs** — container stdout → `/aws/batch/battery-pdm-dev` log group
- **Custom metrics** — emitted from flows via `src/battery_pdm/aws/metrics.py`:
  - `BatteryPDM/ModelAUC` (per ModelName)
  - `BatteryPDM/DriftSignificantFeatures`
  - `BatteryPDM/DriftPredictionPSI`
  - `BatteryPDM/AlertsEmitted` (per Severity dimension)
  - `BatteryPDM/SitesScored`
  - `BatteryPDM/SitesInsufficientData`
- **Dashboard** — `infra/modules/dashboard/main.tf` defines a CloudWatch
  dashboard that visualizes:
  - Model AUC over time (depletion chart)
  - Drift PSI per feature over time
  - Alert counts (HIGH / MEDIUM)
  - Sites scored / insufficient data
  - Recent log lines (live tail)
  - Batch job success/failure counts

## Architectural decisions and rationale

### 1. Why Metaflow over Prefect / Dagster / Airflow

| | Airflow | Prefect 2 | Dagster | Metaflow |
|--|---------|-----------|---------|----------|
| Cron scheduling | ✅ native | ✅ native | ✅ native | ❌ (via EventBridge) |
| Python-first authoring | ⚠️ DAG-as-config | ✅ | ✅ | ✅ (FlowSpec class) |
| Step-level retry | ✅ | ✅ | ✅ | ✅ (@retry) |
| AWS Batch native integration | manual operator | manual | manual | ✅ (@batch decorator) |
| State persistence between runs | ⚠️ via XCom | ✅ via results | ✅ via assets | ✅ via S3 datastore |
| Local dev experience | Painful (Docker compose) | Fine | Fine | Excellent |
| Learning curve | Steep | Medium | Steep | Gentle |

**Why Metaflow won for us:**

1. **AWS Batch is first-class.** `@batch(cpu=2, memory=4096)` decorator
   submits the step to Batch automatically. No glue code.
2. **Local-to-cloud parity.** The exact same `python -m flow run` works
   locally and in AWS — only the env vars change. This matched our
   "minimum-viable" deployment philosophy.
3. **No DAG XML.** Flows are normal Python classes. Easy to read and review.
4. **Used by Netflix in production.** Battle-tested at scale.
5. **Ecosystem of @decorators** (@retry, @catch, @resources, @batch, @schedule)
   that compose cleanly without rewriting the flow.

We don't need Airflow's web UI for orchestration because we're not running
hundreds of flows. We don't need Prefect's hybrid execution because our flows
are independent.

### 2. Why AWS Batch over ECS Service / EKS / Lambda

See the table in [01_PROBLEM_AND_DOMAIN.md](01_PROBLEM_AND_DOMAIN.md) and the
extensive discussion in chat. Short version:
- **Lambda:** 15-min limit, 10GB memory cap. Our retraining flow can run ~10 min
  and uses 8GB. Cutting it too close.
- **ECS long-running service:** would pay 24/7 for compute that runs 5 min/day.
  Wasteful.
- **EKS:** Kubernetes adds operational complexity. We're a single-developer
  portfolio; EKS makes sense at 50+ flows or strict multi-tenancy.
- **Batch on Fargate Spot:** $0 idle, pay-per-second when jobs run, serverless,
  retries built-in. Correct fit.

### 3. Why Fargate Spot over Fargate on-demand

Fargate Spot:
- ~70% cheaper than on-demand
- Can be interrupted with 2-minute notice
- For BATCH jobs (not real-time services), interruption is fine — Batch
  retries automatically

For long-running real-time services (e.g. an inference endpoint) you'd want
on-demand or RIs. We have only batch jobs, so Spot is correct.

### 4. Why S3 over RDS / DynamoDB for data

We're storing immutable, time-partitioned artifacts (parquet files, model
weights, alert reports). S3 is:
- **Cheaper** than RDS for our access pattern (read once a day)
- **Versioned** — accidental deletes are recoverable
- **Encrypted** at rest by default
- **Schema-free** — we can change parquet schemas without migrations

We do NOT use a relational DB. The model performance log is parquet (read by
the dashboard). The reference profile is JSON. No SQL anywhere.

Trade-off: ad-hoc SQL queries over the alerts are slower (need Athena or
Pandas) than they would be in Postgres. Acceptable for the use case.

### 5. Why one Docker image for all flows, not one per flow

The image is ~500MB (numpy, pandas, xgboost, sklearn, mlflow, metaflow, boto3).
If we built one image per flow, we'd:
- Build 5 images per CI run (5× slower)
- Multiply ECR storage cost by 5
- Duplicate Python interpreter and most deps

Sharing one image with different `CMD` arguments is the standard pattern.
Each flow's resource allocation (CPU/memory) is in the Batch job definition,
not the image.

### 6. Why Terraform over CDK / CloudFormation / Pulumi

| | Terraform | CDK | CloudFormation | Pulumi |
|--|-----------|-----|----------------|--------|
| Multi-cloud | ✅ | ❌ AWS-only | ❌ | ✅ |
| HCL (decl) vs code | HCL | TypeScript/Python | YAML/JSON | TS/Python/Go |
| State management | S3 + DDB lock | CloudFormation | CloudFormation | Service backend |
| Learning curve | Medium | Easy if you know TS | Hard (YAML hell) | Medium |
| Modularity | excellent | excellent | mediocre | excellent |
| AWS Provider maturity | excellent | excellent | excellent (native) | good |
| Portability story | ✅ change one tfvars file | partial | ❌ | ✅ |

We picked Terraform because:
1. **Portability across accounts** is a stated requirement. One tfvars file
   change moves the whole stack.
2. **HCL is declarative** — easy to read what the infrastructure IS, not what
   code builds it.
3. **The community has terraform modules for everything.** We didn't have to
   write Kinesis or RDS from scratch.

If this was a real production environment in a multi-language team, CDK or
Pulumi would be appealing. For a portfolio with a Python codebase, Terraform's
language-agnostic declarative HCL wins.

### 7. Why parquet over CSV / JSON / Avro

| | CSV | JSON | Avro | Parquet |
|--|-----|------|------|---------|
| Schema enforcement | ❌ | ❌ | ✅ | ✅ |
| Columnar storage | ❌ | ❌ | ❌ | ✅ |
| Compression | gzip | gzip | snappy | snappy/zstd |
| Pandas/Spark/Athena native | ⚠️ | ⚠️ | ✅ | ✅ |
| Append in place | ✅ | ⚠️ | ❌ | ❌ (full rewrite) |

Parquet wins for analytical workloads. Athena queries S3 parquet natively.
Pandas reads it 5-10× faster than CSV.

The one downside (no append in place) doesn't bite us because we write daily
snapshots, not streaming inserts. For streaming inserts we'd use Kinesis
Firehose to batch into parquet.

## Repo layout

```
project_01/
├── src/battery_pdm/
│   ├── synth/                     ← synthetic data simulator
│   │   ├── simulator.py
│   │   ├── physics.py             ← Arrhenius, sulfation, Shepherd
│   │   ├── load_shedding.py       ← regional schedule generator
│   │   └── config.py
│   ├── (autonomy module removed)
│   │   ├── features.py            ← THE shared feature pipeline
│   │   ├── labels.py
│   │   ├── train.py
│   │   └── registry.py
│   ├── monitoring/
│   │   ├── drift.py               ← PSI + KS drift detection
│   │   ├── data_quality.py        ← scorable / cold-start / null rate
│   │   ├── model_registry.py      ← save/load + feature hash + calibrator
│   │   └── __init__.py
│   ├── flows/                     ← Metaflow FlowSpecs
│   │   ├── training_flow.py
│   │   ├── drain_predictor_flow.py
│   │   ├── failure_scoring_flow.py
│   │   ├── (autonomy flow decommissioned — see notebook 04)
│   │   ├── drift_monitor_flow.py
│   │   └── retraining_flow.py
│   ├── aws/                       ← AWS-specific helpers
│   │   ├── s3_io.py               ← local-or-s3 transparent I/O
│   │   └── metrics.py             ← CloudWatch custom metrics
│   ├── feature_pipeline/          ← legacy lifecycle prep
│   ├── training_pipeline/         ← legacy CV utils
│   └── inference_pipeline/        ← legacy (now superseded by flows)
├── scripts/
│   ├── run_end_to_end.py          ← full pipeline run + drift sim
│   ├── per_region_analysis.py     ← global vs per-region experiment
│   ├── simulate_drift.py          ← Peshawar grid upgrade scenario
│   ├── horizon_signal_strength.py
│   ├── horizon_overlap_analysis.py
│   └── train_*.py                 ← individual model trainers
├── tests/                         ← pytest, 50 tests
│   ├── test_features.py
│   ├── test_drift.py
│   ├── test_labels.py
│   ├── test_physics.py
│   ├── test_integration.py        ← end-to-end MLOps loop
│   └── conftest.py
├── notebooks/                     ← portfolio storytelling
│   ├── 00_system_overview.ipynb
│   ├── 01_drift_detection_demo.ipynb
│   ├── 02_model_evaluation_per_region.ipynb
│   └── _build_notebooks.py        ← source-controlled notebook generator
├── infra/                         ← Terraform IaC
│   ├── bootstrap/main.tf          ← tfstate bucket (run once)
│   ├── main.tf                    ← module wiring
│   ├── variables.tf               ← account-portable values
│   ├── versions.tf
│   ├── terraform.tfvars.example
│   └── modules/
│       ├── s3/
│       ├── ecr/
│       ├── iam/
│       ├── batch/
│       ├── eventbridge/
│       ├── dashboard/
│       └── lambda_simulator/      ← writes new alarm batches
├── lambda/alarm_simulator/        ← Lambda code (continuous data feed)
├── docs/                          ← (you are here)
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Reading next

- [06_PRODUCTION_PATTERNS.md](06_PRODUCTION_PATTERNS.md) — the patterns that
  make this production-grade (drift, calibration, atomic triggers, feature
  hashing)
- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) — the AWS deployment
  details and migration path to a full Metaflow Service
