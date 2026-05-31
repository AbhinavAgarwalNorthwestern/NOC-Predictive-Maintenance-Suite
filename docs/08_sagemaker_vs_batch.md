# 08. SageMaker vs AWS Batch — why we picked Batch

A common interview / review question: *"You're on AWS — why didn't you use
SageMaker?"* This doc answers that explicitly, then walks through which parts
of SageMaker we'd actually adopt for v2.

## TL;DR

We picked **AWS Batch on Fargate Spot** for cron-style ML scoring because:
1. **Cost:** ~70% cheaper than SageMaker Processing for the same compute
2. **Simplicity:** generic containers, no SageMaker-specific contracts
3. **Use case fit:** we don't need real-time inference, distributed training,
   or SageMaker-specific managed services

SageMaker has real strengths we deliberately left unused. They'd be the right
call in different scenarios — laid out below.

## Service-by-service comparison

### Training / scoring compute

| | AWS Batch (chose this) | SageMaker Processing Jobs | SageMaker Training Jobs |
|--|------------------------|---------------------------|-------------------------|
| **Pricing** | Fargate Spot pay-per-second ($) | Per-instance, per-second ($$$) | Same as Processing |
| **Container contract** | Any container, any command | SageMaker requires specific image structure (`/opt/ml/processing/input`, etc.) | Specific structure (`/opt/ml/input/data/...`) |
| **Idle cost** | $0 | $0 | $0 |
| **Per-run cost (5 min, 2vCPU/4GB)** | ~$0.05 | ~$0.15 | ~$0.15 |
| **Logs** | CloudWatch automatically | CloudWatch automatically | CloudWatch automatically |
| **Restart on failure** | Built-in retry policy | Built-in | Built-in |
| **MLflow integration** | DIY (env vars) | Native (when configured) | Native + auto-checkpoint |
| **Distributed training** | Manual | Limited | First-class (multi-node, parameter server, etc.) |

**Why Batch wins for us:** we're running 5-minute single-node cron jobs.
None of SageMaker's premium features (distributed training, automatic
checkpointing, managed feature engineering) move the needle. We pay 3× for
features we don't use.

**Where SageMaker Processing would beat Batch:** if you wanted **SageMaker
Clarify** (managed bias/drift analysis) wired directly to your training
pipeline, or if you needed **SageMaker Pipelines** (which only run on
SageMaker compute).

### Inference

| | We chose | SageMaker Endpoints |
|--|----------|--------------------|
| **Latency** | Daily/weekly batch — minutes | Real-time — <100ms |
| **Cost** | $0 (no endpoint) | $50-200/mo per instance (always-on) |
| **Scaling** | N/A (batch) | Autoscaling, multiple instances |
| **Use case fit** | ✅ our use case is batch screening | If we needed real-time scoring |

**The honest answer for the autonomy model:** an `AC_MAINS_FAIL` event could
plausibly want sub-minute scoring. We currently handle this via daily batch
+ a Metaflow micro-batch flow (1-min cadence). If we wanted true real-time:

```python
# A future serve.py would look like:
from battery_pdm.aws.s3_io import read_parquet
from battery_pdm.monitoring.model_registry import load_calibrator, validate_feature_hash

def model_fn(model_dir):
    """SageMaker contract: load model artifacts."""
    booster = xgb.Booster()
    booster.load_model(f"{model_dir}/booster.json")
    meta = json.load(open(f"{model_dir}/meta.json"))
    calibrator = load_calibrator(model_dir)
    return {"booster": booster, "meta": meta, "calibrator": calibrator}

def predict_fn(input_data, model):
    """SageMaker contract: score a request."""
    validate_feature_hash(model["meta"], list(input_data.columns))
    X = input_data[model["meta"]["feature_cols"]].astype(float).fillna(0.0)
    raw = model["booster"].predict(xgb.DMatrix(X))
    return apply_calibrator(model["calibrator"], raw).tolist()
```

About 30 lines plus a Terraform module for the endpoint. ~$50/month for a
ml.t3.medium instance running 24/7.

**Why we left this for v2:** the daily/weekly cadence covers all 3 of our
documented use cases. Real-time is a "what-if" we documented in
[01_PROBLEM_AND_DOMAIN.md](01_PROBLEM_AND_DOMAIN.md), not a current requirement.

### Model registry

| | We chose | SageMaker Model Registry |
|--|----------|--------------------------|
| **Storage** | S3 (booster.json + meta.json + calibrator.pkl + reference_profile.json) | Managed registry with versioning + approval workflow |
| **Champion/challenger** | Built into RetrainingFlow | Built-in approval state machine |
| **MLflow native** | ✅ (we use MLflow tracking separately) | Has MLflow integration via SageMaker MLflow |
| **Cost** | $0 (S3 storage) | $0 (free service) |
| **Cross-team visibility** | ❌ requires custom dashboard | ✅ built-in UI |

**Why we chose S3 + custom registry:** simpler, free, and the model
metadata is more explicit (you read meta.json directly to understand what
the model is). The trade-off is no cross-team UI.

**What we'd swap in for v2 (if multi-team):** SageMaker Model Registry as
the metadata store, with our scoring flows downloading model artifacts
from there.

### HPO

| | We chose | SageMaker HPO Jobs |
|--|----------|--------------------|
| **Algorithm** | Optuna TPE in a notebook | Bayesian or random in managed jobs |
| **Parallelism** | Sequential (default), can parallelize manually | Native, runs N trials in parallel |
| **Cost** | $0 (uses Batch) | Per-trial compute ($$$) |
| **Trial dashboard** | DIY | Built-in |

**Honest finding:** we ran HPO with Optuna ([NB 03](../notebooks/03_hyperparameter_tuning.ipynb))
and got sub-1% lift. At this scale, native Optuna in a Metaflow `@step` is
plenty. SageMaker HPO Jobs make sense when:
- You need 100+ parallel trials (network architecture search, etc.)
- You want a managed UI to see trial progress
- Your training takes hours per trial (we're at seconds)

### Feature store

| | We chose | SageMaker Feature Store |
|--|----------|-------------------------|
| **Storage** | Parquet on S3, computed in-flow | Managed offline (parquet) + online (DynamoDB) |
| **Point-in-time joins** | DIY via searchsorted | Built-in `as_of_time` API |
| **Cost** | $0 | $0.0001 per write + monthly storage |

We don't have a feature store. Our features are computed on-the-fly because:
- Multiple models share the same feature pipeline (alarm_history, soc_proxy, etc.)
- Recomputing is cheap (~30s for 25k observations)
- Feature stores matter at the **multi-model, multi-team, high-frequency**
  use case

If we had 10 models all needing the same features at sub-second latency:
SageMaker Feature Store would be the move.

## When SageMaker is the right call

If any of these apply to your problem, lean SageMaker:

1. **Real-time inference required** (sub-second latency). Use SageMaker Endpoints.
2. **Distributed training** across multi-node or multi-GPU. Use SageMaker Training.
3. **Multiple ML teams** sharing models. Use SageMaker Model Registry + Studio.
4. **Managed drift detection** via Model Monitor. Less code to maintain.
5. **Compliance** that requires audit trail of every model decision. SageMaker
   logs all of this natively.
6. **Bias / fairness** monitoring via Clarify.

## What we'd actually adopt in a phased upgrade

If this system grew to handle a real customer fleet:

**Phase 2 (~1 week of work):**
1. Add SageMaker Endpoint for the **autonomy model** (real-time scoring on
   `AC_MAINS_FAIL` events)
2. Keep Batch for the daily/weekly screening flows (cost optimization)

**Phase 3 (~2-3 weeks):**
3. Migrate model registry from S3+meta.json to **SageMaker Model Registry**
4. Add SageMaker MLflow as the centralized tracking server
5. Add SageMaker Model Monitor for managed drift detection (deprecating our
   custom PSI implementation OR keeping both for defense in depth)

**Phase 4 (~1 month):**
6. SageMaker Feature Store if we end up with 5+ models sharing features

We deliberately did NOT do this for v1 because the marginal value didn't
justify the complexity. The architectural decision was "ship the
minimum-viable production pattern, document the upgrade path."

## What we built that SageMaker offers as managed

We built ourselves:
- PSI drift detection ↔ SageMaker Model Monitor
- Model registry pattern (S3 + meta.json) ↔ SageMaker Model Registry
- MLflow file backend ↔ SageMaker MLflow
- Isotonic calibration ↔ SageMaker doesn't have this; you'd build it
  yourself either way
- Champion/challenger with atomic triggers ↔ SageMaker has approval workflow
  but the matching logic is still custom

Pros of DIY:
- We learn how the production patterns actually work
- No SageMaker-specific lock-in
- Cost: ~$5/month vs ~$50+ with SageMaker premium services

Cons of DIY:
- More code to maintain (drift detection, registry, etc.)
- No managed UI for non-technical stakeholders to browse runs
- Custom code is the team's responsibility to keep secure / audited

**The portfolio argument:** building these patterns ourselves *and explaining
why we'd swap to SageMaker for a team-scale deployment* shows we understand
both. A candidate who only knows SageMaker has trouble debugging when the
managed service doesn't work.

## Reading next

- [03_MODELS_AND_CHOICES.md](03_MODELS_AND_CHOICES.md) — model choices including
  the measured HPO result
- [05_ARCHITECTURE.md](05_ARCHITECTURE.md) — full architecture with all
  service-level decisions
- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) — the AWS deployment
  + cost analysis
