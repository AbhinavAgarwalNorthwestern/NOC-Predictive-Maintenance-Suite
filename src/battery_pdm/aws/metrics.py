"""Emit custom CloudWatch metrics from anywhere in the codebase.

Gracefully no-ops if not running in AWS (no boto3 / no AWS credentials).
Used by training/scoring/drift flows to feed the Grafana / CloudWatch
dashboard's drift + performance-depletion charts.

Namespace convention: 'BatteryPDM' with dimensions like ModelName, Severity, etc.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

NAMESPACE = "BatteryPDM"


def _get_client():
    """Lazy boto3 client. Returns None if boto3 missing or no credentials."""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError
    except ImportError:
        return None
    try:
        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "ap-south-1"
        )
        return boto3.client("cloudwatch", region_name=region)
    except (NoCredentialsError, PartialCredentialsError):
        return None
    except Exception as exc:
        logger.warning("CloudWatch client unavailable: %s", exc)
        return None


def emit_metric(
    metric_name: str,
    value: float,
    unit: str = "None",
    dimensions: dict[str, Any] | None = None,
) -> bool:
    """Emit a single CloudWatch metric. Safe to call locally — no-ops if no AWS.

    Returns True if successfully emitted.
    """
    client = _get_client()
    if client is None:
        return False

    dims = [{"Name": k, "Value": str(v)} for k, v in (dimensions or {}).items()]
    try:
        client.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[
                {
                    "MetricName": metric_name,
                    "Value": float(value),
                    "Unit": unit,
                    "Dimensions": dims,
                }
            ],
        )
        return True
    except Exception as exc:
        logger.warning("Failed to emit CloudWatch metric %s: %s", metric_name, exc)
        return False


def emit_many(metrics: list[dict]) -> int:
    """Batch-emit up to 20 metrics in one call (CloudWatch API limit).

    Each metric dict: {"name": ..., "value": ..., "unit": ..., "dimensions": ...}
    """
    client = _get_client()
    if client is None:
        return 0
    batches = [metrics[i : i + 20] for i in range(0, len(metrics), 20)]
    emitted = 0
    for batch in batches:
        data = []
        for m in batch:
            data.append(
                {
                    "MetricName": m["name"],
                    "Value": float(m["value"]),
                    "Unit": m.get("unit", "None"),
                    "Dimensions": [
                        {"Name": k, "Value": str(v)}
                        for k, v in (m.get("dimensions") or {}).items()
                    ],
                }
            )
        try:
            client.put_metric_data(Namespace=NAMESPACE, MetricData=data)
            emitted += len(data)
        except Exception as exc:
            logger.warning("Batch metric emit failed: %s", exc)
    return emitted


# Convenience wrappers for common metric types used by the dashboard
def emit_model_performance(model_name: str, auc: float, n_obs: int) -> None:
    emit_many(
        [
            {
                "name": "ModelAUC",
                "value": auc,
                "unit": "None",
                "dimensions": {"ModelName": model_name},
            },
            {
                "name": "ObservationsScored",
                "value": n_obs,
                "unit": "Count",
                "dimensions": {"ModelName": model_name},
            },
        ]
    )


def emit_drift_report(
    model_name: str,
    n_significant: int,
    n_moderate: int,
    prediction_psi: float,
    retrain_recommended: bool,
) -> None:
    emit_many(
        [
            {
                "name": "DriftSignificantFeatures",
                "value": n_significant,
                "unit": "Count",
                "dimensions": {"ModelName": model_name},
            },
            {
                "name": "DriftModerateFeatures",
                "value": n_moderate,
                "unit": "Count",
                "dimensions": {"ModelName": model_name},
            },
            {
                "name": "DriftPredictionPSI",
                "value": prediction_psi,
                "unit": "None",
                "dimensions": {"ModelName": model_name},
            },
            {
                "name": "RetrainRecommended",
                "value": 1 if retrain_recommended else 0,
                "unit": "Count",
                "dimensions": {"ModelName": model_name},
            },
        ]
    )


def emit_alert_counts(high: int, medium: int, low: int, insufficient: int) -> None:
    emit_many(
        [
            {
                "name": "AlertsEmitted",
                "value": high,
                "dimensions": {"Severity": "HIGH"},
            },
            {
                "name": "AlertsEmitted",
                "value": medium,
                "dimensions": {"Severity": "MEDIUM"},
            },
            {"name": "AlertsEmitted", "value": low, "dimensions": {"Severity": "LOW"}},
            {"name": "SitesScored", "value": high + medium + low, "unit": "Count"},
            {"name": "SitesInsufficientData", "value": insufficient, "unit": "Count"},
        ]
    )
