"""Structured JSON logging for production observability.

CloudWatch Logs Insights parses JSON fields automatically — making
queries like "show all flows where AUC dropped below 0.8" trivial.

Usage:
    from battery_pdm._logging import get_logger
    log = get_logger(__name__)
    log.info("scoring_complete", extra={"sites": 250, "auc": 0.89, "flow": "drain_predictor"})
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format log records as one-JSON-line per event for CloudWatch / aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Promote any extra fields (from log.info(..., extra={...})) to top level
        reserved = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in reserved and not key.startswith("_"):
                try:
                    json.dumps(value)  # ensure serializable
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger configured for JSON output (idempotent)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class TimedOp:
    """Context manager for timing operations + structured logging."""

    def __init__(self, logger: logging.Logger, op_name: str, **fields):
        self.logger = logger
        self.op_name = op_name
        self.fields = fields

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        duration = time.perf_counter() - self._t0
        extra = {**self.fields, "duration_sec": round(duration, 3), "op": self.op_name}
        if exc_type is None:
            self.logger.info(f"{self.op_name}_complete", extra=extra)
        else:
            self.logger.error(
                f"{self.op_name}_failed", extra={**extra, "exc_type": exc_type.__name__}
            )
        return False
