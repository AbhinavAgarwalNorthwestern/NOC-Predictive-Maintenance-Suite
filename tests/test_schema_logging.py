"""Tests for schema validation (Pandera-based) + structured logging utilities."""

from __future__ import annotations

import json
import logging
import io

import pandas as pd
import pandera.pandas as pa
import pytest

from battery_pdm.schema_validation import (
    AlarmsSchema,
    SiteStaticSchema,
    validate_scoring_inputs,
)
from battery_pdm._logging import get_logger, JsonFormatter, TimedOp


# ---------- Pandera schema tests ----------


def test_valid_alarms_passes():
    alarms = pd.DataFrame(
        {
            "site_id": ["S1", "S2"],
            "timestamp_h": [10.0, 20.0],
            "alarm_code": ["AC_MAINS_FAIL", "RECTIFIER_FAULT"],
            "severity": ["warning", "critical"],
        }
    )
    # Should not raise
    AlarmsSchema.validate(alarms)


def test_missing_required_column_caught():
    alarms = pd.DataFrame(
        {
            "site_id": ["S1"],
            "timestamp_h": [10.0],
            # missing alarm_code
        }
    )
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        AlarmsSchema.validate(alarms)


def test_negative_load_caught():
    static = pd.DataFrame(
        {
            "site_id": ["S1"],
            "region": ["lahore"],
            "load_A": [-5.0],  # negative!
            "nominal_capacity_ah": [100.0],
            "install_month": [0.0],
            "n_cells": [24],
        }
    )
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        SiteStaticSchema.validate(static)


def test_invalid_region_caught():
    static = pd.DataFrame(
        {
            "site_id": ["S1"],
            "region": ["mars"],  # not in allowed set
            "load_A": [10.0],
            "nominal_capacity_ah": [100.0],
            "install_month": [0.0],
            "n_cells": [24],
        }
    )
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        SiteStaticSchema.validate(static)


def test_nulls_in_non_nullable_caught():
    alarms = pd.DataFrame(
        {
            "site_id": ["S1", None],
            "timestamp_h": [10.0, 20.0],
            "alarm_code": ["AC_MAINS_FAIL", "RECTIFIER_FAULT"],
        }
    )
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        AlarmsSchema.validate(alarms)


def test_duplicate_site_ids_caught():
    static = pd.DataFrame(
        {
            "site_id": ["S1", "S1"],  # duplicates not allowed
            "region": ["lahore", "lahore"],
            "load_A": [10.0, 11.0],
            "nominal_capacity_ah": [100.0, 100.0],
            "install_month": [0.0, 0.0],
            "n_cells": [24, 24],
        }
    )
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        SiteStaticSchema.validate(static)


def test_validate_scoring_inputs_strict_raises_on_failure():
    bad_alarms = pd.DataFrame({"site_id": ["S1"]})  # missing required cols
    static = pd.DataFrame(
        {
            "site_id": ["S1"],
            "region": ["lahore"],
            "load_A": [10.0],
            "nominal_capacity_ah": [100.0],
            "install_month": [0.0],
            "n_cells": [24],
        }
    )
    with pytest.raises(ValueError, match="Scoring input validation failed"):
        validate_scoring_inputs(bad_alarms, static, strict=True)


def test_validate_scoring_inputs_lenient_returns_errors():
    bad_alarms = pd.DataFrame({"site_id": ["S1"]})
    static = pd.DataFrame(
        {
            "site_id": ["S1"],
            "region": ["lahore"],
            "load_A": [10.0],
            "nominal_capacity_ah": [100.0],
            "install_month": [0.0],
            "n_cells": [24],
        }
    )
    errors = validate_scoring_inputs(bad_alarms, static, strict=False)
    assert len(errors["alarms"]) > 0
    assert errors["site_static"] == []


def test_validate_scoring_inputs_all_valid_returns_no_errors():
    alarms = pd.DataFrame(
        {
            "site_id": ["S1"],
            "timestamp_h": [10.0],
            "alarm_code": ["AC_MAINS_FAIL"],
        }
    )
    static = pd.DataFrame(
        {
            "site_id": ["S1"],
            "region": ["lahore"],
            "load_A": [10.0],
            "nominal_capacity_ah": [100.0],
            "install_month": [0.0],
            "n_cells": [24],
        }
    )
    errors = validate_scoring_inputs(alarms, static, strict=False)
    assert errors["alarms"] == []
    assert errors["site_static"] == []


def test_extra_columns_allowed_strict_false():
    """Our schemas use strict=False — upstream can add columns without breaking us."""
    alarms = pd.DataFrame(
        {
            "site_id": ["S1"],
            "timestamp_h": [10.0],
            "alarm_code": ["AC_MAINS_FAIL"],
            "severity": ["critical"],
            "batch_sim_h": [10.0],  # extra
        }
    )
    AlarmsSchema.validate(alarms)


# ---------- Structured logging tests ----------


def test_get_logger_returns_json_formatter():
    log = get_logger("test_battery_pdm_log")
    assert isinstance(log.handlers[0].formatter, JsonFormatter)


def test_log_emits_valid_json():
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    log = logging.getLogger("test_emit_json")
    log.handlers = [handler]
    log.setLevel("INFO")
    log.propagate = False

    log.info("test_event", extra={"sites": 250, "auc": 0.89})
    payload = json.loads(buf.getvalue().strip())
    assert payload["msg"] == "test_event"
    assert payload["sites"] == 250
    assert payload["auc"] == 0.89
    assert "ts" in payload


def test_timed_op_logs_duration():
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    log = logging.getLogger("test_timed")
    log.handlers = [handler]
    log.setLevel("INFO")
    log.propagate = False

    with TimedOp(log, "scoring", n_sites=100):
        pass

    payload = json.loads(buf.getvalue().strip())
    assert payload["op"] == "scoring"
    assert payload["n_sites"] == 100
    assert payload["duration_sec"] >= 0
