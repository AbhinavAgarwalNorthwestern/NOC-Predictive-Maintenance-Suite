"""Schema validation for flow inputs — Pandera-based (industry standard).

Why Pandera over hand-rolled checks:
    - Built on Pydantic v2 — type-safe, declarative, fast
    - Handles dtype coercion, regex patterns, custom validators, nullable columns
    - One-line `Schema.validate(df)` call — fails loud with structured errors
    - Industry standard for DataFrame validation

Usage:
    from battery_pdm.schema_validation import (
        AlarmsSchema, SiteStaticSchema, ScheduleSchema,
        validate_scoring_inputs,
    )

    # Validate one DataFrame:
    AlarmsSchema.validate(alarms)   # raises SchemaError on failure

    # Validate all three at flow start:
    validate_scoring_inputs(alarms, site_static, schedule)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series


# Severity values seen in our simulator
_ALARM_CODES = {
    "AC_MAINS_FAIL",
    "LOAD_DISCONNECT",
    "RECTIFIER_FAULT",
    "BATT_UNDERVOLTAGE",
    "BATT_HIGH_TEMP",
    "CELL_IMBALANCE",
    "REPEAT_FAILURE_FLAG",
    "TICKET_RAISED",
    "TICKET_CLOSED",
}
_SEVERITY_VALUES = {"warning", "critical", "info"}
_REGIONS = {"lahore", "karachi", "peshawar", "quetta", "islamabad"}


class AlarmsSchema(pa.DataFrameModel):
    """Schema for the alarm stream."""

    site_id: Series[str] = pa.Field(nullable=False)
    timestamp_h: Series[float] = pa.Field(ge=0, nullable=False)
    alarm_code: Series[str] = pa.Field(nullable=False)
    # severity is optional in some data sources
    severity: Optional[Series[str]] = pa.Field(nullable=True)

    class Config:
        strict = False  # allow extra columns (e.g., batch_sim_h)
        coerce = True


class SiteStaticSchema(pa.DataFrameModel):
    """Per-site static configuration."""

    site_id: Series[str] = pa.Field(nullable=False, unique=True)
    region: Series[str] = pa.Field(nullable=False, isin=_REGIONS)
    load_A: Series[float] = pa.Field(ge=0, nullable=False)
    nominal_capacity_ah: Series[float] = pa.Field(ge=0, nullable=False)
    install_month: Series[float] = pa.Field(ge=0, nullable=False)
    n_cells: Series[int] = pa.Field(ge=1, nullable=False)
    manufacturer: Optional[Series[str]] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


class ScheduleSchema(pa.DataFrameModel):
    """Regional load-shedding schedule."""

    region: Series[str] = pa.Field(nullable=False, isin=_REGIONS)
    timestamp_h: Series[float] = pa.Field(ge=0, nullable=False)
    scheduled_offgrid: Series[int] = pa.Field(ge=0, le=1, nullable=False)
    severity_score: Optional[Series[float]] = pa.Field(ge=0, le=1, nullable=True)

    class Config:
        strict = False
        coerce = True


def validate_scoring_inputs(
    alarms: pd.DataFrame,
    site_static: pd.DataFrame,
    schedule: pd.DataFrame | None = None,
    strict: bool = True,
) -> dict[str, list[str]]:
    """One-shot validation of the three scoring inputs.

    Returns dict of {input_name: [error_messages]} (empty list if OK).
    If strict=True (default), raises ValueError on any failure.
    """
    errors: dict[str, list[str]] = {"alarms": [], "site_static": [], "schedule": []}

    for df, name, schema in [
        (alarms, "alarms", AlarmsSchema),
        (site_static, "site_static", SiteStaticSchema),
    ]:
        try:
            schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            errors[name] = [
                str(err) for err in exc.failure_cases.to_dict(orient="records")
            ]
        except pa.errors.SchemaError as exc:
            errors[name] = [str(exc)]

    if schedule is not None and not schedule.empty:
        try:
            ScheduleSchema.validate(schedule, lazy=True)
        except pa.errors.SchemaErrors as exc:
            errors["schedule"] = [
                str(err) for err in exc.failure_cases.to_dict(orient="records")
            ]
        except pa.errors.SchemaError as exc:
            errors["schedule"] = [str(exc)]

    if strict and any(errors.values()):
        all_errors = "\n".join(
            f"  [{k}] {msg}" for k, msgs in errors.items() for msg in msgs
        )
        raise ValueError(f"Scoring input validation failed:\n{all_errors}")
    return errors
