"""Feature registry — extensible pattern for adding new features.

Usage:
    @register_feature("discharge_slope_30d", family="per_discharge")
    def compute_discharge_slope_30d(telemetry: pd.DataFrame) -> pd.Series:
        ...

    # Compute all registered features:
    feature_matrix = compute_all_features(telemetry_df, labels_df)

    # Or compute one family:
    discharge_features = compute_family("per_discharge", telemetry_df, labels_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd


@dataclass
class FeatureSpec:
    name: str
    family: str
    compute_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
    description: str = ""


_REGISTRY: dict[str, FeatureSpec] = {}


def register_feature(
    name: str,
    family: str,
    description: str = "",
) -> Callable:
    """Decorator to register a feature computation function."""
    def decorator(fn: Callable[[pd.DataFrame, pd.DataFrame], pd.Series]) -> Callable:
        _REGISTRY[name] = FeatureSpec(
            name=name,
            family=family,
            compute_fn=fn,
            description=description,
        )
        return fn
    return decorator


def get_registry() -> dict[str, FeatureSpec]:
    return _REGISTRY.copy()


def list_features(family: Optional[str] = None) -> list[str]:
    if family is None:
        return list(_REGISTRY.keys())
    return [k for k, v in _REGISTRY.items() if v.family == family]


def compute_feature(name: str, telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    spec = _REGISTRY[name]
    return spec.compute_fn(telemetry, labels)


def compute_family(family: str, telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    features = {}
    for name, spec in _REGISTRY.items():
        if spec.family == family:
            features[name] = spec.compute_fn(telemetry, labels)
    return pd.DataFrame(features)


def compute_all_features(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Compute all registered features, return DataFrame indexed by site_id."""
    features = {}
    for name, spec in _REGISTRY.items():
        features[name] = spec.compute_fn(telemetry, labels)
    df = pd.DataFrame(features)
    return df
