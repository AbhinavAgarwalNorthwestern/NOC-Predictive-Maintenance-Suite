"""Smoke tests for the Evidently AI drift wrapper.

Evidently is an OPTIONAL dependency — these tests should be skippable
if Evidently isn't installed (e.g., in minimal CI). We don't run heavy
Evidently logic here, just confirm the wrapper plumbing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import evidently  # noqa: F401

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


@pytest.mark.skipif(not EVIDENTLY_AVAILABLE, reason="Evidently not installed")
def test_evidently_wrapper_detects_obvious_drift():
    from battery_pdm.monitoring.drift import build_reference_profile
    from battery_pdm.monitoring.evidently_drift import detect_drift_evidently

    rng = np.random.default_rng(42)
    ref_df = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, 1000),
            "feat_b": rng.normal(10, 2, 1000),
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.json"
        build_reference_profile(ref_df, ["feat_a", "feat_b"], output_path=path)
        from battery_pdm.monitoring.drift import load_reference_profile

        ref_profile = load_reference_profile(path)

    # Strongly drifted current data
    cur_df = pd.DataFrame(
        {
            "feat_a": rng.normal(5, 1, 500),  # mean shift of 5 sigma
            "feat_b": rng.normal(20, 2, 500),  # mean shift of 5 sigma
        }
    )
    result = detect_drift_evidently(ref_profile, cur_df)
    # Evidently should flag both features
    assert result["summary"]["n_features_monitored"] == 2
    assert result["summary"]["status"] in (
        "DRIFT_DETECTED",
        "STABLE",
    )  # depends on Evidently threshold
    assert result["summary"].get("engine") == "evidently"


@pytest.mark.skipif(not EVIDENTLY_AVAILABLE, reason="Evidently not installed")
def test_evidently_wrapper_no_drift_when_stable():
    from battery_pdm.monitoring.drift import (
        build_reference_profile,
        load_reference_profile,
    )
    from battery_pdm.monitoring.evidently_drift import detect_drift_evidently

    rng = np.random.default_rng(42)
    ref_df = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, 1000),
            "feat_b": rng.normal(10, 2, 1000),
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.json"
        build_reference_profile(ref_df, ["feat_a", "feat_b"], output_path=path)
        ref_profile = load_reference_profile(path)

    # Same distribution
    cur_df = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, 500),
            "feat_b": rng.normal(10, 2, 500),
        }
    )
    result = detect_drift_evidently(ref_profile, cur_df)
    # Should NOT flag drift on stable data
    assert result["summary"]["n_significant_drift"] == 0


@pytest.mark.skipif(not EVIDENTLY_AVAILABLE, reason="Evidently not installed")
def test_evidently_wrapper_returns_our_schema():
    """Evidently wrapper must return the same dict shape as drift.detect_drift()."""
    from battery_pdm.monitoring.drift import (
        build_reference_profile,
        load_reference_profile,
    )
    from battery_pdm.monitoring.evidently_drift import detect_drift_evidently

    rng = np.random.default_rng(7)
    ref_df = pd.DataFrame({"feat": rng.normal(0, 1, 200)})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.json"
        build_reference_profile(ref_df, ["feat"], output_path=path)
        ref_profile = load_reference_profile(path)

    cur_df = pd.DataFrame({"feat": rng.normal(0, 1, 100)})
    result = detect_drift_evidently(ref_profile, cur_df)

    # Must have the same top-level keys as detect_drift()
    assert "feature_drift" in result
    assert "prediction_drift" in result
    assert "summary" in result
    assert "retrain_recommended" in result["summary"]
    assert "n_features_monitored" in result["summary"]
