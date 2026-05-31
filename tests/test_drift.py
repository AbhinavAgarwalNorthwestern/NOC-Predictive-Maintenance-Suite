"""Tests for drift detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from battery_pdm.monitoring.drift import (
    build_reference_profile,
    compute_psi,
    detect_drift,
    load_reference_profile,
    _psi_single,
    PSI_THRESHOLD_SIGNIFICANT,
)


def test_psi_identical_distributions_is_low():
    """PSI of a distribution against itself should be ~0."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5000)
    psi = _psi_single(x, x.copy())
    assert psi < 0.01


def test_psi_shifted_distribution_is_high():
    """PSI between two clearly different distributions should be > threshold."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(2, 1, 5000)  # mean shift of 2 std
    psi = _psi_single(ref, cur)
    assert psi > PSI_THRESHOLD_SIGNIFICANT


def test_build_and_load_reference_profile():
    df = pd.DataFrame(
        {
            "feat_a": np.arange(100, dtype=float),
            "feat_b": np.arange(100, dtype=float) * 2,
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.json"
        profile = build_reference_profile(df, ["feat_a", "feat_b"], output_path=path)

        assert profile["n_samples"] == 100
        assert "feat_a" in profile["features"]

        loaded = load_reference_profile(path)
        assert loaded["n_samples"] == 100
        assert "feat_a" in loaded["features"]


def test_compute_psi_returns_drift_levels():
    rng = np.random.default_rng(42)
    ref_df = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    cur_df = pd.DataFrame({"x": rng.normal(3, 1, 1000)})  # strong drift

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.json"
        build_reference_profile(ref_df, ["x"], output_path=path)
        ref_profile = load_reference_profile(path)

    result = compute_psi(ref_profile, cur_df)
    assert len(result) == 1
    assert result.iloc[0]["drift_level"] == "SIGNIFICANT"
    assert result.iloc[0]["psi"] > PSI_THRESHOLD_SIGNIFICANT


def test_detect_drift_no_drift_when_stable():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, 1000),
            "feat_b": rng.uniform(0, 1, 1000),
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.json"
        build_reference_profile(df, ["feat_a", "feat_b"], output_path=path)
        ref_profile = load_reference_profile(path)

    # Score with new sample from SAME distribution
    new_df = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, 500),
            "feat_b": rng.uniform(0, 1, 500),
        }
    )
    report = detect_drift(ref_profile, new_df)
    assert not report["summary"]["retrain_recommended"]


def test_detect_drift_triggers_when_significant():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({f"feat_{i}": rng.normal(0, 1, 1000) for i in range(5)})

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.json"
        build_reference_profile(df, list(df.columns), output_path=path)
        ref_profile = load_reference_profile(path)

    # Shift ALL features significantly
    shifted_df = pd.DataFrame({f"feat_{i}": rng.normal(5, 1, 500) for i in range(5)})
    report = detect_drift(ref_profile, shifted_df)
    assert report["summary"]["retrain_recommended"]
    assert report["summary"]["n_significant_drift"] >= 3


def test_prediction_drift_detected():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"feat_a": rng.normal(0, 1, 1000)})
    pred_ref = rng.uniform(0, 0.3, 1000)
    pred_cur = rng.uniform(0.5, 1.0, 500)  # scores shifted higher

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.json"
        build_reference_profile(df, ["feat_a"], predictions=pred_ref, output_path=path)
        ref_profile = load_reference_profile(path)

    report = detect_drift(ref_profile, df, current_predictions=pred_cur)
    assert report["prediction_drift"]["drift_detected"]


def test_psi_handles_nan():
    """PSI should ignore NaN values rather than crash."""
    ref = np.array([1.0, 2.0, 3.0, np.nan, 5.0] * 20)
    cur = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 20)
    psi = _psi_single(ref, cur)
    assert not np.isnan(psi)
