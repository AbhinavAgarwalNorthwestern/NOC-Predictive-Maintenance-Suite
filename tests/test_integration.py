"""End-to-end integration test: Train -> Score -> Drift -> Retrain.

Builds a tiny synthetic alarms dataset, trains the drain predictor, saves
reference profile, scores it, simulates drift, and verifies the drift monitor
flags it. Validates the whole MLOps loop works.

Runs in <10 seconds (no Metaflow execution, just the underlying functions).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_synthetic_dataset():
    """Build a tiny dataset of 20 sites x ~30 alarms each."""
    rng = np.random.default_rng(42)
    sites = []
    alarms = []
    for i in range(20):
        site_id = f"SITE_{i:03d}"
        region = ["lahore", "karachi", "peshawar", "quetta", "islamabad"][i % 5]
        sites.append(
            {
                "site_id": site_id,
                "region": region,
                "manufacturer": "manufacturer_a",
                "install_month": 0.0,
                "load_A": 10.0 + i * 0.5,
                "n_cells": 24,
                "nominal_capacity_ah": 100.0 + i * 5,
                "charger_misconfigured": 0,
                "aging_multiplier": 1.0,
            }
        )
        # ~30 alarms per site spread over 2000 hours
        for cycle in range(15):
            t = cycle * 130.0 + rng.uniform(0, 30)
            alarms.append(
                {
                    "site_id": site_id,
                    "timestamp_h": t,
                    "alarm_code": "AC_MAINS_FAIL",
                    "severity": "warning",
                }
            )
            # LVD half the time
            if rng.random() < 0.5:
                alarms.append(
                    {
                        "site_id": site_id,
                        "timestamp_h": t + 5,
                        "alarm_code": "LOAD_DISCONNECT",
                        "severity": "critical",
                    }
                )
    site_static = pd.DataFrame(sites)
    alarms_df = (
        pd.DataFrame(alarms)
        .sort_values(["site_id", "timestamp_h"])
        .reset_index(drop=True)
    )
    return alarms_df, site_static


def test_full_mlops_loop(tiny_synthetic_dataset, monkeypatch, tmp_path):
    """Train -> Score -> Inject drift -> Detect drift -> Retraining trigger."""
    alarms, site_static = tiny_synthetic_dataset

    # Redirect outputs to tmp_path so we don't pollute real outputs/
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()

    # --- TRAIN: minimal drain predictor ---
    from battery_pdm.common.features import compute_features
    from battery_pdm.synth.load_shedding import build_load_shedding_schedule

    schedule = build_load_shedding_schedule(n_months=6, seed=42)

    # Build labels (daily drain events) — keep tiny
    labels_list = []
    for site_id, g in alarms.groupby("site_id"):
        max_h = g["timestamp_h"].max()
        for t in [200.0, 800.0, 1500.0]:
            if t < max_h - 48:
                lvds = g[
                    (g["alarm_code"] == "LOAD_DISCONNECT")
                    & (g["timestamp_h"] >= t)
                    & (g["timestamp_h"] < t + 48)
                ]
                labels_list.append(
                    {
                        "site_id": site_id,
                        "ref_time_h": t,
                        "drain_event": int(len(lvds) > 0),
                    }
                )
    labels = pd.DataFrame(labels_list)
    assert labels["drain_event"].sum() > 0, "Need at least some positives for training"

    features = compute_features(
        labels=labels.rename(columns={"ref_time_h": "mains_fail_h"}),
        groups=["alarm_history", "site_static", "load_shedding_schedule"],
        inputs={"alarms": alarms, "site_static": site_static, "schedule": schedule},
        ref_time_col="mains_fail_h",
    )
    features = features.rename(columns={"mains_fail_h": "ref_time_h"})
    feature_cols = [
        c
        for c in features.columns
        if c not in ("site_id", "ref_time_h")
        and c not in ("charger_misconfigured", "aging_multiplier")
    ]

    import xgboost as xgb

    y = labels["drain_event"].values
    X = features[feature_cols].astype(float).fillna(0.0)
    scale_pos = max(1.0, (len(y) - y.sum()) / max(y.sum(), 1))
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "max_depth": 3,
            "learning_rate": 0.1,
            "scale_pos_weight": scale_pos,
            "tree_method": "hist",
        },
        xgb.DMatrix(X, label=y),
        num_boost_round=20,
        verbose_eval=False,
    )
    preds = booster.predict(xgb.DMatrix(X))

    # --- SAVE: use model_registry ---
    from battery_pdm.monitoring.model_registry import (
        save_model_artifacts,
        save_reference_profile_for_model,
        append_performance_log,
        validate_feature_hash,
    )

    model_dir = tmp_path / "model"
    meta = save_model_artifacts(
        booster=booster,
        feature_cols=feature_cols,
        feature_groups=["alarm_history"],
        metrics={"roc_auc": 0.8},
        model_dir=model_dir,
    )
    assert "feature_hash" in meta
    assert "model_version" in meta

    save_reference_profile_for_model(
        training_features=features[feature_cols + ["site_id"]],
        feature_cols=feature_cols,
        predictions=preds,
        model_dir=model_dir,
    )
    assert (model_dir / "reference_profile.json").exists()

    append_performance_log(
        model_name="test_model",
        model_version=meta["model_version"],
        metric_name="roc_auc",
        metric_value=0.8,
        n_observations=len(X),
        feature_hash=meta["feature_hash"],
        path=tmp_path / "model_perf.parquet",
    )
    assert (tmp_path / "model_perf.parquet").exists()

    # --- FEATURE HASH VALIDATION ---
    validate_feature_hash(meta, feature_cols)  # should not raise
    with pytest.raises(ValueError):
        validate_feature_hash(meta, feature_cols + ["fake_extra_col"])

    # --- DRIFT DETECTION ---
    from battery_pdm.monitoring.drift import load_reference_profile, detect_drift

    # Inject drift: zero out all alarm-derived features
    drifted_features = features.copy()
    for col in feature_cols:
        if "count" in col or "hours" in col:
            drifted_features[col] = 0.0

    ref_profile = load_reference_profile(model_dir / "reference_profile.json")
    drift_report = detect_drift(
        reference_profile=ref_profile,
        current_features=drifted_features,
        feature_cols=feature_cols,
    )
    # Drift should be detected
    assert drift_report["summary"]["n_significant_drift"] > 0, (
        "Should detect injected drift"
    )


def test_data_quality_flags_insufficient_sites(tiny_synthetic_dataset):
    """Sites with <3 lifetime alarms should be flagged INSUFFICIENT_DATA."""
    from battery_pdm.monitoring.data_quality import assess_scoring_inputs

    alarms, site_static = tiny_synthetic_dataset
    # Add a phantom site with no alarms
    site_static_extended = pd.concat(
        [
            site_static,
            pd.DataFrame(
                [
                    {
                        "site_id": "SITE_PHANTOM",
                        "region": "lahore",
                        "manufacturer": "manufacturer_a",
                        "install_month": 0.0,
                        "load_A": 10.0,
                        "n_cells": 24,
                        "nominal_capacity_ah": 100.0,
                        "charger_misconfigured": 0,
                        "aging_multiplier": 1.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    scorable, report = assess_scoring_inputs(
        site_ids=site_static_extended["site_id"].tolist(),
        alarms=alarms,
        site_static=site_static_extended,
        schedule=None,
        ref_time_h=2000.0,
    )
    assert "SITE_PHANTOM" not in scorable
    assert report.n_sites_insufficient_data >= 1


def test_compute_regional_priors_returns_per_region_drain_rate(tiny_synthetic_dataset):
    """Regional priors should be drain rate per region in [0, 1]."""
    from battery_pdm.monitoring.data_quality import compute_regional_priors

    alarms, site_static = tiny_synthetic_dataset
    priors = compute_regional_priors(alarms, site_static, horizon_h=48)

    # Both regions in the fixture should have a prior
    assert set(priors.keys()) >= {"lahore", "karachi"}
    for region, rate in priors.items():
        assert 0.0 <= rate <= 1.0, f"region {region} prior {rate} out of [0,1]"


def test_compute_regional_priors_empty_data():
    """Empty alarms returns empty dict (no crash)."""
    from battery_pdm.monitoring.data_quality import compute_regional_priors

    empty_alarms = pd.DataFrame(
        columns=["site_id", "timestamp_h", "alarm_code", "severity"]
    )
    empty_static = pd.DataFrame(columns=["site_id", "region"])
    assert compute_regional_priors(empty_alarms, empty_static) == {}


def test_isotonic_calibrator_round_trip(tmp_path):
    """Calibrator can be trained, saved, loaded, and applied; reduces Brier."""
    from battery_pdm.monitoring.model_registry import (
        train_isotonic_calibrator,
        save_calibrator,
        load_calibrator,
        apply_calibrator,
        brier_score,
    )

    rng = np.random.default_rng(42)
    # Synthetic miscalibrated scores: 0.5-0.8 maps to 30% actual rate
    # (model is overconfident in the mid range)
    n = 1000
    raw_scores = rng.uniform(0, 1, n)
    # True labels are biased low at all but the highest scores
    y = (rng.uniform(0, 1, n) < raw_scores**2).astype(int)

    cal = train_isotonic_calibrator(raw_scores, y)
    brier_uncal = brier_score(y, raw_scores)
    brier_cal = brier_score(y, cal.predict(raw_scores))
    assert brier_cal <= brier_uncal, "Calibration must not make Brier worse"

    # Save + load round-trip
    save_calibrator(cal, tmp_path)
    loaded = load_calibrator(tmp_path)
    assert loaded is not None
    np.testing.assert_array_almost_equal(
        loaded.predict(raw_scores),
        cal.predict(raw_scores),
    )

    # apply_calibrator with None is identity (legacy-model fallback)
    np.testing.assert_array_equal(
        apply_calibrator(None, raw_scores),
        raw_scores,
    )


def test_isotonic_calibrator_clips_out_of_range():
    """Inputs outside [0,1] should still produce calibrated outputs in [0,1]."""
    from battery_pdm.monitoring.model_registry import (
        train_isotonic_calibrator,
        apply_calibrator,
    )

    rng = np.random.default_rng(7)
    raw = rng.uniform(0, 1, 200)
    y = (raw > 0.3).astype(int)
    cal = train_isotonic_calibrator(raw, y)
    # Production might (rarely) produce scores outside [0,1] if model is weird
    extreme = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    out = apply_calibrator(cal, extreme)
    assert (out >= 0).all() and (out <= 1).all()


def test_atomic_trigger_claim(tmp_path, monkeypatch):
    """claim_retrain_trigger should be atomic and not double-claim."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "drift_reports").mkdir(parents=True)

    from battery_pdm.monitoring import model_registry as mr

    # Write a trigger
    trigger_data = {"reasons": ["test"]}
    mr.TRIGGER_PATH.write_text(json.dumps(trigger_data))

    # First claim succeeds
    claimed = mr.claim_retrain_trigger()
    assert claimed is not None
    assert claimed["reasons"] == ["test"]

    # Second claim immediately should fail (still processing)
    second = mr.claim_retrain_trigger()
    assert second is None

    # Complete it
    mr.complete_retrain_trigger(promoted=True, reason="ok")

    # Now both processing and pending should be gone
    assert not mr.TRIGGER_PROCESSING_PATH.exists()
    assert not mr.TRIGGER_PATH.exists()
    # Consumed archive should exist
    consumed_files = list(mr.TRIGGER_CONSUMED_DIR.glob("*.json"))
    assert len(consumed_files) == 1
