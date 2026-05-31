"""Shared model registry utilities — used by TrainingFlow, RetrainingFlow, scoring flows.

Responsibilities:
    - Build/save/load reference profile (for drift monitor)
    - Append to model performance history log (for Grafana time-series)
    - Compute and validate feature hash (catches train/inference skew)
    - Save/load isotonic probability calibrator (next to booster.json)
    - Optional MLflow logging (local file store or AWS S3-backed tracking server)

Design: pure functions, no Metaflow imports, so it works in any context.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PERFORMANCE_LOG_PATH = Path("outputs/model_performance_log.parquet")


def compute_feature_hash(feature_cols: list[str]) -> str:
    """Stable hash of the feature column list and order.

    Same hash at training time and inference time = same feature contract.
    """
    serialized = json.dumps(feature_cols, sort_keys=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def validate_feature_hash(meta: dict, feature_cols_used: list[str]) -> None:
    """Raise if the feature hash in meta doesn't match what's being used now."""
    expected = meta.get("feature_hash")
    if expected is None:
        return  # legacy model without hash, allow but warn
    actual = compute_feature_hash(feature_cols_used)
    if expected != actual:
        raise ValueError(
            f"Feature hash mismatch: model expects {expected}, "
            f"got {actual}. Feature contract has drifted between training and inference."
        )


def append_performance_log(
    model_name: str,
    model_version: str,
    metric_name: str,
    metric_value: float,
    n_observations: int,
    feature_hash: str,
    extras: dict | None = None,
    path: Path | str = PERFORMANCE_LOG_PATH,
) -> None:
    """Append one row to the model performance log (append-only parquet).

    This time-series log is the source for the Grafana 'performance depletion' chart.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "n_observations": int(n_observations),
        "feature_hash": feature_hash,
    }
    if extras:
        new_row.update({k: str(v) for k, v in extras.items()})

    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_parquet(path, index=False)


def save_model_artifacts(
    booster,
    feature_cols: list[str],
    feature_groups: list[str],
    metrics: dict,
    model_dir: Path | str,
    extras: dict | None = None,
    versioned: bool = True,
) -> dict:
    """Save model + meta.json with feature_hash. Returns the meta dict that was written.

    If versioned=True, also persists to model_dir/v_<timestamp>/ so prior
    versions are never overwritten. A 'latest' marker file tracks the most
    recent version for auditing.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / "booster.json"))

    feature_hash = compute_feature_hash(feature_cols)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Extract feature importance for monitoring importance drift over time
    try:
        importance = booster.get_score(importance_type="gain")
        ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        feature_importance = {feat: float(gain) for feat, gain in ranked}
    except (AttributeError, ValueError):
        feature_importance = {}

    meta = {
        "model_version": version,
        "feature_cols": feature_cols,
        "feature_groups": feature_groups,
        "feature_hash": feature_hash,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    if extras:
        meta.update(extras)

    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if versioned:
        version_dir = model_dir / f"v_{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(version_dir / "booster.json"))
        (version_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        (model_dir / "latest").write_text(version)

    return meta


def save_reference_profile_for_model(
    training_features: pd.DataFrame,
    feature_cols: list[str],
    predictions: np.ndarray | None,
    model_dir: Path | str,
) -> Path:
    """Save reference profile derived from TRAINING data (not production).

    This is the critical fix: the reference must come from the data the model was
    trained on, not from the first production window the drift monitor happens to see.
    """
    from battery_pdm.monitoring.drift import build_reference_profile

    model_dir = Path(model_dir)
    profile_path = model_dir / "reference_profile.json"
    build_reference_profile(
        features=training_features,
        feature_cols=feature_cols,
        predictions=predictions,
        output_path=profile_path,
    )
    return profile_path


# ---------------------------------------------------------------------------
# MLflow integration — optional, gracefully degrades if MLflow unavailable
# ---------------------------------------------------------------------------


def setup_mlflow_tracking(experiment_name: str = "battery-pdm") -> bool:
    """Configure MLflow tracking. Returns True if MLflow is usable.

    Tracking URI precedence:
        1. MLFLOW_TRACKING_URI env var (set to s3://bucket/mlflow on AWS)
        2. Local file://./mlruns fallback
    """
    try:
        import mlflow
    except ImportError:
        return False

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    else:
        # Local filesystem store
        mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment(experiment_name)
    return True


def log_to_mlflow(
    run_name: str,
    params: dict,
    metrics: dict,
    model=None,
    artifacts: dict | None = None,
    tags: dict | None = None,
) -> str | None:
    """Log a training run to MLflow. Returns run_id or None if MLflow unavailable.

    Safe to call from any flow — silently no-ops if MLflow is not installed.
    """
    try:
        import mlflow
    except ImportError:
        return None

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):
                pass
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        if model is not None:
            try:
                import mlflow.xgboost

                mlflow.xgboost.log_model(model, artifact_path="model")
            except Exception as exc:
                print(f"  MLflow model logging skipped: {exc}")
        if artifacts:
            for name, local_path in artifacts.items():
                p = Path(local_path)
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path=name)
        return run.info.run_id


# ---------------------------------------------------------------------------
# Atomic trigger handling
# ---------------------------------------------------------------------------

TRIGGER_PATH = Path("outputs/drift_reports/retrain_trigger.json")
TRIGGER_PROCESSING_PATH = Path("outputs/drift_reports/retrain_trigger.processing.json")
TRIGGER_CONSUMED_DIR = Path("outputs/drift_reports/consumed_triggers")


def claim_retrain_trigger() -> dict | None:
    """Atomically claim the pending retrain trigger.

    Returns the trigger payload if claimed, None if no trigger exists or
    another worker already claimed it.

    Implementation note: uses ``os.replace()`` (atomic on BOTH POSIX and Windows),
    not ``os.rename()`` (which silently overwrites on Windows, allowing two
    workers to claim the same trigger). To prevent the double-claim race, we
    first check that the processing file doesn't exist, then attempt the rename
    via a temporary "claim sentinel" — fails atomically if another worker has
    already claimed.
    """
    if not TRIGGER_PATH.exists():
        return None
    if TRIGGER_PROCESSING_PATH.exists():
        # Another worker is mid-flight; do not claim
        return None
    try:
        # os.replace is atomic on POSIX and Windows. If TRIGGER_PATH has been
        # deleted between the exists() check and here, os.replace raises FileNotFoundError.
        os.replace(TRIGGER_PATH, TRIGGER_PROCESSING_PATH)
    except (FileNotFoundError, OSError):
        return None
    return json.loads(TRIGGER_PROCESSING_PATH.read_text())


def complete_retrain_trigger(promoted: bool, reason: str = "") -> None:
    """Mark the trigger as fully consumed (only after promotion / archive succeeds)."""
    if not TRIGGER_PROCESSING_PATH.exists():
        return
    TRIGGER_CONSUMED_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = (
        TRIGGER_CONSUMED_DIR
        / f"trigger_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    payload = json.loads(TRIGGER_PROCESSING_PATH.read_text())
    payload["consumed_at"] = datetime.now(timezone.utc).isoformat()
    payload["promoted"] = promoted
    payload["reason"] = reason
    archive_path.write_text(json.dumps(payload, indent=2))
    TRIGGER_PROCESSING_PATH.unlink()


def release_retrain_trigger() -> None:
    """Release a claimed trigger back to pending state (on failure mid-processing).

    Uses ``os.replace()`` for cross-platform atomicity.
    """
    if TRIGGER_PROCESSING_PATH.exists() and not TRIGGER_PATH.exists():
        os.replace(TRIGGER_PROCESSING_PATH, TRIGGER_PATH)


# ---------------------------------------------------------------------------
# Isotonic probability calibration
# ---------------------------------------------------------------------------
#
# XGBoost binary:logistic outputs are usually mis-calibrated (often
# under-confident for our use case). We post-process with isotonic regression
# fit on a HELD-OUT calibration set (different from train + test), then
# persist alongside the booster so scoring flows produce calibrated
# probabilities directly.

CALIBRATOR_FILENAME = "calibrator.pkl"


def train_isotonic_calibrator(raw_scores, y_true):
    """Fit an isotonic regression mapping raw model scores -> calibrated probs.

    Use a separate held-out calibration set (NOT the test set, NOT training).
    Trees + isotonic ~ standard production pattern.

    Parameters
    ----------
    raw_scores : array-like of float in [0, 1] (XGBoost binary:logistic outputs)
    y_true     : array-like of {0, 1}

    Returns
    -------
    Fitted IsotonicRegression with out_of_bounds='clip' (so inference is safe
    if production scores ever drift outside the training-time range).
    """
    from sklearn.isotonic import IsotonicRegression
    import numpy as np

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(raw_scores, dtype=float), np.asarray(y_true, dtype=float))
    return iso


def brier_score(y_true, y_prob) -> float:
    """Brier score = mean squared error between probs and labels. Lower = better."""
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def save_calibrator(calibrator, model_dir) -> None:
    """Pickle the calibrator into <model_dir>/calibrator.pkl."""
    import pickle

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / CALIBRATOR_FILENAME, "wb") as f:
        pickle.dump(calibrator, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_calibrator(model_dir):
    """Load a calibrator if present; return None if missing.

    Returning None lets scoring flows gracefully fall back to raw scores
    (so legacy models without calibrators keep working).
    """
    import pickle

    path = Path(model_dir) / CALIBRATOR_FILENAME
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_calibrator_from_bytes(data: bytes):
    """Deserialize a calibrator from bytes (for S3-loaded artifacts)."""
    import pickle
    import io

    return pickle.load(io.BytesIO(data))


def apply_calibrator(calibrator, raw_scores):
    """Apply a fitted calibrator to model raw scores; safe no-op if calibrator is None."""
    import numpy as np

    if calibrator is None:
        return np.asarray(raw_scores)
    return calibrator.predict(np.asarray(raw_scores, dtype=float))


# ---------------------------------------------------------------------------
# MLflow Model Registry — replaces ad-hoc S3 + meta.json with proper registry
# ---------------------------------------------------------------------------


def register_model(model, model_name: str, run_id: str | None = None) -> str | None:
    """Register a model version in MLflow Model Registry. Returns version number."""
    try:
        import mlflow
        import mlflow.xgboost
    except ImportError:
        return None
    if run_id:
        model_uri = f"runs:/{run_id}/model"
    else:
        with mlflow.start_run() as run:
            mlflow.xgboost.log_model(model, artifact_path="model")
            model_uri = f"runs:/{run.info.run_id}/model"
    result = mlflow.register_model(model_uri, name=model_name)
    return result.version


def transition_model_stage(model_name: str, version: str, stage: str) -> bool:
    """Transition a model version through stages: None -> Staging -> Production -> Archived."""
    try:
        import mlflow

        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
        return True
    except Exception as exc:
        print(f"Stage transition failed: {exc}")
        return False


def load_production_model(model_name: str):
    """Load the current Production version from MLflow Registry."""
    try:
        import mlflow
        import mlflow.xgboost

        return mlflow.xgboost.load_model(f"models:/{model_name}/Production")
    except Exception:
        return None
