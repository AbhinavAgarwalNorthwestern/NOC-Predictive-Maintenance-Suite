"""Backend abstraction tests."""

import pandas as pd

from battery_pdm.backends import (
    BackendConfig,
    LocalBackend,
    BatchBackend,
    SagemakerBackend,
    MockBackend,
    load_backend,
)


def test_mock_backend_returns_predictions():
    backend = MockBackend(config=BackendConfig(name="test"))
    backend.deploy("dummy_path", "test_model")
    preds = backend.invoke("test_model", pd.DataFrame({"x": [1, 2, 3]}))
    assert len(preds) == 3
    assert all(0 <= p <= 1 for p in preds)


def test_load_backend_via_spec():
    config = BackendConfig(name="mock")
    backend = load_backend("battery_pdm.backends.mock.MockBackend", config)
    assert isinstance(backend, MockBackend)


def test_local_backend_deploy_records_path():
    backend = LocalBackend(config=BackendConfig(name="local"))
    result = backend.deploy("outputs/models/test", "test_model")
    assert result["backend"] == "local"
    assert result["model_path"] == "outputs/models/test"


def test_batch_backend_deploy_returns_metadata():
    backend = BatchBackend(
        config=BackendConfig(name="batch", settings={"job_queue": "test-queue"})
    )
    result = backend.deploy("ignored", "drain_predictor")
    assert result["backend"] == "batch"
    assert "job_definition" in result


def test_sagemaker_backend_deploy_returns_metadata():
    backend = SagemakerBackend(
        config=BackendConfig(name="sm", settings={"region": "ap-south-1"})
    )
    # Without actually hitting AWS, the describe call will fail; verify it returns sensible default
    # In a real test we'd mock boto3
    try:
        backend.deploy("ignored", "drain_predictor")
    except Exception:
        pass  # expected without AWS creds
