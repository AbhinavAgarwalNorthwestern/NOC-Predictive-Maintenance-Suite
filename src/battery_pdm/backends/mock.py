"""Mock backend for tests — deterministic, in-memory, no external calls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BackendConfig


class MockBackend:
    """Returns deterministic mock predictions."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.deployments = {}

    def deploy(self, model_path: str, model_name: str) -> dict:
        self.deployments[model_name] = {"path": model_path}
        return {"backend": "mock", "model_name": model_name, "action": "deployed"}

    def invoke(self, model_name: str, features) -> pd.Series:
        n = len(features) if hasattr(features, "__len__") else 1
        rng = np.random.default_rng(42)
        return pd.Series(rng.uniform(0, 1, n))

    def teardown(self, model_name: str) -> None:
        self.deployments.pop(model_name, None)
