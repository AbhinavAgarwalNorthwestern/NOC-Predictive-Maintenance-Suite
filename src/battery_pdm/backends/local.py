"""Local backend — runs scoring as a subprocess. Useful for dev + tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import BackendConfig


class LocalBackend:
    """Runs scoring inline via the same Python process."""

    def __init__(self, config: BackendConfig):
        self.config = config

    def deploy(self, model_path: str, model_name: str) -> dict:
        # For local, "deploy" just records that the model is available
        return {"backend": "local", "model_name": model_name, "model_path": model_path}

    def invoke(self, model_name: str, features: pd.DataFrame) -> pd.Series:
        # Load the model + predict in-process
        import xgboost as xgb
        from battery_pdm.monitoring.model_registry import (
            load_calibrator,
            apply_calibrator,
        )

        model_dir = (
            Path(self.config.settings.get("models_root", "outputs/models")) / model_name
        )
        booster = xgb.Booster()
        booster.load_model(str(model_dir / "booster.json"))
        cal = load_calibrator(model_dir)
        raw = booster.predict(xgb.DMatrix(features.astype(float).fillna(0.0)))
        return pd.Series(apply_calibrator(cal, raw), index=features.index)

    def teardown(self, model_name: str) -> None:
        pass  # no-op for local
