"""Backend protocol. All concrete backends implement deploy() + invoke()."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class BackendConfig:
    """Configuration passed to every backend at instantiation."""

    name: str
    settings: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Interface every backend must satisfy."""

    config: BackendConfig

    def deploy(self, model_path: str, model_name: str) -> dict:
        """Make the model live in this backend. Returns deployment metadata."""
        ...

    def invoke(self, model_name: str, features: Any) -> Any:
        """Score a single batch of features. Returns predictions."""
        ...

    def teardown(self, model_name: str) -> None:
        """Remove the model from this backend."""
        ...


def load_backend(spec: str, config: BackendConfig) -> Backend:
    """Dynamic loader: spec = 'battery_pdm.backends.batch.BatchBackend'."""
    module_path, class_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config=config)
