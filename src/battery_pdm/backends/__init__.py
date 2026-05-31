"""Backend abstraction — deploy + invoke models against any compute target.

Pattern: Strategy. Same flow code, different runtime via config.

Available backends:
    - LocalBackend: runs scoring as a Python subprocess (dev)
    - BatchBackend: submits to AWS Batch (our current production)
    - SagemakerBackend: deploys + invokes a SageMaker endpoint (real-time)
    - MockBackend: in-memory for tests
"""

from .base import Backend, BackendConfig, load_backend
from .local import LocalBackend
from .batch import BatchBackend
from .sagemaker import SagemakerBackend
from .mock import MockBackend

__all__ = [
    "Backend",
    "BackendConfig",
    "load_backend",
    "LocalBackend",
    "BatchBackend",
    "SagemakerBackend",
    "MockBackend",
]
