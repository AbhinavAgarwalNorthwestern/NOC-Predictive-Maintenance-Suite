"""AWS Batch backend — submits scoring as a Batch job."""

from __future__ import annotations

from typing import Any

from .base import BackendConfig


class BatchBackend:
    """Submits scoring jobs to AWS Batch. Our current production pattern."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "batch", region_name=self.config.settings.get("region", "ap-south-1")
            )
        return self._client

    def deploy(self, model_path: str, model_name: str) -> dict:
        # For Batch, "deploy" = make sure the image + job definition exist
        # (assumed already done via Terraform). Returns metadata for invocation.
        return {
            "backend": "batch",
            "model_name": model_name,
            "job_queue": self.config.settings["job_queue"],
            "job_definition": f"{self.config.settings.get('name_prefix', 'battery-pdm-dev')}-{model_name}",
        }

    def invoke(self, model_name: str, features: Any = None) -> dict:
        # In Batch, you submit a JOB rather than invoking inline.
        # features arg is ignored — the Batch job reads from S3 directly.
        prefix = self.config.settings.get("name_prefix", "battery-pdm-dev")
        response = self.client.submit_job(
            jobName=f"{model_name}-{__import__('time').strftime('%Y%m%d-%H%M%S')}",
            jobQueue=self.config.settings["job_queue"],
            jobDefinition=f"{prefix}-{model_name}",
        )
        return {"job_id": response["jobId"], "job_arn": response["jobArn"]}

    def teardown(self, model_name: str) -> None:
        pass  # job definitions stay
