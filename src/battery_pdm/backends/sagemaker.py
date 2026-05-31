"""SageMaker backend — deploys model as a real-time endpoint."""

from __future__ import annotations


from .base import BackendConfig


class SagemakerBackend:
    """Deploys models as SageMaker real-time endpoints (<100ms scoring)."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self._sm = None
        self._sm_runtime = None

    @property
    def sm(self):
        if self._sm is None:
            import boto3

            self._sm = boto3.client(
                "sagemaker",
                region_name=self.config.settings.get("region", "ap-south-1"),
            )
        return self._sm

    @property
    def sm_runtime(self):
        if self._sm_runtime is None:
            import boto3

            self._sm_runtime = boto3.client(
                "sagemaker-runtime",
                region_name=self.config.settings.get("region", "ap-south-1"),
            )
        return self._sm_runtime

    def deploy(self, model_path: str, model_name: str) -> dict:
        """Create or update Model + EndpointConfig + Endpoint."""
        # In production: package model_path into S3 tarball, then create resources.
        # This is a simplified illustration.
        endpoint_name = f"{model_name}-endpoint"
        # Check if endpoint already exists
        try:
            self.sm.describe_endpoint(EndpointName=endpoint_name)
            return {
                "backend": "sagemaker",
                "endpoint_name": endpoint_name,
                "action": "exists",
            }
        except self.sm.exceptions.ClientError:
            pass
        return {
            "backend": "sagemaker",
            "endpoint_name": endpoint_name,
            "action": "would_create",
            "note": "Use Terraform sagemaker_endpoint module to provision",
        }

    def invoke(self, model_name: str, features) -> dict:
        """Real-time scoring via SageMaker endpoint."""
        import json

        endpoint_name = f"{model_name}-endpoint"
        response = self.sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(
                {
                    "features": features.to_dict(orient="records")
                    if hasattr(features, "to_dict")
                    else features
                }
            ),
        )
        return json.loads(response["Body"].read())

    def teardown(self, model_name: str) -> None:
        endpoint_name = f"{model_name}-endpoint"
        try:
            self.sm.delete_endpoint(EndpointName=endpoint_name)
        except self.sm.exceptions.ClientError:
            pass
