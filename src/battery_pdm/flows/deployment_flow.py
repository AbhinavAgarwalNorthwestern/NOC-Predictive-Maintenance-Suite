"""DeploymentFlow — deploy latest model from MLflow Registry to SageMaker endpoint.

Pulls the production-registered model, packages it as a SageMaker model.tar.gz,
deploys to the endpoint, and runs a smoke test with synthetic data.

This is code-reviewable + auditable deployment, not a manual CLI step.

Run:
    python -m battery_pdm.flows.deployment_flow run
    python -m battery_pdm.flows.deployment_flow run --target sagemaker --endpoint-name drain-endpoint
"""

from __future__ import annotations

import json
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, Parameter, card, current, project, step
from metaflow.cards import Markdown

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@project(name="battery_pdm")
class DeploymentFlow(FlowSpec):
    """Deploy the latest registered model to a serving endpoint."""

    model_name = Parameter(
        "model-name",
        default="drain_predictor_48h",
        help="Name of the model in MLflow Registry (or local model dir name).",
    )
    target = Parameter(
        "target",
        default="local",
        help="Deployment target: 'local' (validate only), 'sagemaker', 'batch' (backend).",
    )
    endpoint_name = Parameter(
        "endpoint-name",
        default="",
        help="SageMaker endpoint name. If empty, uses '{model_name}-endpoint'.",
    )
    models_bucket = Parameter(
        "models-bucket",
        default="",
        help="S3 bucket for model artifacts. Required for sagemaker target.",
    )
    smoke_test_samples = Parameter(
        "smoke-test-samples",
        type=int,
        default=5,
        help="Number of synthetic samples for post-deployment smoke test.",
    )

    @step
    def start(self):
        """Load model artifacts from registry or local path."""
        from battery_pdm.monitoring.model_registry import (
            load_production_model,
            setup_mlflow_tracking,
        )

        self.endpoint = self.endpoint_name or f"{self.model_name}-endpoint"

        if setup_mlflow_tracking("battery-pdm"):
            self.registry_model = load_production_model(self.model_name)
        else:
            self.registry_model = None

        local_model_dir = Path("outputs/models") / self.model_name
        if self.registry_model is not None:
            print(
                f"Loaded model '{self.model_name}' from MLflow Registry (Production stage)"
            )
            self.model_source = "mlflow_registry"
            self.local_model_dir = str(local_model_dir)
        elif local_model_dir.exists() and (local_model_dir / "booster.json").exists():
            print(
                f"MLflow Registry unavailable — using local model at {local_model_dir}"
            )
            self.model_source = "local"
            self.local_model_dir = str(local_model_dir)
        else:
            raise FileNotFoundError(
                f"No model found. Either register '{self.model_name}' in MLflow "
                f"or ensure outputs/models/{self.model_name}/booster.json exists."
            )

        self.next(self.package_model)

    @step
    def package_model(self):
        """Package model artifacts into model.tar.gz for SageMaker."""

        model_dir = Path(self.local_model_dir) if self.model_source == "local" else None

        if self.model_source == "local":
            self.meta = json.loads((model_dir / "meta.json").read_text())
            self.model_version = self.meta.get("model_version", "unknown")
        else:
            self.model_version = "registry_production"
            model_dir = Path("outputs/models") / self.model_name
            self.meta = json.loads((model_dir / "meta.json").read_text())

        if self.target == "sagemaker":
            tar_path = Path(tempfile.gettempdir()) / f"{self.model_name}_model.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                for artifact in ["booster.json", "meta.json", "calibrator.pkl"]:
                    artifact_path = model_dir / artifact
                    if artifact_path.exists():
                        tar.add(str(artifact_path), arcname=artifact)
                serve_path = Path(__file__).parent.parent / "inference" / "serve.py"
                if serve_path.exists():
                    tar.add(str(serve_path), arcname="serve.py")
            self.tar_path = str(tar_path)
            print(
                f"Packaged model.tar.gz at {tar_path} ({tar_path.stat().st_size / 1024:.1f} KB)"
            )
        else:
            self.tar_path = None
            print(f"Target is '{self.target}' — skipping tar packaging")

        self.next(self.deploy)

    @card(type="default")
    @step
    def deploy(self):
        """Deploy model to the target platform."""
        if self.target == "local":
            print("Target=local: skipping deployment (validation-only mode)")
            self.deployment_status = "skipped_local"

        elif self.target == "sagemaker":
            self._deploy_sagemaker()
            self.deployment_status = "deployed"

        else:
            from battery_pdm.backends.base import load_backend, BackendConfig

            config = BackendConfig(
                name=self.target, settings={"endpoint": self.endpoint}
            )
            backend = load_backend(
                f"battery_pdm.backends.{self.target}.{self.target.title()}Backend",
                config,
            )
            result = backend.deploy(self.local_model_dir, self.model_name)
            self.deployment_status = "deployed"
            print(f"Backend deployment result: {result}")

        current.card.append(Markdown(f"# Deployment: {self.model_name}"))
        current.card.append(Markdown(f"**Target:** {self.target}"))
        current.card.append(Markdown(f"**Endpoint:** {self.endpoint}"))
        current.card.append(Markdown(f"**Model version:** {self.model_version}"))
        current.card.append(Markdown(f"**Status:** {self.deployment_status}"))

        self.next(self.smoke_test)

    @step
    def smoke_test(self):
        """Send synthetic samples to verify the deployed endpoint works."""
        import numpy as np

        feature_cols = self.meta["feature_cols"]
        n = self.smoke_test_samples

        np.random.seed(42)
        synthetic = pd.DataFrame(
            {col: np.random.uniform(0, 1, size=n) for col in feature_cols}
        )

        if self.target == "local" or self.deployment_status == "skipped_local":
            import xgboost as xgb
            from battery_pdm.monitoring.model_registry import (
                load_calibrator,
                apply_calibrator,
            )

            model_dir = (
                Path(self.local_model_dir) if self.model_source == "local" else None
            )
            booster = xgb.Booster()
            booster.load_model(str(model_dir / "booster.json"))
            calibrator = load_calibrator(model_dir)

            X = synthetic[feature_cols].astype(float)
            raw = booster.predict(xgb.DMatrix(X))
            preds = apply_calibrator(calibrator, raw)
            self.smoke_results = {
                "predictions": [float(p) for p in preds],
                "n_samples": n,
            }
            print(
                f"Local smoke test: {n} samples scored, mean prediction={np.mean(preds):.4f}"
            )

        elif self.target == "sagemaker":
            self.smoke_results = self._invoke_sagemaker(synthetic, feature_cols)
            print(f"SageMaker smoke test: {self.smoke_results}")

        else:
            self.smoke_results = {
                "status": "skipped",
                "reason": f"no smoke test for target={self.target}",
            }

        self.next(self.end)

    @step
    def end(self):
        """Log deployment to MLflow for audit trail."""
        from battery_pdm.monitoring.model_registry import (
            setup_mlflow_tracking,
            log_to_mlflow,
        )

        if setup_mlflow_tracking("battery-pdm-deployments"):
            log_to_mlflow(
                run_name=f"deploy_{self.model_name}_{self.model_version}",
                params={
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "target": self.target,
                    "endpoint": self.endpoint,
                },
                metrics={"smoke_test_samples": self.smoke_test_samples},
                tags={"event": "deployment", "status": self.deployment_status},
            )

        print(
            f"DeploymentFlow complete. Model={self.model_name} Target={self.target} "
            f"Status={self.deployment_status}"
        )

    def _deploy_sagemaker(self):
        """Upload model.tar.gz to S3 and update SageMaker endpoint."""
        import boto3

        if not self.models_bucket:
            raise ValueError("--models-bucket required for sagemaker target")

        s3 = boto3.client("s3")
        s3_key = f"{self.model_name}/model.tar.gz"
        s3.upload_file(self.tar_path, self.models_bucket, s3_key)
        print(f"Uploaded model.tar.gz to s3://{self.models_bucket}/{s3_key}")

        sm = boto3.client("sagemaker")
        try:
            sm.describe_endpoint(EndpointName=self.endpoint)
            print(
                f"Endpoint '{self.endpoint}' exists — will be updated on next Terraform apply "
                f"(model data URL now points to new artifact)"
            )
        except sm.exceptions.ClientError:
            print(
                f"Endpoint '{self.endpoint}' does not exist yet — "
                f"run 'just tf-apply' to create it via Terraform"
            )

    def _invoke_sagemaker(self, df, feature_cols):
        """Invoke the SageMaker endpoint with test data."""
        import boto3

        runtime = boto3.client("sagemaker-runtime")
        payload = json.dumps({"features": df[feature_cols].to_dict(orient="records")})

        try:
            response = runtime.invoke_endpoint(
                EndpointName=self.endpoint,
                ContentType="application/json",
                Body=payload,
            )
            result = json.loads(response["Body"].read())
            return {"status": "success", "predictions": result.get("predictions", [])}
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}


if __name__ == "__main__":
    DeploymentFlow()
