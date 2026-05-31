# Battery PdM image — runs Metaflow flows + Python scoring scripts.
# Single image for all flows; the CMD picks which flow to run at container start.
# Designed for AWS Batch / Fargate execution and local docker-compose dev.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy lock and project metadata first for cache reuse
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project || uv sync --no-install-project

# Copy the project source
COPY src /app/src
COPY scripts /app/scripts
COPY tests /app/tests

# Install the project itself
RUN uv pip install --no-deps -e .

# AWS-specific deps (boto3, CloudWatch logging handler)
RUN uv pip install boto3==1.35.* watchtower==3.3.* awscli

# Default writable workdir
RUN mkdir -p /app/outputs

# Entrypoint syncs S3 data to local before flow, syncs results back after
RUN chmod +x /app/scripts/batch_entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import battery_pdm" || exit 1

ENTRYPOINT ["/app/scripts/batch_entrypoint.sh"]
CMD ["python", "-m", "battery_pdm.flows.drain_predictor_flow", "run"]
