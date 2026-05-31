FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps (dashboard + flows)
COPY dashboard/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    streamlit plotly pandas numpy scikit-learn pyarrow matplotlib metaflow boto3 pandera && \
    pip install --no-cache-dir --timeout=300 --retries=3 --no-deps xgboost && \
    pip install --no-cache-dir --timeout=300 --retries=3 scipy

# Copy source code
COPY src/ /app/src/
COPY dashboard/ /app/dashboard/

# Set Python path so battery_pdm is importable
ENV PYTHONPATH=/app/src
ENV USERNAME=batch-runner
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Data is mounted from S3 sync or EFS
# In production: sidecar container syncs S3 -> /app/data every 5 min
VOLUME /app/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "/app/dashboard/app.py", \
    "--server.port=8501", "--server.address=0.0.0.0"]
