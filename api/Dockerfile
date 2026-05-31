FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    fastapi uvicorn pandas numpy scikit-learn pyarrow boto3 && \
    pip install --no-cache-dir --timeout=300 --retries=3 --no-deps xgboost && \
    pip install --no-cache-dir --timeout=300 --retries=3 scipy

COPY src/ /app/src/
COPY api/ /app/api/

ENV PYTHONPATH=/app/src
ENV USERNAME=api-server

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
