#!/bin/bash

MODELS_BUCKET="${MODELS_BUCKET:-battery-pdm-dev-models-998716768706}"
DATA_BUCKET="${DATA_BUCKET:-battery-pdm-dev-data-998716768706}"
ALERTS_BUCKET="${ALERTS_BUCKET:-battery-pdm-dev-alerts-998716768706}"
LOCAL="/app/outputs"

echo "=== Downloading data from S3 ==="
mkdir -p $LOCAL/models $LOCAL/drain_alerts $LOCAL/failure_alerts $LOCAL/drift_reports

echo "Downloading alarms.parquet..."
aws s3 cp "s3://$DATA_BUCKET/alarms.parquet" "$LOCAL/alarms.parquet"
echo "Downloading site_static.parquet..."
aws s3 cp "s3://$DATA_BUCKET/site_static.parquet" "$LOCAL/site_static.parquet"
echo "Downloading labels.parquet..."
aws s3 cp "s3://$DATA_BUCKET/labels.parquet" "$LOCAL/labels.parquet" || echo "WARN: labels.parquet not found"
echo "Downloading load_shedding_schedule.parquet..."
aws s3 cp "s3://$DATA_BUCKET/load_shedding_schedule.parquet" "$LOCAL/load_shedding_schedule.parquet" || echo "WARN: schedule not found"

echo "Syncing models..."
aws s3 sync "s3://$MODELS_BUCKET/" "$LOCAL/models/" || echo "WARN: models sync failed"
echo "Syncing alerts..."
aws s3 sync "s3://$ALERTS_BUCKET/drain_alerts/" "$LOCAL/drain_alerts/" || echo "WARN: drain_alerts sync failed"
aws s3 sync "s3://$ALERTS_BUCKET/failure_alerts/" "$LOCAL/failure_alerts/" || echo "WARN: failure_alerts sync failed"

echo "=== Downloaded files ==="
ls -lh $LOCAL/*.parquet
echo "=== Models ==="
ls $LOCAL/models/

echo "=== Running flow ==="
"$@"
EXIT_CODE=$?

echo "=== Syncing results back to S3 ==="
aws s3 sync "$LOCAL/models/" "s3://$MODELS_BUCKET/" || echo "WARN: models upload failed"
aws s3 sync "$LOCAL/drain_alerts/" "s3://$ALERTS_BUCKET/drain_alerts/" || echo "WARN: drain_alerts upload failed"
aws s3 sync "$LOCAL/failure_alerts/" "s3://$ALERTS_BUCKET/failure_alerts/" || echo "WARN: failure_alerts upload failed"
aws s3 sync "$LOCAL/drift_reports/" "s3://$ALERTS_BUCKET/drift_reports/" || echo "WARN: drift_reports upload failed"

echo "=== Done (exit code: $EXIT_CODE) ==="
exit $EXIT_CODE
