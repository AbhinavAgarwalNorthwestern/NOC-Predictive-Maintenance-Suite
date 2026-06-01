#!/bin/bash
# Backfill missing scoring runs after an AWS regional outage.
#
# Scenario: ap-south-1 was down from <START> to <END>. EventBridge couldn't
# trigger Step Functions, so drain_alerts/ and drift_reports/ have gaps.
# Once region is back, this script replays missed scoring runs.
#
# All flows are idempotent (same input -> same output), so re-running is safe.
#
# Usage:
#   bash scripts/backfill_after_outage.sh 2026-06-01 2026-06-03
#   (backfills daily scoring for the date range, inclusive)

set -e

START_DATE="${1:?Usage: $0 YYYY-MM-DD YYYY-MM-DD}"
END_DATE="${2:?Usage: $0 YYYY-MM-DD YYYY-MM-DD}"

REGION="ap-south-1"
ACCOUNT="998716768706"

echo "Backfilling drain_predictor + drift_monitor + failure_scoring from $START_DATE to $END_DATE"

current="$START_DATE"
while [[ "$current" < "$END_DATE" || "$current" == "$END_DATE" ]]; do
    echo
    echo "=== Backfill for $current ==="

    for flow in drain-predictor drift-monitor failure-scoring; do
        echo "  Triggering $flow..."
        aws stepfunctions start-execution \
            --state-machine-arn "arn:aws:states:$REGION:$ACCOUNT:stateMachine:battery-pdm-$flow" \
            --name "backfill-$flow-$current" \
            --region "$REGION" \
            --output text \
            --query "executionArn"
    done

    # Advance to next day (GNU date on Linux, BSD date on macOS)
    current=$(date -I -d "$current + 1 day" 2>/dev/null || date -v +1d -j -f %Y-%m-%d "$current" +%Y-%m-%d)
done

echo
echo "Backfill triggered. Monitor with:"
echo "  aws stepfunctions list-executions --state-machine-arn arn:aws:states:$REGION:$ACCOUNT:stateMachine:battery-pdm-drain-predictor --region $REGION --max-results 10"
echo
echo "Each execution takes ~5 min. Total backfill time ~ N_DAYS x 5 min."
