"""Register Metaflow flows as AWS Step Functions state machines.

Equivalent to `METAFLOW_PROFILE=production python -m flow step-functions create`
but runs on Windows without fcntl dependency.

Each flow becomes a single-state state machine that submits a Batch job.
This is the proactive retraining pattern: flows run as Batch jobs orchestrated by Step Functions.
"""

import json
import subprocess
import sys

REGION = "ap-south-1"
ACCOUNT = "998716768706"
SFN_ROLE = f"arn:aws:iam::{ACCOUNT}:role/battery-pdm-dev-sfn-role"
JOB_QUEUE = f"arn:aws:batch:{REGION}:{ACCOUNT}:job-queue/battery-pdm-dev-queue"

FLOWS = {
    "drain-predictor": {
        "job_def": f"arn:aws:batch:{REGION}:{ACCOUNT}:job-definition/battery-pdm-dev-drain_predictor",
        "comment": "Daily 00:30 — score all sites for 48h drain risk",
    },
    "drift-monitor": {
        "job_def": f"arn:aws:batch:{REGION}:{ACCOUNT}:job-definition/battery-pdm-dev-drift_monitor",
        "comment": "Daily 01:00 — detect feature/prediction drift",
    },
    "failure-scoring": {
        "job_def": f"arn:aws:batch:{REGION}:{ACCOUNT}:job-definition/battery-pdm-dev-failure_scoring",
        "comment": "Weekly Sunday 02:00 — score failure risk (survival model)",
    },
    "retraining": {
        "job_def": f"arn:aws:batch:{REGION}:{ACCOUNT}:job-definition/battery-pdm-dev-retraining",
        "comment": "Weekly Saturday 03:00 — champion/challenger CV-gated retraining",
    },
    "shadow-promotion": {
        "job_def": f"arn:aws:batch:{REGION}:{ACCOUNT}:job-definition/battery-pdm-dev-shadow_promotion",
        "comment": "Daily 01:30 — validate shadow model on realized labels",
    },
    "rollback-monitor": {
        "job_def": f"arn:aws:batch:{REGION}:{ACCOUNT}:job-definition/battery-pdm-dev-rollback_monitor",
        "comment": "Daily 02:00 - auto-revert if promoted model degrades",
    },
    "retraining-failure": {
        "job_def": f"arn:aws:batch:{REGION}:{ACCOUNT}:job-definition/battery-pdm-dev-retraining_failure",
        "comment": "Weekly Saturday 03:30 - failure model retraining (shadow mode)",
    },
}


def build_state_machine(flow_name: str, flow_config: dict) -> dict:
    return {
        "Comment": flow_config["comment"],
        "StartAt": "SubmitBatchJob",
        "States": {
            "SubmitBatchJob": {
                "Type": "Task",
                "Resource": "arn:aws:states:::batch:submitJob.sync",
                "Parameters": {
                    "JobName.$": f"States.Format('battery-pdm-{flow_name}-{{}}', $$.Execution.Name)",
                    "JobDefinition": flow_config["job_def"],
                    "JobQueue": JOB_QUEUE,
                    "ContainerOverrides": {
                        "Environment": [
                            {"Name": "METAFLOW_DEFAULT_DATASTORE", "Value": "s3"},
                            {"Name": "METAFLOW_DATASTORE_SYSROOT_S3", "Value": f"s3://battery-pdm-dev-data-{ACCOUNT}/metaflow"},
                            {"Name": "METAFLOW_DEFAULT_METADATA", "Value": "local"},
                            {"Name": "METAFLOW_CARD_S3ROOT", "Value": f"s3://battery-pdm-dev-data-{ACCOUNT}/metaflow/cards"},
                            {"Name": "METAFLOW_USER", "Value": "batch-runner"},
                        ]
                    },
                },
                "Retry": [
                    {
                        "ErrorEquals": ["Batch.ServerException", "States.TaskFailed"],
                        "IntervalSeconds": 60,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0,
                    }
                ],
                "Catch": [
                    {
                        "ErrorEquals": ["States.ALL"],
                        "Next": "JobFailed",
                    }
                ],
                "Next": "JobSucceeded",
            },
            "JobSucceeded": {
                "Type": "Succeed",
            },
            "JobFailed": {
                "Type": "Fail",
                "Error": "BatchJobFailed",
                "Cause": f"{flow_name} Batch job failed - check CloudWatch logs at /aws/batch/battery-pdm-dev",
            },
        },
    }


def main():
    created = []
    updated = []
    errors = []

    for flow_name, flow_config in FLOWS.items():
        sfn_name = f"battery-pdm-{flow_name}"
        definition = json.dumps(build_state_machine(flow_name, flow_config))

        check = subprocess.run(
            ["aws", "stepfunctions", "describe-state-machine",
             "--state-machine-arn", f"arn:aws:states:{REGION}:{ACCOUNT}:stateMachine:{sfn_name}",
             "--region", REGION],
            capture_output=True, text=True,
        )

        if check.returncode == 0:
            result = subprocess.run(
                ["aws", "stepfunctions", "update-state-machine",
                 "--state-machine-arn", f"arn:aws:states:{REGION}:{ACCOUNT}:stateMachine:{sfn_name}",
                 "--definition", definition,
                 "--role-arn", SFN_ROLE,
                 "--region", REGION],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                updated.append(sfn_name)
                print(f"  UPDATED  {sfn_name}")
            else:
                errors.append((sfn_name, result.stderr))
                print(f"  ERROR    {sfn_name}: {result.stderr.strip()}")
        else:
            result = subprocess.run(
                ["aws", "stepfunctions", "create-state-machine",
                 "--name", sfn_name,
                 "--definition", definition,
                 "--role-arn", SFN_ROLE,
                 "--region", REGION,
                 "--type", "STANDARD"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                arn = json.loads(result.stdout).get("stateMachineArn", "")
                created.append(sfn_name)
                print(f"  CREATED  {sfn_name} -> {arn}")
            else:
                errors.append((sfn_name, result.stderr))
                print(f"  ERROR    {sfn_name}: {result.stderr.strip()}")

    print(f"\nDone: {len(created)} created, {len(updated)} updated, {len(errors)} errors")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
