"""
LocalStack Bootstrap Script — Idempotent AWS Resource Provisioning.

Creates and verifies all AWS resources needed by the Cooling Digital Twin
against a running LocalStack instance (default: http://localhost:4566).

Resources created:
  S3:            cooling-twin-telemetry, cooling-twin-models, cooling-twin-validation
  SNS:           cooling-twin-alerts
  EventBridge:   cooling-digital-twin (custom event bus)
  Step Functions: CoolingTwinRetrainingStateMachine  (production definition, ARNs substituted)
                  CoolingTwinRetrainingTestStateMachine  (Pass-only Community test definition)

Usage:
  python scripts/bootstrap_localstack.py [--endpoint http://localhost:4566]

Environment variables (all optional, default to LocalStack dev creds):
  AWS_ENDPOINT_URL      — LocalStack URL (default: http://localhost:4566)
  AWS_ACCESS_KEY_ID     — fake key (default: test)
  AWS_SECRET_ACCESS_KEY — fake secret (default: test)
  AWS_DEFAULT_REGION    — region (default: us-east-1)

Note on LocalStack Community vs Pro:
  - S3, SNS, SQS, EventBridge, Step Functions: Community (free).
  - Timestream, IoT SiteWise, TwinMaker: Pro / Enterprise only.
    Timestream creation is attempted and gracefully skipped if unavailable.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bootstrap_localstack")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
DEFAULT_REGION   = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ACCESS_KEY       = os.environ.get("AWS_ACCESS_KEY_ID", "test")
SECRET_KEY       = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")

S3_BUCKETS = [
    "cooling-twin-telemetry",
    "cooling-twin-models",
    "cooling-twin-validation",
]

SNS_TOPIC_NAME   = "cooling-twin-alerts"
EVENT_BUS_NAME   = "cooling-digital-twin"
STATE_MACHINE_NAME      = "CoolingTwinRetrainingStateMachine"
TEST_STATE_MACHINE_NAME = "CoolingTwinRetrainingTestStateMachine"
TIMESTREAM_DB    = os.environ.get("TIMESTREAM_DB", "CoolingTelemetry")
TIMESTREAM_TABLE = os.environ.get("TIMESTREAM_TABLE", "TelemetryMetrics")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_WORKFLOW = PROJECT_ROOT / "src" / "aws" / "orchestration" / "step_functions_workflow.json"
TEST_WORKFLOW       = PROJECT_ROOT / "src" / "aws" / "orchestration" / "step_functions_localstack_test_workflow.json"


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _client(service: str, endpoint: str) -> boto3.client:
    return boto3.client(
        service,
        endpoint_url=endpoint,
        region_name=DEFAULT_REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )


def _check_localstack(endpoint: str) -> bool:
    """Return True if LocalStack is reachable at endpoint."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{endpoint}/_localstack/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Resource provisioning helpers
# ---------------------------------------------------------------------------

def provision_s3(endpoint: str) -> None:
    s3 = _client("s3", endpoint)
    for bucket in S3_BUCKETS:
        try:
            s3.create_bucket(Bucket=bucket)
            logger.info("[S3]  Created bucket: %s", bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou"):
                logger.info("[S3]  Bucket already exists (skipped): %s", bucket)
            else:
                logger.warning("[S3]  Failed to create %s: %s", bucket, e)


def provision_sns(endpoint: str) -> str:
    """Create SNS topic and return its ARN."""
    sns = _client("sns", endpoint)
    try:
        resp = sns.create_topic(Name=SNS_TOPIC_NAME)
        arn = resp["TopicArn"]
        logger.info("[SNS] Topic ARN: %s", arn)
        return arn
    except ClientError as e:
        logger.warning("[SNS] Failed to create topic: %s", e)
        return f"arn:aws:sns:{DEFAULT_REGION}:000000000000:{SNS_TOPIC_NAME}"


def provision_eventbridge(endpoint: str) -> str:
    """Create custom EventBridge event bus and return its ARN."""
    events = _client("events", endpoint)
    try:
        resp = events.create_event_bus(Name=EVENT_BUS_NAME)
        arn = resp["EventBusArn"]
        logger.info("[EventBridge] Bus ARN: %s", arn)
        return arn
    except ClientError as e:
        if "already exists" in str(e).lower():
            logger.info("[EventBridge] Bus already exists (skipped): %s", EVENT_BUS_NAME)
            return f"arn:aws:events:{DEFAULT_REGION}:000000000000:event-bus/{EVENT_BUS_NAME}"
        logger.warning("[EventBridge] Failed to create bus: %s", e)
        return ""


def _build_production_definition(sns_arn: str) -> str:
    """
    Substitute ${...} CloudFormation-style tokens in the production workflow
    with valid-format LocalStack ARN placeholders.

    The substituted definition is deployed as-is so the production architecture
    is accurately represented. LocalStack Community will start execution and
    proceed through Choice states, then fail at the first SageMaker integration
    state (which is expected and documented in docs/evidence/step_functions_run.md).
    """
    region = DEFAULT_REGION
    account = "000000000000"

    replacements = {
        "${DriftDetectionLambdaArn}":   f"arn:aws:lambda:{region}:{account}:function:DriftDetectionLambda",
        "${ModelRegistrationLambdaArn}":f"arn:aws:lambda:{region}:{account}:function:ModelRegistrationLambda",
        "${ValidationGateLambdaArn}":   f"arn:aws:lambda:{region}:{account}:function:ValidationGateLambda",
        "${SageMakerTrainingImage}":     f"{account}.dkr.ecr.{region}.amazonaws.com/cooling-twin:latest",
        "${SageMakerRoleArn}":          f"arn:aws:iam::{account}:role/SageMakerExecutionRole",
        "${TrainingDataS3Uri}":          "s3://cooling-twin-telemetry/training/",
        "${ModelOutputS3Uri}":           "s3://cooling-twin-models/output/",
        "${ValidationReportS3Uri}":      "s3://cooling-twin-validation/reports/",
        "${SageMakerEndpointName}":      "cooling-twin-endpoint",
        "${AlertTopicArn}":              sns_arn,
    }
    with open(PRODUCTION_WORKFLOW, encoding="utf-8") as f:
        definition = f.read()
    for token, value in replacements.items():
        definition = definition.replace(token, value)
    return definition


def provision_state_machines(endpoint: str, sns_arn: str) -> dict:
    """Create state machines; return dict of name -> ARN.

    Production SM: always attempted (definition substituted inline).
    Test SM:        only created when step_functions_localstack_test_workflow.json
                   is already present on disk.  If the file is missing (Commit 4
                   has not yet landed) the test SM is silently skipped.
    """
    sfn = _client("stepfunctions", endpoint)
    region = DEFAULT_REGION
    account = "000000000000"
    role_arn = f"arn:aws:iam::{account}:role/StepFunctionsRole"
    results = {}

    # Collect (name, definition_string) pairs to create
    definitions_to_create = [(STATE_MACHINE_NAME, _build_production_definition(sns_arn))]

    if TEST_WORKFLOW.exists():
        definitions_to_create.append((TEST_STATE_MACHINE_NAME, TEST_WORKFLOW.read_text(encoding="utf-8-sig")))
    else:
        logger.info(
            "[StepFunctions] Test workflow not found at %s — "
            "test SM will be created in Commit 4 (skipping now).",
            TEST_WORKFLOW,
        )

    for name, defn in definitions_to_create:
        try:
            resp = sfn.create_state_machine(
                name=name,
                definition=defn,
                roleArn=role_arn,
                type="STANDARD",
            )
            arn = resp["stateMachineArn"]
            logger.info("[StepFunctions] Created %s: %s", name, arn)
            results[name] = arn
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "StateMachineAlreadyExists" or "already exists" in str(e).lower():
                existing = sfn.list_state_machines()
                for sm in existing.get("stateMachines", []):
                    if sm["name"] == name:
                        logger.info(
                            "[StepFunctions] Already exists (skipped) %s: %s",
                            name, sm["stateMachineArn"],
                        )
                        results[name] = sm["stateMachineArn"]
                        break
            else:
                logger.warning("[StepFunctions] Failed to create %s: %s", name, e)
    return results


def provision_timestream(endpoint: str) -> bool:
    """
    Attempt to create Timestream DB and table.
    Gracefully skips if LocalStack Community doesn't support Timestream.
    Returns True if succeeded.
    """
    try:
        ts = _client("timestream-write", endpoint)
        try:
            ts.create_database(DatabaseName=TIMESTREAM_DB)
            logger.info("[Timestream] Created database: %s", TIMESTREAM_DB)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConflictException":
                logger.info("[Timestream] Database already exists: %s", TIMESTREAM_DB)
            else:
                raise
        try:
            ts.create_table(
                DatabaseName=TIMESTREAM_DB,
                TableName=TIMESTREAM_TABLE,
                RetentionProperties={
                    "MemoryStoreRetentionPeriodInHours": 24,
                    "MagneticStoreRetentionPeriodInDays": 7,
                },
            )
            logger.info("[Timestream] Created table: %s.%s", TIMESTREAM_DB, TIMESTREAM_TABLE)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConflictException":
                logger.info("[Timestream] Table already exists: %s.%s", TIMESTREAM_DB, TIMESTREAM_TABLE)
            else:
                raise
        return True
    except Exception as e:
        logger.warning(
            "[Timestream] SKIPPED \u2014 not available on LocalStack Community: %s. "
            "The backend will use the in-memory fallback store instead.", e
        )
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(endpoint: str) -> None:
    logger.info("=== LocalStack Bootstrap ===")
    logger.info("Endpoint : %s", endpoint)
    logger.info("Region   : %s", DEFAULT_REGION)

    if not _check_localstack(endpoint):
        logger.error(
            "LocalStack is not reachable at %s.\n"
            "Start it first:  docker compose -f deployment/docker/docker-compose.yml up -d localstack",
            endpoint,
        )
        sys.exit(1)

    provision_s3(endpoint)
    sns_arn = provision_sns(endpoint)
    provision_eventbridge(endpoint)

    if not TEST_WORKFLOW.exists():
        logger.info(
            "[StepFunctions] Test workflow JSON not yet on disk; "
            "run bootstrap again after Commit 4 to create the test SM."
        )
    sm_arns = provision_state_machines(endpoint, sns_arn)
    for name, arn in sm_arns.items():
        logger.info("  %s -> %s", name, arn)

    ts_ok = provision_timestream(endpoint)

    logger.info("=== Bootstrap Summary ===")
    logger.info("S3 buckets   : %d created/verified", len(S3_BUCKETS))
    logger.info("SNS topic    : %s", SNS_TOPIC_NAME)
    logger.info("EventBridge  : %s", EVENT_BUS_NAME)
    logger.info("Timestream   : %s", "OK" if ts_ok else "SKIPPED (Community edition)")
    logger.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap LocalStack resources for Cooling Digital Twin")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="LocalStack endpoint URL")
    args = parser.parse_args()
    main(args.endpoint)
