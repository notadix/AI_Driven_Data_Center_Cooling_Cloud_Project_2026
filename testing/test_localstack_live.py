"""Live LocalStack checks (skipped automatically when LocalStack is not running)."""

import json
import os
import subprocess
import sys
import urllib.request

import pytest

LOCALSTACK_URL = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _up() -> bool:
    try:
        with urllib.request.urlopen(f"{LOCALSTACK_URL}/_localstack/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


live = pytest.mark.skipif(not _up(), reason="LocalStack not running - start with: docker compose up -d localstack")


@live
def test_every_workflow_branch_routes_correctly_on_localstack(tmp_path):
    """Runs scripts/run_stepfunctions_evidence.py: 3 drift severities x 2 validation outcomes."""
    env = dict(os.environ, AWS_ENDPOINT_URL=LOCALSTACK_URL, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_stepfunctions_evidence.py")],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with open(os.path.join(ROOT, "results", "stepfunctions_localstack_run.json")) as f:
        data = json.load(f)
    assert data["branches_verified"] == 6 and data["all_executions_succeeded"]


@live
def test_production_state_machine_is_rejected_only_because_sagemaker_is_not_emulated():
    """The production definition needs the SageMaker task integration, which Community does not emulate.
    It must fail for THAT reason (not because of a malformed definition)."""
    import boto3
    from botocore.exceptions import ClientError

    sfn = boto3.client("stepfunctions", endpoint_url=LOCALSTACK_URL, region_name="us-east-1",
                       aws_access_key_id="test", aws_secret_access_key="test")
    with open(os.path.join(ROOT, "src", "aws", "orchestration", "step_functions_workflow.json"), encoding="utf-8") as f:
        raw = f.read()
    import re
    # the bootstrap script substitutes ${...} tokens with ARNs; any well-formed ARN is enough for this probe
    definition = re.sub(r"\$\{[^}]+\}", "arn:aws:lambda:us-east-1:000000000000:function:probe", raw)
    try:
        sfn.create_state_machine(name="ProdDefinitionProbe", definition=definition,
                                 roleArn="arn:aws:iam::000000000000:role/r", type="STANDARD")
    except ClientError as e:
        assert "sagemaker" in str(e).lower() or "InvalidDefinition" in str(e)
        assert "Unsupported service" in str(e) or "sagemaker" in str(e).lower()
    else:
        pytest.skip("this LocalStack edition accepts the SageMaker integration; nothing to assert")


@live
def test_timestream_outage_degrades_to_memory_on_community():
    """Timestream is Pro-only on LocalStack Community: the client must trip its breaker, keep the data
    and serve reads instead of returning nothing."""
    os.environ["AWS_ENDPOINT_URL"] = LOCALSTACK_URL
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    from database.timestream_client import TimestreamClient

    c = TimestreamClient(local_mode=False)
    for i in range(5):
        c.write_telemetry({"facility_id": "DC-EAST-01", "crac_id": "CRAC-01", "rack_id": "R", "pue": 1.05,
                           "server_inlet_temp_c": 22.0})
    health = json.loads(urllib.request.urlopen(f"{LOCALSTACK_URL}/_localstack/health").read())
    if health["services"].get("timestream-write") in ("available", "running"):
        pytest.skip("Timestream is available on this LocalStack edition")
    assert c._cloud_failures == 1                       # tripped once, not once per record
    assert len(c.get_telemetry_history("DC-EAST-01")) == 5
