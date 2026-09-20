"""
Run the Pass/Choice test workflow on LocalStack for every branch and record what happened.

The checked-in test workflow injects a fixed drift result, so a single run only exercises one
branch. This derives in-memory variants (drift severity x validation outcome) from the same
definition, so every Choice state is exercised. Nothing here touches real AWS.

    docker compose -f deployment/docker/docker-compose.yml up -d localstack
    python scripts/run_stepfunctions_evidence.py

Writes results/stepfunctions_localstack_run.json
"""
import copy
import json
import os
import sys
import time

import boto3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFINITION = os.path.join(PROJECT_ROOT, "src", "aws", "orchestration", "step_functions_localstack_test_workflow.json")
ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
ROLE = "arn:aws:iam::000000000000:role/StepFunctionsRole"


def client():
    return boto3.client("stepfunctions", endpoint_url=ENDPOINT, region_name="us-east-1",
                        aws_access_key_id="test", aws_secret_access_key="test")


def variant(base, severity, passed):
    d = copy.deepcopy(base)
    d["States"]["DetectModelDrift"]["Result"]["drift_severity"] = severity
    d["States"]["DetectModelDrift"]["Result"]["drift_detected"] = severity != "NONE"
    d["States"]["RunModelValidation"]["Result"]["passed"] = passed
    return d


def run(sfn, name, definition):
    for sm in sfn.list_state_machines()["stateMachines"]:
        if sm["name"] == name:
            sfn.delete_state_machine(stateMachineArn=sm["stateMachineArn"])
            time.sleep(0.5)
    arn = sfn.create_state_machine(name=name, definition=json.dumps(definition), roleArn=ROLE, type="STANDARD")["stateMachineArn"]
    ex = sfn.start_execution(stateMachineArn=arn, input=json.dumps({"facility_id": "DC-EAST-01"}))["executionArn"]
    for _ in range(60):
        desc = sfn.describe_execution(executionArn=ex)
        if desc["status"] != "RUNNING":
            break
        time.sleep(0.5)
    hist = sfn.get_execution_history(executionArn=ex, maxResults=200)["events"]
    path = [e["stateEnteredEventDetails"]["name"] for e in hist if e["type"].endswith("StateEntered")]
    return {"status": desc["status"], "path": path, "events": len(hist), "output": json.loads(desc.get("output", "null"))}


def main() -> None:
    with open(DEFINITION, encoding="utf-8") as f:
        base = json.load(f)
    sfn = client()
    expected = {
        ("CRITICAL", True): "EmergencyRetraining", ("MODERATE", True): "PrioritizedRetraining",
        ("NONE", True): "StandardRetraining",
    }
    results = []
    for severity in ("CRITICAL", "MODERATE", "NONE"):
        for passed in (True, False):
            r = run(sfn, f"EvidenceRun-{severity}-{'pass' if passed else 'fail'}", variant(base, severity, passed))
            r.update(drift_severity=severity, validation_passed=passed)
            results.append(r)
            print(f"{severity:9s} validation={'pass' if passed else 'fail'} -> {r['status']:9s} {' > '.join(r['path'])}", flush=True)

    ok = all(r["status"] == "SUCCEEDED" for r in results)
    for r in results:
        want = expected.get((r["drift_severity"], True), None) or {
            "CRITICAL": "EmergencyRetraining", "MODERATE": "PrioritizedRetraining", "NONE": "StandardRetraining"}[r["drift_severity"]]
        assert want in r["path"], (r["drift_severity"], r["path"])
        assert ("DeployModel" in r["path"]) == r["validation_passed"], r["path"]
        assert ("AlertValidationFailure" in r["path"]) == (not r["validation_passed"]), r["path"]

    out = {"endpoint": ENDPOINT, "state_machine_definition": os.path.relpath(DEFINITION, PROJECT_ROOT),
           "all_executions_succeeded": ok, "branches_verified": len(results), "runs": results}
    with open(os.path.join(PROJECT_ROOT, "results", "stepfunctions_localstack_run.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("all branches routed correctly" if ok else "SOME EXECUTIONS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
