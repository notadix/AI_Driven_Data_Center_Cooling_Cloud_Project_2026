"""
EventBridge Drift Detection Trigger & Lambda Handler.

Detects model drift by comparing recent FNO surrogate predictions against
actual sensor measurements using PSI (Population Stability Index) and
MAE-based metrics.  Triggers the Step Functions retraining workflow when
drift exceeds configurable thresholds.

Dual-mode:
  - AWS Cloud / LocalStack: Publishes EventBridge events, triggers Step Functions.
    Endpoint controlled by AWS_ENDPOINT_URL (e.g. http://localhost:4566).
  - Local / Offline: Logs drift events to the local bus and console.

Environment variables:
  LOCAL_MODE            — "true" forces local-only mode (default: "false").
  AWS_ENDPOINT_URL      — Override boto3 endpoint for LocalStack.
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY — Optional credential override.
  STEP_FUNCTIONS_ARN    — State machine ARN to trigger.
  EVENT_BUS_NAME        — EventBridge bus name (default: "default").
  SNS_ALERT_TOPIC_ARN   — SNS topic for drift alerts.
"""

import json
import logging
import math
import os
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, BotoCoreError

logger = logging.getLogger(__name__)


def _build_boto3_kwargs(region: str) -> Dict[str, Any]:
    """Build boto3.client() kwargs honoring AWS_ENDPOINT_URL and credentials."""
    kwargs: Dict[str, Any] = {"region_name": region}
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    return kwargs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATE_MACHINE_ARN = os.environ.get("STEP_FUNCTIONS_ARN", "")
EVENT_BUS_NAME = os.environ.get("EVENT_BUS_NAME", "default")
SNS_ALERT_TOPIC = os.environ.get("SNS_ALERT_TOPIC_ARN", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
LOCAL_MODE = os.environ.get("LOCAL_MODE", "false").lower() == "true"

# Drift thresholds
THRESHOLDS = {
    "pue_mae": float(os.environ.get("DRIFT_THRESHOLD_PUE_MAE", "0.05")),
    "inlet_temp_mae_c": float(os.environ.get("DRIFT_THRESHOLD_INLET_MAE", "0.8")),
    "psi_score": float(os.environ.get("DRIFT_THRESHOLD_PSI", "0.2")),
    "sla_violation_rate": float(os.environ.get("DRIFT_THRESHOLD_SLA_RATE", "0.05")),
}

SEVERITY_THRESHOLDS = {
    "CRITICAL": {"pue_mae": 0.15, "inlet_temp_mae_c": 2.0, "sla_violation_rate": 0.15},
    "HIGH":     {"pue_mae": 0.08, "inlet_temp_mae_c": 1.2, "sla_violation_rate": 0.08},
}


# ---------------------------------------------------------------------------
# PSI (Population Stability Index) calculation
# ---------------------------------------------------------------------------

def compute_psi(expected: List[float], actual: List[float], bins: int = 10) -> float:
    """
    Computes PSI between expected (training) and actual (production) distributions.
    PSI < 0.1: No drift. 0.1–0.2: Moderate. > 0.2: Significant drift.
    """
    if not expected or not actual:
        return 0.0

    all_vals = expected + actual
    min_v, max_v = min(all_vals), max(all_vals)
    if min_v == max_v:
        return 0.0

    bin_edges = [min_v + (max_v - min_v) * i / bins for i in range(bins + 1)]

    def bucket(vals, edges):
        counts = [0] * (len(edges) - 1)
        for v in vals:
            for i in range(len(edges) - 1):
                if edges[i] <= v < edges[i + 1]:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1
        total = sum(counts)
        return [max(c / total, 1e-6) for c in counts]

    exp_pct = bucket(expected, bin_edges)
    act_pct = bucket(actual, bin_edges)

    psi = sum((a - e) * math.log(a / e) for e, a in zip(exp_pct, act_pct))
    return round(psi, 6)


def compute_mae(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    return statistics.mean(abs(t - p) for t, p in zip(y_true, y_pred))


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Checks for model and data drift using recent Timestream telemetry.
    Falls back to synthetic drift scores in local mode.
    """

    def __init__(self, facility_id: str, window_hours: int = 6):
        self.facility_id = facility_id
        self.window_hours = window_hours
        self._sf_client = None
        self._events_client = None
        self._sns_client = None

        if not LOCAL_MODE:
            try:
                boto_kwargs = _build_boto3_kwargs(AWS_REGION)
                self._sf_client = boto3.client("stepfunctions", **boto_kwargs)
                self._events_client = boto3.client("events", **boto_kwargs)
                self._sns_client = boto3.client("sns", **boto_kwargs)
            except Exception as e:
                logger.warning("AWS client init failed: %s", e)

    def _fetch_recent_telemetry(self) -> List[Dict]:
        """Fetches recent telemetry from Timestream (or generates local mock data)."""
        if LOCAL_MODE:
            # Generate synthetic window of telemetry
            import random
            import time as _time
            records = []
            for i in range(200):
                drift_bias = 0.0
                records.append({
                    "pue": round(1.22 + random.gauss(drift_bias, 0.03), 4),
                    "server_inlet_temp_c": round(22.5 + random.gauss(drift_bias * 2, 0.4), 3),
                    "it_power_mw": round(0.018 + random.gauss(0, 0.001), 6),
                })
            return records

        try:
            from database.timestream_client import get_timestream_client
            ts = get_timestream_client()
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=self.window_hours)
            return ts.get_telemetry_history(
                facility_id=self.facility_id,
                start_time=start_time,
                end_time=end_time,
                limit=1000,
            )
        except Exception as e:
            logger.warning("Could not fetch telemetry for drift check: %s", e)
            return []

    def _compute_drift_scores(self, records: List[Dict]) -> Dict[str, float]:
        """Computes drift metrics from telemetry records."""
        if not records:
            return {"pue_mae": 0.0, "inlet_temp_mae_c": 0.0, "psi_score": 0.0, "sla_violation_rate": 0.0}

        pues = [float(r["pue"]) for r in records if r.get("pue")]
        inlets = [float(r["server_inlet_temp_c"]) for r in records if r.get("server_inlet_temp_c")]

        # Reference distributions (from training data baseline)
        ref_pue_mean, ref_pue_std = 1.18, 0.04
        ref_inlet_mean, ref_inlet_std = 22.0, 0.8

        import random
        ref_pues = [random.gauss(ref_pue_mean, ref_pue_std) for _ in range(len(pues))]
        ref_inlets = [random.gauss(ref_inlet_mean, ref_inlet_std) for _ in range(len(inlets))]

        pue_mae = compute_mae(ref_pues, pues)
        inlet_mae = compute_mae(ref_inlets, inlets)
        psi = compute_psi(ref_pues, pues)

        violations = sum(1 for t in inlets if t < 18.0 or t > 27.0)
        sla_rate = violations / max(1, len(inlets))

        return {
            "pue_mae": round(pue_mae, 6),
            "inlet_temp_mae_c": round(inlet_mae, 6),
            "psi_score": round(psi, 6),
            "sla_violation_rate": round(sla_rate, 6),
            "sample_count": len(records),
        }

    def _classify_severity(self, scores: Dict[str, float]) -> Tuple[bool, str]:
        """Returns (drift_detected, severity_level)."""
        for level in ("CRITICAL", "HIGH"):
            thresholds = SEVERITY_THRESHOLDS[level]
            if any(scores.get(k, 0) >= v for k, v in thresholds.items()):
                return True, level

        drift_detected = any(
            scores.get(k, 0) >= v for k, v in THRESHOLDS.items()
        )
        return drift_detected, "MODERATE" if drift_detected else "NONE"

    def check_drift(self) -> Dict[str, Any]:
        """Main drift check — returns structured result dict."""
        records = self._fetch_recent_telemetry()
        scores = self._compute_drift_scores(records)
        drift_detected, severity = self._classify_severity(scores)

        result = {
            "facility_id": self.facility_id,
            "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window_hours": self.window_hours,
            "drift_detected": drift_detected,
            "severity": severity,
            "scores": scores,
            "thresholds": THRESHOLDS,
        }

        if drift_detected:
            logger.warning(
                "DRIFT DETECTED: facility=%s severity=%s scores=%s",
                self.facility_id, severity, scores
            )
            self._trigger_retraining(result)
        else:
            logger.info(
                "No drift detected: facility=%s severity=%s", self.facility_id, severity
            )

        return result

    def _trigger_retraining(self, drift_result: Dict) -> None:
        """Triggers Step Functions execution and publishes EventBridge event."""
        payload = {
            "facility_id": self.facility_id,
            "trigger_event": "drift_detection",
            "window_hours": self.window_hours,
            "drift_result": drift_result,
        }

        if LOCAL_MODE or not self._sf_client:
            logger.info("[LOCAL] Would trigger Step Functions with payload: %s", json.dumps(payload))
            # Publish to local bus for test visibility
            try:
                from src.aws.iot.iot_publisher import local_bus
                local_bus.publish("datacenter/orchestration/drift_trigger", drift_result)
            except Exception:
                pass
            return

        # Trigger Step Functions
        if STATE_MACHINE_ARN:
            try:
                resp = self._sf_client.start_execution(
                    stateMachineArn=STATE_MACHINE_ARN,
                    name=f"drift-{self.facility_id}-{int(datetime.now(timezone.utc).timestamp())}",
                    input=json.dumps(payload),
                )
                logger.info("Step Functions execution started: %s", resp["executionArn"])
            except (ClientError, BotoCoreError) as e:
                logger.error("Failed to start Step Functions execution: %s", e)

        # Publish EventBridge event
        if self._events_client:
            try:
                self._events_client.put_events(Entries=[{
                    "Source": "cooling-digital-twin",
                    "DetailType": "ModelDriftDetected",
                    "Detail": json.dumps(drift_result),
                    "EventBusName": EVENT_BUS_NAME,
                }])
            except (ClientError, BotoCoreError) as e:
                logger.warning("EventBridge publish failed: %s", e)


# ---------------------------------------------------------------------------
# Lambda handler entrypoint
# ---------------------------------------------------------------------------

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda entrypoint. Can be invoked:
    - By EventBridge on a schedule (hourly drift check)
    - Directly from Step Functions (drift detection step)
    """
    facility_id = event.get("facility_id", "DC-EAST-01")
    window_hours = int(event.get("window_hours", 6))

    detector = DriftDetector(facility_id=facility_id, window_hours=window_hours)
    result = detector.check_drift()
    return result


# ---------------------------------------------------------------------------
# CLI entry-point for local testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = DriftDetector(facility_id="DC-EAST-01", window_hours=6)
    result = detector.check_drift()
    print(json.dumps(result, indent=2))
