"""
Amazon Timestream client for cooling telemetry ingestion and analytical queries.

Dual-mode:
  - AWS Cloud / LocalStack: Uses boto3 timestream-write + timestream-query clients.
    Endpoint is controlled via AWS_ENDPOINT_URL (e.g. http://localhost:4566).
  - Local / Offline: Falls back to an in-memory ring buffer with identical query API.

Environment variables:
  LOCAL_MODE            — "true" to force in-memory mode (default: "false")
  AWS_ENDPOINT_URL      — Override boto3 endpoint (e.g. LocalStack). Unset = real AWS.
  AWS_ACCESS_KEY_ID     — AWS / LocalStack access key (default: unset)
  AWS_SECRET_ACCESS_KEY — AWS / LocalStack secret key (default: unset)
  AWS_REGION            — AWS region (default: us-east-1)
"""

import logging
import os
import re
import time
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, BotoCoreError

logger = logging.getLogger(__name__)


def _build_boto3_kwargs(region: str) -> Dict[str, Any]:
    """Build keyword arguments for boto3.client() that honour AWS_ENDPOINT_URL
    and explicit credentials without breaking LOCAL_MODE=true paths.
    When AWS_ENDPOINT_URL is unset the returned dict contains only region_name
    so boto3 behaves exactly as before."""
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

# facility_id/crac_id are interpolated directly into the query strings below
# (f-string text, not a parameterized Timestream query) — reject anything
# outside a safe identifier charset before it ever reaches SQL, regardless
# of whether the caller already validated it (callers include a FastAPI
# router with its own pattern check, but also a Lambda handler that reads
# facility_id straight off an arbitrary event payload).
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _is_safe_identifier(value: Optional[str]) -> bool:
    return value is not None and bool(_SAFE_ID_RE.match(value))


# ---------------------------------------------------------------------------
# In-memory fallback store (used in local / offline mode)
# ---------------------------------------------------------------------------

class InMemoryTimestreamStore:
    """Thread-safe ring buffer that mirrors Timestream's multi-measure record API."""

    def __init__(self, maxlen: int = 50_000):
        self._records: deque = deque(maxlen=maxlen)
        # The simulator thread appends while API handlers read. Iterating a deque that another thread
        # is appending to raises "deque mutated during iteration" (seen as an intermittent HTTP 500),
        # so writes take this lock and readers work from a snapshot taken under it.
        self._lock = threading.Lock()

    def write(self, records: List[Dict]) -> None:
        with self._lock:
            for r in records:
                self._records.append(r)

    def snapshot(self) -> List[Dict]:
        """Consistent copy of the buffer, oldest first."""
        with self._lock:
            return list(self._records)

    def query_latest(
        self,
        measure_name: str,
        facility_id: Optional[str] = None,
        crac_id: Optional[str] = None,
        rack_id: Optional[str] = None,
    ) -> Optional[Dict]:
        for r in reversed(self.snapshot()):
            if r.get("measure_name") != measure_name:
                continue
            dims = r.get("dimensions", {})
            if facility_id and dims.get("facility_id") != facility_id:
                continue
            if crac_id and dims.get("crac_id") != crac_id:
                continue
            if rack_id and dims.get("rack_id") != rack_id:
                continue
            return r
        return None

    def query_history(
        self,
        measure_name: str,
        start_time: datetime,
        end_time: datetime,
        facility_id: Optional[str] = None,
        crac_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        results = []
        for r in self.snapshot():
            ts = r.get("timestamp_dt")
            if ts is None:
                continue
            if ts < start_time or ts > end_time:
                continue
            if r.get("measure_name") != measure_name:
                continue
            dims = r.get("dimensions", {})
            if facility_id and dims.get("facility_id") != facility_id:
                continue
            if crac_id and dims.get("crac_id") != crac_id:
                continue
            results.append(r)
        # Keep the most recent `limit` records (still in ascending time order).
        # Breaking at the first `limit` matches returned the OLDEST records in
        # the window, so drift detection and the history API saw stale data
        # once more than `limit` records had accumulated.
        return results[-limit:] if limit else results

    def aggregate_avg(
        self,
        measure_name: str,
        hours: int = 1,
        facility_id: Optional[str] = None,
    ) -> Optional[float]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        vals = []
        for r in self.snapshot():
            ts = r.get("timestamp_dt")
            if ts is None or ts < cutoff:
                continue
            if r.get("measure_name") != measure_name:
                continue
            dims = r.get("dimensions", {})
            if facility_id and dims.get("facility_id") != facility_id:
                continue
            mv = r.get("measure_value")
            if mv is not None:
                vals.append(float(mv))
        return sum(vals) / len(vals) if vals else None

    def get_all_latest(self) -> Dict[str, Dict]:
        seen: Dict[str, Dict] = {}
        for r in reversed(self.snapshot()):
            key = f"{r.get('dimensions', {}).get('facility_id')}:{r.get('dimensions', {}).get('crac_id')}"
            if key not in seen:
                seen[key] = r
        return seen


# ---------------------------------------------------------------------------
# Timestream Client
# ---------------------------------------------------------------------------

class TimestreamClient:
    """
    Unified Timestream client with automatic local-mode fallback.
    Environment variables:
      TIMESTREAM_DB         — database name (default: CoolingTelemetry)
      TIMESTREAM_TABLE      — table name   (default: TelemetryMetrics)
      AWS_REGION            — AWS region   (default: us-east-1)
      LOCAL_MODE            — "true" to force in-memory mode
    """

    DB_NAME = os.environ.get("TIMESTREAM_DB", "CoolingTelemetry")
    TABLE_NAME = os.environ.get("TIMESTREAM_TABLE", "TelemetryMetrics")
    REGION = os.environ.get("AWS_REGION", "us-east-1")

    TELEMETRY_MEASURES = [
        "server_inlet_temp_c", "server_outlet_temp_c", "fws_supply_temp_c",
        "return_temp_c", "flow_rate_lpm", "pump_speed_pct", "fan_speed_pct",
        "valve_split_pct", "it_power_mw", "cooling_power_mw", "pue", "wue",
        "grid_carbon_gco2_kwh",
    ]

    def __init__(self, local_mode: Optional[bool] = None):
        if local_mode is None:
            local_mode = os.environ.get("LOCAL_MODE", "false").lower() == "true"
        self.local_mode = local_mode
        self._store = InMemoryTimestreamStore()
        self._write_client = None
        self._query_client = None
        # Circuit breaker: when Timestream errors (service not available on LocalStack Community,
        # throttling, an outage, expired credentials) telemetry keeps flowing into the in-memory
        # store and reads are served from it, instead of every write logging an error and history /
        # analytics coming back empty. The cloud is retried after CLOUD_RETRY_S.
        self._cloud_down_until = 0.0
        self._cloud_failures = 0

        if not local_mode:
            try:
                boto_kwargs = _build_boto3_kwargs(self.REGION)
                self._write_client = boto3.client("timestream-write", **boto_kwargs)
                self._query_client = boto3.client("timestream-query", **boto_kwargs)
                endpoint = os.environ.get("AWS_ENDPOINT_URL", "real AWS")
                logger.info(
                    "Connected to Amazon Timestream: %s.%s (endpoint: %s)",
                    self.DB_NAME, self.TABLE_NAME, endpoint,
                )
            except Exception as e:
                logger.warning("Timestream init failed, using in-memory mode: %s", e)
                self.local_mode = True

    CLOUD_RETRY_S = float(os.environ.get("TIMESTREAM_RETRY_S", "60"))

    def _cloud_ready(self) -> bool:
        return not self.local_mode and time.time() >= self._cloud_down_until

    def _local_reads(self) -> bool:
        """True when reads must be served from the in-memory store."""
        return not self._cloud_ready()

    def _cloud_failed(self, exc: Exception) -> None:
        first = self._cloud_failures == 0
        self._cloud_failures += 1
        self._cloud_down_until = time.time() + self.CLOUD_RETRY_S
        if first:
            logger.warning(
                "Timestream unavailable (%s); serving telemetry from the in-memory store, retrying in %.0fs",
                str(exc)[:160], self.CLOUD_RETRY_S,
            )

    def _cloud_ok(self) -> None:
        if self._cloud_failures:
            logger.info("Timestream reachable again; resuming cloud reads and writes")
        self._cloud_failures = 0

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def write_telemetry(self, payload: Dict[str, Any]) -> bool:
        """Writes a single telemetry payload as a Timestream multi-measure record."""
        now_ms = int(time.time() * 1000)
        dimensions = [
            {"Name": "facility_id", "Value": str(payload.get("facility_id", "unknown"))},
            {"Name": "crac_id",     "Value": str(payload.get("crac_id", "unknown"))},
            {"Name": "rack_id",     "Value": str(payload.get("rack_id", "unknown"))},
        ]
        measure_values = []
        for m in self.TELEMETRY_MEASURES:
            if m in payload:
                measure_values.append({
                    "Name": m,
                    "Value": str(payload[m]),
                    "Type": "DOUBLE",
                })

        record = {
            "Dimensions": dimensions,
            "MeasureName": "telemetry",
            "MeasureValueType": "MULTI",
            "MeasureValues": measure_values,
            "Time": str(now_ms),
            "TimeUnit": "MILLISECONDS",
        }

        # Always write to local store (for offline queries and WS streaming)
        local_record = {
            "measure_name": "telemetry",
            "dimensions": {d["Name"]: d["Value"] for d in dimensions},
            "measures": {m["Name"]: float(m["Value"]) for m in measure_values},
            "measure_value": payload.get("pue"),
            "timestamp_dt": datetime.now(timezone.utc),
            "raw": payload,
        }
        self._store.write([local_record])

        if not self._cloud_ready():
            return True     # local mode, or the cloud is temporarily down: the record is kept in memory

        try:
            self._write_client.write_records(
                DatabaseName=self.DB_NAME,
                TableName=self.TABLE_NAME,
                Records=[record],
            )
            self._cloud_ok()
            return True
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "RejectedRecordsException":
                logger.warning("Timestream rejected records: %s", e)   # bad data, not an outage
            else:
                self._cloud_failed(e)
        except BotoCoreError as e:
            self._cloud_failed(e)
        return False

    def write_telemetry_batch(self, payloads: List[Dict[str, Any]]) -> int:
        """Batch write up to 100 telemetry records. Returns count of successful writes."""
        success = 0
        # Timestream supports max 100 records per call
        for i in range(0, len(payloads), 100):
            batch = payloads[i:i + 100]
            for p in batch:
                if self.write_telemetry(p):
                    success += 1
        return success

    # ------------------------------------------------------------------
    # Query path
    # ------------------------------------------------------------------

    def get_latest_pue(self, facility_id: str, hours: int = 1) -> Optional[float]:
        """Returns average PUE over the last N hours for a facility."""
        if self._local_reads():
            return self._store.aggregate_avg("telemetry", hours=hours, facility_id=facility_id)

        if not _is_safe_identifier(facility_id):
            logger.warning("Rejected unsafe facility_id in get_latest_pue: %r", facility_id)
            return None

        query = (
            f"SELECT avg(measure_value::double) "
            f"FROM \"{self.DB_NAME}\".\"{self.TABLE_NAME}\" "
            f"WHERE measure_name = 'pue' "
            f"AND facility_id = '{facility_id}' "
            f"AND time > ago({hours}h)"
        )
        result, ok = self._scalar_query(query)
        if not ok:
            return self._store.aggregate_avg("telemetry", hours=hours, facility_id=facility_id)
        return result

    def get_sla_violation_rate(self, facility_id: str, hours: int = 24) -> float:
        """Returns fraction of readings where server inlet temp was outside ASHRAE range (18–27°C)."""
        if self._local_reads():
            return self._local_sla_rate(facility_id, hours)

        if not _is_safe_identifier(facility_id):
            logger.warning("Rejected unsafe facility_id in get_sla_violation_rate: %r", facility_id)
            return 0.0

        query = (
            f"SELECT "
            f"  COUNT(CASE WHEN measure_value::double < 18.0 OR measure_value::double > 27.0 THEN 1 END) * 1.0 "
            f"  / NULLIF(COUNT(*), 0) "
            f"FROM \"{self.DB_NAME}\".\"{self.TABLE_NAME}\" "
            f"WHERE measure_name = 'server_inlet_temp_c' "
            f"AND facility_id = '{facility_id}' "
            f"AND time > ago({hours}h)"
        )
        result, ok = self._scalar_query(query)
        if not ok:
            return self._local_sla_rate(facility_id, hours)
        return float(result or 0.0)

    def _local_sla_rate(self, facility_id: str, hours: int) -> float:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        total, violations = 0, 0
        for r in self._store.snapshot():
            ts = r.get("timestamp_dt")
            if ts is None or ts < cutoff:
                continue
            if r.get("dimensions", {}).get("facility_id") != facility_id:
                continue
            inlet = r.get("measures", {}).get("server_inlet_temp_c")
            if inlet is not None:
                total += 1
                if inlet < 18.0 or inlet > 27.0:
                    violations += 1
        return violations / max(1, total)

    def get_telemetry_history(
        self,
        facility_id: str,
        crac_id: Optional[str] = None,
        measures: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Returns time-ordered telemetry history."""
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        if start_time is None:
            start_time = end_time - timedelta(hours=1)

        def local():
            rows = self._store.query_history(
                "telemetry", start_time, end_time,
                facility_id=facility_id, crac_id=crac_id, limit=limit,
            )
            return [r.get("raw", {}) for r in rows]

        if self._local_reads():
            return local()

        if not _is_safe_identifier(facility_id) or (crac_id is not None and not _is_safe_identifier(crac_id)):
            logger.warning("Rejected unsafe identifier in get_telemetry_history: facility_id=%r crac_id=%r", facility_id, crac_id)
            return []

        time_pred = (
            f"AND time BETWEEN '{start_time.strftime('%Y-%m-%d %H:%M:%S')}' "
            f"AND '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'"
        )
        crac_pred = f"AND crac_id = '{crac_id}'" if crac_id else ""
        query = (
            f"SELECT time, facility_id, crac_id, rack_id, measure_name, measure_value::double "
            f"FROM \"{self.DB_NAME}\".\"{self.TABLE_NAME}\" "
            f"WHERE facility_id = '{facility_id}' {crac_pred} {time_pred} "
            f"ORDER BY time DESC LIMIT {int(limit)}"
        )
        # Newest `limit` rows, returned oldest-first.
        rows, ok = self._tabular_query(query)
        return list(reversed(rows)) if ok else local()

    def get_spatial_snapshot(self) -> List[Dict[str, Any]]:
        """Returns latest telemetry per CRAC for spatial heatmap rendering."""
        def local():
            latest = self._store.get_all_latest()
            return [r.get("raw", {}) for r in latest.values() if r.get("raw")]

        if self._local_reads():
            return local()

        query = (
            f"WITH ranked AS ("
            f"  SELECT *, ROW_NUMBER() OVER (PARTITION BY facility_id, crac_id ORDER BY time DESC) AS rn "
            f"  FROM \"{self.DB_NAME}\".\"{self.TABLE_NAME}\" "
            f"  WHERE measure_name = 'server_inlet_temp_c' "
            f") SELECT facility_id, crac_id, time, measure_value::double AS server_inlet_temp_c "
            f"FROM ranked WHERE rn = 1"
        )
        rows, ok = self._tabular_query(query)
        return rows if ok else local()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scalar_query(self, query_string: str):
        """Returns (value, ok). ok=False means the cloud query failed (the breaker is tripped)."""
        try:
            resp = self._query_client.query(QueryString=query_string)
            self._cloud_ok()
            rows = resp.get("Rows", [])
            if rows and rows[0]["Data"]:
                scalar = rows[0]["Data"][0].get("ScalarValue")
                return (float(scalar) if scalar else None), True
            return None, True
        except (ClientError, BotoCoreError) as e:
            self._cloud_failed(e)
            return None, False

    def _tabular_query(self, query_string: str):
        """Returns (rows, ok). ok=False means the cloud query failed (the breaker is tripped)."""
        results = []
        try:
            paginator = self._query_client.get_paginator("query")
            for page in paginator.paginate(QueryString=query_string):
                col_info = page.get("ColumnInfo", [])
                for row in page.get("Rows", []):
                    row_dict = {}
                    for i, col in enumerate(col_info):
                        name = col["Name"]
                        data = row["Data"][i]
                        row_dict[name] = data.get("ScalarValue") or data.get("NullValue")
                    results.append(row_dict)
            self._cloud_ok()
            return results, True
        except (ClientError, BotoCoreError) as e:
            self._cloud_failed(e)
            return [], False


# Module-level singleton
_default_client: Optional[TimestreamClient] = None


def get_timestream_client() -> TimestreamClient:
    global _default_client
    if _default_client is None:
        _default_client = TimestreamClient()
    return _default_client
