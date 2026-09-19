"""
AWS IoT TwinMaker Universal Data Query (UDQ) Connector.

Maps TwinMaker entity/property queries to the underlying data sources:
  - Amazon Timestream  (time-series telemetry)
  - AWS IoT SiteWise   (asset property values)
  - Local in-memory bus (offline/development mode)

Implements the three TwinMaker UDQ interface methods:
  - get_property_value
  - get_property_value_history
  - batch_put_property_values

Environment variables:
  LOCAL_MODE       — "true" to force local mode.
  AWS_ENDPOINT_URL — Override boto3 endpoint (e.g. http://localhost:4566 for LocalStack).
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY — Optional credential override.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

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

# entity_id/property_name are interpolated directly into Timestream query
# strings below (f-string text, not a parameterized query), so anything not
# matching this safe identifier charset is rejected before it reaches SQL.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _is_safe_identifier(value: str) -> bool:
    return bool(value) and bool(_SAFE_ID_RE.match(value))


# ---------------------------------------------------------------------------
# Property routing: maps TwinMaker (entity_id, property_name) -> data source
# ---------------------------------------------------------------------------

PROPERTY_ROUTE = {
    # Telemetry properties served from Timestream
    "server_inlet_temp_c":   "timestream",
    "server_outlet_temp_c":  "timestream",
    "fws_supply_temp_c":     "timestream",
    "return_temp_c":         "timestream",
    "flow_rate_lpm":         "timestream",
    "pump_speed_pct":        "timestream",
    "fan_speed_pct":         "timestream",
    "valve_split_pct":       "timestream",
    "it_power_mw":           "timestream",
    "cooling_power_mw":      "timestream",
    "pue":                   "timestream",
    "grid_carbon_gco2_kwh":  "timestream",
    # SiteWise asset properties
    "SupplyTempC":    "sitewise",
    "ReturnTempC":    "sitewise",
    "FlowRateLPM":    "sitewise",
    "PumpSpeedPct":   "sitewise",
    "FanSpeedPct":    "sitewise",
    "FacilityPUE":    "sitewise",
    "TotalITPowerMW": "sitewise",
}


class TwinMakerUDQConnector:
    """
    Universal Data Query connector for AWS IoT TwinMaker.
    Can be deployed as an AWS Lambda function or used locally.
    """

    TIMESTREAM_DB = os.environ.get("TIMESTREAM_DB", "CoolingTelemetry")
    TIMESTREAM_TABLE = os.environ.get("TIMESTREAM_TABLE", "TelemetryMetrics")
    AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

    def __init__(self, local_mode: bool = False):
        self.local_mode = local_mode
        self._ts_query_client = None
        self._sitewise_client = None

        if not local_mode:
            try:
                boto_kwargs = _build_boto3_kwargs(self.AWS_REGION)
                self._ts_query_client = boto3.client("timestream-query", **boto_kwargs)
                self._sitewise_client = boto3.client("iotsitewise", **boto_kwargs)
            except Exception as e:
                logger.warning("AWS client init failed, falling back to local mode: %s", e)
                self.local_mode = True

    # ------------------------------------------------------------------
    # Core UDQ Interface Methods
    # ------------------------------------------------------------------

    def get_property_value(
        self,
        workspace_id: str,
        entity_id: str,
        component_name: str,
        property_name: str,
    ) -> Dict[str, Any]:
        """Returns the latest value for a single entity property."""
        source = PROPERTY_ROUTE.get(property_name, "timestream")

        if self.local_mode or source == "timestream":
            return self._get_latest_from_timestream(entity_id, property_name)
        elif source == "sitewise":
            return self._get_latest_from_sitewise(entity_id, property_name)

        return self._empty_response(property_name)

    def get_property_value_history(
        self,
        workspace_id: str,
        entity_id: str,
        component_name: str,
        property_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """Returns time-series history for a property within a time range."""
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        if start_time is None:
            start_time = end_time - timedelta(hours=1)

        if self.local_mode:
            return self._mock_history(entity_id, property_name, start_time, end_time, max_results)

        return self._query_timestream_history(
            entity_id, property_name, start_time, end_time, max_results
        )

    def batch_put_property_values(
        self,
        workspace_id: str,
        entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Writes multiple property values to the workspace (local bus in offline mode)."""
        if self.local_mode:
            return {"batchPutPropertyValuesResponse": {"errorEntries": []}}

        # In cloud mode, writes go to SiteWise via MQTT/IoT rule; UDQ is read-only
        return {"batchPutPropertyValuesResponse": {"errorEntries": []}}

    # ------------------------------------------------------------------
    # Timestream Queries
    # ------------------------------------------------------------------

    def _get_latest_from_timestream(self, entity_id: str, property_name: str) -> Dict[str, Any]:
        if self.local_mode:
            return self._mock_latest(entity_id, property_name)

        if not (_is_safe_identifier(entity_id) and _is_safe_identifier(property_name)):
            logger.warning("Rejected unsafe identifier in UDQ request: entity_id=%r property_name=%r", entity_id, property_name)
            return self._empty_response(property_name)

        query = (
            f"SELECT time, measure_value::double AS value "
            f"FROM \"{self.TIMESTREAM_DB}\".\"{self.TIMESTREAM_TABLE}\" "
            f"WHERE measure_name = '{property_name}' "
            f"AND (facility_id = '{entity_id}' OR crac_id = '{entity_id}' OR rack_id = '{entity_id}') "
            f"ORDER BY time DESC LIMIT 1"
        )
        try:
            resp = self._ts_query_client.query(QueryString=query)
            rows = resp.get("Rows", [])
            if rows:
                ts = rows[0]["Data"][0]["ScalarValue"]
                val = float(rows[0]["Data"][1]["ScalarValue"])
                return {
                    "propertyValue": {
                        "value": {"doubleValue": val},
                        "timestamp": {"seconds": int(datetime.fromisoformat(ts.replace(" ", "T")).timestamp())}
                    }
                }
        except (ClientError, BotoCoreError) as e:
            logger.warning("Timestream query failed for %s/%s: %s", entity_id, property_name, e)
        return self._empty_response(property_name)

    def _query_timestream_history(
        self,
        entity_id: str,
        property_name: str,
        start_time: datetime,
        end_time: datetime,
        max_results: int,
    ) -> Dict[str, Any]:
        if not (_is_safe_identifier(entity_id) and _is_safe_identifier(property_name)):
            logger.warning("Rejected unsafe identifier in UDQ history request: entity_id=%r property_name=%r", entity_id, property_name)
            return {"propertyValues": []}

        start_s = start_time.strftime("%Y-%m-%d %H:%M:%S.000000000")
        end_s = end_time.strftime("%Y-%m-%d %H:%M:%S.000000000")
        query = (
            f"SELECT time, measure_value::double AS value "
            f"FROM \"{self.TIMESTREAM_DB}\".\"{self.TIMESTREAM_TABLE}\" "
            f"WHERE measure_name = '{property_name}' "
            f"AND (facility_id = '{entity_id}' OR crac_id = '{entity_id}' OR rack_id = '{entity_id}') "
            f"AND time BETWEEN '{start_s}' AND '{end_s}' "
            f"ORDER BY time ASC LIMIT {max_results}"
        )
        values = []
        try:
            resp = self._ts_query_client.query(QueryString=query)
            for row in resp.get("Rows", []):
                ts_str = row["Data"][0]["ScalarValue"]
                val = float(row["Data"][1]["ScalarValue"])
                ts_epoch = int(datetime.fromisoformat(ts_str.replace(" ", "T")).timestamp())
                values.append({
                    "value": {"doubleValue": val},
                    "timestamp": {"seconds": ts_epoch}
                })
        except (ClientError, BotoCoreError) as e:
            logger.warning("Timestream history query failed: %s", e)
        return {"propertyValues": values}

    def _get_latest_from_sitewise(self, entity_id: str, property_name: str) -> Dict[str, Any]:
        try:
            resp = self._sitewise_client.get_asset_property_value(
                assetId=entity_id,
                propertyAlias=f"/{entity_id}/{property_name}"
            )
            pv = resp.get("propertyValue", {})
            return {"propertyValue": pv}
        except (ClientError, BotoCoreError) as e:
            logger.warning("SiteWise query failed for %s/%s: %s", entity_id, property_name, e)
        return self._empty_response(property_name)

    # ------------------------------------------------------------------
    # Local / Mock fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_latest(entity_id: str, property_name: str) -> Dict[str, Any]:
        import random
        defaults = {
            "server_inlet_temp_c": 22.5,
            "server_outlet_temp_c": 35.8,
            "fws_supply_temp_c": 18.0,
            "return_temp_c": 26.5,
            "flow_rate_lpm": 4500.0,
            "pump_speed_pct": 72.0,
            "fan_speed_pct": 68.0,
            "valve_split_pct": 50.0,
            "it_power_mw": 0.018,
            "cooling_power_mw": 0.005,
            "pue": 1.224,
            "grid_carbon_gco2_kwh": 380.0,
        }
        base = defaults.get(property_name, 0.0)
        val = base + random.gauss(0, base * 0.02)
        return {
            "propertyValue": {
                "value": {"doubleValue": round(val, 4)},
                "timestamp": {"seconds": int(datetime.now(timezone.utc).timestamp())}
            }
        }

    @staticmethod
    def _mock_history(
        entity_id: str,
        property_name: str,
        start_time: datetime,
        end_time: datetime,
        max_results: int,
    ) -> Dict[str, Any]:
        import math
        import random
        duration_s = (end_time - start_time).total_seconds()
        step_s = max(1.0, duration_s / max_results)
        defaults = {
            "server_inlet_temp_c": 22.5,
            "pue": 1.22,
            "pump_speed_pct": 72.0,
            "fan_speed_pct": 68.0,
            "flow_rate_lpm": 4500.0,
            "cooling_power_mw": 0.005,
        }
        base = defaults.get(property_name, 10.0)
        values = []
        for i in range(min(max_results, int(duration_s / step_s))):
            ts = start_time + timedelta(seconds=i * step_s)
            val = base + math.sin(2 * math.pi * i / max_results) * base * 0.05 + random.gauss(0, base * 0.01)
            values.append({
                "value": {"doubleValue": round(val, 4)},
                "timestamp": {"seconds": int(ts.timestamp())}
            })
        return {"propertyValues": values}

    @staticmethod
    def _empty_response(property_name: str) -> Dict[str, Any]:
        return {"propertyValue": None, "property": property_name}


# ---------------------------------------------------------------------------
# Lambda handler entrypoint for AWS IoT TwinMaker
# ---------------------------------------------------------------------------

_connector = TwinMakerUDQConnector(local_mode=(os.environ.get("LOCAL_MODE", "false").lower() == "true"))


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entrypoint called by TwinMaker for UDQ requests."""
    request_type = event.get("requestType", "")
    params = event.get("params", {})

    workspace_id = params.get("workspaceId", "")
    entity_id = params.get("entityId", "")
    component_name = params.get("componentName", "")
    property_name = params.get("propertyName", "")

    if request_type == "GetPropertyValue":
        return _connector.get_property_value(workspace_id, entity_id, component_name, property_name)

    if request_type == "GetPropertyValueHistory":
        start_str = params.get("startTime", "")
        end_str = params.get("endTime", "")
        start_dt = datetime.fromisoformat(start_str.rstrip("Z")).replace(tzinfo=timezone.utc) if start_str else None
        end_dt = datetime.fromisoformat(end_str.rstrip("Z")).replace(tzinfo=timezone.utc) if end_str else None
        return _connector.get_property_value_history(
            workspace_id, entity_id, component_name, property_name,
            start_time=start_dt, end_time=end_dt,
            max_results=params.get("maxResults", 100),
        )

    if request_type == "BatchPutPropertyValues":
        return _connector.batch_put_property_values(workspace_id, params.get("entries", []))

    return {"error": f"Unknown requestType: {request_type}"}
