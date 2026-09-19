"""
Prometheus metrics for the AI-Driven Cooling Digital Twin backend.

Design decisions
----------------
1. Scrape-time updates: all gauge values are refreshed inside update_metrics()
   that fires exactly once per scrape (called from the /metrics endpoint),
   keeping the handler free from async complexity and avoiding double-counting.

2. Dedicated CollectorRegistry: we do NOT use the default prometheus_client
   global registry to avoid process-level metric leakage from third-party
   libraries.  Every metric is registered on COOLING_REGISTRY.

3. No _total suffix on Gauges: Prometheus convention reserves _total for
   Counters (monotonic).  Gauges use plain names such as
   cooling_sla_violation_rate instead of cooling_sla_violations_total.

4. Labels: all per-CRAC/facility metrics carry facility_id and crac_id labels
   so Prometheus and Grafana can filter/aggregate by site.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    generate_latest,
)

if TYPE_CHECKING:
    from src.aws.iot.iot_publisher import IoTSimulator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dedicated registry (not the prometheus_client global)
# ---------------------------------------------------------------------------

COOLING_REGISTRY = CollectorRegistry(auto_describe=True)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

SUPPLY_TEMP_C = Gauge(
    "cooling_crac_supply_temp_c",
    "CRAC supply water temperature in degrees C",
    ["facility_id", "crac_id"],
    registry=COOLING_REGISTRY,
)
RETURN_TEMP_C = Gauge(
    "cooling_crac_return_temp_c",
    "CRAC return water temperature in degrees C",
    ["facility_id", "crac_id"],
    registry=COOLING_REGISTRY,
)
SERVER_INLET_TEMP_C = Gauge(
    "cooling_rack_server_inlet_temp_c",
    "Server rack inlet temperature in degrees C",
    ["facility_id", "crac_id"],
    registry=COOLING_REGISTRY,
)
PUMP_SPEED_PCT = Gauge(
    "cooling_crac_pump_speed_pct",
    "CRAC pump speed percentage",
    ["facility_id", "crac_id"],
    registry=COOLING_REGISTRY,
)
FAN_SPEED_PCT = Gauge(
    "cooling_crac_fan_speed_pct",
    "CRAC fan speed percentage",
    ["facility_id", "crac_id"],
    registry=COOLING_REGISTRY,
)
PUE = Gauge(
    "cooling_facility_pue",
    "Power Usage Effectiveness (PUE)",
    ["facility_id", "crac_id"],
    registry=COOLING_REGISTRY,
)
GRID_CARBON = Gauge(
    "cooling_crac_grid_carbon_gco2_kwh",
    "Grid carbon intensity in gCO2/kWh",
    ["facility_id", "crac_id"],
    registry=COOLING_REGISTRY,
)
SLA_VIOLATION_RATE = Gauge(
    "cooling_sla_violation_rate",
    "Fraction of CRACs in this facility currently in SLA breach (0 to 1)",
    ["facility_id"],
    registry=COOLING_REGISTRY,
)
ACTIVE_CRACS = Gauge(
    "cooling_active_cracs",
    "Number of CRACs actively simulated in this facility",
    ["facility_id"],
    registry=COOLING_REGISTRY,
)
CONTROL_MODE = Gauge(
    "cooling_crac_control_mode",
    "Control mode of a CRAC: 1=auto, 0=manual",
    ["facility_id", "crac_id"],
    registry=COOLING_REGISTRY,
)
LAST_SCRAPE_TIMESTAMP = Gauge(
    "cooling_metrics_last_scrape_timestamp_seconds",
    "Unix timestamp of the last successful /metrics scrape",
    registry=COOLING_REGISTRY,
)

# ---------------------------------------------------------------------------
# Scrape-time update helpers
# ---------------------------------------------------------------------------

ASHRAE_INLET_HIGH = 27.0
ASHRAE_INLET_LOW = 18.0


def update_metrics(simulator: "IoTSimulator") -> None:
    """Refresh all gauge values from live simulator state.

    Called once per scrape by generate_metrics_output().
    Reads PhysicsSimulator state directly: no I/O, no async required.
    """
    from src.backend.api.v1.control import get_crac_mode

    facility_crac_counts: dict = {}
    facility_breach_counts: dict = {}
    latest = simulator.get_latest_telemetry()

    for key, sim in simulator._simulators.items():
        fid = sim.facility_id
        cid = sim.crac_id

        SUPPLY_TEMP_C.labels(fid, cid).set(sim.supply_c)
        PUMP_SPEED_PCT.labels(fid, cid).set(sim.pump_pct)
        FAN_SPEED_PCT.labels(fid, cid).set(sim.fan_pct)
        GRID_CARBON.labels(fid, cid).set(sim.carbon_gco2_kwh)

        # Return/inlet temperature and PUE come from the latest published
        # telemetry so they match what the dashboard and REST API report
        # (they used to be approximated from setpoints, which ignored the
        # chiller load and free-air mixing, and gave a different PUE).
        payload = latest.get(f"datacenter/cooling/telemetry/{fid}/{cid}") or {}
        inlet_c = payload.get("server_inlet_temp_c")
        if payload.get("return_temp_c") is not None:
            RETURN_TEMP_C.labels(fid, cid).set(float(payload["return_temp_c"]))
        if inlet_c is not None:
            SERVER_INLET_TEMP_C.labels(fid, cid).set(float(inlet_c))
        if payload.get("pue") is not None:
            PUE.labels(fid, cid).set(float(payload["pue"]))

        mode = get_crac_mode(cid, facility_id=fid)
        CONTROL_MODE.labels(fid, cid).set(1 if mode == "auto" else 0)

        facility_crac_counts[fid] = facility_crac_counts.get(fid, 0) + 1
        if inlet_c is not None and (float(inlet_c) > ASHRAE_INLET_HIGH or float(inlet_c) < ASHRAE_INLET_LOW):
            facility_breach_counts[fid] = facility_breach_counts.get(fid, 0) + 1

    for fid, total in facility_crac_counts.items():
        ACTIVE_CRACS.labels(fid).set(total)
        breaches = facility_breach_counts.get(fid, 0)
        SLA_VIOLATION_RATE.labels(fid).set(round(breaches / max(1, total), 4))

    LAST_SCRAPE_TIMESTAMP.set(time.time())


def generate_metrics_output(simulator: "IoTSimulator") -> bytes:
    """Update gauges then serialise to Prometheus text format."""
    update_metrics(simulator)
    return generate_latest(COOLING_REGISTRY)