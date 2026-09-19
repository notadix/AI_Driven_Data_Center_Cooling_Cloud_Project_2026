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


def update_metrics(simulator: "IoTSimulator") -> None:
    """Refresh all gauge values from live simulator state.

    Called once per scrape by generate_metrics_output().
    Reads PhysicsSimulator state directly: no I/O, no async required.
    """
    from src.backend.api.v1.control import get_crac_mode

    facility_crac_counts: dict = {}
    facility_breach_counts: dict = {}

    for key, sim in simulator._simulators.items():
        fid = sim.facility_id
        cid = sim.crac_id

        SUPPLY_TEMP_C.labels(fid, cid).set(sim.supply_c)
        # Rough return-water delta: Q = m_dot * cp * dT => dT = Q / (m_dot * cp)
        # Use fixed m_dot=20 LPM, cp=4.186 kJ/(kg*C) as a scrape-time approximation
        return_c = sim.supply_c + (sim.it_power_kw * 3.0 / (4.186 * 20.0))
        RETURN_TEMP_C.labels(fid, cid).set(round(return_c, 2))

        inlet_c = sim.supply_c + 3.5 + sim.ambient_c * 0.05
        SERVER_INLET_TEMP_C.labels(fid, cid).set(round(inlet_c, 2))

        PUMP_SPEED_PCT.labels(fid, cid).set(sim.pump_pct)
        FAN_SPEED_PCT.labels(fid, cid).set(sim.fan_pct)
        GRID_CARBON.labels(fid, cid).set(sim.carbon_gco2_kwh)

        it_kw   = sim.it_power_kw
        pump_kw = 15.0 * (sim.pump_pct / 100.0) ** 3
        fan_kw  = 8.0  * (sim.fan_pct  / 100.0) ** 3
        pue = (it_kw + pump_kw + fan_kw) / max(0.1, it_kw)
        PUE.labels(fid, cid).set(round(pue, 4))

        mode = get_crac_mode(cid, facility_id=fid)
        CONTROL_MODE.labels(fid, cid).set(1 if mode == "auto" else 0)

        facility_crac_counts[fid] = facility_crac_counts.get(fid, 0) + 1
        if inlet_c > ASHRAE_INLET_HIGH:
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