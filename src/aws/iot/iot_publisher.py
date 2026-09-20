"""
AWS IoT Core MQTT Publisher & Physics-Grounded Telemetry Simulator.

Dual-mode operation:
  - AWS Cloud / LocalStack Mode: Publishes to AWS IoT Core over MQTT/TLS using
    boto3 IoT Data Plane.  When AWS_ENDPOINT_URL is set (e.g. LocalStack at
    http://localhost:4566) that URL is used as the iot-data endpoint, allowing
    local development without real AWS credentials.
  - Local Offline Mode: Publishes to an in-memory async queue consumed by the
    FastAPI backend.

MQTT Topics:
  Telemetry: datacenter/cooling/telemetry/{facility_id}/{crac_id}
  Control:   datacenter/cooling/control/{crac_id}
"""

import asyncio
import json
import logging
import math
import os
import random
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from src.digital_twin.physics_dynamics import LiquidCoolingPhysics, ZONE_SCALE, estimate_relative_humidity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regional climate constants — single source of truth, mirroring
# src/aws/serverless/lambda_weather_fetcher.py::REGIONAL_CLIMATE_BASE
# ---------------------------------------------------------------------------

# Ambient baseline temperatures (°C dry-bulb) per facility.
# These values are used by PhysicsSimulator to initialise ambient_c so that
# different facilities produce measurably different thermal behaviour in the UI.
REGIONAL_CLIMATE_BASE = {
    "DC-EAST-01": {"base_dry_bulb": 22.0, "rh_base": 60.0, "diurnal_range": 8.0},
    "DC-WEST-02": {"base_dry_bulb": 18.0, "rh_base": 50.0, "diurnal_range": 9.0},
    "DC-EU-01":   {"base_dry_bulb": 15.0, "rh_base": 75.0, "diurnal_range": 6.0},
}

# Grid carbon intensity (gCO₂/kWh) per facility — used as the starting value
# for the PhysicsSimulator's carbon_gco2_kwh random walk.
FACILITY_GRID_CARBON_GCOEKWH: Dict[str, float] = {
    "DC-EAST-01": 320.0,   # US East (mid-Atlantic, heavier coal/gas mix)
    "DC-WEST-02": 180.0,   # US West (high renewables — Pacific Northwest hydro)
    "DC-EU-01":   210.0,   # EU Frankfurt (wind + nuclear)
}


# ---------------------------------------------------------------------------
# Telemetry & Control Data Contracts
# ---------------------------------------------------------------------------

@dataclass
class TelemetryPayload:
    timestamp_iso: str
    facility_id: str
    crac_id: str
    rack_id: str
    server_inlet_temp_c: float
    server_outlet_temp_c: float
    fws_supply_temp_c: float
    return_temp_c: float
    flow_rate_lpm: float
    pump_speed_pct: float
    fan_speed_pct: float
    valve_split_pct: float
    it_power_mw: float
    cooling_power_mw: float
    pue: float
    wue: float
    grid_carbon_gco2_kwh: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TelemetryPayload":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ControlPayload:
    control: Dict[str, float]
    safety_status: str = "NORMAL"
    v_reward: float = 0.0
    v_cost: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ---------------------------------------------------------------------------
# Physics-grounded telemetry generator
# ---------------------------------------------------------------------------

class PhysicsSimulator:
    """Generates physically plausible telemetry for one CRAC unit and rack row."""

    ASHRAE_INLET_LOW = 18.0
    ASHRAE_INLET_HIGH = 27.0
    ASHRAE_INLET_CRITICAL = 32.0
    SPECIFIC_HEAT_WATER = 4.186     # kJ/(kg·°C)
    WATER_DENSITY = 1.0             # kg/L

    def __init__(
        self,
        facility_id: str,
        crac_id: str,
        rack_id: str,
        ambient_c: float = 22.0,
        grid_carbon_gco2_kwh: float = 320.0,
        it_power_kw: float = 18.0,
        day_steps: int = 144,
    ):
        # Simulator steps per simulated 24 h day. 144 (default) = one step is 10 simulated minutes, so at a 1 s
        # publish interval a day lasts 2.4 minutes. Raise it (SIM_DAY_STEPS) for a slower, calmer day; the daily
        # drift per step shrinks in proportion and the noise with its square root.
        self._day_steps = max(1, int(day_steps))
        self._tscale = 144.0 / self._day_steps
        self.facility_id = facility_id
        self.crac_id = crac_id
        self.rack_id = rack_id
        self.ambient_c = ambient_c
        self.it_power_kw = it_power_kw

        # Physical state
        self.supply_c = 18.5
        self.pump_pct = 75.0
        self.fan_pct = 70.0
        self.valve_split_pct = 50.0
        # Initialise carbon at the per-facility grid intensity so that
        # facilities with cleaner grids (e.g. DC-WEST-02 at 180 gCO₂/kWh)
        # show lower values in the UI from the very first telemetry frame.
        self.carbon_gco2_kwh = grid_carbon_gco2_kwh
        self._physics = LiquidCoolingPhysics()
        # One step = 10 simulated minutes (144 steps per day): the last 24 IT readings are the
        # 4-hour history the load forecaster expects.
        self.it_history: deque = deque(maxlen=24)
        self._step = 0

        # Pending control action from control topic
        self._pending_control: Optional[Dict[str, float]] = None

    def apply_control(self, control: Dict[str, float]) -> None:
        self._pending_control = control

    def _cubic_pump_power(self, pump_pct: float, rated_kw: float = 15.0) -> float:
        frac = pump_pct / 100.0
        return rated_kw * (frac ** 3)

    def _fan_power(self, fan_pct: float, rated_kw: float = 8.0) -> float:
        frac = fan_pct / 100.0
        return rated_kw * (frac ** 3)

    def _chiller_cop(self, ambient_c: float) -> float:
        base_cop = 3.5
        return max(1.5, base_cop - 0.04 * max(0.0, ambient_c - 15.0))

    def _estimate_relative_humidity(self, ambient_c: float) -> float:
        return estimate_relative_humidity(ambient_c)

    def _compute_wue(self, chiller_kw: float, it_power_kw: float, ambient_c: float) -> float:
        """Water Usage Effectiveness (L/kWh-IT); delegates to the shared
        physics model (src/digital_twin/physics_dynamics.py) so the RL
        environment and the live simulator use one water model."""
        return self._physics.wue(chiller_kw, it_power_kw, ambient_c)

    def step(self) -> TelemetryPayload:
        self._step += 1
        hour = (self._step % self._day_steps) * (24.0 / self._day_steps)
        k, kn = self._tscale, math.sqrt(self._tscale)

        # Ambient & IT load dynamics
        self.ambient_c = float(
            max(-5.0, min(45.0, self.ambient_c
                + math.sin(2 * math.pi * (hour - 8) / 24.0) * 0.4 * k
                + random.gauss(0, 0.15 * kn)))
        )
        # it_power_kw is ONE representative rack (10-28 kW); steps are the same size as the RL environment's
        # hall-scale walk (150 kW drift / 80 kW noise) divided by ZONE_SCALE. The old +-200 / 100 steps were
        # 10x the whole range, so the load just alternated between the two clamps.
        self.it_power_kw = float(
            max(10.0, min(28.0, self.it_power_kw
                + math.sin(2 * math.pi * (hour - 9) / 24.0) * 0.15 * k
                + random.gauss(0, 0.08 * kn)))
        )
        self.carbon_gco2_kwh = float(
            max(120.0, min(580.0, self.carbon_gco2_kwh + random.gauss(0, 10.0 * kn)))
        )

        # Apply pending control action
        if self._pending_control:
            ctrl = self._pending_control
            self.supply_c = float(max(14.0, min(24.0, self.supply_c + ctrl.get("delta_supply_c", 0.0))))
            if "pump_speed_pct" in ctrl:
                self.pump_pct = float(max(35.0, min(100.0, ctrl["pump_speed_pct"])))
            if "fan_speed_pct" in ctrl:
                self.fan_pct = float(max(30.0, min(100.0, ctrl["fan_speed_pct"])))
            if "valve_split_pct" in ctrl:
                self.valve_split_pct = float(max(0.0, min(100.0, ctrl["valve_split_pct"])))
            self._pending_control = None

        self.it_history.append(self.it_power_kw)

        # Thermal + power model: the SAME Frontier-calibrated physics the RL
        # environment uses (src/digital_twin/physics_dynamics.py), evaluated at
        # hall scale. Each simulated CRAC reports one representative rack whose
        # load is 1/ZONE_SCALE of the hall, so heat/power are scaled up to the
        # hall, run through the shared physics, and per-rack values are scaled
        # back down for reporting. (Previously this class used a separate
        # rack-scale model with pump/fan ratings sized for the whole hall,
        # giving PUE ~2 and observations the trained agent had never seen.)
        z = ZONE_SCALE
        phys = self._physics
        it_zone_kw = self.it_power_kw * z
        flow_lpm = phys.flow_lpm(self.pump_pct)
        return_c, inlet_c, outlet_c = phys.thermal_balance(it_zone_kw, self.supply_c, flow_lpm, self.ambient_c)

        # Free-air economizer mixing (identical to DataCenterCoolingEnv._obs).
        # Outside air can cut chiller load but cannot cool the rack inlet below
        # the supply setpoint.
        free_cool_frac = max(0.0, min(1.0, (self.valve_split_pct / 100.0) * (1.0 - max(0.0, (self.ambient_c - 18.0) / 20.0))))
        mixed_inlet_c = max(self.supply_c, (1.0 - free_cool_frac) * inlet_c + free_cool_frac * min(self.ambient_c, 22.0))
        server_outlet_c = outlet_c + (mixed_inlet_c - inlet_c)
        server_inlet_c = mixed_inlet_c

        _, _, chiller_zone_kw, cooling_zone_kw, _ = phys.power_and_pue(
            it_zone_kw, self.supply_c, self.pump_pct, self.fan_pct, self.ambient_c
        )
        cooling_zone_kw = cooling_zone_kw * (1.0 - 0.3 * free_cool_frac)
        pue = (it_zone_kw + cooling_zone_kw + phys.c.FIXED_OVERHEAD_KW) / max(1.0, it_zone_kw)

        # Per-rack reporting values.
        cooling_kw = cooling_zone_kw / z
        chiller_kw = chiller_zone_kw / z
        wue = self._compute_wue(chiller_kw, self.it_power_kw, self.ambient_c)

        return TelemetryPayload(
            timestamp_iso=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            facility_id=self.facility_id,
            crac_id=self.crac_id,
            rack_id=self.rack_id,
            server_inlet_temp_c=round(server_inlet_c, 3),
            server_outlet_temp_c=round(server_outlet_c, 3),
            fws_supply_temp_c=round(self.supply_c, 3),
            return_temp_c=round(return_c, 3),
            flow_rate_lpm=round(flow_lpm, 1),
            pump_speed_pct=round(self.pump_pct, 1),
            fan_speed_pct=round(self.fan_pct, 1),
            valve_split_pct=round(self.valve_split_pct, 1),
            it_power_mw=round(self.it_power_kw / 1000.0, 6),
            cooling_power_mw=round(cooling_kw / 1000.0, 6),
            pue=round(pue, 4),
            wue=round(wue, 4),
            grid_carbon_gco2_kwh=round(self.carbon_gco2_kwh, 1),
        )


# ---------------------------------------------------------------------------
# In-memory telemetry bus for local / offline mode
# ---------------------------------------------------------------------------

class LocalTelemetryBus:
    """Thread-safe in-memory pub/sub bus for local simulation without AWS."""

    def __init__(self, maxsize: int = 1000):
        self._queue: asyncio.Queue = None  # initialised lazily per event loop
        self._subscribers: List[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._latest: Dict[str, Dict] = {}

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not q]

    def publish(self, topic: str, payload: Dict) -> None:
        with self._lock:
            self._latest[topic] = payload
            for sub in list(self._subscribers):
                try:
                    sub.put_nowait({"topic": topic, "payload": payload})
                except asyncio.QueueFull:
                    pass

    def get_latest(self, topic: Optional[str] = None) -> Dict:
        with self._lock:
            if topic:
                return self._latest.get(topic, {})
            return dict(self._latest)


# Singleton bus instance (imported by FastAPI backend)
local_bus = LocalTelemetryBus()


# ---------------------------------------------------------------------------
# AWS IoT Core Publisher
# ---------------------------------------------------------------------------

class AWSIoTPublisher:
    """Publishes MQTT messages via AWS IoT Core Data Plane using boto3.

    The endpoint for the iot-data client is resolved in order:
      1. AWS_IOT_ENDPOINT env var (original custom IoT endpoint like
         xxxx.iot.us-east-1.amazonaws.com — used as-is with https://).
      2. AWS_ENDPOINT_URL env var (LocalStack-style, already a full URL
         including scheme, used directly).
      Both paths pass AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY if present.
    """

    def __init__(
        self,
        endpoint: str,
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.region = region

        # Derive endpoint_url: prefer explicit AWS_ENDPOINT_URL (LocalStack)
        # so we never blindly prepend https:// to an http:// address.
        aws_endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        if aws_endpoint_url:
            endpoint_url = aws_endpoint_url
        else:
            # Traditional IoT Core custom endpoint hostname — always HTTPS.
            ep = endpoint.lstrip("https://").lstrip("http://")
            endpoint_url = f"https://{ep}"

        boto_kwargs: Dict[str, Any] = {
            "region_name": region,
            "endpoint_url": endpoint_url,
        }
        # Credential pass-through (env takes precedence over constructor args)
        key = os.environ.get("AWS_ACCESS_KEY_ID") or aws_access_key_id
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY") or aws_secret_access_key
        if key and secret:
            boto_kwargs["aws_access_key_id"] = key
            boto_kwargs["aws_secret_access_key"] = secret

        self._client = boto3.client("iot-data", **boto_kwargs)

    def publish(self, topic: str, payload: str, qos: int = 1) -> bool:
        try:
            self._client.publish(topic=topic, payload=payload.encode("utf-8"), qos=qos)
            return True
        except (ClientError, BotoCoreError) as e:
            logger.warning("AWS IoT publish failed for topic %s: %s", topic, e)
            return False


# ---------------------------------------------------------------------------
# IoT Simulator — orchestrates simulators + publisher
# ---------------------------------------------------------------------------

class IoTSimulator:
    """
    Orchestrates multiple PhysicsSimulator instances across facilities and CRACs.
    Supports AWS Cloud and Local offline modes.
    """

    TELEMETRY_TOPIC_TEMPLATE = "datacenter/cooling/telemetry/{facility_id}/{crac_id}"
    CONTROL_TOPIC_TEMPLATE = "datacenter/cooling/control/{crac_id}"

    def __init__(
        self,
        topology: Optional[List[Dict[str, str]]] = None,
        publish_interval_s: float = 1.0,
        aws_endpoint: Optional[str] = None,
        aws_region: str = "us-east-1",
        day_steps: int = 144,
    ):
        # Default topology: 3 facilities × 4 CRACs, matching postgres_schema.sql seed data.
        # ambient_c and grid_carbon_gco2_kwh are derived from REGIONAL_CLIMATE_BASE and
        # FACILITY_GRID_CARBON_GCOEKWH so there is a single source of truth for these values.
        _rcb = REGIONAL_CLIMATE_BASE
        _gcc = FACILITY_GRID_CARBON_GCOEKWH
        self.topology = topology or [
            # DC-EAST-01 (US East, mid-Atlantic — warmest, highest grid carbon)
            {"facility_id": "DC-EAST-01", "crac_id": "CRAC-01", "rack_id": "RACK-A01",
             "ambient_c": _rcb["DC-EAST-01"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-EAST-01"]},
            {"facility_id": "DC-EAST-01", "crac_id": "CRAC-02", "rack_id": "RACK-E01",
             "ambient_c": _rcb["DC-EAST-01"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-EAST-01"]},
            {"facility_id": "DC-EAST-01", "crac_id": "CRAC-03", "rack_id": "RACK-A05",
             "ambient_c": _rcb["DC-EAST-01"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-EAST-01"]},
            {"facility_id": "DC-EAST-01", "crac_id": "CRAC-04", "rack_id": "RACK-E05",
             "ambient_c": _rcb["DC-EAST-01"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-EAST-01"]},
            # DC-WEST-02 (US West, Pacific Northwest — coolest US site, high renewables)
            {"facility_id": "DC-WEST-02", "crac_id": "CRAC-01", "rack_id": "RACK-A01",
             "ambient_c": _rcb["DC-WEST-02"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-WEST-02"]},
            {"facility_id": "DC-WEST-02", "crac_id": "CRAC-02", "rack_id": "RACK-E01",
             "ambient_c": _rcb["DC-WEST-02"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-WEST-02"]},
            {"facility_id": "DC-WEST-02", "crac_id": "CRAC-03", "rack_id": "RACK-A05",
             "ambient_c": _rcb["DC-WEST-02"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-WEST-02"]},
            {"facility_id": "DC-WEST-02", "crac_id": "CRAC-04", "rack_id": "RACK-E05",
             "ambient_c": _rcb["DC-WEST-02"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-WEST-02"]},
            # DC-EU-01 (EU Frankfurt — lowest ambient, mixed-clean grid)
            {"facility_id": "DC-EU-01", "crac_id": "CRAC-01", "rack_id": "RACK-A01",
             "ambient_c": _rcb["DC-EU-01"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-EU-01"]},
            {"facility_id": "DC-EU-01", "crac_id": "CRAC-02", "rack_id": "RACK-E01",
             "ambient_c": _rcb["DC-EU-01"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-EU-01"]},
            {"facility_id": "DC-EU-01", "crac_id": "CRAC-03", "rack_id": "RACK-A05",
             "ambient_c": _rcb["DC-EU-01"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-EU-01"]},
            {"facility_id": "DC-EU-01", "crac_id": "CRAC-04", "rack_id": "RACK-E05",
             "ambient_c": _rcb["DC-EU-01"]["base_dry_bulb"], "grid_carbon_gco2_kwh": _gcc["DC-EU-01"]},
        ]
        self.publish_interval_s = publish_interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Initialise physics simulators; one per (facility_id, crac_id) pair.
        # Regional ambient_c and grid_carbon_gco2_kwh come from the topology entry
        # (which is pre-populated from REGIONAL_CLIMATE_BASE / FACILITY_GRID_CARBON_GCOEKWH
        # for the default topology, or from the caller-supplied topology dict).
        self._simulators: Dict[str, PhysicsSimulator] = {}
        for t in self.topology:
            key = f"{t['facility_id']}:{t['crac_id']}"
            self._simulators[key] = PhysicsSimulator(
                facility_id=t["facility_id"],
                crac_id=t["crac_id"],
                rack_id=t["rack_id"],
                ambient_c=float(t.get("ambient_c", 22.0)),
                grid_carbon_gco2_kwh=float(
                    t.get("grid_carbon_gco2_kwh",
                          FACILITY_GRID_CARBON_GCOEKWH.get(t["facility_id"], 320.0))
                ),
                day_steps=day_steps,
            )

        # Publisher: AWS Cloud or Local
        self._aws_publisher: Optional[AWSIoTPublisher] = None
        if aws_endpoint:
            self._aws_publisher = AWSIoTPublisher(aws_endpoint, aws_region)

    def _build_topic(self, template: str, **kwargs) -> str:
        return template.format(**kwargs)

    def _publish_payload(self, topic: str, payload_json: str) -> None:
        success_aws = False
        if self._aws_publisher:
            success_aws = self._aws_publisher.publish(topic, payload_json)
            if not success_aws:
                logger.warning("IoT Core publish failed for %s; telemetry still delivered locally", topic)
        payload_dict = json.loads(payload_json)
        # Always publish to local bus regardless
        try:
            local_bus.publish(topic, payload_dict)
        except Exception as e:
            logger.debug("Local bus publish error: %s", e)
        # Persist to Timestream (or its in-memory fallback) so history/analytics
        # queries have data to serve even when running without a real AWS account.
        try:
            from database.timestream_client import get_timestream_client
            get_timestream_client().write_telemetry(payload_dict)
        except Exception as e:
            logger.debug("Timestream write error: %s", e)

    def apply_control_action(
        self,
        crac_id: str,
        control: Dict[str, float],
        facility_id: Optional[str] = None,
    ) -> None:
        """Apply a control action to the simulator for the given CRAC.

        Args:
            crac_id:     CRAC unit ID (e.g. 'CRAC-01').
            control:     Dict of actuator setpoints.
            facility_id: Optional facility scope.  When provided only the
                         simulator matching that exact (facility_id, crac_id)
                         pair is updated — necessary when multiple facilities
                         share the same crac_id values (e.g. every facility
                         has a 'CRAC-01').  When omitted the first simulator
                         whose crac_id matches is updated (backward-compat
                         with single-facility usage).
        """
        for key, sim in self._simulators.items():
            if facility_id is not None:
                if sim.facility_id == facility_id and sim.crac_id == crac_id:
                    sim.apply_control(control)
                    return
            else:
                if sim.crac_id == crac_id:
                    sim.apply_control(control)
                    return

    def _run_loop(self) -> None:
        logger.info("IoT Simulator started. Publishing every %.1fs", self.publish_interval_s)
        while self._running:
            for key, sim in self._simulators.items():
                try:
                    payload = sim.step()
                    topic = self._build_topic(
                        self.TELEMETRY_TOPIC_TEMPLATE,
                        facility_id=payload.facility_id,
                        crac_id=payload.crac_id,
                    )
                    self._publish_payload(topic, payload.to_json())
                    logger.debug("Published to %s | PUE=%.3f | inlet=%.1f°C",
                                 topic, payload.pue, payload.server_inlet_temp_c)
                except Exception as e:
                    logger.error("Simulation step error for %s: %s", key, e)
            time.sleep(self.publish_interval_s)
        logger.info("IoT Simulator stopped.")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="iot-simulator")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def get_latest_telemetry(self) -> Dict[str, Dict]:
        return local_bus.get_latest()

    def get_simulator_state(self, facility_id: str, crac_id: str) -> Optional[PhysicsSimulator]:
        key = f"{facility_id}:{crac_id}"
        return self._simulators.get(key)


# ---------------------------------------------------------------------------
# Factory & default instance
# ---------------------------------------------------------------------------

def create_simulator_from_env() -> IoTSimulator:
    """Creates IoT simulator configured from environment variables."""
    topology_str = os.environ.get("IOT_TOPOLOGY", "")
    topology = json.loads(topology_str) if topology_str else None
    interval = float(os.environ.get("IOT_PUBLISH_INTERVAL_S", "1.0"))
    aws_endpoint = os.environ.get("AWS_IOT_ENDPOINT", None)
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    return IoTSimulator(
        topology=topology,
        publish_interval_s=interval,
        aws_endpoint=aws_endpoint,
        aws_region=aws_region,
        day_steps=int(os.environ.get("SIM_DAY_STEPS", "144")),
    )


# Module-level singleton used by FastAPI backend
_default_simulator: Optional[IoTSimulator] = None


def get_simulator() -> IoTSimulator:
    global _default_simulator
    if _default_simulator is None:
        _default_simulator = create_simulator_from_env()
    return _default_simulator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sim = IoTSimulator(publish_interval_s=2.0)
    sim.start()
    try:
        while True:
            time.sleep(10)
            latest = sim.get_latest_telemetry()
            for topic, payload in latest.items():
                print(f"[{topic}] PUE={payload.get('pue'):.3f} | inlet={payload.get('server_inlet_temp_c'):.1f}°C")
    except KeyboardInterrupt:
        sim.stop()
