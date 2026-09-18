"""
AWS IoT Core MQTT Publisher & Physics-Grounded Telemetry Simulator.

Dual-mode operation:
  - AWS Cloud Mode: Publishes to AWS IoT Core over MQTT/TLS using boto3 IoT Data Plane.
  - Local Offline Mode: Publishes to an in-memory async queue consumed by the FastAPI backend.

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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, BotoCoreError

logger = logging.getLogger(__name__)


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
        it_power_kw: float = 18.0,
    ):
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
        self.carbon_gco2_kwh = 320.0
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

    def step(self) -> TelemetryPayload:
        self._step += 1
        hour = (self._step % 144) * (24.0 / 144.0)

        # Ambient & IT load dynamics
        self.ambient_c = float(
            max(-5.0, min(45.0, self.ambient_c
                + math.sin(2 * math.pi * (hour - 8) / 24.0) * 0.4
                + random.gauss(0, 0.15)))
        )
        self.it_power_kw = float(
            max(10.0, min(28.0, self.it_power_kw
                + math.sin(2 * math.pi * (hour - 9) / 24.0) * 200.0
                + random.gauss(0, 100.0)))
        )
        self.carbon_gco2_kwh = float(
            max(120.0, min(580.0, self.carbon_gco2_kwh + random.gauss(0, 10.0)))
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

        # Thermal balance: Q = m_dot * Cp * Delta_T
        flow_lpm = 2000.0 + (self.pump_pct / 100.0) * 5500.0
        flow_kg_s = (flow_lpm / 60.0) * self.WATER_DENSITY
        delta_t_water = self.it_power_kw / max(0.1, flow_kg_s * self.SPECIFIC_HEAT_WATER)
        return_c = self.supply_c + delta_t_water

        # Rack inlet: mixing of supply and ambient
        free_cool_frac = max(0.0, min(1.0, (self.valve_split_pct / 100.0) * (1.0 - max(0.0, (self.ambient_c - 18.0) / 20.0))))
        server_inlet_c = (1.0 - free_cool_frac) * self.supply_c + free_cool_frac * min(self.ambient_c, 22.0)
        server_outlet_c = server_inlet_c + (self.it_power_kw / max(0.1, flow_lpm / 60.0 * 0.25))

        # Power calculations
        pump_kw = self._cubic_pump_power(self.pump_pct)
        fan_kw = self._fan_power(self.fan_pct)
        cop = self._chiller_cop(self.ambient_c)
        chiller_load_kw = self.it_power_kw * (1.0 - free_cool_frac)
        chiller_kw = chiller_load_kw / cop
        cooling_kw = pump_kw + fan_kw + chiller_kw
        total_kw = self.it_power_kw + cooling_kw
        pue = total_kw / max(0.01, self.it_power_kw)

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
    """Publishes MQTT messages via AWS IoT Core Data Plane using boto3."""

    def __init__(
        self,
        endpoint: str,
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.region = region
        self._client = boto3.client(
            "iot-data",
            endpoint_url=f"https://{endpoint}",
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

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
    ):
        # Default topology: 1 facility, 2 CRACs, 1 rack each
        self.topology = topology or [
            {"facility_id": "DC-EAST-01", "crac_id": "CRAC-01", "rack_id": "RACK-A01"},
            {"facility_id": "DC-EAST-01", "crac_id": "CRAC-02", "rack_id": "RACK-B01"},
        ]
        self.publish_interval_s = publish_interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Initialise physics simulators
        self._simulators: Dict[str, PhysicsSimulator] = {}
        for t in self.topology:
            key = f"{t['facility_id']}:{t['crac_id']}"
            self._simulators[key] = PhysicsSimulator(
                facility_id=t["facility_id"],
                crac_id=t["crac_id"],
                rack_id=t["rack_id"],
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
        # Always publish to local bus regardless
        try:
            local_bus.publish(topic, json.loads(payload_json))
        except Exception as e:
            logger.debug("Local bus publish error: %s", e)

    def apply_control_action(self, crac_id: str, control: Dict[str, float]) -> None:
        for key, sim in self._simulators.items():
            if sim.crac_id == crac_id:
                sim.apply_control(control)
                break

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
