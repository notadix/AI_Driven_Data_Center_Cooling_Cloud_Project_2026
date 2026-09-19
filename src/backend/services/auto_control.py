"""
Auto-control loop — closes the gap between the 'auto' operator mode and
anything actually driving the CRACs while in it.

Previously, 'auto' mode only gated manual setpoint overrides (see
api/v1/control.py); nothing ever computed or applied a control action on
its own. A CRAC left in 'auto' mode never changed its setpoints unless a
human (or a test) explicitly POSTed one -- the "AUTO (Safe-PPO)" label in
the operator UI had no controller behind it at all.

This loop periodically applies an action to every CRAC currently in
'auto' mode:
  - Uses the trained Safe-PPO checkpoint at models/safe_ppo_agent_v1.pt
    if one has been produced by src/ai/rl/train_rl.py.
  - Falls back to the deterministic PID baseline controller
    (src/ai/rl/reward_functions.BaselineControllers.pid) otherwise, since
    no trained checkpoint ships in source control.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from src.digital_twin.physics_dynamics import ZONE_SCALE

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "safe_ppo_agent_v1.pt")


def _build_obs(state: Any, payload: Dict[str, Any]) -> Optional[np.ndarray]:
    """Builds the 10-dim observation vector matching cooling_sim_env.py's
    ordering, combining live PhysicsSimulator state (for ambient_c, which
    isn't part of the published telemetry payload) with the latest
    published telemetry (for everything else).

    The policy is trained on the hall-scale Gymnasium environment (IT load
    ~10-28 MW), while each simulated CRAC reports ONE representative rack
    (~10-28 kW). Extensive quantities (IT power, cooling power) are therefore
    multiplied by ZONE_SCALE so the agent sees the operating range it was
    trained on; intensive quantities (temperatures, flow, PUE) are passed
    through unchanged."""
    try:
        return np.array(
            [
                float(state.it_power_kw) * ZONE_SCALE,
                float(state.ambient_c),
                float(payload["grid_carbon_gco2_kwh"]),
                float(payload["fws_supply_temp_c"]),
                float(payload["return_temp_c"]),
                float(payload["flow_rate_lpm"]),
                float(payload["server_inlet_temp_c"]),
                float(payload["server_outlet_temp_c"]),
                float(payload["cooling_power_mw"]) * 1000.0 * ZONE_SCALE,
                float(payload["pue"]),
            ],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.debug("Auto-control: could not build observation: %s", e)
        return None


def _action_to_control(a: np.ndarray) -> Dict[str, float]:
    """Maps a [-1, 1]^4 action to the same physical setpoint ranges the
    Gymnasium env and the ControlAction API model use."""
    return {
        "delta_supply_c": float(np.clip(a[0] * 1.5, -2.0, 2.0)),
        "pump_speed_pct": float(np.clip(35.0 + (a[1] + 1.0) * 0.5 * 65.0, 35.0, 100.0)),
        "fan_speed_pct": float(np.clip(30.0 + (a[2] + 1.0) * 0.5 * 70.0, 30.0, 100.0)),
        # Same 0-40% range the environment maps this action to during training
        # (cooling_sim_env.py, sagemaker/model_handler.py); it used to be 0-100%.
        "valve_split_pct": float(np.clip((a[3] + 1.0) * 0.5 * 40.0, 0.0, 40.0)),
    }


class AutoControlLoop:
    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._shield = None
        from src.backend.services.sensor_guard import SensorGuard

        self._guard = SensorGuard()
        # Per-CRAC online calibration of the shield's inlet-temperature model (plant drift).
        self._calibrators: Dict[Any, Any] = {}
        self._pending_pred: Dict[Any, float] = {}
        self._agent = self._try_load_agent()

    def _try_load_agent(self):
        if not os.path.exists(CHECKPOINT_PATH):
            logger.info(
                "No trained Safe-PPO checkpoint at %s; auto mode will use the PID baseline controller.",
                CHECKPOINT_PATH,
            )
            return None
        try:
            import torch
            from src.ai.rl.safe_ppo import SafePPOAgent

            # weights_only=False: always this project's own locally-trained checkpoint.
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            hp = ckpt.get("hyperparams", {"state_dim": 10, "action_dim": 4})
            agent = SafePPOAgent(state_dim=hp["state_dim"], action_dim=hp["action_dim"], device="cpu")
            agent.ac.load_state_dict(ckpt["ac_state_dict"])
            agent.ac.eval()
            if ckpt.get("shield", False):
                from src.ai.rl.safety_shield import SafetyShield

                self._shield = SafetyShield()
            logger.info(
                "Auto-control loop loaded Safe-PPO checkpoint: %s (safety shield: %s)",
                CHECKPOINT_PATH, "on" if self._shield else "off",
            )
            return agent
        except Exception as e:
            logger.warning("Failed to load Safe-PPO checkpoint (%s); falling back to PID baseline.", e)
            return None

    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        """The policy's proposal, vetoed by the model-based safety shield when the
        checkpoint was trained with one (the size of the edit is kept in
        self._last_correction so the action can be logged as SHIELDED)."""
        if self._agent is not None:
            action, *_ = self._agent.select_action(obs, det=True)
            if self._shield is not None:
                action, correction = self._shield.filter(obs, action)
                self._last_correction = correction
            else:
                self._last_correction = 0.0
            return action
        from src.ai.rl.reward_functions import BaselineControllers

        self._last_correction = 0.0
        return BaselineControllers.pid(obs)

    def compute_control(self, facility_id: str, crac_id: str, state: Any, payload: Dict[str, Any]):
        """One control decision for one CRAC. Returns (control, source, safety_status) or None.

        The telemetry is validated first (SensorGuard); a corrupt payload is repaired from the last
        good reading, and if the fault persists (or there is no good reading to repair from) the
        conservative PID baseline drives the CRAC instead of the learned policy."""
        clean, flags, fallback = self._guard.validate(facility_id, crac_id, payload)
        obs = _build_obs(state, clean)
        if obs is None:
            return None
        if flags:
            logger.warning("Sensor fault on %s/%s: %s%s", facility_id, crac_id, ",".join(flags),
                           " -> PID fallback" if fallback else " (repaired)")
        if fallback:
            from src.ai.rl.reward_functions import BaselineControllers

            action = BaselineControllers.pid(obs)
            self._last_correction = 0.0
            return _action_to_control(action), "baseline_pid", "SENSOR_FAULT"

        key = (facility_id, crac_id)
        if self._shield is not None:
            from src.ai.rl.safety_shield import OnlineInletCalibrator

            cal = self._calibrators.setdefault(key, OnlineInletCalibrator())
            # Learn from the last action's outcome -- but never from a corrupt reading.
            if not flags and key in self._pending_pred:
                cal.update(float(clean["server_inlet_temp_c"]), self._pending_pred[key])
            self._shield.calibrator = cal

        action = self._select_action(obs)
        if self._shield is not None:
            self._pending_pred[key] = float(self._shield.predict_inlet(float(obs[3]), float(obs[1]), action[0], action[3]))
        source = "rl_agent" if self._agent else "baseline_pid"
        if flags:
            status = "SENSOR_REPAIRED"
        elif getattr(self, "_last_correction", 0.0) > 0.0:
            status = "SHIELDED"
        else:
            status = "NORMAL"
        return _action_to_control(action), source, status

    async def _run_loop(self) -> None:
        from src.aws.iot.iot_publisher import get_simulator, local_bus
        from src.backend.api.v1.control import get_crac_mode, record_action

        sim = get_simulator()
        while self._running:
            await asyncio.sleep(self.interval_s)
            for t in sim.topology:
                facility_id = t["facility_id"]
                crac_id = t["crac_id"]
                # Per-facility-CRAC isolation: check mode for this exact pair
                if get_crac_mode(crac_id, facility_id=facility_id) != "auto":
                    continue
                state = sim.get_simulator_state(facility_id, crac_id)
                topic = f"datacenter/cooling/telemetry/{facility_id}/{crac_id}"
                payload = local_bus.get_latest(topic)
                if state is None or not payload:
                    continue
                try:
                    decision = self.compute_control(facility_id, crac_id, state, payload)
                    if decision is None:
                        continue
                    control, source, status = decision
                    sim.apply_control_action(crac_id, control, facility_id=facility_id)
                    record_action(
                        crac_id, control, source=source, safety_status=status, facility_id=facility_id,
                    )
                except Exception as e:
                    logger.error("Auto-control step failed for %s/%s: %s", facility_id, crac_id, e)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._run_loop())
        logger.info(
            "Auto-control loop started (interval=%.1fs, controller=%s)",
            self.interval_s,
            "Safe-PPO" if self._agent else "PID baseline",
        )

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()


_default_loop: Optional[AutoControlLoop] = None


def get_auto_control_loop() -> AutoControlLoop:
    global _default_loop
    if _default_loop is None:
        interval = float(os.environ.get("AUTO_CONTROL_INTERVAL_S", "2.0"))
        _default_loop = AutoControlLoop(interval_s=interval)
    return _default_loop
