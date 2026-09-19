"""
Model-based safety shield.

The RL policy proposes an action; before it is applied the shield predicts,
with the calibrated twin physics, the rack-inlet temperature the action would
produce, and replaces the action with the CLOSEST safe one whenever the
prediction leaves the SLA envelope (ASHRAE TC9.9 recommended 18-27 C, with a
margin to absorb one step of ambient / load drift).

This turns the SLA from a learned soft penalty (which the Lagrangian method
only satisfies on average, and which failed to converge for most seeds) into a
constraint enforced at run time, as the project report asks ("hard safety
guarantees ... no constraint violations"). The guarantee is only as good as the
twin: it holds exactly inside the environment and to within the calibration
error on the live simulator.

Only the supply-temperature and free-air valve actions influence inlet
temperature in the physics (pump and fan speeds change energy, not inlet), so
they are the only components the shield edits.
"""

from typing import Tuple

import numpy as np

try:  # scripts run with src/digital_twin on sys.path; the backend imports it as a package
    from physics_dynamics import CoolingConstants, load_calibrated_constants
except ImportError:  # pragma: no cover
    from src.digital_twin.physics_dynamics import CoolingConstants, load_calibrated_constants

INLET_MARGIN_LOW_C = 0.5     # keep inlet >= 18.5 C
INLET_MARGIN_HIGH_C = 1.0    # keep inlet <= 26.0 C


class OnlineInletCalibrator:
    """Tracks the (slowly varying) gap between measured and twin-predicted rack inlet
    temperature, so the shield's model follows plant drift (fouling, sensor bias,
    component degradation) instead of silently trusting a stale calibration.

    Each update moves the bias estimate by `gain` x the innovation (measured minus
    the prediction that already included the current bias). Innovations are clipped
    so a single bad reading cannot move the estimate by more than `max_step` degC."""

    def __init__(self, gain: float = 0.5, max_step: float = 2.0, max_bias: float = 8.0):
        self.gain = gain
        self.max_step = max_step
        self.max_bias = max_bias
        self.bias_c = 0.0

    def update(self, measured_inlet_c: float, predicted_inlet_c: float) -> float:
        innovation = float(np.clip(measured_inlet_c - predicted_inlet_c, -self.max_step, self.max_step))
        self.bias_c = float(np.clip(self.bias_c + self.gain * innovation, -self.max_bias, self.max_bias))
        return self.bias_c


class SafetyShield:
    def __init__(self, constants: CoolingConstants = None,
                 margin_low_c: float = INLET_MARGIN_LOW_C, margin_high_c: float = INLET_MARGIN_HIGH_C,
                 calibrator: "OnlineInletCalibrator" = None):
        self.c = constants or load_calibrated_constants()
        self.calibrator = calibrator
        self.lo = self.c.SLA_MIN_INLET_C + margin_low_c
        self.hi = self.c.SLA_MAX_INLET_C - margin_high_c
        self._grid = np.linspace(-1.0, 1.0, 81)

    # -- prediction ------------------------------------------------------
    def predict_inlet(self, supply_c: float, ambient_c: float, a0, a3):
        """Inlet temperature after applying (supply delta action a0, valve action a3).
        Mirrors DataCenterCoolingEnv._obs / PhysicsSimulator.step; vectorised over a0."""
        c = self.c
        supply = np.clip(supply_c + np.asarray(a0) * 1.5, 14.0, 24.0)
        valve_pct = (np.asarray(a3) + 1.0) * 0.5 * 40.0
        base_inlet = supply + c.INLET_OFFSET_C + c.INLET_AMBIENT_COEF * max(0.0, ambient_c - 20.0)
        free = np.clip((valve_pct / 100.0) * (1.0 - max(0.0, (ambient_c - 18.0) / 20.0)), 0.0, 1.0)
        mixed = (1.0 - free) * base_inlet + free * min(ambient_c, 22.0)
        bias = self.calibrator.bias_c if self.calibrator is not None else 0.0
        return np.maximum(supply, mixed) + bias

    # -- shield ----------------------------------------------------------
    def is_safe(self, obs: np.ndarray, action: np.ndarray) -> bool:
        inlet = float(self.predict_inlet(float(obs[3]), float(obs[1]), action[0], action[3]))
        return self.lo <= inlet <= self.hi

    def filter(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """Returns (safe_action, correction) where correction = L1 distance moved."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        supply_c, ambient_c = float(obs[3]), float(obs[1])
        if self.is_safe(obs, action):
            return action, 0.0

        # Search the supply-delta action closest to the proposal that is predicted safe.
        inlet = self.predict_inlet(supply_c, ambient_c, self._grid, action[3])
        ok = (inlet >= self.lo) & (inlet <= self.hi)
        safe = action.copy()
        if ok.any():
            cand = self._grid[ok]
            safe[0] = cand[np.argmin(np.abs(cand - action[0]))]
        else:
            # Nothing in the grid is safe with this valve setting: try every valve setting too.
            best = None
            for a3 in np.linspace(-1.0, 1.0, 9):
                inlet = self.predict_inlet(supply_c, ambient_c, self._grid, a3)
                ok = (inlet >= self.lo) & (inlet <= self.hi)
                if ok.any():
                    cand = self._grid[ok]
                    a0 = cand[np.argmin(np.abs(cand - action[0]))]
                    cost = abs(a0 - action[0]) + abs(a3 - action[3])
                    if best is None or cost < best[0]:
                        best = (cost, a0, a3)
            if best is not None:
                safe[0], safe[3] = best[1], best[2]
            else:
                # Emergency: most cooling / coldest supply if too hot, otherwise warmest.
                too_hot = float(self.predict_inlet(supply_c, ambient_c, action[0], action[3])) > self.hi
                safe[0] = -1.0 if too_hot else 1.0
                safe[3] = -1.0 if too_hot else safe[3]
        return safe.astype(np.float32), float(np.abs(safe - action).sum())


try:
    import gymnasium as gym

    class ShieldedEnv(gym.Wrapper):
        """Applies the SafetyShield to every action before it reaches the environment.

        info["shield_correction"] is the L1 size of the edit (0 when the policy's
        action was already safe); a small penalty discourages relying on the shield
        so the learned policy stays close to safe on its own.
        """

        CORRECTION_PENALTY = 0.05

        def __init__(self, env, shield: SafetyShield = None, adapt: bool = False):
            """adapt=True attaches an OnlineInletCalibrator so the shield tracks plant drift."""
            super().__init__(env)
            calibrator = OnlineInletCalibrator() if adapt else None
            self.shield = shield or SafetyShield(env.unwrapped.physics.c, calibrator=calibrator)
            self._obs = None
            self._predicted_inlet = None

        def reset(self, **kwargs):
            self._obs, info = self.env.reset(**kwargs)
            return self._obs, info

        def step(self, action):
            safe, correction = self.shield.filter(self._obs, action)
            predicted = float(self.shield.predict_inlet(float(self._obs[3]), float(self._obs[1]), safe[0], safe[3]))
            obs, reward, term, trunc, info = self.env.step(safe)
            self._obs = obs
            if self.shield.calibrator is not None:
                # Learn from the measured inlet how far the plant has drifted from the twin.
                self.shield.calibrator.update(float(info["inlet_c"]), predicted)
                info["shield_bias_c"] = self.shield.calibrator.bias_c
            info["shield_correction"] = correction
            info["shield_active"] = bool(correction > 0.0)
            return obs, reward - self.CORRECTION_PENALTY * correction, term, trunc, info
except ImportError:  # gymnasium is only needed for training / evaluation
    ShieldedEnv = None
