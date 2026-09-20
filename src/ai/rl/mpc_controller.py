"""
One-step model-predictive controller on the calibrated twin.

A learned policy is only worth deploying if it beats what a model-based controller
gets from the same physics with no training. This is that controller: at each step
it enumerates supply-temperature and pump-speed actions, predicts with the twin the
rack-inlet temperature and the cooling power each would produce, and applies the
cheapest one the safety shield accepts. Fan speed goes to its minimum (it has no
thermal effect in the model) and the free-air valve is opened as far as is safe.

It is myopic (one step) and uses the same model as the shield, so it is a strong
baseline inside the twin and exactly as trustworthy as the twin.
"""

import numpy as np

try:
    from safety_shield import SafetyShield
except ImportError:  # pragma: no cover
    from src.ai.rl.safety_shield import SafetyShield


class MPCController:
    def __init__(self, env, n_supply: int = 41, n_pump: int = 21):
        self.phys = env.unwrapped.physics
        self.shield = SafetyShield(self.phys.c)
        self.shield.coupling = float(getattr(env.unwrapped, "flow_coupling", 0.0))
        self.a0 = np.linspace(-1.0, 1.0, n_supply)
        self.a1 = np.linspace(-1.0, 1.0, n_pump)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        it_kw, ambient, supply_now = float(obs[0]), float(obs[1]), float(obs[3])
        c = self.phys.c
        a3 = 1.0                                         # free-air valve fully open: cheaper, and the shield clips if unsafe
        A0, A1 = np.meshgrid(self.a0, self.a1, indexing="ij")
        inlet = self.shield.predict_inlet(supply_now, ambient, A0, a3, it_kw, A1)
        safe = (inlet >= self.shield.lo) & (inlet <= self.shield.hi)

        supply = np.clip(supply_now + A0 * 1.5, 14.0, 24.0)
        pump_pct = 35.0 + (A1 + 1.0) * 0.5 * 65.0
        power = np.empty_like(supply)
        for i in range(supply.shape[0]):
            for j in range(supply.shape[1]):
                power[i, j] = self.phys.power_and_pue(it_kw, float(supply[i, j]), float(pump_pct[i, j]), 30.0, ambient)[3]
        free = np.clip(0.4 * (1.0 - max(0.0, (ambient - 18.0) / 20.0)), 0.0, 1.0)
        power = power * (1.0 - 0.3 * free)
        if safe.any():
            power = np.where(safe, power, np.inf)
            i, j = np.unravel_index(int(np.argmin(power)), power.shape)
            return np.array([self.a0[i], self.a1[j], -1.0, a3], dtype=np.float32)
        return np.array([-1.0, 1.0, -1.0, -1.0], dtype=np.float32)      # nothing predicted safe: maximum cooling
