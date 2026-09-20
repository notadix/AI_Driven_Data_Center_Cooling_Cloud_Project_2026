"""
Synchronised (data-assimilating) digital twin.

The calibrated physics model is a static map from operating point to plant
response, so it cannot follow effects the physics leaves out (thermal inertia,
staging of pumps and towers). A digital twin that is *synchronised* with its
plant corrects the physics with the plant's latest measurements. Here that is a
ridge regression on

    the last `LAGS` measured values of the quantity,
    the physics prediction now and one step ago,
    the operating point now and its change since the last step,

predicting the value at the next 10-minute step. It is evaluated one step ahead
on held-out data against the honest baseline "repeat the last measurement"
(persistence), because a synchronised twin that cannot beat persistence adds
nothing.

Measured on the held-out 30% of Frontier2023 (results/twin_fidelity.json):
inlet temperature, outlet temperature and PUE reach well below the report's ~2%
MAPE; return temperature and cooling power do NOT beat persistence (their
10-minute fluctuations are not explained by any signal available in the data),
so for those two the twin is no better than repeating the last reading.
"""

from typing import Dict, List

import numpy as np
from sklearn.linear_model import Ridge

LAGS = 3
TARGETS = ["server_inlet_temp_c", "server_outlet_temp_c", "pue", "return_temp_c", "cooling_power_kw"]


def consecutive_rows(timestamps, lags: int = LAGS) -> np.ndarray:
    """Indices t whose previous `lags` rows are exactly 10 minutes apart."""
    ts = np.asarray(timestamps).astype("datetime64[s]")
    dt = np.diff(ts).astype(int)
    n = len(ts)
    ok = np.zeros(n, dtype=bool)
    for t in range(lags, n):
        ok[t] = bool(np.all(dt[t - lags:t] == 600))
    return np.where(ok)[0]


def _features(y: np.ndarray, phys: np.ndarray, x_in: np.ndarray, idx: np.ndarray) -> np.ndarray:
    lags = np.column_stack([y[idx - j] for j in range(1, LAGS + 1)])
    return np.column_stack([lags, phys[idx], phys[idx - 1], x_in[idx], x_in[idx] - x_in[idx - 1]])


class SyncedTwin:
    """One ridge model per target quantity."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models: Dict[str, Ridge] = {}
        self._mu: Dict[str, np.ndarray] = {}
        self._sd: Dict[str, np.ndarray] = {}

    def fit(self, truth: Dict[str, np.ndarray], phys: Dict[str, np.ndarray], x_in: np.ndarray,
            idx: np.ndarray, targets: List[str] = TARGETS) -> "SyncedTwin":
        for k in targets:
            F = _features(truth[k], phys[k], x_in, idx)
            self._mu[k], self._sd[k] = F.mean(0), F.std(0) + 1e-9
            self.models[k] = Ridge(alpha=self.alpha).fit((F - self._mu[k]) / self._sd[k], truth[k][idx])
        return self

    def predict(self, k: str, truth: Dict[str, np.ndarray], phys: Dict[str, np.ndarray], x_in: np.ndarray,
                idx: np.ndarray) -> np.ndarray:
        F = _features(truth[k], phys[k], x_in, idx)
        return self.models[k].predict((F - self._mu[k]) / self._sd[k])


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true) / np.abs(true)) * 100.0)
