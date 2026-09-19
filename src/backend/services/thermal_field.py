"""
Live inference for the 2D Fourier Neural Operator thermal surrogate.

Builds the FNO's 3-channel input (per-rack workload, supply-temperature field,
per-rack coolant flow) for one CRAC zone exactly the way
dataset/preprocess_telemetry.build_spatial_tensors builds the training data,
normalises it with the training statistics, runs the trained model and returns
the predicted 8x8 rack coolant-temperature field.

Caveat (also in docs/RESULTS.md): the training target is derived analytically
from the same inputs (supply + heat / (flow * cp)); the Frontier2023 data has no
per-rack temperature sensors. The surrogate therefore reproduces that thermal
model 1000x faster than solving it -- it is not validated against measured rack
temperatures.
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fno_surrogate_v1.pt")
STATS_PATH = os.path.join(PROJECT_ROOT, "models", "fno_normalization_stats.json")

GRID = 8
N_RACKS = GRID * GRID


def _hotspot() -> np.ndarray:
    y_g, x_g = np.meshgrid(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID), indexing="ij")
    hot = 1.0 + 0.35 * np.exp(-(x_g ** 2 + y_g ** 2) / 0.8)
    return hot / hot.mean(), x_g, y_g


class ThermalFieldService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._stats: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._tried = False

    def _load(self) -> None:
        if self._tried:
            return
        with self._lock:
            if self._tried:
                return
            self._tried = True
            try:
                import torch
                from src.ai.surrogate.fno_model import FNO2d

                ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
                model = FNO2d(**ckpt["architecture"])
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                with open(STATS_PATH, encoding="utf-8") as f:
                    self._stats = json.load(f)
                self._model = model
            except Exception as e:  # missing checkpoint / stats
                self._error = str(e)
                logger.warning("FNO thermal surrogate unavailable: %s", e)

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def build_input(self, it_zone_mw: float, supply_c: float, flow_lpm: float) -> np.ndarray:
        hot, x_g, y_g = _hotspot()
        ch0 = (it_zone_mw / N_RACKS) * hot
        ch1 = supply_c + 0.05 * (x_g + y_g)
        ch2 = (flow_lpm / N_RACKS) * (1.0 / (hot + 0.1))
        return np.stack([ch0, ch1, ch2]).astype(np.float32)

    def predict(self, it_zone_mw: float, supply_c: float, flow_lpm: float) -> Dict[str, Any]:
        """Returns the rack coolant-temperature field (deg C, 8x8) and inference latency."""
        if not self.available:
            raise RuntimeError(self._error or "FNO surrogate not loaded")
        import torch

        lo = np.asarray(self._stats["channel_min"], dtype=np.float32)
        hi = np.asarray(self._stats["channel_max"], dtype=np.float32)
        x = self.build_input(it_zone_mw, supply_c, flow_lpm)
        x = np.clip((x - lo[:3, None, None]) / (hi[:3, None, None] - lo[:3, None, None] + 1e-6), 0.0, 1.0)
        t0 = time.perf_counter()
        with torch.no_grad():
            y = self._model(torch.tensor(x[None])).numpy()[0, 0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        temp = y * (hi[3] - lo[3] + 1e-6) + lo[3]
        return {
            "field_c": np.round(temp, 3).tolist(),
            "min_c": round(float(temp.min()), 3),
            "max_c": round(float(temp.max()), 3),
            "mean_c": round(float(temp.mean()), 3),
            "latency_ms": round(latency_ms, 3),
        }


_service: Optional[ThermalFieldService] = None


def get_thermal_field_service() -> ThermalFieldService:
    global _service
    if _service is None:
        _service = ThermalFieldService()
    return _service
