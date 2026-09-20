"""
Live inference for the 2D Fourier Neural Operator thermal surrogate.

The surrogate is trained (scripts/train_fno_pde.py) to reproduce the 2D advection-diffusion transport
solver in src/digital_twin/thermal_solver.py: given the per-rack IT load, the supply-temperature field and
the per-rack coolant flow of one CRAC zone it returns the 8x8 rack coolant-temperature field, ~18x faster
than the solver at 128x128 resolution (results/fno_pde_eval.json).

Caveat (also in docs/RESULTS.md): the solver is a 2D reduced-order model, not 3D CFD, and Frontier2023 has no
per-rack temperature sensors, so the surrogate is validated against the solver, not against measured rack
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
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fno_pde_v1.pt")
STATS_PATH = os.path.join(PROJECT_ROOT, "models", "fno_pde_normalization_stats.json")

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
        if self._tried:          # only set AFTER loading finished (see below)
            return
        with self._lock:         # concurrent first requests wait here instead of seeing a half-loaded service
            if self._tried:
                return
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
            finally:
                self._tried = True

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def build_input(self, it_zone_mw: float, supply_c: float, flow_lpm: float) -> np.ndarray:
        from src.digital_twin.thermal_solver import rack_inputs
        return rack_inputs(it_zone_mw, supply_c, flow_lpm)

    def predict(self, it_zone_mw: float, supply_c: float, flow_lpm: float) -> Dict[str, Any]:
        """Returns the rack coolant-temperature field (deg C, 8x8) and inference latency."""
        if not self.available:
            raise RuntimeError(self._error or "FNO surrogate not loaded")
        import torch

        lo = np.asarray(self._stats["input_min"], dtype=np.float32).reshape(3, 1, 1)
        hi = np.asarray(self._stats["input_max"], dtype=np.float32).reshape(3, 1, 1)
        lo_y, hi_y = float(self._stats["target_min"]), float(self._stats["target_max"])
        x = np.clip((self.build_input(it_zone_mw, supply_c, flow_lpm) - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        t0 = time.perf_counter()
        with torch.no_grad():
            y = self._model(torch.tensor(x[None])).numpy()[0, 0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        temp = y * (hi_y - lo_y + 1e-6) + lo_y
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
