"""
Sensor-fault guard for the control loop.

Real telemetry contains dropouts, NaNs, out-of-range readings and spikes. A
policy fed a corrupt observation can output a dangerous action, so every
payload is validated before it reaches the controller:

  * missing / non-numeric / non-finite fields        -> flagged "missing"
  * physically impossible values (range check)      -> replaced by last good, flagged "range"
  * implausible step changes (spike / rate check)   -> replaced by last good, flagged "spike"

A reading is replaced with the last good value for that (facility, CRAC) so the
observation stays well-formed, and after `max_bad_steps` consecutive bad
payloads the loop is told to fall back to the conservative rule-based
controller instead of trusting a stale, repaired observation.
"""

import math
from typing import Any, Dict, List, Tuple

# field -> (min, max, max absolute change per step)
FIELD_LIMITS: Dict[str, Tuple[float, float, float]] = {
    "grid_carbon_gco2_kwh": (0.0, 1500.0, 250.0),
    "fws_supply_temp_c": (0.0, 45.0, 6.0),
    "return_temp_c": (0.0, 90.0, 20.0),
    "flow_rate_lpm": (0.0, 40000.0, 15000.0),
    "server_inlet_temp_c": (0.0, 60.0, 8.0),
    "server_outlet_temp_c": (0.0, 90.0, 20.0),
    "cooling_power_mw": (0.0, 1.0, 0.5),
    "pue": (1.0, 3.5, 1.0),
}


class SensorGuard:
    def __init__(self, max_bad_steps: int = 3):
        self.max_bad_steps = max_bad_steps
        self._last_good: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._bad_streak: Dict[Tuple[str, str], int] = {}
        self._last_raw: Dict[Tuple[str, str], Dict[str, float]] = {}

    def validate(self, facility_id: str, crac_id: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], bool]:
        """Returns (clean_payload, flags, fallback). `fallback` is True once the fault has persisted
        long enough (or no good reading exists yet) that the controller must not trust the data."""
        key = (facility_id, crac_id)
        last = self._last_good.get(key)
        clean = dict(payload)
        flags: List[str] = []

        for name, (lo, hi, max_step) in FIELD_LIMITS.items():
            raw = payload.get(name)
            try:
                v = float(raw)
                ok_number = math.isfinite(v)
            except (TypeError, ValueError):
                ok_number = False

            if not ok_number:
                flags.append(f"missing:{name}")
                if last is not None and name in last:
                    clean[name] = last[name]
                continue
            if v < lo or v > hi:
                flags.append(f"range:{name}")
                if last is not None and name in last:
                    clean[name] = last[name]
                continue
            if last is not None and name in last and abs(v - last[name]) > max_step:
                # A single-step jump is a spike; but if the previous raw reading was already at the
                # new level the plant really moved (a genuine step change) -- accept it.
                prev_raw = self._last_raw.get(key, {}).get(name)
                if prev_raw is None or abs(v - prev_raw) > max_step:
                    flags.append(f"spike:{name}")
                    clean[name] = last[name]
                    self._last_raw.setdefault(key, {})[name] = v
                    continue
            self._last_raw.setdefault(key, {})[name] = v
            clean[name] = v

        if flags:
            streak = self._bad_streak.get(key, 0) + 1
            self._bad_streak[key] = streak
        else:
            self._bad_streak[key] = 0
            self._last_good[key] = {n: float(clean[n]) for n in FIELD_LIMITS}
            streak = 0

        no_history = last is None and any(f.startswith(("missing", "range")) for f in flags)
        fallback = no_history or streak >= self.max_bad_steps
        return clean, flags, fallback
