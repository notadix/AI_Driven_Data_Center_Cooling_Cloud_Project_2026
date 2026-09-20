"""
Measure the latency of the closed control loop.

  * decision latency: one CRAC's `AutoControlLoop.compute_control` (sensor guard + observation +
    Safe-PPO policy + safety shield + calibrator), 5,000 calls
  * API latency: control status, telemetry, FNO thermal field and load forecast endpoints
    (in-process ASGI client, so this is compute time without network transit)
  * loop budget: the control period (AUTO_CONTROL_INTERVAL_S, default 2 s) and telemetry period
    (IOT_PUBLISH_INTERVAL_S, default 1 s) that bound end-to-end reaction time

Writes results/control_latency.json.

    python scripts/measure_control_latency.py
"""
import json
import os
import sys
import time
import types

import numpy as np

os.environ.setdefault("LOCAL_MODE", "true")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from src.backend.main import app  # noqa: E402
from src.backend.services.auto_control import AutoControlLoop  # noqa: E402

GOOD = {
    "grid_carbon_gco2_kwh": 300.0, "fws_supply_temp_c": 21.0, "return_temp_c": 30.0, "flow_rate_lpm": 19500.0,
    "server_inlet_temp_c": 23.5, "server_outlet_temp_c": 33.0, "cooling_power_mw": 0.00085, "pue": 1.046,
}


def pct(a, q):
    return round(float(np.percentile(a, q)), 3)


def summary(ms):
    return {"p50_ms": pct(ms, 50), "p95_ms": pct(ms, 95), "p99_ms": pct(ms, 99), "max_ms": round(float(np.max(ms)), 3), "n": len(ms)}


def main() -> None:
    loop = AutoControlLoop()
    state = types.SimpleNamespace(it_power_kw=19.0, ambient_c=22.0)
    for i in range(200):                                    # warm up
        loop.compute_control("F", "C", state, dict(GOOD, timestamp_iso=f"w{i}"))
    ms = []
    for i in range(5000):
        payload = dict(GOOD, server_inlet_temp_c=23.0 + (i % 7) * 0.2, timestamp_iso=f"t{i}")
        t = time.perf_counter()
        loop.compute_control("F", "C", state, payload)
        ms.append((time.perf_counter() - t) * 1000)

    api = {}
    with TestClient(app) as c:
        time.sleep(4)
        for name, path in (("control_status", "/api/v1/control/status/DC-EAST-01"),
                           ("telemetry_latest", "/api/v1/telemetry/latest/DC-EAST-01"),
                           ("explain", "/api/v1/control/explain/DC-EAST-01/CRAC-01"),
                           ("thermal_field_fno", "/api/v1/forecast/thermal-field/DC-EAST-01/CRAC-01"),
                           ("load_forecast", "/api/v1/forecast/load/DC-EAST-01"),
                           ("metrics", "/metrics")):
            for _ in range(5):
                c.get(path)
            lat = []
            for _ in range(60):
                t = time.perf_counter()
                r = c.get(path)
                lat.append((time.perf_counter() - t) * 1000)
                assert r.status_code == 200, (path, r.status_code)
            api[name] = summary(lat)

    control_period = float(os.environ.get("AUTO_CONTROL_INTERVAL_S", "2.0"))
    publish_period = float(os.environ.get("IOT_PUBLISH_INTERVAL_S", "1.0"))
    dec = summary(ms)
    out = {
        "decision_latency_per_crac": dec,
        "api_latency_in_process": api,
        "loop_budget": {
            "telemetry_publish_period_s": publish_period,
            "control_period_s": control_period,
            "worst_case_reaction_s": publish_period + control_period,
            "decision_share_of_control_period_pct": round(100.0 * dec["p99_ms"] / 1000.0 / control_period, 3),
            "cracs_per_tick": 12,
            "decision_time_all_cracs_p99_ms": round(12 * dec["p99_ms"], 2),
        },
        "note": "In-process timings (no network). Worst-case reaction = a disturbance just after a telemetry sample "
                "waits for the next sample, then for the next control tick.",
    }
    with open(os.path.join(PROJECT_ROOT, "results", "control_latency.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
