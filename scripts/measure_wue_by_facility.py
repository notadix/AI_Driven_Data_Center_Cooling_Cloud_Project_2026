"""
Measure per-facility PUE and WUE from the real IoT PhysicsSimulator.

Runs the default 3-facility topology (12 CRACs) for a fixed number of steps at
the simulator's default setpoints (no controller in the loop) and writes the
per-facility means to results/wue_by_facility.json. Seeded for reproducibility.

    python scripts/measure_wue_by_facility.py
"""
import json
import os
import random
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.aws.iot.iot_publisher import IoTSimulator  # noqa: E402

STEPS = 600
SEED = 42


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    sim = IoTSimulator()
    acc = {}
    for _ in range(STEPS):
        for _, s in sim._simulators.items():
            p = s.step()
            a = acc.setdefault(s.facility_id, {"pue": [], "wue": [], "ambient_c": []})
            a["pue"].append(p.pue)
            a["wue"].append(p.wue)
            a["ambient_c"].append(s.ambient_c)

    out = {
        "method": (
            f"{STEPS} steps x 4 CRACs per facility on the IoT PhysicsSimulator at default "
            f"setpoints (no controller), seed={SEED}"
        ),
        "facilities": {
            fid: {
                "samples": len(v["pue"]),
                "mean_pue": round(float(np.mean(v["pue"])), 4),
                "mean_wue_l_per_kwh": round(float(np.mean(v["wue"])), 4),
                "mean_ambient_c": round(float(np.mean(v["ambient_c"])), 2),
            }
            for fid, v in acc.items()
        },
    }
    path = os.path.join(PROJECT_ROOT, "results", "wue_by_facility.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
