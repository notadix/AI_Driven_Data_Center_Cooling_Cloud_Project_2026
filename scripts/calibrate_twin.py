"""
Calibrate the digital twin's physics constants to the real Frontier2023 data
and measure model fidelity (MAPE) on a held-out period.

  * fit   : first 70% of the year (chronological)
  * report: last 30% (held out), for the ORIGINAL and the CALIBRATED constants

Writes:
  src/digital_twin/calibrated_constants.json   (loaded by LiquidCoolingPhysics)
  results/twin_fidelity.json                   (MAPE before/after)

    python scripts/calibrate_twin.py
"""
import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.digital_twin.physics_dynamics import (  # noqa: E402
    CoolingConstants, LiquidCoolingPhysics,
)

DATA = os.path.join(PROJECT_ROOT, "dataset", "raw", "frontier2023_cooling_telemetry.parquet")
OUT_CONST = os.path.join(PROJECT_ROOT, "src", "digital_twin", "calibrated_constants.json")
OUT_RESULT = os.path.join(PROJECT_ROOT, "results", "twin_fidelity.json")

THERMAL_PARAMS = ["HEAT_CAPTURE", "INLET_OFFSET_C", "INLET_AMBIENT_COEF", "OUTLET_K"]
POWER_PARAMS = ["PUMP_RATED_KW", "COP_A", "COP_B"]
FIT_PARAMS = THERMAL_PARAMS + POWER_PARAMS
# Frontier's reported PUE is exactly (IT + cooling) / IT, so there is no
# separate fixed overhead term; the "free-cooling" chiller shortcut is
# disabled because Frontier's ambient is below its supply temperature almost
# all year, i.e. it is not a distinct regime in the real plant.
FIXED = {"FIXED_OVERHEAD_KW": 0.0, "FREE_COOL_MARGIN_C": 1000.0, "COP_MAX": 40.0}
FAN_TO_PUMP_RATIO = 120.0 / 450.0  # fan rating is tied to the pump rating (not separately observable)


def load_rows() -> pd.DataFrame:
    df = pd.read_parquet(DATA).sort_values("timestamp").reset_index(drop=True)
    # Drop downtime / near-idle rows: zero flow makes the thermal balance undefined.
    df = df[(df.flow_rate_lpm > 1000) & (df.it_power_mw > 1.0)].reset_index(drop=True)
    return df


def predict(c: CoolingConstants, d: pd.DataFrame) -> dict:
    """Vectorised twin outputs for the measured operating points."""
    it_kw = d.it_power_mw.values * 1000.0
    flow = d.flow_rate_lpm.values
    supply = d.fws_supply_temp_c.values
    amb = d.ambient_temp_c.values

    mass = (flow / 60.0) * c.DENSITY_KG_L + 1e-6
    ret = supply + it_kw * c.HEAT_CAPTURE / (mass * c.CP_KJ_KGK)
    inlet = supply + c.INLET_OFFSET_C + c.INLET_AMBIENT_COEF * np.maximum(0.0, amb - 20.0)
    outlet = inlet + np.minimum(20.0, it_kw * c.OUTLET_K / np.maximum(1.0, mass * c.CP_KJ_KGK))

    # Operating point of the plant inferred from the measured flow.
    pump_frac = np.clip((flow - c.FLOW_MIN_LPM) / c.FLOW_SPAN_LPM, 0.2, 1.0)
    pump_kw = c.PUMP_RATED_KW * pump_frac ** 3
    fan_kw = c.PUMP_RATED_KW * FAN_TO_PUMP_RATIO * pump_frac ** 3
    lift = np.maximum(1.0, amb - supply)
    cop = np.clip(c.COP_A - c.COP_B * lift, c.COP_MIN, c.COP_MAX)
    chiller = np.where(amb < supply - c.FREE_COOL_MARGIN_C, c.FREE_COOL_CHILLER_KW, it_kw * c.HEAT_CAPTURE / cop)
    cooling = pump_kw + fan_kw + chiller
    pue = (it_kw + cooling + c.FIXED_OVERHEAD_KW) / np.maximum(1.0, it_kw)
    return {"return_temp_c": ret, "server_inlet_temp_c": inlet, "server_outlet_temp_c": outlet,
            "cooling_power_kw": cooling, "pue": pue}


def truth(d: pd.DataFrame) -> dict:
    return {"return_temp_c": d.fws_return_temp_c.values, "server_inlet_temp_c": d.server_inlet_temp_c.values,
            "server_outlet_temp_c": d.server_outlet_temp_c.values,
            "cooling_power_kw": d.cooling_power_mw.values * 1000.0, "pue": d.pue.values}


def score(c: CoolingConstants, d: pd.DataFrame) -> dict:
    p, t = predict(c, d), truth(d)
    out = {}
    for k in p:
        err = np.abs(p[k] - t[k])
        out[k] = {"mape_pct": round(float(np.mean(err / np.abs(t[k])) * 100.0), 3),
                  "mae": round(float(np.mean(err)), 4)}
    out["mean_mape_pct"] = round(float(np.mean([v["mape_pct"] for v in out.values()])), 3)
    return out


def with_params(base: CoolingConstants, names, theta) -> CoolingConstants:
    c = dataclasses.replace(base, **FIXED)
    for name, v in zip(names, theta):
        setattr(c, name, float(v))
    return c


def main() -> None:
    d = load_rows()
    split = int(len(d) * 0.7)
    train, test = d.iloc[:split], d.iloc[split:]
    original = CoolingConstants()

    # Sanity: the vectorised model must match LiquidCoolingPhysics point-wise.
    phys = LiquidCoolingPhysics(original)
    row = test.iloc[5]
    ret, inlet, outlet = phys.thermal_balance(row.it_power_mw * 1000, row.fws_supply_temp_c,
                                              row.flow_rate_lpm, row.ambient_temp_c)
    pv = predict(original, test.iloc[[5]])
    assert abs(pv["return_temp_c"][0] - ret) < 1e-6 and abs(pv["server_outlet_temp_c"][0] - outlet) < 1e-6

    before = score(original, test)

    t_fit = truth(train.iloc[::5])
    d_fit = train.iloc[::5]

    keys_thermal = ["return_temp_c", "server_inlet_temp_c", "server_outlet_temp_c"]
    keys_power = ["cooling_power_kw"]

    def fit(names, keys, lo, hi):
        def residuals(theta):
            p = predict(with_params(original, names, theta), d_fit)
            return np.concatenate([(p[k] - t_fit[k]) / np.abs(t_fit[k]) for k in keys])
        x0 = np.array([getattr(original, n) for n in names])
        return least_squares(residuals, x0, bounds=(lo, hi)).x

    th = fit(THERMAL_PARAMS, keys_thermal, [0.5, 0.0, 0.0, 0.0], [1.2, 6.0, 0.5, 2.0])
    base_after_thermal = with_params(original, THERMAL_PARAMS, th)
    original_for_power = base_after_thermal
    def fit_power():
        def residuals(theta):
            p = predict(with_params(original_for_power, POWER_PARAMS, theta), d_fit)
            return (p["cooling_power_kw"] - t_fit["cooling_power_kw"]) / t_fit["cooling_power_kw"]
        x0 = np.array([getattr(original, n) for n in POWER_PARAMS])
        return least_squares(residuals, x0, bounds=([10.0, 2.0, 0.0], [3000.0, 200.0, 5.0])).x
    pw = fit_power()
    calibrated = with_params(base_after_thermal, POWER_PARAMS, pw)
    after = score(calibrated, test)
    after_train = score(calibrated, train)

    consts = {n: round(float(getattr(calibrated, n)), 6) for n in FIT_PARAMS}
    consts.update(FIXED)
    os.makedirs(os.path.dirname(OUT_RESULT), exist_ok=True)
    with open(OUT_CONST, "w", encoding="utf-8") as f:
        json.dump({"source": "scripts/calibrate_twin.py on Frontier2023 (fit: first 70% of rows)",
                   "constants": consts}, f, indent=2)
    result = {
        "method": "Chronological split of Frontier2023 (rows with flow>1000 LPM and IT>1 MW): "
                  "constants fitted on the first 70%, error measured on the held-out last 30%.",
        "rows": {"total": int(len(d)), "fit": int(len(train)), "held_out": int(len(test))},
        "held_out_original_constants": before,
        "held_out_calibrated_constants": after,
        "train_calibrated_constants": after_train,
        "calibrated_constants": consts,
        "report_target_mape_pct": 2.0,
    }
    with open(OUT_RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"before": before, "after": after, "constants": consts}, indent=2))


if __name__ == "__main__":
    main()
