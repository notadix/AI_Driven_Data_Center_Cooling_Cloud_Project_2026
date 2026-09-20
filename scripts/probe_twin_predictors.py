"""
Can a better model reach the ~2% one-step MAPE target for return temperature and cooling power?

Tries a gradient-boosted residual model (L1 loss, predicts the change from the last reading) on the last 6 readings,
the operating point and its change, against "repeat the last measurement". Chronological split: fit on the first 70%
of the usable rows, test on the last 30% (same rows as results/twin_fidelity.json).

Writes results/twin_predictor_probe.json.

    python scripts/probe_twin_predictors.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.calibrate_twin import load_rows  # noqa: E402
from src.digital_twin.synced_twin import consecutive_rows, mape  # noqa: E402

LAGS = 6


def main() -> None:
    df = load_rows()
    ts = df.timestamp.values
    idx = consecutive_rows(ts, LAGS)
    cut = int(0.7 * len(df))
    hours = pd.to_datetime(ts).hour.values.astype(float)
    x_in = np.column_stack([df.it_power_mw.values, df.flow_rate_lpm.values, df.fws_supply_temp_c.values])
    targets = {"return_temp_c": df.fws_return_temp_c.values, "cooling_power_kw": df.cooling_power_mw.values * 1000.0}

    def feats(y, i):
        lag = np.column_stack([y[i - j] for j in range(1, LAGS + 1)])
        d = np.column_stack([y[i - 1] - y[i - j] for j in range(2, LAGS + 1)])
        return np.column_stack([lag, d, x_in[i], x_in[i] - x_in[i - 1], x_in[i - 1], lag.mean(1), lag.std(1), hours[i]])

    tr, te = idx[idx < cut], idx[idx >= cut]
    out = {"method": "HistGradientBoosting (L1 loss) on the change from the last reading; chronological 70/30 split",
           "train_rows": int(len(tr)), "test_rows": int(len(te))}
    for k, y in targets.items():
        m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05, max_depth=5, random_state=0)
        m.fit(feats(y, tr), y[tr] - y[tr - 1])
        pred = y[te - 1] + m.predict(feats(y, te))
        out[k] = {"gbm_residual_mape_pct": round(mape(pred, y[te]), 3), "persistence_mape_pct": round(mape(y[te - 1], y[te]), 3)}
        out[k]["meets_2pct"] = out[k]["gbm_residual_mape_pct"] <= 2.0
    with open(os.path.join(PROJECT_ROOT, "results", "twin_predictor_probe.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
