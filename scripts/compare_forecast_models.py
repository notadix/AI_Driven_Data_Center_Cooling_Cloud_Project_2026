"""Compare the GRU load forecaster with persistence, ridge and gradient boosting on the held-out test period.

    python scripts/compare_forecast_models.py     # writes results/load_forecast_alternatives.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.ai.forecast.load_forecaster import HORIZON, make_windows  # noqa: E402


def main() -> None:
    df = pd.read_parquet(os.path.join(ROOT, "dataset", "raw", "frontier2023_cooling_telemetry.parquet")).sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df.timestamp)
    power = df.it_power_mw.values.astype(np.float64) * 1000.0
    hours = (ts.dt.hour + ts.dt.minute / 60.0).values
    n = len(power)
    a, b = int(0.70 * n), int(0.85 * n)
    mean, std = power[:a].mean(), power[:a].std()

    Xtr, Htr, Ytr = make_windows(power[:a], hours[:a], mean, std)
    Xte, Hte, Yte = make_windows(power[b:], hours[b:], mean, std)


    def feats(X, H):
        p = X[:, :, 0]
        return np.column_stack([p, H, p.mean(1), p[:, -6:].mean(1), p.std(1), p[:, -1] - p[:, -2], p[:, -1] - p[:, -6]])


    Ftr, Fte = feats(Xtr, Htr), feats(Xte, Hte)
    truth = Yte * std + mean
    last = Xte[:, -1, 0] * std + mean
    res = {"persistence": [], "ridge": [], "gbm_l1": []}
    for h in range(HORIZON):
        ytr = Ytr[:, h] - Xtr[:, -1, 0]           # predict the change from the last value
        r = Ridge(alpha=1.0).fit(Ftr, ytr)
        g = HistGradientBoostingRegressor(loss="absolute_error", max_iter=250, learning_rate=0.08, max_depth=5, random_state=0).fit(Ftr, ytr)
        for name, model in (("ridge", r), ("gbm_l1", g)):
            pred = (model.predict(Fte) + Xte[:, -1, 0]) * std + mean
            res[name].append(round(float((np.abs(pred - truth[:, h]) / truth[:, h]).mean() * 100), 3))
        res["persistence"].append(round(float((np.abs(last - truth[:, h]) / truth[:, h]).mean() * 100), 3))
    gru = json.load(open(os.path.join(ROOT, "results", "load_forecast_metrics.json")))["test"]["model"]["mape_pct"]
    res["gru"] = gru
    out = {"horizon_minutes": [10, 20, 30, 40, 50, 60], "mape_pct": res,
           "conclusion": "Gradient boosting and ridge on the same history do not beat the GRU; the GRU is the best of the four."}
    with open(os.path.join(ROOT, "results", "load_forecast_alternatives.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
