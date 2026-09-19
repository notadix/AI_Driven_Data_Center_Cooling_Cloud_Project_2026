"""
Train and evaluate the IT-load forecaster on the real Frontier2023 IT-power
series (chronological 70/15/15 split) and write

  models/load_forecaster_v1.pt
  results/load_forecast_metrics.json

    python scripts/train_load_forecaster.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.ai.forecast.load_forecaster import HORIZON, LOOKBACK, LoadForecaster, evaluate, make_windows  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "dataset", "raw", "frontier2023_cooling_telemetry.parquet")


def main(epochs: int = 25, seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = pd.read_parquet(DATA).sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df.timestamp)
    power = df.it_power_mw.values.astype(np.float64) * 1000.0
    hours = (ts.dt.hour + ts.dt.minute / 60.0).values

    n = len(power)
    a, b = int(0.70 * n), int(0.85 * n)
    mean, std = float(power[:a].mean()), float(power[:a].std())

    model = LoadForecaster()
    model.mean.fill_(mean)
    model.std.fill_(std)

    Xtr, Htr, Ytr = (torch.tensor(v) for v in make_windows(power[:a], hours[:a], mean, std))
    Xva, Hva, Yva = (torch.tensor(v) for v in make_windows(power[a:b], hours[a:b], mean, std))

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.L1Loss()
    best, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx], Htr[idx]), Ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva, Hva), Yva).item()
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:02d} val L1 (normalised) {val:.4f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    metrics = {
        "data": "Frontier2023 IT power, 10-minute samples, chronological 70/15/15 split",
        "lookback_steps": LOOKBACK, "horizon_steps": HORIZON,
        "validation": evaluate(model, power[a:b], hours[a:b]),
        "test": evaluate(model, power[b:], hours[b:]),
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "lookback": LOOKBACK, "horizon": HORIZON, "mean": mean, "std": std},
               os.path.join(PROJECT_ROOT, "models", "load_forecaster_v1.pt"))
    with open(os.path.join(PROJECT_ROOT, "results", "load_forecast_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    t = metrics["test"]
    print("TEST MAPE % by horizon (10..60 min)")
    for name in ("model", "persistence", "hour_of_day_mean"):
        print(f"  {name:<17}", t[name]["mape_pct"])


if __name__ == "__main__":
    main()
