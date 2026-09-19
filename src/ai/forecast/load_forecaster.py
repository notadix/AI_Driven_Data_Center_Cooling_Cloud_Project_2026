"""
IT-load (heat-load) forecaster.

Predicts the next `HORIZON` 10-minute steps of facility IT power from the last
`LOOKBACK` steps plus time-of-day, so cooling can be planned ahead of load
changes rather than after them. A small GRU trained on the real Frontier2023
IT-power series; evaluated against two honest baselines (persistence and the
hour-of-day mean) on a chronological held-out split.
"""

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

LOOKBACK = 24      # 4 hours of history
HORIZON = 6        # forecast 60 minutes ahead
STEP_MIN = 10


class LoadForecaster(nn.Module):
    def __init__(self, hidden: int = 64, horizon: int = HORIZON):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + 2, 64), nn.GELU(), nn.Linear(64, horizon))
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))

    def forward(self, x: torch.Tensor, hour_feat: torch.Tensor) -> torch.Tensor:
        """x: (B, LOOKBACK, 3) = [normalised power, sin(hour), cos(hour)];
        hour_feat: (B, 2) = sin/cos of the forecast origin hour. Returns normalised
        forecasts = last observed value + a learned change, so persistence is the default."""
        out, _ = self.gru(x)
        delta = self.head(torch.cat([out[:, -1], hour_feat], dim=-1))
        return x[:, -1, :1] + delta

    @torch.no_grad()
    def predict_kw(self, history_kw: np.ndarray, hour_of_day: float) -> np.ndarray:
        """history_kw: last LOOKBACK IT-power readings (kW). Returns HORIZON forecasts (kW)."""
        h = np.asarray(history_kw, dtype=np.float32)
        if h.shape[0] != LOOKBACK:
            raise ValueError(f"need exactly {LOOKBACK} history points")
        hours = (hour_of_day - (LOOKBACK - 1 - np.arange(LOOKBACK)) * STEP_MIN / 60.0) % 24.0
        ang = 2 * math.pi * hours / 24.0
        x = np.stack([(h - float(self.mean)) / float(self.std), np.sin(ang), np.cos(ang)], axis=-1)[None]
        hf = np.array([[math.sin(2 * math.pi * hour_of_day / 24.0), math.cos(2 * math.pi * hour_of_day / 24.0)]], dtype=np.float32)
        y = self(torch.tensor(x, dtype=torch.float32), torch.tensor(hf))
        return (y[0].numpy() * float(self.std) + float(self.mean)).astype(float)


def make_windows(power: np.ndarray, hours: np.ndarray, mean: float, std: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """power: (T,) kW; hours: (T,) hour-of-day. Returns X (N, LOOKBACK, 3), H (N, 2), Y (N, HORIZON), normalised."""
    T = len(power)
    n = T - LOOKBACK - HORIZON + 1
    norm = (power - mean) / std
    ang = 2 * math.pi * hours / 24.0
    feats = np.stack([norm, np.sin(ang), np.cos(ang)], axis=-1).astype(np.float32)
    idx = np.arange(n)[:, None] + np.arange(LOOKBACK)[None, :]
    X = feats[idx]
    origin = np.arange(n) + LOOKBACK - 1
    H = np.stack([np.sin(ang[origin]), np.cos(ang[origin])], axis=-1).astype(np.float32)
    Y = norm[origin[:, None] + 1 + np.arange(HORIZON)[None, :]].astype(np.float32)
    return X, H, Y


def evaluate(model: LoadForecaster, power: np.ndarray, hours: np.ndarray) -> Dict:
    """MAE (kW) and MAPE (%) per horizon for the model and two baselines."""
    mean, std = float(model.mean), float(model.std)
    X, H, Y = make_windows(power, hours, mean, std)
    with torch.no_grad():
        pred = model(torch.tensor(X), torch.tensor(H)).numpy()
    truth = Y * std + mean
    pred = pred * std + mean
    persistence = np.repeat((X[:, -1, 0] * std + mean)[:, None], HORIZON, axis=1)

    hour_bins = np.floor(hours).astype(int) % 24
    hourly_mean = np.array([power[hour_bins == h].mean() for h in range(24)])
    origin_hour = hour_bins[np.arange(len(X)) + LOOKBACK - 1]
    seasonal = np.stack(
        [hourly_mean[(origin_hour + int(round((k + 1) * STEP_MIN / 60.0))) % 24] for k in range(HORIZON)], axis=1
    )

    def stats(p):
        err = np.abs(p - truth)
        return {"mae_kw": [round(float(v), 1) for v in err.mean(axis=0)],
                "mape_pct": [round(float(v), 3) for v in (err / truth).mean(axis=0) * 100.0]}

    return {"samples": int(len(X)), "model": stats(pred), "persistence": stats(persistence),
            "hour_of_day_mean": stats(seasonal),
            "horizon_minutes": [(k + 1) * STEP_MIN for k in range(HORIZON)]}
