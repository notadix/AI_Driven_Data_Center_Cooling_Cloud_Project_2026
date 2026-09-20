"""
Train and evaluate the 2D Fourier Neural Operator against the 2D transport SOLVER
(src/digital_twin/thermal_solver.py) instead of an analytic formula.

Operating points are real Frontier2023 measurements (IT power, coolant supply temperature,
coolant flow), sampled from a chronological 70/15/15 split so the test operating points come
from the last part of the year. The FNO learns the map

    (per-rack IT, supply-temperature field, per-rack flow)  ->  8x8 rack coolant-temperature field

which requires the non-local coupling (advection + diffusion + hotspots) the solver resolves.

Writes:
  models/fno_pde_v1.pt, models/fno_pde_normalization_stats.json
  results/fno_pde_eval.json    (accuracy vs the solver, speed-up, baselines)

    python scripts/train_fno_pde.py [--train 6000 --epochs 60 --grid 128]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.ai.surrogate.fno_model import FNO2d  # noqa: E402
from src.digital_twin.thermal_solver import rack_inputs, solve_field, time_solver  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "dataset", "raw", "frontier2023_cooling_telemetry.parquet")


def _solve(args):
    inp, n = args
    return solve_field(inp, n)


def sample_points(df: pd.DataFrame, k: int, rng):
    idx = rng.choice(len(df), size=k, replace=False)
    return df.iloc[idx][["it_power_mw", "fws_supply_temp_c", "flow_rate_lpm"]].values


def make_split(points, n, workers):
    inputs = np.stack([rack_inputs(*p) for p in points])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        targets = np.stack(list(pool.map(_solve, [(x, n) for x in inputs], chunksize=32)))
    return inputs.astype(np.float32), targets.astype(np.float32)[:, None]


class RelL2(nn.Module):
    def forward(self, p, t):
        return (torch.linalg.vector_norm(p - t, dim=(1, 2, 3)) / (torch.linalg.vector_norm(t, dim=(1, 2, 3)) + 1e-8)).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_parquet(DATA).sort_values("timestamp").reset_index(drop=True)
    df = df[(df.flow_rate_lpm > 1000) & (df.it_power_mw > 1.0)].reset_index(drop=True)
    a, b = int(0.70 * len(df)), int(0.85 * len(df))

    t0 = time.time()
    Xtr, Ytr = make_split(sample_points(df.iloc[:a], args.train, rng), args.grid, args.workers)
    Xva, Yva = make_split(sample_points(df.iloc[a:b], args.val, rng), args.grid, args.workers)
    Xte, Yte = make_split(sample_points(df.iloc[b:], args.test, rng), args.grid, args.workers)
    print(f"solver dataset: {len(Xtr)}/{len(Xva)}/{len(Xte)} fields in {time.time() - t0:.0f}s", flush=True)

    lo_x, hi_x = Xtr.min(axis=(0, 2, 3), keepdims=True), Xtr.max(axis=(0, 2, 3), keepdims=True)
    lo_y, hi_y = float(Ytr.min()), float(Ytr.max())
    nx = lambda X: np.clip((X - lo_x) / (hi_x - lo_x + 1e-6), 0.0, 1.0)
    ny = lambda Y: (Y - lo_y) / (hi_y - lo_y + 1e-6)
    tX, tY = torch.tensor(nx(Xtr)), torch.tensor(ny(Ytr))
    vX, vY = torch.tensor(nx(Xva)), torch.tensor(ny(Yva))
    sX = torch.tensor(nx(Xte))

    model = FNO2d(in_channels=3, out_channels=1, modes1=4, modes2=4, width=32, num_layers=4)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rel, mse = RelL2(), nn.MSELoss()
    best, best_state = float("inf"), None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tX))
        for i in range(0, len(perm) - 63, 64):
            j = perm[i:i + 64]
            opt.zero_grad()
            loss = rel(model(tX[j]), tY[j]) + 0.5 * mse(model(tX[j]), tY[j])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = mse(model(vX), vY).item()
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep:02d} val MSE (normalised) {v:.6f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()

    # ---- evaluation on the held-out operating points, in degrees C
    def denorm(y):
        return y * (hi_y - lo_y + 1e-6) + lo_y

    with torch.no_grad():
        pred = denorm(model(sX).numpy())
    true = Yte
    err = np.abs(pred - true)
    ss_res, ss_tot = float(((pred - true) ** 2).sum()), float(((true - true.mean()) ** 2).sum())
    # baseline the FNO must beat: a lumped model with NO spatial structure. Even granting it the solver's exact
    # field mean (an oracle a real lumped model does not have), a uniform field cannot represent the hall's gradient.
    uniform_oracle = np.broadcast_to(true.mean(axis=(2, 3), keepdims=True), true.shape)
    lump_err = np.abs(uniform_oracle - true)
    # spatial-structure check: correlation of the predicted deviation from the field mean with the solver's
    dev_p = (pred - pred.mean(axis=(2, 3), keepdims=True)).ravel()
    dev_t = (true - true.mean(axis=(2, 3), keepdims=True)).ravel()
    spatial_corr = float(np.corrcoef(dev_p, dev_t)[0, 1])

    # ---- speed: FNO vs the solver on the same operating point
    x1 = torch.tensor(nx(Xte[:1]))
    with torch.no_grad():
        for _ in range(20):
            model(x1)
        lats = []
        for _ in range(300):
            t = time.perf_counter()
            model(x1)
            lats.append((time.perf_counter() - t) * 1000)
    fno_ms = float(np.median(lats))
    solver_ms = {n: time_solver(Xte[0], n, 3) * 1000 for n in (64, 128, 192)}

    result = {
        "target": "2D advection-diffusion solver (src/digital_twin/thermal_solver.py), 8x8 rack-averaged coolant temperature",
        "operating_points": "real Frontier2023 IT power / supply temperature / flow, chronological 70/15/15 split",
        "samples": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "solver_grid": args.grid,
        "test_metrics": {
            "r2": round(1 - ss_res / ss_tot, 5),
            "mae_c": round(float(err.mean()), 4),
            "rmse_c": round(float(np.sqrt((err ** 2).mean())), 4),
            "max_err_c": round(float(err.max()), 4),
            "spatial_pattern_correlation": round(spatial_corr, 4),
            "uniform_field_oracle_mae_c": round(float(lump_err.mean()), 4),
            "field_range_c": [round(float(true.min()), 2), round(float(true.max()), 2)],
        },
        "latency_ms": {"fno_median_cpu": round(fno_ms, 3),
                       "solver_median_cpu_by_grid": {str(k): round(v, 2) for k, v in solver_ms.items()}},
        "speedup_vs_solver": {str(k): round(v / fno_ms, 1) for k, v in solver_ms.items()},
        "note": "The solver is a 2D reduced-order transport model, not 3D CFD; the speed-up is versus this solver.",
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "architecture": {
        "in_channels": 3, "out_channels": 1, "modes1": 4, "modes2": 4, "width": 32, "num_layers": 4},
        "best_val_loss": float(best)}, os.path.join(PROJECT_ROOT, "models", "fno_pde_v1.pt"))
    with open(os.path.join(PROJECT_ROOT, "models", "fno_pde_normalization_stats.json"), "w") as f:
        json.dump({"input_min": lo_x.squeeze().tolist(), "input_max": hi_x.squeeze().tolist(),
                   "target_min": lo_y, "target_max": hi_y, "grid": [8, 8],
                   "channels": ["workload_mw_per_rack", "supply_temp_c", "flow_lpm_per_rack"]}, f, indent=2)
    with open(os.path.join(PROJECT_ROOT, "results", "fno_pde_eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
