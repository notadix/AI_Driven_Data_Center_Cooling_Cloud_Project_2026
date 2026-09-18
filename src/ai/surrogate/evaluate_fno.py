import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_model import FNO2d

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def evaluate(model_path: str = None):
    ckpt_path = model_path or os.path.join(MODELS_DIR, "fno_surrogate_v1.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train_fno.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt["architecture"]
    model = FNO2d(**arch).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_arr = np.load(os.path.join(PROCESSED_DIR, "test_spatial.npy"))
    X = torch.tensor(test_arr[:, :3], dtype=torch.float32).to(device)
    Y = test_arr[:, 3:4]

    with open(os.path.join(PROCESSED_DIR, "normalization_stats.json")) as f:
        stats = json.load(f)
    t_min, t_max = stats["channel_min"][3], stats["channel_max"][3]

    # Latency benchmark
    with torch.no_grad():
        for _ in range(20):
            _ = model(X[:1])
        lats = []
        for i in range(min(300, len(X))):
            t0 = time.perf_counter()
            _ = model(X[i : i + 1])
            lats.append((time.perf_counter() - t0) * 1000.0)

        preds = model(X).cpu().numpy()

    Y_c = Y * (t_max - t_min) + t_min
    P_c = preds * (t_max - t_min) + t_min

    yf, pf = Y_c.flatten(), P_c.flatten()
    metrics = {
        "r2": float(r2_score(yf, pf)),
        "mae_c": float(mean_absolute_error(yf, pf)),
        "rmse_c": float(np.sqrt(mean_squared_error(yf, pf))),
        "max_err_c": float(np.max(np.abs(yf - pf))),
        "mean_latency_ms": float(np.mean(lats)),
        "p95_latency_ms": float(np.percentile(lats, 95)),
        "samples": int(len(X)),
    }

    print("\n" + "=" * 48)
    print("  FNO EVALUATION RESULTS")
    print("=" * 48)
    print(f"  R²           : {metrics['r2']:.4f}   (target ≥ 0.95)")
    print(f"  MAE          : {metrics['mae_c']:.4f} °C  (target ≤ 0.40)")
    print(f"  RMSE         : {metrics['rmse_c']:.4f} °C")
    print(f"  Max Error    : {metrics['max_err_c']:.4f} °C")
    print(f"  Mean Latency : {metrics['mean_latency_ms']:.2f} ms")
    print(f"  P95  Latency : {metrics['p95_latency_ms']:.2f} ms")
    print("=" * 48)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "fno_eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.model)
