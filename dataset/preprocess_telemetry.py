"""
Converts raw Frontier2023 telemetry into normalized (B, C, H, W) spatial tensors
for Fourier Neural Operator training. Produces train/val/test .npy splits.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(CURRENT_DIR, "raw")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed")

GRID_H, GRID_W = 8, 8
N_RACKS = GRID_H * GRID_W


def build_spatial_tensors(df: pd.DataFrame) -> np.ndarray:
    N = len(df)
    tensors = np.zeros((N, 4, GRID_H, GRID_W), dtype=np.float32)

    y_g, x_g = np.meshgrid(np.linspace(-1, 1, GRID_H), np.linspace(-1, 1, GRID_W), indexing="ij")
    hotspot = 1.0 + 0.35 * np.exp(-(x_g ** 2 + y_g ** 2) / 0.8)
    hotspot /= hotspot.mean()

    for i, row in enumerate(df.itertuples(index=False)):
        tensors[i, 0] = (row.it_power_mw / N_RACKS) * hotspot
        tensors[i, 1] = row.fws_supply_temp_c + 0.05 * (x_g + y_g)
        tensors[i, 2] = (row.flow_rate_lpm / N_RACKS) * (1.0 / (hotspot + 0.1))
        temp_rise = (tensors[i, 0] * 1000.0 * 0.92) / (tensors[i, 2] * 0.064 + 1e-4)
        tensors[i, 3] = tensors[i, 1] + temp_rise

    return tensors


def process_and_save(input_parquet: str, output_dir: str = PROCESSED_DIR):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_parquet):
        print(f"[!] Raw file not found at {input_parquet}. Generating now...")
        from download_dataset import generate_frontier2023
        generate_frontier2023(save_path=input_parquet)

    df = pd.read_parquet(input_parquet)
    tensors = build_spatial_tensors(df)

    N = len(tensors)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)

    train = tensors[:n_train]
    val = tensors[n_train : n_train + n_val]
    test = tensors[n_train + n_val :]

    ch_min = train.min(axis=(0, 2, 3), keepdims=True)
    ch_max = train.max(axis=(0, 2, 3), keepdims=True)
    eps = 1e-6

    np.save(os.path.join(output_dir, "train_spatial.npy"), (train - ch_min) / (ch_max - ch_min + eps))
    np.save(os.path.join(output_dir, "val_spatial.npy"), (val - ch_min) / (ch_max - ch_min + eps))
    np.save(os.path.join(output_dir, "test_spatial.npy"), (test - ch_min) / (ch_max - ch_min + eps))

    stats = {
        "channel_min": ch_min.squeeze().tolist(),
        "channel_max": ch_max.squeeze().tolist(),
        "channels": ["workload_mw", "supply_temp_c", "flow_rate_lpm", "rack_temp_c"],
        "grid": [GRID_H, GRID_W],
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
    }
    with open(os.path.join(output_dir, "normalization_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[✓] train={train.shape} | val={val.shape} | test={test.shape}  →  {output_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(RAW_DIR, "frontier2023_cooling_telemetry.parquet"))
    parser.add_argument("--output_dir", default=PROCESSED_DIR)
    parser.add_argument("--verify", action="store_true", help="Run quick integrity check")
    args = parser.parse_args()

    stats = process_and_save(args.input, args.output_dir)

    if args.verify:
        print("[*] Verifying saved tensors...")
        for split in ("train", "val", "test"):
            arr = np.load(os.path.join(args.output_dir, f"{split}_spatial.npy"))
            assert arr.ndim == 4 and arr.shape[1] == 4, f"Shape error for {split}: {arr.shape}"
            assert arr.min() >= -1e-4 and arr.max() <= 1.0 + 1e-4, f"Normalization error for {split}"
        print("[✓] All tensors verified.")


if __name__ == "__main__":
    main()
