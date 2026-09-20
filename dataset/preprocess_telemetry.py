"""
Converts raw Frontier2023 telemetry into normalized (B, C, H, W) spatial tensors
for Fourier Neural Operator training. Produces train/val/test .npy splits.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# See dataset/download_dataset.py for why this is needed on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

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
        # Real facility data includes genuine zero-flow downtime rows (pumps
        # off), where this formula's 1e-4 epsilon guard against division by
        # zero isn't enough on its own -- it still produces a near-infinite
        # temp_rise (observed: up to ~142,000C on the real Frontier2023
        # dataset) that would dominate min-max normalization and squash
        # every legitimate reading into a tiny sliver near zero. The
        # synthetic generator's flow never goes near zero, so this was
        # latent until real data was used. Clamped to 60C, comfortably
        # above the realistic p99 (~50C) seen in real data.
        temp_rise = np.clip(temp_rise, 0.0, 60.0)
        tensors[i, 3] = tensors[i, 1] + temp_rise

    return tensors


def process_and_save(input_parquet: str, output_dir: str = PROCESSED_DIR, use_synthetic: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_parquet):
        if use_synthetic:
            print(f"[!] Raw file not found at {input_parquet}. Generating synthetic data...")
            from download_dataset import generate_frontier2023
            generate_frontier2023(save_path=input_parquet)
        else:
            print(f"[!] Raw file not found at {input_parquet}. Downloading real Frontier2023 data...")
            from download_dataset import download_frontier2023_raw, load_real_frontier2023
            xlsx_path = download_frontier2023_raw()
            load_real_frontier2023(xlsx_path, save_path=input_parquet)

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

    # Splits are sequential by time (train=Jan-Jul, val/test later in the
    # year), so on real seasonal data a handful of val/test readings can
    # fall slightly outside the range train's min/max were computed from
    # (e.g. a hotter summer day in val than anything seen in train). Clip
    # after normalizing so those readings saturate at the boundary instead
    # of producing values outside [0, 1] -- standard practice for held-out
    # splits normalized against training statistics, and harmless for the
    # synthetic generator's data, which doesn't have this seasonal skew.
    def _normalize(arr):
        return np.clip((arr - ch_min) / (ch_max - ch_min + eps), 0.0, 1.0)

    np.save(os.path.join(output_dir, "train_spatial.npy"), _normalize(train))
    np.save(os.path.join(output_dir, "val_spatial.npy"), _normalize(val))
    np.save(os.path.join(output_dir, "test_spatial.npy"), _normalize(test))

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
    parser.add_argument("--synthetic", action="store_true",
                         help="Use the fabricated synthetic generator instead of real Frontier2023 data "
                              "when --input doesn't exist yet.")
    args = parser.parse_args()

    process_and_save(args.input, args.output_dir, use_synthetic=args.synthetic)

    if args.verify:
        print("[*] Verifying saved tensors...")
        for split in ("train", "val", "test"):
            arr = np.load(os.path.join(args.output_dir, f"{split}_spatial.npy"))
            assert arr.ndim == 4 and arr.shape[1] == 4, f"Shape error for {split}: {arr.shape}"
            assert arr.min() >= -1e-4 and arr.max() <= 1.0 + 1e-4, f"Normalization error for {split}"
        print("[✓] All tensors verified.")


if __name__ == "__main__":
    main()
