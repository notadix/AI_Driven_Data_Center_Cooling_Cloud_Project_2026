import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# See dataset/download_dataset.py for why this is needed on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_model import FNO2d

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class RelativeL2Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.norm(pred - target, p=2, dim=(-2, -1))
        norm = torch.norm(target, p=2, dim=(-2, -1))
        return torch.mean(diff / (norm + 1e-6))


def load_splits(data_dir: str):
    train = np.load(os.path.join(data_dir, "train_spatial.npy"))
    val = np.load(os.path.join(data_dir, "val_spatial.npy"))
    to_tensor = lambda arr: (
        torch.tensor(arr[:, :3], dtype=torch.float32),
        torch.tensor(arr[:, 3:4], dtype=torch.float32),
    )
    return to_tensor(train), to_tensor(val)


def ensure_preprocessed(data_dir: str):
    if not os.path.exists(os.path.join(data_dir, "train_spatial.npy")):
        print("[!] Preprocessed data not found. Running preprocessing pipeline...")
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "dataset"))
        from preprocess_telemetry import process_and_save
        raw = os.path.join(PROJECT_ROOT, "dataset", "raw", "frontier2023_cooling_telemetry.parquet")
        process_and_save(raw, data_dir)


def train(
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    smoke_test: bool = False,
    output: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training FNO on {device}")

    ensure_preprocessed(PROCESSED_DIR)
    (X_tr, Y_tr), (X_val, Y_val) = load_splits(PROCESSED_DIR)

    if smoke_test:
        X_tr, Y_tr, X_val, Y_val = X_tr[:256], Y_tr[:256], X_val[:64], Y_val[:64]
        epochs = min(epochs, 2)

    tr_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

    model = FNO2d(in_channels=3, out_channels=1, modes1=4, modes2=4, width=32, num_layers=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_rel = RelativeL2Loss()
    loss_mse = nn.MSELoss()

    best_val = float("inf")
    save_path = output or os.path.join(MODELS_DIR, "fno_surrogate_v1.pt")
    os.makedirs(MODELS_DIR, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_rel(pred, yb) + 0.5 * loss_mse(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        sched.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                val_losses.append(loss_mse(model(xv), yv).item())

        avg_tr = np.mean(tr_losses)
        avg_val = np.mean(val_losses)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train: {avg_tr:.6f} | Val MSE: {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                # float() here matters: np.mean() returns numpy.float64, and
                # PyTorch >=2.6 defaults torch.load to weights_only=True,
                # which rejects checkpoints containing numpy scalar types --
                # this went undiscovered until a checkpoint was actually
                # saved and reloaded for the first time in this repo.
                "best_val_loss": float(best_val),
                "architecture": {"in_channels": 3, "out_channels": 1, "modes1": 4, "modes2": 4, "width": 32, "num_layers": 4},
            }, save_path)

    print(f"[✓] Best checkpoint saved → {save_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.smoke_test, args.output)


if __name__ == "__main__":
    main()
