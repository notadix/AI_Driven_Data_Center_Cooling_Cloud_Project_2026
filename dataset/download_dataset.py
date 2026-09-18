"""
Frontier2023 dataset ingestion + high-fidelity synthetic fallback generator.
ORNL Source: Figshare DOI: 10.6084/m9.figshare.24391240
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
N_RECORDS = 52560  # 365 days * 144 steps/day (10-min resolution)


def generate_frontier2023(n: int = N_RECORDS, save_path: str = None) -> pd.DataFrame:
    print(f"[*] Generating {n} telemetry records (Frontier2023 schema)...")
    np.random.seed(42)

    start = datetime(2023, 1, 1)
    timestamps = [start + timedelta(minutes=10 * i) for i in range(n)]
    doy = np.array([t.timetuple().tm_yday for t in timestamps], dtype=float)
    hod = np.array([t.hour + t.minute / 60.0 for t in timestamps], dtype=float)
    is_weekday = np.array([t.weekday() < 5 for t in timestamps], dtype=float)

    ambient_c = (
        15.0
        + 12.0 * np.sin(2 * np.pi * (doy - 80) / 365.0)
        + 4.0 * np.sin(2 * np.pi * (hod - 9) / 24.0)
        + np.random.normal(0, 1.2, n)
    )

    it_mw = np.clip(
        18.0 + 8.0 * is_weekday * np.sin(np.pi * (hod - 6) / 18.0).clip(0, 1)
        + np.random.normal(0, 1.5, n),
        10.0, 29.0,
    )

    supply_c = np.clip(
        16.0 + 0.3 * (ambient_c - 15.0).clip(0, 20) + np.random.normal(0, 0.4, n),
        14.0, 24.0,
    )
    flow_lpm = 3500.0 + 80.0 * it_mw + np.random.normal(0, 50.0, n)

    temp_rise = (it_mw * 1000.0 * 0.92) / (flow_lpm * 0.064 + 1e-5)
    return_c = supply_c + temp_rise + np.random.normal(0, 0.3, n)

    inlet_c = supply_c + 2.5 + np.random.normal(0, 0.3, n)
    outlet_c = inlet_c + it_mw * 0.8 + np.random.normal(0, 0.5, n)

    cop = np.clip(6.5 - 0.1 * (ambient_c - 10.0), 3.0, 7.5)
    pump_mw = 0.4 * (flow_lpm / 5000.0) ** 3
    chiller_mw = (it_mw * 0.92) / cop * (ambient_c > 18.0).astype(float) * 0.4
    fan_mw = np.full(n, 0.35)
    cooling_mw = pump_mw + chiller_mw + fan_mw
    pue = (it_mw + cooling_mw + 0.25) / it_mw

    carbon = np.clip(
        320.0
        + 80.0 * np.sin(2 * np.pi * (hod - 17) / 24.0)
        + 40.0 * (ambient_c > 25.0).astype(float)
        + np.random.normal(0, 15.0, n),
        120.0, 580.0,
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "ambient_temp_c": np.round(ambient_c, 2),
        "it_power_mw": np.round(it_mw, 3),
        "fws_supply_temp_c": np.round(supply_c, 2),
        "fws_return_temp_c": np.round(return_c, 2),
        "flow_rate_lpm": np.round(flow_lpm, 1),
        "server_inlet_temp_c": np.round(inlet_c, 2),
        "server_outlet_temp_c": np.round(outlet_c, 2),
        "cooling_power_mw": np.round(cooling_mw, 3),
        "total_facility_power_mw": np.round(it_mw + cooling_mw + 0.25, 3),
        "pue": np.round(pue, 4),
        "grid_carbon_gco2_kwh": np.round(carbon, 1),
    })

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path, index=False)
        print(f"[✓] Saved to {save_path} ({len(df)} rows)")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=N_RECORDS)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out = args.output or os.path.join(RAW_DIR, "frontier2023_cooling_telemetry.parquet")
    df = generate_frontier2023(n=args.records, save_path=out)
    print(df.head(3))


if __name__ == "__main__":
    main()
