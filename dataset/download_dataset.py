"""
Frontier2023 dataset ingestion + high-fidelity synthetic fallback generator.
ORNL Source: Figshare DOI: 10.6084/m9.figshare.24391240

Real-data path (default): downloads the actual "Frontier HPC & Facility Data.xlsx"
workbook from Figshare and converts its measured columns (compute power, coolant
supply/return temp, coolant flow, facility accessory power, total power, PUE) into
the same schema generate_frontier2023() fabricates. Ambient temperature, per-rack
inlet/outlet temperature, and grid carbon intensity are NOT measured anywhere in
the source dataset (confirmed by inspecting every sheet) -- those three columns
are derived using the same formulas/constants already used elsewhere in this
codebase (src/digital_twin/physics_dynamics.py's inlet-mixing offset, and
src/aws/serverless/lambda_carbon_fetcher.py's regional diurnal carbon model), not
invented fresh, and are clearly marked as derived below.

Synthetic path (--synthetic flag): the original generate_frontier2023() fallback,
unchanged, for offline dev/CI or when network access to Figshare is unavailable.
"""

import hashlib
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Windows' console defaults to cp1252, which can't encode the checkmark/cross
# glyphs this script (and others in dataset/, src/ai/) print -- UnicodeEncodeError
# on every run from a plain `python script.py` invocation there. Reconfigure to
# UTF-8 defensively; unavailable on some redirected-output setups, hence the guard.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
N_RECORDS = 52560  # 365 days * 144 steps/day (10-min resolution)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Figshare article 24391240, "Energy dataset of the Frontier supercomputer"
# (Grant, D. et al., Nature Scientific Data, CC BY 4.0). File id/checksum
# confirmed by querying https://api.figshare.com/v2/articles/24391240.
FIGSHARE_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/47812750"
FIGSHARE_FILE_MD5 = "f1832f7215e529f12f38d4087620d6e2"
FIGSHARE_SHEET_NAME = "Frontier2023"
GPM_TO_LPM = 3.785411784


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


def download_frontier2023_raw(dest_path: str = None, force: bool = False) -> str:
    """Downloads the real Frontier2023 Figshare workbook, verifying its MD5
    checksum against the value confirmed via the Figshare API. Skips the
    download if a file already exists at dest_path with a matching checksum."""
    import urllib.request

    dest_path = dest_path or os.path.join(RAW_DIR, "frontier2023_facility_data.xlsx")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if not force and os.path.exists(dest_path):
        existing_md5 = hashlib.md5(open(dest_path, "rb").read()).hexdigest()
        if existing_md5 == FIGSHARE_FILE_MD5:
            print(f"[*] Using cached, checksum-verified dataset at {dest_path}")
            return dest_path
        print(f"[!] Cached file at {dest_path} failed checksum verification, re-downloading...")

    print(f"[*] Downloading Frontier2023 dataset from Figshare ({FIGSHARE_DOWNLOAD_URL})...")
    req = urllib.request.Request(FIGSHARE_DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())

    downloaded_md5 = hashlib.md5(open(dest_path, "rb").read()).hexdigest()
    if downloaded_md5 != FIGSHARE_FILE_MD5:
        raise ValueError(
            f"Downloaded file checksum mismatch: expected {FIGSHARE_FILE_MD5}, got {downloaded_md5}. "
            "The Figshare artifact may have changed; verify the DOI (10.6084/m9.figshare.24391240) manually."
        )
    print(f"[✓] Downloaded and verified {dest_path} (MD5 {downloaded_md5})")
    return dest_path


def load_real_frontier2023(xlsx_path: str, save_path: str = None) -> pd.DataFrame:
    """Converts the real Frontier2023 workbook into the same schema
    generate_frontier2023() fabricates. See module docstring for which
    columns are real measurements vs. derived approximations."""
    raw = pd.read_excel(xlsx_path, sheet_name=FIGSHARE_SHEET_NAME)
    # Row 0 is a units row (e.g. "(deg C)", "(gpm)", "(MW)"), not data.
    raw = raw.iloc[1:].reset_index(drop=True)

    numeric_cols = [c for c in raw.columns if c != "Date/Time"]
    for c in numeric_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    # ~0.2% of rows are missing "Overall-average Coolant Return Temp" in the
    # source file; linear interpolation is standard practice for short gaps
    # in a 10-minute-resolution time series and matches the report's own
    # documented preprocessing plan ("handle the ~5% gaps by ... interpolation").
    raw[numeric_cols] = raw[numeric_cols].interpolate(method="linear", limit_direction="both")

    timestamps = pd.to_datetime(raw["Date/Time"])
    n = len(raw)
    hod = timestamps.dt.hour + timestamps.dt.minute / 60.0
    doy = timestamps.dt.dayofyear

    # --- Real, measured columns (used directly from the workbook) ---
    it_mw = raw["Frontier Compute Power"].to_numpy()
    supply_c = raw["Overall Coolant Supply Temp"].to_numpy()
    return_c = raw["Overall-average Coolant Return Temp"].to_numpy()
    flow_lpm = raw["Overall Coolant FLow"].to_numpy() * GPM_TO_LPM
    cooling_mw = raw["Frontier Facility accessory Power"].to_numpy()
    total_mw = raw["Frontier Total Power"].to_numpy()
    pue_col = next(c for c in raw.columns if c.strip() == "Power Usage Effectiveness")
    pue = raw[pue_col].to_numpy()

    # --- Derived columns: not measured anywhere in the source dataset ---
    # Ambient dry-bulb: Frontier is sited at Oak Ridge, TN; no on-site ambient
    # sensor is published, so this reuses the same annual+diurnal sinusoidal
    # shape (and comparable amplitude/mean for that climate) already used in
    # generate_frontier2023() above, rather than inventing a new model.
    ambient_c = (
        15.0
        + 12.0 * np.sin(2 * np.pi * (doy.to_numpy() - 80) / 365.0)
        + 4.0 * np.sin(2 * np.pi * (hod.to_numpy() - 9) / 24.0)
    )
    # Per-rack inlet/outlet: the dataset has no per-rack sensors at all (the
    # report's own limitation: "no per-rack ... detail"). Reuses the same
    # supply-to-inlet offset src/digital_twin/physics_dynamics.py's
    # LiquidCoolingPhysics.thermal_balance() assumes (+2.0-2.5C) rather than
    # a fresh, unvalidated number.
    inlet_c = supply_c + 2.5
    outlet_c = inlet_c + it_mw * 0.8

    # Grid carbon intensity: not in this facility dataset at all. Reuses the
    # existing regional diurnal carbon model from
    # src/aws/serverless/lambda_carbon_fetcher.py (us-east-1 profile, the
    # same region this project already treats Oak Ridge/TVA territory as)
    # instead of a second, inconsistent carbon model.
    sys.path.insert(0, PROJECT_ROOT)
    from src.aws.serverless.lambda_carbon_fetcher import compute_diurnal_carbon

    carbon = np.array([
        compute_diurnal_carbon("us-east-1", timestamp=ts.to_pydatetime())["carbon_intensity_gco2_kwh"]
        for ts in timestamps
    ])

    df = pd.DataFrame({
        "timestamp": timestamps.dt.tz_localize(None),
        "ambient_temp_c": np.round(ambient_c, 2),
        "it_power_mw": np.round(it_mw, 3),
        "fws_supply_temp_c": np.round(supply_c, 2),
        "fws_return_temp_c": np.round(return_c, 2),
        "flow_rate_lpm": np.round(flow_lpm, 1),
        "server_inlet_temp_c": np.round(inlet_c, 2),
        "server_outlet_temp_c": np.round(outlet_c, 2),
        "cooling_power_mw": np.round(cooling_mw, 3),
        "total_facility_power_mw": np.round(total_mw, 3),
        "pue": np.round(pue, 4),
        "grid_carbon_gco2_kwh": np.round(carbon, 1),
    })

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path, index=False)
        print(f"[✓] Saved real Frontier2023 data to {save_path} ({len(df)} rows)")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=N_RECORDS,
                         help="Row count for --synthetic mode only; the real dataset's row count is fixed.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true",
                         help="Use the fabricated synthetic generator instead of downloading real data "
                              "(for offline dev/CI, or when Figshare is unreachable).")
    parser.add_argument("--force-download", action="store_true",
                         help="Re-download the real dataset even if a checksum-verified copy is cached.")
    args = parser.parse_args()

    out = args.output or os.path.join(RAW_DIR, "frontier2023_cooling_telemetry.parquet")

    if args.synthetic:
        df = generate_frontier2023(n=args.records, save_path=out)
    else:
        xlsx_path = download_frontier2023_raw(force=args.force_download)
        df = load_real_frontier2023(xlsx_path, save_path=out)

    print(df.head(3))


if __name__ == "__main__":
    main()
