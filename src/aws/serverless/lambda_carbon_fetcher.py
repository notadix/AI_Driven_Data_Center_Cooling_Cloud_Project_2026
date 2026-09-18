"""
AWS Lambda: Green Grid Marginal Carbon Intensity Fetcher.

Ingests marginal carbon emissions (gCO2e/kWh) and renewable generation breakdown
from regional grid APIs (WattTime / Electricity Maps) to support multi-objective
sustainability optimization within the data center cooling digital twin.

Fallback mode provides realistic diurnal carbon variations based on regional
generation profiles when external credentials are not configured.
"""

import json
import logging
import math
import os
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Regional baseline parameters (gCO2e/kWh, renewable mix %)
REGIONAL_GRID_PROFILES = {
    "us-east-1": {"base_carbon": 340.0, "diurnal_amp": 75.0, "peak_hour": 18, "renewable_base": 28.0},
    "us-west-2": {"base_carbon": 140.0, "diurnal_amp": 45.0, "peak_hour": 20, "renewable_base": 68.0},
    "eu-west-1": {"base_carbon": 210.0, "diurnal_amp": 55.0, "peak_hour": 19, "renewable_base": 52.0},
    "ap-southeast-1": {"base_carbon": 420.0, "diurnal_amp": 60.0, "peak_hour": 17, "renewable_base": 18.0},
}


def compute_diurnal_carbon(region: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    """Generates physically plausible regional grid marginal carbon intensity and fuel mix."""
    profile = REGIONAL_GRID_PROFILES.get(region, REGIONAL_GRID_PROFILES["us-east-1"])
    ts = timestamp or datetime.now(timezone.utc)
    hour = ts.hour + ts.minute / 60.0

    # Diurnal variation: lower during midday solar peak, higher during evening ramp
    solar_dip = 40.0 * math.exp(-((hour - 12.5) ** 2) / 10.0)
    evening_peak = profile["diurnal_amp"] * math.exp(-((hour - profile["peak_hour"]) ** 2) / 8.0)
    carbon_intensity = max(50.0, profile["base_carbon"] + evening_peak - solar_dip)

    # Renewable mix inversely correlates with carbon intensity
    renewable_pct = max(5.0, min(95.0, profile["renewable_base"] + (solar_dip * 0.4) - (evening_peak * 0.2)))

    # Classification
    if carbon_intensity < 180.0:
        level = "ULTRA_CLEAN"
    elif carbon_intensity < 300.0:
        level = "CLEAN"
    elif carbon_intensity < 420.0:
        level = "MODERATE"
    else:
        level = "DIRTY"

    return {
        "region": region,
        "timestamp": ts.isoformat(),
        "carbon_intensity_gco2_kwh": round(carbon_intensity, 2),
        "renewable_mix_pct": round(renewable_pct, 1),
        "grid_signal_level": level,
        "fuel_mix": {
            "solar_pct": round(max(0.0, 30.0 * math.exp(-((hour - 12.5) ** 2) / 8.0)), 1),
            "wind_pct": round(max(5.0, renewable_pct * 0.5), 1),
            "hydro_nuclear_pct": round(max(10.0, renewable_pct * 0.4), 1),
            "gas_coal_pct": round(max(5.0, 100.0 - renewable_pct), 1),
        },
        "source": "digital_twin_grid_model",
    }


def fetch_electricity_maps(zone: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Optional external HTTP query to Electricity Maps API."""
    url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"
    req = urllib.request.Request(url, headers={"auth-token": api_key})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return {
                "region": zone,
                "timestamp": data.get("datetime", datetime.now(timezone.utc).isoformat()),
                "carbon_intensity_gco2_kwh": float(data.get("carbonIntensity", 300.0)),
                "source": "electricity_maps_api",
            }
    except Exception as e:
        logger.warning("External Electricity Maps API fetch failed: %s. Falling back to local model.", e)
        return None


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    AWS Lambda entry point.
    Accepts:
      event = {"region": "us-east-1", "facility_id": "DC-EAST-01"}
    """
    query_params = event.get("queryStringParameters") or {}
    region = event.get("region") or query_params.get("region") or os.environ.get("AWS_REGION", "us-east-1")
    facility_id = event.get("facility_id") or query_params.get("facility_id", "DC-EAST-01")

    api_key = os.environ.get("ELECTRICITY_MAPS_API_KEY")
    result = None
    if api_key:
        result = fetch_electricity_maps(region, api_key)

    if not result:
        result = compute_diurnal_carbon(region)

    result["facility_id"] = facility_id

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps({"status": "ok", "data": result}),
    }
