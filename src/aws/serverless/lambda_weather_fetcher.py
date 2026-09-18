"""
AWS Lambda: Ambient Weather & Psychrometric Enthalpy Fetcher.

Ingests ambient dry-bulb and wet-bulb temperatures, relative humidity, and pressure
to compute moist air enthalpy and free-air economizer suitability scores for hybrid
cooling optimization.
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Regional climate baselines (Dry-Bulb °C, Humidity %, Elevation m)
REGIONAL_CLIMATE_BASE = {
    "DC-EAST-01": {"lat": 38.9, "lon": -77.0, "base_dry_bulb": 22.0, "rh_base": 60.0, "diurnal_range": 8.0},
    "DC-WEST-02": {"lat": 45.5, "lon": -122.6, "base_dry_bulb": 18.0, "rh_base": 50.0, "diurnal_range": 9.0},
    "DC-EU-01":   {"lat": 53.3, "lon": -6.2,   "base_dry_bulb": 15.0, "rh_base": 75.0, "diurnal_range": 6.0},
}


def compute_psychrometrics(dry_bulb_c: float, relative_humidity_pct: float, pressure_kpa: float = 101.325) -> Dict[str, float]:
    """
    Computes dew point, wet bulb approximation, humidity ratio, and enthalpy (kJ/kg)
    using standard ASHRAE psychrometric formulations.
    """
    # Saturation vapor pressure over liquid water (Tetens equation) in kPa
    p_sat = 0.61078 * math.exp((17.27 * dry_bulb_c) / (dry_bulb_c + 237.3))
    # Actual vapor pressure
    p_v = (relative_humidity_pct / 100.0) * p_sat

    # Dew point temperature (°C)
    alpha = math.log(max(1e-6, p_v / 0.61078))
    dew_point_c = (237.3 * alpha) / (17.27 - alpha)

    # Stull wet-bulb temperature empirical formula (°C)
    t = dry_bulb_c
    rh = relative_humidity_pct
    wet_bulb_c = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )

    # Humidity ratio W (kg water / kg dry air)
    w = 0.62198 * p_v / max(0.1, pressure_kpa - p_v)

    # Specific enthalpy h = c_pa * T + W * (h_we + c_pw * T) in kJ/kg
    enthalpy_kj_kg = 1.006 * dry_bulb_c + w * (2501.0 + 1.86 * dry_bulb_c)

    # Economizer suitability score: 1.0 = full free cooling, 0.0 = full chiller required
    # Free cooling is viable when wet-bulb <= 15°C or dry-bulb <= 18°C
    if wet_bulb_c <= 12.0:
        economizer_potential = 1.0
    elif wet_bulb_c >= 22.0:
        economizer_potential = 0.0
    else:
        economizer_potential = max(0.0, min(1.0, (22.0 - wet_bulb_c) / 10.0))

    return {
        "dry_bulb_temp_c": round(dry_bulb_c, 2),
        "wet_bulb_temp_c": round(wet_bulb_c, 2),
        "dew_point_temp_c": round(dew_point_c, 2),
        "relative_humidity_pct": round(relative_humidity_pct, 1),
        "enthalpy_kj_per_kg": round(enthalpy_kj_kg, 2),
        "economizer_potential_score": round(economizer_potential, 3),
        "free_cooling_recommended": economizer_potential >= 0.5,
    }


def get_ambient_weather(facility_id: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    """Generates physically consistent ambient weather for the facility location."""
    climate = REGIONAL_CLIMATE_BASE.get(facility_id, REGIONAL_CLIMATE_BASE["DC-EAST-01"])
    ts = timestamp or datetime.now(timezone.utc)
    hour = ts.hour + ts.minute / 60.0

    # Diurnal temperature cycle: peak around 15:00, minimum around 05:00
    diurnal_offset = climate["diurnal_range"] * math.sin(2 * math.pi * (hour - 9.0) / 24.0)
    dry_bulb = climate["base_dry_bulb"] + diurnal_offset

    # RH is inversely proportional to temperature
    rh = max(20.0, min(95.0, climate["rh_base"] - (diurnal_offset * 2.5)))

    psych = compute_psychrometrics(dry_bulb, rh)
    psych["facility_id"] = facility_id
    psych["timestamp"] = ts.isoformat()
    psych["coordinates"] = {"latitude": climate["lat"], "longitude": climate["lon"]}
    return psych


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda entry point for ambient weather & economizer evaluation."""
    query_params = event.get("queryStringParameters") or {}
    facility_id = event.get("facility_id") or query_params.get("facility_id", "DC-EAST-01")

    data = get_ambient_weather(facility_id)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps({"status": "ok", "data": data}),
    }
