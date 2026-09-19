"""Diurnal grid carbon-intensity profiles for the RL environment.

Mirrors src/aws/serverless/lambda_carbon_fetcher.compute_diurnal_carbon (a unit
test keeps the two in sync); duplicated here because the RL training scripts
run without the project root on sys.path and the Lambda module lives under the
AWS package.
"""
import math

REGIONAL_GRID_PROFILES = {
    "us-east-1": {"base_carbon": 340.0, "diurnal_amp": 75.0, "peak_hour": 18},
    "us-west-2": {"base_carbon": 140.0, "diurnal_amp": 45.0, "peak_hour": 20},
    "eu-west-1": {"base_carbon": 210.0, "diurnal_amp": 55.0, "peak_hour": 19},
    "ap-southeast-1": {"base_carbon": 420.0, "diurnal_amp": 60.0, "peak_hour": 17},
}

# Simulated facility -> grid region (same mapping the dashboard uses).
FACILITY_REGIONS = {
    "DC-EAST-01": "us-east-1",
    "DC-WEST-02": "us-west-2",
    "DC-EU-01": "eu-west-1",
}


# Regional climate baselines (mean dry-bulb degC, half-range degC); kept in sync with
# lambda_weather_fetcher.REGIONAL_CLIMATE_BASE by a unit test.
REGIONAL_CLIMATE = {
    "us-east-1": (22.0, 8.0),
    "us-west-2": (18.0, 9.0),
    "eu-west-1": (15.0, 6.0),
}


def diurnal_carbon_gco2_kwh(region: str, hour: float) -> float:
    profile = REGIONAL_GRID_PROFILES.get(region, REGIONAL_GRID_PROFILES["us-east-1"])
    solar_dip = 40.0 * math.exp(-((hour - 12.5) ** 2) / 10.0)
    evening_peak = profile["diurnal_amp"] * math.exp(-((hour - profile["peak_hour"]) ** 2) / 8.0)
    return max(50.0, profile["base_carbon"] + evening_peak - solar_dip)
