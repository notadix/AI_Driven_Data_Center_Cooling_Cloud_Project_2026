"""Phase 3: carbon-aware scheduling and the shared water model."""

import numpy as np
import pytest

from src.aws.serverless.lambda_carbon_fetcher import compute_diurnal_carbon, REGIONAL_GRID_PROFILES as LAMBDA_PROFILES
from src.aws.serverless.lambda_weather_fetcher import compute_psychrometrics
from src.digital_twin.carbon_profiles import REGIONAL_GRID_PROFILES, diurnal_carbon_gco2_kwh
from src.digital_twin.cooling_sim_env import DataCenterCoolingEnv
from src.digital_twin.physics_dynamics import (
    LiquidCoolingPhysics, estimate_relative_humidity, wet_bulb_stull,
)
from src.ai.scheduler.carbon_aware_scheduler import plan_shift, daily_emissions_kg
from datetime import datetime, timezone


class TestDuplicatedModelsStayInSync:
    @pytest.mark.parametrize("region", list(REGIONAL_GRID_PROFILES))
    @pytest.mark.parametrize("hour", [0, 6, 12, 13, 18, 22])
    def test_carbon_profile_matches_lambda(self, region, hour):
        ts = datetime(2026, 1, 1, hour, 0, tzinfo=timezone.utc)
        lam = compute_diurnal_carbon(region, ts)["carbon_intensity_gco2_kwh"]
        assert abs(lam - diurnal_carbon_gco2_kwh(region, float(hour))) < 0.01

    def test_profiles_have_same_regions_and_parameters(self):
        for region, p in REGIONAL_GRID_PROFILES.items():
            for k in ("base_carbon", "diurnal_amp", "peak_hour"):
                assert LAMBDA_PROFILES[region][k] == p[k]

    @pytest.mark.parametrize("t", [-2.0, 8.0, 15.0, 22.0, 31.0])
    def test_wet_bulb_matches_psychrometrics(self, t):
        rh = estimate_relative_humidity(t)
        ref = compute_psychrometrics(t, rh)["wet_bulb_temp_c"]
        assert abs(ref - wet_bulb_stull(t, rh)) < 0.01


class TestWaterModel:
    def test_water_scales_with_chiller_load_and_humidity(self):
        phys = LiquidCoolingPhysics()
        assert phys.water_use_l_per_hr(2000.0, 20.0) == pytest.approx(2 * phys.water_use_l_per_hr(1000.0, 20.0))
        assert phys.water_use_l_per_hr(1000.0, 30.0) > phys.water_use_l_per_hr(1000.0, 5.0)

    def test_iot_simulator_wue_uses_shared_model(self):
        from src.aws.iot.iot_publisher import PhysicsSimulator
        sim = PhysicsSimulator("DC-EAST-01", "CRAC-01", "RACK-A01", ambient_c=20.0)
        assert sim._compute_wue(1.0, 18.0, 20.0) == pytest.approx(sim._physics.wue(1.0, 18.0, 20.0))

    def test_env_reports_water_and_carbon(self):
        env = DataCenterCoolingEnv()
        obs, _ = env.reset(seed=1)
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert info["water_l_hr"] > 0 and info["wue"] > 0 and 50 <= info["carbon_gco2_kwh"] <= 580
        assert info["facility_emissions_kg_hr"] > info["emissions_kg_hr"]


class TestEnvironmentCarbonAndProfile:
    def test_carbon_follows_diurnal_profile(self):
        env = DataCenterCoolingEnv(region="us-east-1")
        env.reset(seed=3)
        cs = []
        for _ in range(144):
            _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
            cs.append(info["carbon_gco2_kwh"])
        assert max(cs) - min(cs) > 60.0     # evening peak vs midday dip

    def test_regions_differ(self):
        def mean_carbon(region):
            env = DataCenterCoolingEnv(region=region)
            env.reset(seed=5)
            return np.mean([env.step(np.zeros(4, dtype=np.float32))[4]["carbon_gco2_kwh"] for _ in range(144)])
        assert mean_carbon("us-west-2") < mean_carbon("eu-west-1") < mean_carbon("us-east-1")

    def test_it_profile_is_followed(self):
        profile = np.full(24, 15000.0); profile[12] = 25000.0
        env = DataCenterCoolingEnv(it_profile_kw=profile)
        env.reset(seed=0)
        seen = {}
        for _ in range(144):
            _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
            seen[round(env._hour())] = info["it_kw"]
        assert seen[12] > seen[3] + 5000


class TestScheduler:
    HOURS = np.arange(24)
    CARBON = np.array([diurnal_carbon_gco2_kwh("us-east-1", float(h)) for h in HOURS])
    BASE = np.full(24, 19000.0)

    def test_energy_is_conserved_and_emissions_do_not_increase(self):
        plan = plan_shift(self.BASE, self.CARBON, flexible_fraction=0.25, max_delay_h=8)
        assert plan.shifted_kw.sum() == pytest.approx(self.BASE.sum(), rel=1e-6)
        assert plan.shifted_emissions_kg <= plan.baseline_emissions_kg + 1e-6
        assert plan.reduction_pct > 0.5

    def test_capacity_and_deadline_respected(self):
        plan = plan_shift(self.BASE, self.CARBON, flexible_fraction=0.5, max_delay_h=6, capacity_factor=1.10)
        assert plan.shifted_kw.max() <= 1.10 * self.BASE.max() + 1e-6
        assert all(0 < m["delay_h"] <= 6 for m in plan.moves)

    def test_no_flexibility_means_no_change(self):
        plan = plan_shift(self.BASE, self.CARBON, flexible_fraction=0.0)
        assert np.allclose(plan.shifted_kw, self.BASE) and plan.reduction_pct == 0.0

    def test_flat_carbon_gives_no_saving(self):
        plan = plan_shift(self.BASE, np.full(24, 300.0), flexible_fraction=0.3)
        assert abs(plan.reduction_pct) < 1e-6

    def test_more_flexibility_never_hurts(self):
        r = [plan_shift(self.BASE, self.CARBON, flexible_fraction=f, max_delay_h=8).reduction_pct for f in (0.1, 0.2, 0.4)]
        assert r[0] <= r[1] + 1e-9 <= r[2] + 2e-9

    def test_water_weight_trades_off_against_carbon(self):
        wue = np.linspace(0.05, 0.6, 24)      # water intensity rises through the day
        carbon_only = plan_shift(self.BASE, self.CARBON, 0.3, 8)
        with_water = plan_shift(self.BASE, self.CARBON, 0.3, 8, water_l_per_kwh=wue, water_weight_g_per_l=2000.0)
        water = lambda p: float((p.shifted_kw * wue).sum())
        assert water(with_water) <= water(carbon_only) + 1e-6
        assert with_water.shifted_emissions_kg >= carbon_only.shifted_emissions_kg - 1e-6

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            plan_shift(self.BASE, self.CARBON[:10])
        with pytest.raises(ValueError):
            plan_shift(self.BASE, self.CARBON, flexible_fraction=1.5)


class TestCarbonPlanApi:
    @pytest.fixture(scope="class")
    def client(self):
        import os
        os.environ["LOCAL_MODE"] = "true"
        from fastapi.testclient import TestClient
        from src.backend.main import app
        with TestClient(app) as c:
            yield c

    @pytest.mark.parametrize("facility", ["DC-EAST-01", "DC-WEST-02", "DC-EU-01"])
    def test_plan_for_every_facility(self, client, facility):
        r = client.get(f"/api/v1/optimization/carbon-plan/{facility}")
        assert r.status_code == 200
        d = r.json()["data"]
        assert len(d["shifted_kw"]) == 24 and d["emissions_reduction_pct"] >= 0
        assert abs(sum(d["shifted_kw"]) - sum(d["baseline_kw"])) < 24.0   # energy conserved (rounding)

    def test_unknown_facility_404_and_bad_params_422(self, client):
        assert client.get("/api/v1/optimization/carbon-plan/DC-NOPE").status_code == 404
        assert client.get("/api/v1/optimization/carbon-plan/DC-EAST-01?flexible_fraction=2").status_code == 422


def test_climate_table_matches_weather_lambda():
    from src.digital_twin.carbon_profiles import REGIONAL_CLIMATE
    from src.aws.serverless.lambda_weather_fetcher import REGIONAL_CLIMATE_BASE
    from src.digital_twin.carbon_profiles import FACILITY_REGIONS
    for facility, region in FACILITY_REGIONS.items():
        base, half = REGIONAL_CLIMATE[region]
        ref = REGIONAL_CLIMATE_BASE[facility]
        assert base == ref["base_dry_bulb"] and half == ref["diurnal_range"]


def test_env_climate_option_changes_ambient_range():
    def start_ambients(region, climate):
        env = DataCenterCoolingEnv(region=region, climate=climate)
        return [env.reset(seed=s)[0][1] for s in range(40)]
    eu = start_ambients("eu-west-1", True)
    default = start_ambients("eu-west-1", False)
    assert max(eu) <= 21.0 + 1e-6 and min(eu) >= 9.0 - 1e-6
    assert max(default) > 25.0          # original 10-32 degC range untouched when climate=False
