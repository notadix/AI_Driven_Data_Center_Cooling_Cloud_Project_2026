"""The 2D transport solver, and the FNO surrogate trained to reproduce it."""

import json
import os

import numpy as np
import pytest

from src.digital_twin.thermal_solver import (
    RACKS, energy_balance_error, hotspot_profile, rack_inputs, solve_field,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
X = rack_inputs(it_mw=11.6, supply_c=20.0, flow_lpm=14500.0)


class TestSolverPhysics:
    def test_shape_and_finite(self):
        f = solve_field(X, 64)
        assert f.shape == (RACKS, RACKS) and np.isfinite(f).all()

    def test_coolant_never_colder_than_supply_and_warms_along_the_flow(self):
        f = solve_field(X, 64)
        assert f.min() >= X[1].min() - 1e-6                 # heat only ever raises the coolant temperature
        assert f[:, -1].mean() > f[:, 0].mean() + 3.0        # outlet side is warmer than the inlet side

    def test_heat_is_conserved_to_a_few_percent(self):
        assert energy_balance_error(X, 128) < 0.06

    def test_grid_converged(self):
        a, b = solve_field(X, 96), solve_field(X, 192)
        assert np.abs(a - b).max() < 0.15                    # deg C

    def test_more_heat_is_hotter_more_flow_is_cooler(self):
        base = solve_field(X, 64).mean()
        assert solve_field(rack_inputs(18.0, 20.0, 14500.0), 64).mean() > base
        assert solve_field(rack_inputs(11.6, 20.0, 22000.0), 64).mean() < base

    def test_supply_temperature_shifts_the_field_one_for_one(self):
        d = solve_field(rack_inputs(11.6, 25.0, 14500.0), 64) - solve_field(X, 64)
        assert np.allclose(d, 5.0, atol=0.02)

    def test_hotspot_racks_are_the_hottest_in_their_column(self):
        f = solve_field(X, 96)
        centre = f[3:5, 3:5].mean()
        assert centre > f[0, 3:5].mean() and centre > f[7, 3:5].mean()

    def test_hotspot_profile_has_mean_one(self):
        assert hotspot_profile().mean() == pytest.approx(1.0)

    def test_rack_grid_must_divide_the_solver_grid(self):
        with pytest.raises(ValueError):
            solve_field(X, 100)


class TestFnoSurrogateOfTheSolver:
    @pytest.fixture(scope="class")
    def metrics(self):
        with open(os.path.join(ROOT, "results", "fno_pde_eval.json")) as f:
            return json.load(f)

    def test_recorded_accuracy_and_speed(self, metrics):
        m = metrics["test_metrics"]
        assert m["r2"] > 0.999 and m["mae_c"] < 0.1
        assert m["mae_c"] < m["uniform_field_oracle_mae_c"] / 20      # it learned the spatial structure
        assert metrics["speedup_vs_solver"]["128"] > 5

    def test_live_service_agrees_with_the_solver_on_unseen_operating_points(self):
        from src.backend.services.thermal_field import get_thermal_field_service
        svc = get_thermal_field_service()
        assert svc.available
        errs = []
        for it, sup, flow in [(9.0, 18.0, 13000.0), (12.5, 21.0, 15500.0), (15.0, 24.0, 19000.0), (7.0, 16.0, 12000.0)]:
            truth = solve_field(rack_inputs(it, sup, flow), 128)
            pred = np.array(svc.predict(it, sup, flow)["field_c"])
            errs.append(np.abs(pred - truth).mean())
        assert max(errs) < 0.3                                          # deg C mean abs error

    def test_field_responds_to_load(self):
        from src.backend.services.thermal_field import get_thermal_field_service
        svc = get_thermal_field_service()
        assert svc.predict(18.0, 20.0, 16000.0)["mean_c"] > svc.predict(8.0, 20.0, 16000.0)["mean_c"]
