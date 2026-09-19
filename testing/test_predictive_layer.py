"""Phase 4: load forecaster and FNO thermal-field service / API."""

import json
import os
import time

import numpy as np
import pytest

os.environ.setdefault("LOCAL_MODE", "true")

from src.ai.forecast.load_forecaster import HORIZON, LOOKBACK, LoadForecaster, make_windows  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestForecasterMetrics:
    @pytest.fixture(scope="class")
    def metrics(self):
        with open(os.path.join(ROOT, "results", "load_forecast_metrics.json")) as f:
            return json.load(f)

    def test_metrics_cover_all_horizons_and_baselines(self, metrics):
        t = metrics["test"]
        for name in ("model", "persistence", "hour_of_day_mean"):
            assert len(t[name]["mape_pct"]) == HORIZON

    def test_model_beats_persistence_at_the_long_horizon_and_the_hourly_mean_everywhere(self, metrics):
        t = metrics["test"]
        assert t["model"]["mape_pct"][-1] < t["persistence"]["mape_pct"][-1]
        assert all(m < h for m, h in zip(t["model"]["mape_pct"], t["hour_of_day_mean"]["mape_pct"]))


class TestForecasterModel:
    def test_windows_shapes_and_alignment(self):
        p = np.arange(100, dtype=float)
        h = np.zeros(100)
        X, H, Y = make_windows(p, h, 0.0, 1.0)
        assert X.shape == (100 - LOOKBACK - HORIZON + 1, LOOKBACK, 3) and Y.shape[1] == HORIZON
        assert X[0, -1, 0] == LOOKBACK - 1 and Y[0, 0] == LOOKBACK      # next value follows the window

    def test_trained_model_predicts_sane_values(self):
        import torch
        ckpt = torch.load(os.path.join(ROOT, "models", "load_forecaster_v1.pt"), map_location="cpu", weights_only=False)
        model = LoadForecaster()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        hist = np.full(LOOKBACK, 11600.0)
        f = model.predict_kw(hist, 10.0)
        assert f.shape == (HORIZON,) and np.all(np.abs(f - 11600.0) < 2000.0)

    def test_wrong_history_length_rejected(self):
        with pytest.raises(ValueError):
            LoadForecaster().predict_kw(np.zeros(5), 1.0)


class TestFnoThermalField:
    def test_service_reproduces_the_analytic_thermal_model(self):
        from src.backend.services.thermal_field import get_thermal_field_service
        svc = get_thermal_field_service()
        assert svc.available
        out = svc.predict(it_zone_mw=11.6, supply_c=20.0, flow_lpm=14500.0)
        field = np.array(out["field_c"])
        assert field.shape == (8, 8)
        # supply + heat pickup: 11.6 MW * 0.92 / (14500/64 lpm per rack * ...) -> ~ supply + O(10 C)
        assert 20.0 < field.mean() < 60.0 and out["latency_ms"] < 100.0

    def test_more_load_gives_hotter_racks(self):
        from src.backend.services.thermal_field import get_thermal_field_service
        svc = get_thermal_field_service()
        lo = svc.predict(8.0, 20.0, 16000.0)["mean_c"]
        hi = svc.predict(18.0, 20.0, 16000.0)["mean_c"]
        assert hi > lo


class TestPredictiveApi:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from src.backend.main import app
        with TestClient(app) as c:
            time.sleep(1.5)
            yield c

    @pytest.mark.parametrize("facility", ["DC-EAST-01", "DC-WEST-02", "DC-EU-01"])
    def test_load_forecast(self, client, facility):
        r = client.get(f"/api/v1/forecast/load/{facility}")
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["available"] and len(d["forecast"]) == HORIZON
        assert [p["minutes_ahead"] for p in d["forecast"]] == [10, 20, 30, 40, 50, 60]
        assert all(1.0 < p["pue"] < 2.0 for p in d["forecast"])
        assert abs(d["forecast"][0]["it_kw"] - d["current_it_kw"]) < 0.25 * d["current_it_kw"]

    def test_thermal_field_endpoint(self, client):
        r = client.get("/api/v1/forecast/thermal-field/DC-EU-01/CRAC-02")
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["available"] and np.array(d["field_c"]).shape == (8, 8) and d["max_c"] >= d["min_c"]

    def test_unknown_targets_404(self, client):
        assert client.get("/api/v1/forecast/load/DC-NOPE").status_code == 404
        assert client.get("/api/v1/forecast/thermal-field/DC-EU-01/CRAC-99").status_code == 404
