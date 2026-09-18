import os
import sys
import json
import io
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ai", "surrogate"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ai", "rl"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "digital_twin"))

from fno_model import FNO2d
from safe_ppo import SafePPOAgent, ActorCritic

_fno_model = None
_ppo_agent = None
_norm_stats = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_norm_stats(model_dir: str):
    stats_path = os.path.join(model_dir, "normalization_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return None


def initialize(context):
    global _fno_model, _ppo_agent, _norm_stats
    model_dir = context.system_properties.get("model_dir", "/opt/ml/model")

    fno_path = os.path.join(model_dir, "fno_surrogate_v1.pt")
    if os.path.exists(fno_path):
        # weights_only=False: these are always this project's own
        # locally-produced checkpoints, not externally-sourced files.
        ckpt = torch.load(fno_path, map_location=_device, weights_only=False)
        arch = ckpt.get("architecture", {})
        _fno_model = FNO2d(**arch).to(_device)
        _fno_model.load_state_dict(ckpt["model_state_dict"])
        _fno_model.eval()

    ppo_path = os.path.join(model_dir, "safe_ppo_agent_v1.pt")
    if os.path.exists(ppo_path):
        ckpt = torch.load(ppo_path, map_location=_device, weights_only=False)
        hp = ckpt.get("hyperparams", {"state_dim": 10, "action_dim": 4})
        _ppo_agent = ActorCritic(hp["state_dim"], hp["action_dim"]).to(_device)
        _ppo_agent.load_state_dict(ckpt["ac_state_dict"])
        _ppo_agent.eval()

    _norm_stats = _load_norm_stats(model_dir)


def preprocess(request):
    body = request[0].get("body")
    if isinstance(body, (bytes, bytearray)):
        data = json.loads(body.decode("utf-8"))
    else:
        data = json.loads(body)
    return data


def inference(data: dict) -> dict:
    request_type = data.get("type", "setpoints")

    if request_type == "thermal_field" and _fno_model is not None:
        fields = data["input_fields"]
        x = torch.tensor(fields, dtype=torch.float32, device=_device).unsqueeze(0)
        with torch.no_grad():
            pred = _fno_model(x).squeeze(0).cpu().tolist()
        return {"type": "thermal_field", "temperature_field": pred}

    if request_type == "setpoints" and _ppo_agent is not None:
        obs = np.array(data["observation"], dtype=np.float32)
        s = torch.tensor(obs, dtype=torch.float32, device=_device).unsqueeze(0)
        with torch.no_grad():
            mean, _, vr, vc = _ppo_agent(s)
            action = mean.squeeze(0).cpu().tolist()
        delta_supply = action[0] * 1.5
        pump_pct = 35.0 + (action[1] + 1.0) * 0.5 * 65.0
        fan_pct = 30.0 + (action[2] + 1.0) * 0.5 * 70.0
        valve_pct = (action[3] + 1.0) * 0.5 * 40.0
        return {
            "type": "setpoints",
            "control": {
                "delta_supply_c": round(delta_supply, 3),
                "pump_speed_pct": round(pump_pct, 2),
                "fan_speed_pct": round(fan_pct, 2),
                "valve_split_pct": round(valve_pct, 2),
            },
            "v_reward": float(vr.item()),
            "v_cost": float(vc.item()),
        }

    return {"error": "No valid model loaded for request type: " + request_type}


def postprocess(inference_output: dict) -> list:
    return [{"body": json.dumps(inference_output).encode("utf-8"), "content_type": "application/json"}]


def handle(data, context):
    if not _fno_model and not _ppo_agent:
        initialize(context)
    processed = preprocess(data)
    output = inference(processed)
    return postprocess(output)
