"""
Feature attribution for Safe-PPO control decisions.

Uses gradient x input (saliency) attribution — a standard, cheap
approximation of SHAP for differentiable policies — to explain how much
each observation dimension pushed the actor's setpoint decision.
"""

import numpy as np
import torch

OBS_FEATURE_NAMES = [
    "it_power_kw", "ambient_c", "grid_carbon_gco2", "supply_c", "return_c",
    "flow_lpm", "server_inlet_c", "server_outlet_c", "cooling_kw", "pue",
]


def compute_feature_attribution(agent, obs: np.ndarray) -> dict:
    """
    Returns a dict of {feature_name: normalized_attribution} in [-1, 1],
    summing (in absolute value) to 1.0, explaining the actor's mean action
    for the given observation.
    """
    s = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    s.requires_grad_(True)

    mean, _, _, _ = agent.ac(s)
    agent.ac.zero_grad()
    mean.sum().backward()

    grads = s.grad.squeeze(0).detach().cpu().numpy()
    attribution = grads * obs
    total = float(np.sum(np.abs(attribution))) + 1e-8
    normalized = attribution / total

    return {name: float(v) for name, v in zip(OBS_FEATURE_NAMES, normalized)}
