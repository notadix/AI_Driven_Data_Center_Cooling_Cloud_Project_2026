import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Dict
import numpy as np


# Observation bounds of DataCenterCoolingEnv (kept in sync by a unit test).
# The policy normalises raw observations to [-1, 1] with these fixed bounds
# internally: raw inputs span ~1e-2 (PUE offsets) to ~1e4 (IT kW, flow LPM), which
# made the first layer's pre-activations dominated by whichever feature had the
# biggest units. Doing it inside the network keeps every caller (live control,
# SageMaker handler, explainability) passing raw observations unchanged.
DEFAULT_OBS_LOW = [5000.0, -5.0, 50.0, 10.0, 15.0, 1000.0, 10.0, 20.0, 50.0, 1.0]
DEFAULT_OBS_HIGH = [30000.0, 45.0, 700.0, 30.0, 80.0, 30000.0, 35.0, 75.0, 4500.0, 2.0]


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 10, action_dim: int = 4, hidden: int = 128):
        super().__init__()
        if state_dim == len(DEFAULT_OBS_LOW):
            low = torch.tensor(DEFAULT_OBS_LOW, dtype=torch.float32)
            high = torch.tensor(DEFAULT_OBS_HIGH, dtype=torch.float32)
        else:  # unknown observation layout: identity normalisation
            low = torch.zeros(state_dim)
            high = torch.full((state_dim,), 2.0)
        self.register_buffer("obs_mid", (high + low) / 2.0)
        self.register_buffer("obs_half", (high - low) / 2.0)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.v_reward = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.v_cost = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, s: torch.Tensor):
        s = (s - self.obs_mid) / self.obs_half
        h = self.shared(s)
        mean = torch.tanh(self.actor_mean(h))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std, self.v_reward(h), self.v_cost(h)

    def act(self, s: torch.Tensor, deterministic=False):
        mean, std, vr, vc = self.forward(s)
        if deterministic:
            return mean, None, vr, vc
        dist = Normal(mean, std)
        a = torch.clamp(dist.sample(), -1.0, 1.0)
        lp = dist.log_prob(a).sum(-1, keepdim=True)
        return a, lp, vr, vc

    def evaluate(self, s: torch.Tensor, a: torch.Tensor):
        mean, std, vr, vc = self.forward(s)
        dist = Normal(mean, std)
        return dist.log_prob(a).sum(-1, keepdim=True), dist.entropy().sum(-1, keepdim=True), vr, vc


class SafePPOAgent:
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 4,
        lr: float = 3e-4,
        lr_lag: float = 5e-3,
        gamma: float = 0.99,
        lam_gae: float = 0.95,
        clip: float = 0.2,
        cost_limit: float = 0.05,
        device: str = "cpu",
        constrained: bool = True,
    ):
        # constrained=False is the standard-PPO baseline: the Lagrangian cost
        # advantage is ignored and safety is only a soft penalty already
        # present in the environment reward.
        self.constrained = constrained
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam_gae = lam_gae
        self.clip = clip
        self.cost_limit = cost_limit

        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)

        self.log_lam = nn.Parameter(torch.zeros(1, device=self.device))
        self.opt_lam = torch.optim.Adam([self.log_lam], lr=lr_lag)

    @property
    def lam(self) -> float:
        return float(F.softplus(self.log_lam).detach())

    def select_action(self, obs: np.ndarray, det: bool = False) -> Tuple[np.ndarray, float, float, float]:
        s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, lp, vr, vc = self.ac.act(s, deterministic=det)
        return (
            a.squeeze(0).cpu().numpy(),
            lp.item() if lp is not None else 0.0,
            vr.item(),
            vc.item(),
        )

    def gae(self, rews, costs, vrs, vcs, dones, last_vr, last_vc):
        n = len(rews)
        adv_r = np.zeros(n, np.float32)
        adv_c = np.zeros(n, np.float32)
        g_r = g_c = 0.0
        for t in reversed(range(n)):
            nxt_vr = last_vr if t == n - 1 else vrs[t + 1]
            nxt_vc = last_vc if t == n - 1 else vcs[t + 1]
            m = 0.0 if dones[t] else 1.0
            g_r = rews[t] + self.gamma * nxt_vr * m - vrs[t] + self.gamma * self.lam_gae * m * g_r
            g_c = costs[t] + self.gamma * nxt_vc * m - vcs[t] + self.gamma * self.lam_gae * m * g_c
            adv_r[t], adv_c[t] = g_r, g_c
        ret_r = adv_r + np.array(vrs, np.float32)
        ret_c = adv_c + np.array(vcs, np.float32)
        return (
            torch.tensor(adv_r, device=self.device),
            torch.tensor(adv_c, device=self.device),
            torch.tensor(ret_r, device=self.device),
            torch.tensor(ret_c, device=self.device),
        )

    def update(self, states, actions, old_lps, adv_r, adv_c, ret_r, ret_c, epochs=5, bs=64) -> Dict:
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)
        composite = adv_r - self.lam * adv_c if self.constrained else adv_r

        N = states.size(0)
        p_losses, vr_losses, vc_losses = [], [], []

        for _ in range(epochs):
            idx = torch.randperm(N)
            for s in range(0, N, bs):
                b = idx[s : s + bs]
                lps, ent, vr, vc = self.ac.evaluate(states[b], actions[b])
                ratio = torch.exp(lps - old_lps[b])
                adv = composite[b].unsqueeze(-1)
                loss_p = -torch.min(ratio * adv, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv).mean()
                loss_p -= 0.01 * ent.mean()
                loss_v = F.mse_loss(vr.squeeze(-1), ret_r[b]) + F.mse_loss(vc.squeeze(-1), ret_c[b])
                self.opt.zero_grad()
                (loss_p + 0.5 * loss_v).backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()
                p_losses.append(loss_p.item())
                vr_losses.append(F.mse_loss(vr.squeeze(-1), ret_r[b]).item())
                vc_losses.append(F.mse_loss(vc.squeeze(-1), ret_c[b]).item())

        mean_cost = ret_c.mean().item()
        if self.constrained:
            self.opt_lam.zero_grad()
            (-F.softplus(self.log_lam) * (mean_cost - self.cost_limit)).backward()
            self.opt_lam.step()

        return {
            "loss_policy": float(np.mean(p_losses)),
            "loss_vr": float(np.mean(vr_losses)),
            "loss_vc": float(np.mean(vc_losses)),
            "lagrangian": self.lam,
            "mean_cost": float(mean_cost),
        }
