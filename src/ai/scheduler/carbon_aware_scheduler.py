"""
Carbon-aware flexible-workload scheduler.

Deferrable IT work (batch jobs, training runs, checkpointing, backups) can be
delayed by up to `max_delay_h` hours. Given a carbon-intensity forecast, this
plans WHEN that flexible share of the load runs so that the facility's total
emissions are minimised, subject to

  * energy conservation  -- every unit of deferred work still runs
  * deadline             -- a job can only move later, by at most `max_delay_h`
  * capacity             -- IT power in any hour stays <= `capacity_factor` x the
                            baseline peak (no over-subscribing the plant)

The plan is an exact linear program (scipy.optimize.linprog / HiGHS), not a
heuristic. The objective weights each hour's IT energy by

    carbon_t * PUE_t          (cooling energy follows the IT load it removes)

so the cooling plant's own carbon is part of the optimisation, and a per-hour
water factor can be added to trade carbon against water use.

The day is treated as periodic (yesterday's deferred work lands in today's
early hours), which is the steady-state of a repeating daily schedule.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import linprog


@dataclass
class ShiftPlan:
    baseline_kw: np.ndarray            # (H,) IT power before shifting
    shifted_kw: np.ndarray             # (H,) IT power after shifting
    carbon_gco2_kwh: np.ndarray        # (H,) forecast used
    baseline_emissions_kg: float
    shifted_emissions_kg: float
    flexible_fraction: float
    max_delay_h: int
    moved_energy_kwh: float
    moves: List[Dict[str, float]] = field(default_factory=list)

    @property
    def reduction_pct(self) -> float:
        if self.baseline_emissions_kg <= 0:
            return 0.0
        return 100.0 * (self.baseline_emissions_kg - self.shifted_emissions_kg) / self.baseline_emissions_kg

    def to_dict(self) -> Dict:
        return {
            "flexible_fraction": self.flexible_fraction,
            "max_delay_h": self.max_delay_h,
            "baseline_kw": [round(float(v), 1) for v in self.baseline_kw],
            "shifted_kw": [round(float(v), 1) for v in self.shifted_kw],
            "carbon_gco2_kwh": [round(float(v), 1) for v in self.carbon_gco2_kwh],
            "baseline_emissions_kg": round(self.baseline_emissions_kg, 2),
            "shifted_emissions_kg": round(self.shifted_emissions_kg, 2),
            "emissions_reduction_pct": round(self.reduction_pct, 3),
            "moved_energy_kwh": round(self.moved_energy_kwh, 1),
            "moves": self.moves,
        }


def daily_emissions_kg(it_kw: np.ndarray, carbon_gco2_kwh: np.ndarray, pue: Sequence[float] | float = 1.05,
                       step_h: float = 1.0) -> float:
    """Facility emissions (kg CO2) for an hourly IT profile: IT energy x PUE x carbon intensity."""
    pue_arr = np.broadcast_to(np.asarray(pue, dtype=float), it_kw.shape)
    return float(np.sum(it_kw * step_h * pue_arr * carbon_gco2_kwh) / 1000.0)


def plan_shift(
    baseline_it_kw: Sequence[float],
    carbon_gco2_kwh: Sequence[float],
    flexible_fraction: float = 0.20,
    max_delay_h: int = 8,
    capacity_factor: float = 1.15,
    pue: Sequence[float] | float = 1.05,
    water_l_per_kwh: Optional[Sequence[float]] = None,
    water_weight_g_per_l: float = 0.0,
    step_h: float = 1.0,
) -> ShiftPlan:
    """Optimal delay-only shift of the flexible share of an hourly IT profile.

    water_l_per_kwh / water_weight_g_per_l optionally fold water use into the
    objective as an equivalent number of gCO2 per litre, so carbon and water
    can be traded off explicitly (0 = carbon only).
    """
    base = np.asarray(baseline_it_kw, dtype=float)
    carbon = np.asarray(carbon_gco2_kwh, dtype=float)
    H = len(base)
    if carbon.shape != base.shape:
        raise ValueError("carbon forecast and baseline profile must have the same length")
    if not 0.0 <= flexible_fraction <= 1.0:
        raise ValueError("flexible_fraction must be within [0, 1]")
    max_delay_h = int(max(0, min(max_delay_h, H - 1)))
    pue_arr = np.broadcast_to(np.asarray(pue, dtype=float), base.shape)

    cost = carbon * pue_arr
    if water_l_per_kwh is not None and water_weight_g_per_l > 0.0:
        cost = cost + water_weight_g_per_l * np.asarray(water_l_per_kwh, dtype=float)

    fixed = base * (1.0 - flexible_fraction)
    flex = base * flexible_fraction * step_h   # kWh of flexible work released at each hour

    # Variable x[i, d] = share of hour i's flexible work executed d hours later (0 <= d <= max_delay_h).
    D = max_delay_h + 1
    n = H * D
    c = np.zeros(n)
    for i in range(H):
        for d in range(D):
            c[i * D + d] = flex[i] * cost[(i + d) % H]

    # Each hour's flexible work must be fully executed: sum_d x[i, d] = 1.
    A_eq = np.zeros((H, n))
    for i in range(H):
        A_eq[i, i * D:(i + 1) * D] = 1.0
    b_eq = np.ones(H)

    # Capacity: fixed load + flexible work landing in hour j <= capacity_factor * baseline peak.
    cap_kw = capacity_factor * float(base.max())
    A_ub = np.zeros((H, n))
    for i in range(H):
        for d in range(D):
            A_ub[(i + d) % H, i * D + d] += flex[i] / step_h
    b_ub = cap_kw - fixed

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, 1.0), method="highs")
    if not res.success:
        # Infeasible only if the capacity cap is below the baseline itself: fall back to no shift.
        shifted = base.copy()
        moves: List[Dict[str, float]] = []
    else:
        x = res.x.reshape(H, D)
        shifted = fixed.copy()
        moves = []
        for i in range(H):
            for d in range(D):
                share = float(x[i, d])
                shifted[(i + d) % H] += flex[i] * share / step_h
                if d > 0 and share > 1e-6:
                    moves.append({"from_hour": i, "to_hour": (i + d) % H, "delay_h": d,
                                  "energy_kwh": round(float(flex[i] * share), 1)})

    base_kg = daily_emissions_kg(base, carbon, pue_arr, step_h)
    new_kg = daily_emissions_kg(shifted, carbon, pue_arr, step_h)
    moved = float(sum(m["energy_kwh"] for m in moves))
    return ShiftPlan(base, shifted, carbon, base_kg, new_kg, flexible_fraction, max_delay_h, moved, moves)
