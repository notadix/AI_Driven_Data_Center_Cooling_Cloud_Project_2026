"""
Reduced-order 2D coolant-transport solver for the data-hall floor.

A steady advection-diffusion model of the coolant temperature over the hall:

    u(x,y) * dT/dx  =  kappa * laplacian(T)  +  S(x,y)

* coolant enters along the left edge at the (per-row) supply temperature and moves in +x
  with a local velocity proportional to the per-rack coolant flow,
* every rack is a heat source (its IT load, times the fraction the liquid loop captures),
  with the hotspot profile the project's rack layout uses (hotter in the middle of the hall),
* heat is also carried sideways by diffusion, so a hot rack warms its neighbours,
* the source is scaled so the hall-average temperature rise at the outlet equals the
  lumped energy balance  dT = q_captured / (m_dot * c_p)  (the model the twin's
  `LiquidCoolingPhysics.thermal_balance` uses), i.e. this solver ADDS the spatial structure
  that the lumped model cannot represent and conserves the same total heat.

This is a 2D reduced-order transport model, NOT a 3D CFD solution: it has no turbulence
model, buoyancy or geometry. It is used as the physics the FNO surrogate is trained to
reproduce and as the baseline for measuring the surrogate's speed-up.

Discretisation: first-order upwind advection + second-order central diffusion on an
n x n finite-volume grid (n a multiple of the 8x8 rack grid), sparse direct solve.
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

RACKS = 8                      # 8 x 8 racks
HEAT_CAPTURE = 0.92            # fraction of IT heat captured by the liquid loop
CP_KJ_KGK = 3.85               # 25% propylene glycol
DENSITY_KG_L = 1.04
PECLET = 20.0                  # u * L / kappa (domain-scale advection vs diffusion)


def hotspot_profile(racks: int = RACKS) -> np.ndarray:
    """Relative rack heat (mean 1): hotter towards the hall centre."""
    y, x = np.meshgrid(np.linspace(-1, 1, racks), np.linspace(-1, 1, racks), indexing="ij")
    hot = 1.0 + 0.35 * np.exp(-(x ** 2 + y ** 2) / 0.8)
    return hot / hot.mean()


def rack_inputs(it_mw: float, supply_c: float, flow_lpm: float, racks: int = RACKS) -> np.ndarray:
    """The 3-channel 8x8 input the FNO sees: per-rack IT (MW), supply temperature, per-rack flow (LPM)."""
    y, x = np.meshgrid(np.linspace(-1, 1, racks), np.linspace(-1, 1, racks), indexing="ij")
    hot = hotspot_profile(racks)
    ch0 = (it_mw / racks ** 2) * hot
    ch1 = supply_c + 0.05 * (x + y)
    ch2 = (flow_lpm / racks ** 2) * (1.0 / (hot + 0.1))
    return np.stack([ch0, ch1, ch2]).astype(np.float32)


def _upsample(field: np.ndarray, n: int) -> np.ndarray:
    return np.kron(field, np.ones((n // field.shape[0], n // field.shape[1])))


def solve_field(inputs: np.ndarray, n: int = 128, return_fine: bool = False):
    """Solve for the coolant temperature.

    inputs: (3, 8, 8) from `rack_inputs`. Returns the 8x8 rack-averaged temperature field (deg C),
    or (8x8, fine n x n) when return_fine=True.
    """
    it_mw_map, supply_map, flow_map = inputs
    if n % RACKS:
        raise ValueError("n must be a multiple of the 8x8 rack grid")

    q_rack = it_mw_map * 1000.0 * HEAT_CAPTURE                             # kW captured per rack
    m_rack = flow_map / 60.0 * DENSITY_KG_L                                 # kg/s per rack
    dT_lumped = q_rack.mean() / (m_rack.mean() * CP_KJ_KGK)                 # hall-average lumped rise (deg C)

    q_rel = _upsample(q_rack / q_rack.mean(), n)                            # heat pattern, mean 1
    u_rel = _upsample(flow_map / flow_map.mean(), n)                        # velocity pattern, mean 1
    T_in = np.repeat(_upsample(supply_map, n)[:, :1], 1, axis=1)[:, 0]      # inlet temperature per row

    dx = 1.0 / n
    kappa = 1.0 / PECLET
    # theta = T - T_in(row). Source normalised so the mean outlet rise equals dT_lumped:
    #   integral_0^1 S dx = dT_lumped * u  =>  S = dT_lumped * q_rel  (u_ref = 1)
    S = dT_lumped * q_rel

    idx = np.arange(n * n).reshape(n, n)                                    # row = y, col = x
    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r.ravel()); cols.append(c.ravel()); vals.append(np.broadcast_to(v, r.shape).ravel())

    diff = kappa / dx ** 2
    adv = u_rel / dx
    # interior + boundaries in one pass using masks
    center = idx
    # advection (upwind, flow in +x): u * (theta_i - theta_{i-1}) / dx ; inlet column has theta_{-1} = 0
    add(center, center, adv)
    add(center[:, 1:], center[:, :-1], -adv[:, 1:])
    # diffusion in x: -kappa * (theta_{i+1} - 2 theta_i + theta_{i-1}) / dx^2 ; Neumann at outlet, Dirichlet(0) at inlet
    add(center, center, 2 * diff)
    add(center[:, :-1], center[:, 1:], -diff)
    add(center[:, 1:], center[:, :-1], -diff)
    add(center[:, -1:], center[:, -1:], -diff)                              # Neumann outlet: ghost = interior
    # diffusion in y with Neumann walls
    add(center, center, 2 * diff)
    add(center[:-1, :], center[1:, :], -diff)
    add(center[1:, :], center[:-1, :], -diff)
    add(center[:1, :], center[:1, :], -diff)
    add(center[-1:, :], center[-1:, :], -diff)

    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))
    theta = spla.spsolve(A.tocsc(), S.ravel()).reshape(n, n)
    T = theta + T_in[:, None]

    block = n // RACKS
    coarse = T.reshape(RACKS, block, RACKS, block).mean(axis=(1, 3))
    return (coarse, T) if return_fine else coarse


def energy_balance_error(inputs: np.ndarray, n: int = 128) -> float:
    """|mean outlet rise - lumped rise| / lumped rise: how well the solver conserves the heat."""
    _, T = solve_field(inputs, n, return_fine=True)
    T_in = _upsample(inputs[1], n)[:, 0]
    outlet_rise = float((T[:, -1] - T_in).mean())
    q = inputs[0] * 1000.0 * HEAT_CAPTURE
    m = inputs[2] / 60.0 * DENSITY_KG_L
    lumped = float(q.mean() / (m.mean() * CP_KJ_KGK))
    return abs(outlet_rise - lumped) / lumped


def time_solver(inputs: np.ndarray, n: int = 128, repeats: int = 5) -> float:
    """Median wall-clock seconds for one solve."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        solve_field(inputs, n)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
