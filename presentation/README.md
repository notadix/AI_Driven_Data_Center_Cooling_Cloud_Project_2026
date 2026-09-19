# Presentation figures

Every figure is drawn from a measured file in `results/` (or a real rollout of the selected
agent) by `scripts/make_result_charts.py`; regenerate with

```bash
python scripts/make_result_charts.py
```

| Figure | Source | Shows |
|---|---|---|
| `rl_benchmark_comparison.png` | `results/rl_benchmark.json` | cooling energy, PUE and SLA-violating steps for constant setpoint, PID, Guideline-36-style rule and the selected Safe-PPO agent (30 unseen days) |
| `sla_compliance.png` | `results/rl_benchmark.json` | SLA-violating steps for every controller, including standard PPO and Lagrangian-without-shield ablations (5 seeds each) |
| `seed_variance.png` | `results/rl_benchmark.json` | per-seed cooling reduction; seeds with > 1% violations are marked "unsafe" |
| `twin_fidelity.png` | `results/twin_fidelity.json` | MAPE before/after calibrating the twin to Frontier2023, held-out 30% |
| `load_forecast.png` | `results/load_forecast_metrics.json` | GRU load forecast vs persistence vs hour-of-day mean |
| `carbon_water.png` | `results/carbon_water.json` | CO₂ from load shifting, cooling-energy and water savings per facility |
| `transfer_learning.png` | `results/transfer_learning.json` | zero-shot / fine-tuned / from-scratch on a new facility |
| `fno_metrics.png` | `results/fno_eval_metrics.json` | FNO error and latency (target field is an analytic model, see `docs/RESULTS.md` §2) |
| `pue_trajectory.png` | real rollout, seed 42 | one simulated day (144 steps of 10 minutes) for the controllers |

See `docs/RESULTS.md` for method and limitations before quoting any number.
