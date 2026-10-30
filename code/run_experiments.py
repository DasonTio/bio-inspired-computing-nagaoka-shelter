"""Final experiments: 30 seeds x {GA, PSO} x 3 evacuation scenarios, matched budget.

Outputs (results/):
  runs.csv     one row per run (score, its components, evaluations, runtime, seed)
  curves.npz   best-so-far score at every 500 evaluations, per run
  baselines.csv  nearest, greedy and the linear exact reference per scenario
"""
import json, time, numpy as np, pandas as pd
from shelter_model import Problem
from algorithms import genetic_algorithm, particle_swarm
import baselines as bl

BUDGET, SEEDS, SCENARIOS, STEP = 50000, range(30), [0.2, 0.3, 0.4], 500
cfg = json.load(open("results/chosen_params.json"))
grid = np.arange(STEP, BUDGET + 1, STEP)


def on_grid(curve):
    ev, f = np.array(curve).T
    return f[np.searchsorted(ev, grid, side="right") - 1]


runs, curves, base = [], {}, []
for r in SCENARIOS:
    p = Problem("nagaoka_cells_2020.csv", "nagaoka_shelters_raw.csv", evac_rate=r)
    for name, x in [("nearest", bl.nearest(p)), ("greedy", bl.greedy(p))]:
        c = p.components(x)
        base.append(dict(scenario=r, method=name, F=p.evaluate(x), **{k: v for k, v in c.items() if k != "load"}))
    for alg, fn, kw in [("GA", genetic_algorithm, cfg["GA"]), ("PSO", particle_swarm, cfg["PSO"])]:
        for s in SEEDS:
            t0, e0 = time.time(), p.n_evals
            x, f, cur = fn(p, BUDGET, s, **kw)
            c = p.components(x)
            runs.append(dict(scenario=r, alg=alg, seed=s, F=f, evals=p.n_evals - e0,
                             runtime_s=time.time() - t0,
                             **{k: v for k, v in c.items() if k != "load"}))
            curves[f"{alg}_{r}_{s}"] = on_grid(cur)
        print(f"scenario {r} {alg} done", flush=True)

pd.DataFrame(runs).to_csv("results/runs.csv", index=False)
pd.DataFrame(base).to_csv("results/baselines.csv", index=False)
np.savez("results/curves.npz", grid=grid, **curves)
print("saved")
