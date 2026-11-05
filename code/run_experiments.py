"""Final experiments: 30 seeds x {GA, PSO} x 3 evacuation scenarios, matched budget.

Can be run in parts (each part takes about 1.5 minutes):
  python run_experiments.py 0.3 GA      one scenario and one algorithm
  python run_experiments.py merge       combine the parts and add the baselines
  python run_experiments.py             everything in one go

Outputs (../results/):
  runs.csv       one row per run (score, its components, evaluations, runtime, seed)
  curves.npz     best-so-far score at every 500 evaluations, per run
  baselines.csv  nearest and greedy per scenario
  best_assign.npz  best assignment (shelter id per cell) of each algorithm per scenario
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd
from shelter_model import Problem
from algorithms import genetic_algorithm, particle_swarm
import baselines as bl

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PARTS = os.path.join(RES, "parts")
BUDGET, SEEDS, SCENARIOS, STEP = 50000, range(30), [0.2, 0.3, 0.4], 500
cfg = json.load(open(os.path.join(RES, "chosen_params.json")))
ALGS = {"GA": genetic_algorithm, "PSO": particle_swarm}
grid = np.arange(STEP, BUDGET + 1, STEP)


def on_grid(curve):
    ev, f = np.array(curve).T
    return f[np.searchsorted(ev, grid, side="right") - 1]


def part(r, alg):
    p = Problem(evac_rate=r)
    runs, curves, best = [], {}, None
    for s in SEEDS:
        t0, e0 = time.time(), p.n_evals
        x, f, cur = ALGS[alg](p, BUDGET, s, **cfg[alg])
        c = p.components(x)
        runs.append(dict(scenario=r, alg=alg, seed=s, F=f, evals=p.n_evals - e0,
                         runtime_s=time.time() - t0, feasible_assignment=bool(((x >= 0) & (x < p.n_valid)).all()),
                         **{k: v for k, v in c.items() if k != "load"}))
        curves[f"{alg}_{r}_{s}"] = on_grid(cur)
        if best is None or f < best[1]:
            best = (p.cand[np.arange(p.n), x], f)
    os.makedirs(PARTS, exist_ok=True)
    pd.DataFrame(runs).to_csv(os.path.join(PARTS, f"runs_{r}_{alg}.csv"), index=False)
    np.savez(os.path.join(PARTS, f"curves_{r}_{alg}.npz"), best_assign=best[0], **curves)
    print(f"scenario {r} {alg}: median {np.median([q['F'] for q in runs]):.0f}", flush=True)


def merge():
    base = []
    for r in SCENARIOS:
        p = Problem(evac_rate=r)
        for name, x in [("nearest", bl.nearest(p)), ("greedy", bl.greedy(p))]:
            c = p.components(x)
            base.append(dict(scenario=r, method=name, F=p.evaluate(x), **{k: v for k, v in c.items() if k != "load"}))
    runs = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PARTS, "runs_*.csv")))])
    curves, best = {"grid": grid}, {}
    for f in sorted(glob.glob(os.path.join(PARTS, "curves_*.npz"))):
        z = np.load(f)
        tag = os.path.basename(f)[len("curves_"):-4]
        best[tag] = z["best_assign"]
        curves.update({k: z[k] for k in z.files if k != "best_assign"})
    runs.to_csv(os.path.join(RES, "runs.csv"), index=False)
    pd.DataFrame(base).to_csv(os.path.join(RES, "baselines.csv"), index=False)
    np.savez(os.path.join(RES, "curves.npz"), **curves)
    np.savez(os.path.join(RES, "best_assign.npz"), **best)
    print(f"merged {len(runs)} runs ->", RES)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        part(float(sys.argv[1]), sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "merge":
        merge()
    else:
        for r in SCENARIOS:
            for alg in ALGS:
                part(r, alg)
        merge()
