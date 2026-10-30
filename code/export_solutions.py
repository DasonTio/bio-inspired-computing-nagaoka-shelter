"""Preview of GA vs PSO for the data viewer (index.html), before tuning.

For one scenario: nearest rule, greedy rule, and GA and PSO with their default
parameters over a few seeds at the full budget. Saves the best assignment of each
method, every run's final score and the best-so-far curves on a common grid.

Usage:  python export_solutions.py 30        (evacuation rate in %, writes results/preview_30.json)
The final experiment is run_experiments.py (30 seeds, tuned parameters).
"""
import json, sys, time
import numpy as np
from shelter_model import Problem
from algorithms import genetic_algorithm, particle_swarm
import baselines as bl

RATE = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BUDGET, SEEDS, STEP = 50000, range(10), 500
grid = np.arange(STEP, BUDGET + 1, STEP)


def on_grid(curve):
	ev, f = np.array(curve).T
	return f[np.searchsorted(ev, grid, side="right") - 1]


def summary(p, x):
	c = p.components(x)
	c.pop("load")
	c["F"] = float(p.evaluate(x))
	c["reassigned_cells"] = int((x != 0).sum())
	return c


p = Problem(evac_rate=RATE / 100)
out = {"rate": RATE, "budget": BUDGET, "seeds": list(SEEDS), "grid": grid.tolist(), "methods": {}}
for name, x in [("Nearest (current practice)", bl.nearest(p)), ("Greedy (simple heuristic)", bl.greedy(p))]:
	out["methods"][name] = {"assign": p.cand[np.arange(p.n), x].tolist(), "stats": summary(p, x)}

for name, fn in [("GA", genetic_algorithm), ("PSO", particle_swarm)]:
	finals, curves, best = [], [], None
	for s in SEEDS:
		t0 = time.time()
		x, f, cur = fn(p, BUDGET, s)
		finals.append(dict(seed=s, F=float(f), runtime_s=round(time.time() - t0, 2), **{k: v for k, v in summary(p, x).items() if k != "F"}))
		curves.append(on_grid(cur))
		if best is None or f < best[1]:
			best = (x, f, s)
	C = np.array(curves)
	out["methods"][name] = {
		"assign": p.cand[np.arange(p.n), best[0]].tolist(), "best_seed": best[2],
		"stats": summary(p, best[0]), "runs": finals,
		"curve_median": np.median(C, 0).tolist(), "curve_q1": np.percentile(C, 25, 0).tolist(), "curve_q3": np.percentile(C, 75, 0).tolist(),
	}
	print(name, "median", round(float(np.median([r["F"] for r in finals]))), flush=True)
json.dump(out, open(f"../results/preview_{RATE}.json", "w"))
print("saved", f"../results/preview_{RATE}.json")
