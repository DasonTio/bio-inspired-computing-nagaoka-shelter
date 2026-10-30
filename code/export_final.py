"""Convert the final experiment (results/runs.csv, curves.npz, best_assign.npz, baselines.csv)
into results/final_<rate>.json for the GA vs PSO tab of index.html.
Run after run_experiments.py merge:  python export_final.py
"""
import json, os
import numpy as np, pandas as pd
from shelter_model import Problem

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
runs = pd.read_csv(os.path.join(RES, "runs.csv"))
base = pd.read_csv(os.path.join(RES, "baselines.csv"))
cv = np.load(os.path.join(RES, "curves.npz"))
best = np.load(os.path.join(RES, "best_assign.npz"))
params = json.load(open(os.path.join(RES, "chosen_params.json")))
STATS = ["walk_person_km", "crowding", "max_load_ratio", "overflow_people", "overloaded_shelters", "mean_walk_km", "max_walk_km"]


def stats_of(p, assign):
	x = np.array([list(p.cand[i]).index(j) for i, j in enumerate(assign)])
	c = p.components(x); c.pop("load")
	c["F"] = float(p.evaluate(x)); c["reassigned_cells"] = int((x != 0).sum())
	return c


for r in sorted(runs.scenario.unique()):
	p = Problem(evac_rate=r)
	out = {"rate": int(round(r * 100)), "budget": int(runs.evals.max()), "seeds": sorted(runs.seed.unique().tolist()),
		"grid": cv["grid"].tolist(), "final": True, "params": {k: v for k, v in params.items() if not k.startswith("_")}, "methods": {}}
	for name, key in [("Nearest (current practice)", "nearest"), ("Greedy (simple heuristic)", "greedy")]:
		x = np.zeros(p.n, int) if key == "nearest" else __import__("baselines").greedy(p)
		out["methods"][name] = {"assign": p.cand[np.arange(p.n), x].tolist(), "stats": stats_of(p, p.cand[np.arange(p.n), x])}
	for alg in ["GA", "PSO"]:
		g = runs[(runs.scenario == r) & (runs.alg == alg)]
		C = np.array([cv[f"{alg}_{r}_{s}"] for s in g.seed])
		a = best[f"{r}_{alg}"]
		out["methods"][alg] = {
			"assign": a.tolist(), "best_seed": int(g.loc[g.F.idxmin(), "seed"]), "stats": stats_of(p, a),
			"runs": [dict(seed=int(q.seed), F=float(q.F), runtime_s=float(q.runtime_s)) for q in g.itertuples()],
			"curve_median": np.median(C, 0).tolist(), "curve_q1": np.percentile(C, 25, 0).tolist(), "curve_q3": np.percentile(C, 75, 0).tolist(),
		}
	json.dump(out, open(os.path.join(RES, f"final_{out['rate']}.json"), "w"))
	print("saved final", out["rate"], {k: round(v["stats"]["F"]) for k, v in out["methods"].items()})
