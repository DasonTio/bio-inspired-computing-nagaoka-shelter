"""PSO parameter study on the tuning seeds 1000-1004 (final runs use seeds 0-29), 30% scenario.

Each call adds rows to results/tuning_pso.csv (one row per run), so it can be split:
	python tune_pso.py vmax 0.5 0.2          velocity limit x swarm size {20, 50, 100}
	python tune_pso.py w 50 0.02 0.4 0.6 0.9  inertia weight, given swarm size and velocity limit
	python tune_pso.py ring 0.5 0.2 0.1       ring topology: velocity limit x swarm size {20, 50}
	python tune_pso.py summary
"""
import os, sys, time
import numpy as np, pandas as pd
from shelter_model import Problem
from algorithms import particle_swarm

B, SEEDS = 50000, range(1000, 1005)
OUT = "../results/tuning_pso.csv"


def run(p, stage, **kw):
	rows = []
	for s in SEEDS:
		t0 = time.time()
		_, f, _ = particle_swarm(p, B, s, **kw)
		rows.append(dict(stage=stage, seed=s, F=f, runtime_s=round(time.time() - t0, 2), **kw))
	old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
	pd.concat([old, pd.DataFrame(rows)]).to_csv(OUT, index=False)
	print(stage, kw, "median", round(np.median([r["F"] for r in rows])), flush=True)


mode = sys.argv[1]
if mode == "summary":
	d = pd.read_csv(OUT)
	d = d.fillna({"w": 0.729, "topology": "global"}) if "topology" in d else d.fillna({"w": 0.729}).assign(topology="global")
	keys = ["stage", "topology", "swarm_size", "vmax_frac", "w"]
	print(d.groupby(keys)["F"].agg(["median", "min", "max"]).round(0).sort_values("median").to_string())
	sys.exit()
p = Problem(evac_rate=0.3)
if mode == "vmax":
	for v in map(float, sys.argv[2:]):
		for sw in [20, 50, 100]:
			run(p, 1, swarm_size=sw, vmax_frac=v)
elif mode == "ring":
	for v in map(float, sys.argv[2:]):
		for sw in [20, 50]:
			run(p, 3, swarm_size=sw, vmax_frac=v, topology="ring")
elif mode == "w":
	sw, v = int(sys.argv[2]), float(sys.argv[3])
	for w in map(float, sys.argv[4:]):
		run(p, 2, swarm_size=sw, vmax_frac=v, w=w)
