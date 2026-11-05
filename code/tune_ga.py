"""GA parameter study on the tuning seeds 1000-1004 (final runs use seeds 0-29), 30% scenario.
Same seeds, budget and scenario as tune_pso.py. Each call adds rows to results/tuning_ga.csv:
	python tune_ga.py pm 1 2             mutation rate (x/n) x population {30, 50, 100}
	python tune_ga.py tour 30 2 2 5      tournament size, given population and mutation x/n
	python tune_ga.py summary
"""
import os, sys, time
import numpy as np, pandas as pd
from shelter_model import Problem
from algorithms import genetic_algorithm

B, SEEDS = 50000, range(1000, 1005)
OUT = "../results/tuning_ga.csv"


def run(p, stage, pm_mult, **kw):
	rows = []
	for s in SEEDS:
		t0 = time.time()
		_, f, _ = genetic_algorithm(p, B, s, mutation_rate=pm_mult / p.n, **kw)
		rows.append(dict(stage=stage, seed=s, F=f, runtime_s=round(time.time() - t0, 2), mutation_x_per_n=pm_mult, **kw))
	old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
	pd.concat([old, pd.DataFrame(rows)]).to_csv(OUT, index=False)
	print(stage, f"pm={pm_mult}/n", kw, "median", round(np.median([r["F"] for r in rows])), flush=True)


mode = sys.argv[1]
if mode == "summary":
	d = pd.read_csv(OUT).fillna({"tournament": 3})
	print(d.groupby(["stage", "pop_size", "mutation_x_per_n", "tournament"])["F"].agg(["median", "min", "max"]).round(0).sort_values("median").to_string())
	sys.exit()
p = Problem(evac_rate=0.3)
if mode == "pm":
	for m in map(float, sys.argv[2:]):
		for pop in [30, 50, 100]:
			run(p, 1, m, pop_size=pop)
elif mode == "tour":
	pop, m = int(sys.argv[2]), float(sys.argv[3])
	for k in map(int, sys.argv[4:]):
		run(p, 2, m, pop_size=pop, tournament=k)
