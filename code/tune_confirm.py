"""Second tuning stage: the best few settings of tune_ga.py and tune_pso.py on 20 tuning seeds
(1000-1019; the final runs use 0-29), because 5 seeds could not separate close settings.
Same scenario (30%) and budget (50,000) for both algorithms. Adds rows to results/tuning_confirm.csv.
	python tune_confirm.py GA 0 1        settings number 0 and 1 of the GA list
	python tune_confirm.py summary
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from shelter_model import Problem
from algorithms import genetic_algorithm, particle_swarm

B, SEEDS = 50000, range(1000, 1020)
OUT = "../results/tuning_confirm.csv"
N = 1403
CANDIDATES = {
	"GA": [dict(pop_size=100, mutation_rate=1 / N), dict(pop_size=30, mutation_rate=2 / N),
		dict(pop_size=50, mutation_rate=2 / N), dict(pop_size=50, mutation_rate=5 / N)],
	"PSO": [dict(swarm_size=20, vmax_frac=0.2, topology="ring"), dict(swarm_size=50, vmax_frac=0.2, topology="ring"),
		dict(swarm_size=20, vmax_frac=0.1, topology="ring")],
}
FN = {"GA": genetic_algorithm, "PSO": particle_swarm}


def label(alg, kw):
	return ", ".join(f"{k}={round(v * N, 1)}/n" if k == "mutation_rate" else f"{k}={v}" for k, v in kw.items())


if sys.argv[1] == "summary":
	d = pd.read_csv(OUT)
	for alg, g in d.groupby("alg"):
		s = g.groupby("setting")["F"].agg(["median", "mean", "std", "min", "max"]).sort_values("median").round(0)
		print(alg); print(s.to_string())
		best = s.index[0]
		for other in s.index[1:]:
			p = mannwhitneyu(g[g.setting == best].F, g[g.setting == other].F).pvalue
			print(f"  best vs {other}: Mann-Whitney p = {p:.3f}")
		print()
	sys.exit()
alg = sys.argv[1]
p = Problem(evac_rate=0.3)
for i in map(int, sys.argv[2:]):
	kw = CANDIDATES[alg][i]
	rows = []
	for s in SEEDS:
		t0 = time.time()
		_, f, _ = FN[alg](p, B, s, **kw)
		rows.append(dict(alg=alg, setting=label(alg, kw), seed=s, F=f, runtime_s=round(time.time() - t0, 2)))
	old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
	pd.concat([old, pd.DataFrame(rows)]).to_csv(OUT, index=False)
	print(alg, label(alg, kw), "median", round(np.median([r["F"] for r in rows])), flush=True)
