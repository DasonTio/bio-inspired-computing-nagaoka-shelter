"""Parameter sensitivity on tuning seeds 1000-1004 (final runs use seeds 0-29)."""
import itertools, json, numpy as np
from shelter_model import Problem
from algorithms import genetic_algorithm, particle_swarm
p = Problem("nagaoka_cells_2020.csv", "nagaoka_shelters_raw.csv", evac_rate=0.3)
B, seeds = 50000, range(1000, 1005)
rows = []
for pop, pm_mult in itertools.product([30, 50, 100], [1, 2, 5]):
    f = [genetic_algorithm(p, B, s, pop_size=pop, mutation_rate=pm_mult / p.n)[1] for s in seeds]
    rows.append(dict(alg="GA", pop_size=pop, mutation=f"{pm_mult}/n", median=float(np.median(f)), all=f))
for sw, vfrac in itertools.product([30, 50, 100], [0.05, 0.2, 0.5]):
    f = [particle_swarm(p, B, s, swarm_size=sw, vmax_frac=vfrac)[1] for s in seeds]
    rows.append(dict(alg="PSO", swarm_size=sw, vmax_frac=vfrac, median=float(np.median(f)), all=f))
json.dump(rows, open("results/tuning.json", "w"), indent=1)
for r in rows:
    print({k: (round(v) if isinstance(v, float) and k == "median" else v) for k, v in r.items() if k != "all"})
