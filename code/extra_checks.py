"""Extra checks for the report (all on the 30% scenario unless stated). Results go to ../results/.
	python extra_checks.py sens base|w1|w100|official|noest   model sensitivity: nearest, greedy, GA x 10 seeds
	python extra_checks.py pso_steps                         cells changed per PSO move, 10 seeds
	python extra_checks.py linear                            GA vs the MILP reference on the linear objective
	python extra_checks.py summary                           print the collected tables
Seeds 0-9 are a subset of the final seeds; the settings are the tuned ones in chosen_params.json.
"""
import json, os, shutil, subprocess, sys, tempfile, time
import numpy as np, pandas as pd
from shelter_model import Problem
from algorithms import genetic_algorithm, particle_swarm
import baselines as bl

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
cfg = json.load(open(os.path.join(RES, "chosen_params.json")))
B, SEEDS = 50000, range(10)


def add(name, rows):
	f = os.path.join(RES, name)
	old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
	pd.concat([old, pd.DataFrame(rows)]).to_csv(f, index=False)


def phys(p, x):
	c = p.components(x)
	return dict(walk_km_per_evacuee=c["mean_walk_km"], overflow_people=c["overflow_people"],
		overloaded_shelters=c["overloaded_shelters"], max_load_ratio=c["max_load_ratio"])


def no_estimates_dir():
	d = os.path.join(tempfile.gettempdir(), "nagaoka_noest")
	if not os.path.exists(os.path.join(d, "shelters.csv")):
		shutil.rmtree(d, ignore_errors=True)
		shutil.copytree(os.path.join(HERE, "..", "dataset"), d)
		s = open(os.path.join(d, "build_dataset.py")).read().replace("USE_ESTIMATES = True ", "USE_ESTIMATES = False")
		open(os.path.join(d, "build_dataset.py"), "w").write(s)
		subprocess.run([sys.executable, "build_dataset.py"], cwd=d, check=True, capture_output=True)
	return d


mode = sys.argv[1]
if mode == "sens":
	case = sys.argv[2]
	kw = {"base": {}, "w1": dict(penalty=1.0), "w100": dict(penalty=100.0), "w3": dict(penalty=3.0), "w30": dict(penalty=30.0), "official": dict(capacity="official"),
		"noest": dict(data_dir=no_estimates_dir())}[case]
	p = Problem(evac_rate=0.3, **kw)
	rows = []
	for name, x in [("nearest", bl.nearest(p)), ("greedy", bl.greedy(p))]:
		rows.append(dict(case=case, method=name, seed=-1, F=p.evaluate(x), **phys(p, x)))
	for s in SEEDS:
		x, f, _ = genetic_algorithm(p, B, s, **cfg["GA"])
		rows.append(dict(case=case, method="GA", seed=s, F=f, **phys(p, x)))
	add("sensitivity.csv", rows)
	print(case, "done", flush=True)
elif mode == "pso_steps":
	p = Problem(evac_rate=0.3)
	orig = p.decode_continuous
	rows = []
	for s in SEEDS:
		last, k, ch = {}, [0], []
		sw = cfg["PSO"]["swarm_size"]
		def dec(z):
			x = orig(z); i = k[0] % sw
			if k[0] >= sw: ch.append(int((x != last[i]).sum()))
			last[i] = x; k[0] += 1
			return x
		p.decode_continuous = dec
		_, f, cur = particle_swarm(p, B, s, **cfg["PSO"])
		ch = np.array(ch); n = len(ch)
		ev, fv = np.array(cur).T
		at = lambda e: fv[np.searchsorted(ev, e) - 1]
		rows.append(dict(seed=s, changed_first_500=np.median(ch[:500]), changed_quarter=np.median(ch[n // 4 - 250:n // 4 + 250]),
			changed_last_500=np.median(ch[-500:]), F_25k=at(25000), F_50k=f, gain_last_half_pct=100 * (1 - f / at(25000))))
	p.decode_continuous = orig
	add("pso_steps.csv", rows)
	print("pso_steps done", flush=True)
elif mode == "linear":
	p = Problem(evac_rate=0.3, objective="linear")
	t0 = time.time(); F_milp, x_milp, status, gap = bl.exact_ilp(p); t_milp = time.time() - t0
	rows = [dict(method="MILP (HiGHS, 120 s limit)", seed=-1, F=F_milp, gap_reported=gap, runtime_s=t_milp, status=status),
		dict(method="nearest", seed=-1, F=p.evaluate(bl.nearest(p))), dict(method="greedy", seed=-1, F=p.evaluate(bl.greedy(p)))]
	for s in SEEDS:
		_, f, _ = genetic_algorithm(p, B, s, **cfg["GA"])
		rows.append(dict(method="GA", seed=s, F=f))
	add("linear_reference.csv", rows)
	print("linear done", flush=True)
elif mode == "summary":
	for f in ["sensitivity.csv", "pso_steps.csv", "linear_reference.csv"]:
		fp = os.path.join(RES, f)
		if not os.path.exists(fp): continue
		d = pd.read_csv(fp); print("==", f)
		if f == "sensitivity.csv":
			print(d.groupby(["case", "method"])[["F", "walk_km_per_evacuee", "overflow_people", "overloaded_shelters", "max_load_ratio"]].median().round(3).to_string())
		elif f == "pso_steps.csv":
			print(d.round(1).to_string()); print(d.median().round(2).to_string())
		else:
			print(d.to_string())
