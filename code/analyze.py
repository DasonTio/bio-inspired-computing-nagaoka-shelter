"""Statistics and figures for the report."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

runs = pd.read_csv("results/runs.csv")
base = pd.read_csv("results/baselines.csv")
cv = np.load("results/curves.npz")
grid = cv["grid"]


def a12(x, y):
    """Vargha-Delaney A12: probability that a random GA run beats (is lower than) a random PSO run."""
    x, y = np.asarray(x), np.asarray(y)
    return ((x[:, None] < y[None]).sum() + 0.5 * (x[:, None] == y[None]).sum()) / (len(x) * len(y))


rows = []
for (r, alg), g in runs.groupby(["scenario", "alg"]):
    rows.append(dict(scenario=f"{int(r*100)}%", alg=alg, best=g.F.min(), median=g.F.median(),
                     mean=g.F.mean(), std=g.F.std(), worst=g.F.max(),
                     overflow_median=g.overflow_people.median(), max_ratio_median=g.max_load_ratio.median(),
                     mean_walk_km=g.mean_walk_km.median(),
                     no_overflow_runs=f"{(g.overflow_people < 1).mean():.0%}",
                     evals=int(g.evals.iloc[0]), runtime_s=g.runtime_s.median()))
summary = pd.DataFrame(rows)
summary.to_csv("results/summary.csv", index=False)

tests = []
for r, g in runs.groupby("scenario"):
    ga, pso = g[g.alg == "GA"].F.values, g[g.alg == "PSO"].F.values
    greedy = base[(base.scenario == r) & (base.method == "greedy")].F.iloc[0]
    nearest = base[(base.scenario == r) & (base.method == "nearest")].F.iloc[0]
    tests.append(dict(scenario=f"{int(r*100)}%", mannwhitney_p=mannwhitneyu(ga, pso).pvalue,
                      A12_GA_better=a12(ga, pso),
                      GA_vs_nearest=f"{1-np.median(ga)/nearest:.1%} lower",
                      GA_vs_greedy=f"{1-np.median(ga)/greedy:.1%} lower",
                      GA_runs_beating_greedy=f"{(ga < greedy).mean():.0%}"))
pd.DataFrame(tests).to_csv("results/tests.csv", index=False)
print(summary.round(2).to_string(index=False)); print(); print(pd.DataFrame(tests).to_string(index=False))

# Figure 1: convergence (median + IQR) per scenario, x = evaluations
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
for ax, r in zip(axes, sorted(runs.scenario.unique())):
    for alg, col in [("GA", "#2a6fbb"), ("PSO", "#d0602a")]:
        M = np.array([cv[f"{alg}_{r}_{s}"] for s in range(30)])
        q1, med, q3 = np.percentile(M, [25, 50, 75], axis=0)
        ax.plot(grid, med, color=col, label=alg); ax.fill_between(grid, q1, q3, color=col, alpha=.2)
    for m, ls in [("greedy", "--"), ("nearest", ":")]:
        ax.axhline(base[(base.scenario == r) & (base.method == m)].F.iloc[0], color="grey", ls=ls, lw=1, label=m)
    ax.set_yscale("log"); ax.set_title(f"{int(r*100)}% evacuate"); ax.set_xlabel("objective evaluations")
axes[0].set_ylabel("best score so far (log)"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/fig_convergence.png", dpi=160)

# Figure 2: final-score distributions
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, r in zip(axes, sorted(runs.scenario.unique())):
    g = runs[runs.scenario == r]
    ax.boxplot([g[g.alg == "GA"].F, g[g.alg == "PSO"].F], tick_labels=["GA", "PSO"])
    ax.axhline(base[(base.scenario == r) & (base.method == "greedy")].F.iloc[0], color="grey", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_title(f"{int(r*100)}% evacuate (30 runs each)")
axes[0].set_ylabel("final score (log)")
fig.tight_layout(); fig.savefig("results/fig_final_scores.png", dpi=160)
print("figures saved")
