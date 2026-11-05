"""Figure: how uneven the demand is (residents per 500 m cell). Writes ../results/fig_demand_profile.png"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
pop = pd.read_csv(os.path.join(HERE, "..", "dataset", "cells.csv"))["population"].to_numpy()
INK, MUTED, GRID, BLUE = "#1f2328", "#5b636d", "#e3e6ea", "#2a6fbb"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
	"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})

# (a) residents per cell, log bins
bins = np.logspace(0, np.log10(pop.max() * 1.05), 30)
a.hist(pop, bins=bins, color=BLUE, edgecolor="white", linewidth=1)
a.set_xscale("log")
a.set_xlabel("Residents per 500 m cell (log scale)")
a.set_ylabel("Number of cells")
a.grid(axis="y", color=GRID, linewidth=0.8); a.set_axisbelow(True)
for s in ("top", "right"): a.spines[s].set_visible(False)
top = a.get_ylim()[1]
a.set_ylim(0, top * 1.12); top = top * 1.12
for v, lab, ha, h in [(np.median(pop), f"median {np.median(pop):.0f}", "right", 0.98), (pop.mean(), f"mean {pop.mean():.0f}", "left", 0.98), (pop.max(), f"max {pop.max():,}", "right", 0.86)]:
	a.axvline(v, color=INK, linewidth=1, linestyle=(0, (3, 2)))
	a.text(v * (0.9 if ha == "right" else 1.1), top * h, lab, ha=ha, va="top", color=INK, fontsize=9,
		bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
a.set_title("(a) Most cells are small, a few are very large", loc="left", fontsize=10, color=INK)

# (b) concentration: cells sorted from largest to smallest
s = np.sort(pop)[::-1]
x = np.arange(1, len(s) + 1) / len(s) * 100
y = np.cumsum(s) / s.sum() * 100
b.plot(x, y, color=BLUE, linewidth=2)
b.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
b.text(66, 57, "if every cell were equal", color=MUTED, fontsize=9)
for share in (10, 25):
	k = int(np.ceil(len(s) * share / 100)); v = y[k - 1]
	b.plot([share], [v], "o", color=BLUE, markersize=6, markeredgecolor="white", markeredgewidth=1.5)
	b.annotate(f"largest {share}% of cells\nhold {v:.0f}% of residents", (share, v), xytext={10: (52, 18), 25: (36, 68)}[share],
		fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
b.set_xlim(0, 100); b.set_ylim(0, 101)
b.set_xlabel("Share of cells, largest first (%)"); b.set_ylabel("Share of residents (%)")
b.grid(color=GRID, linewidth=0.8); b.set_axisbelow(True)
for sp in ("top", "right"): b.spines[sp].set_visible(False)
b.set_title("(b) Residents are concentrated in few cells", loc="left", fontsize=10, color=INK)

fig.tight_layout()
out = os.path.join(HERE, "..", "results", "fig_demand_profile.png")
fig.savefig(out, dpi=200)
k10 = int(np.ceil(len(s) * .10)); k25 = int(np.ceil(len(s) * .25)); half = np.searchsorted(y, 50) + 1
print(f"top10% {y[k10-1]:.1f}%  top25% {y[k25-1]:.1f}%  cells for half of residents {half} ({half/len(s)*100:.1f}%)  cells<=54 {np.mean(pop<=54)*100:.0f}%")
