"""Collision-study plots (dataviz reference palette, same conventions as
make_plots.py).  Reads results/geometry_scorecard.csv and
results/sweeps/collision_margin_weight.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "results"
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11,
})


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return np.nan


# ------------------------------------------------------------------ #
# 1. geometry accuracy by category (3 models)
# ------------------------------------------------------------------ #
recs = list(csv.DictReader(open(ROOT / "geometry_scorecard.csv")))
cats = ["plates/self", "fork/self", "fork/inter-arm", "fingers/inter-arm",
        "arm-links/self"]
labels = ["plates", "fork (self)", "fork (inter-arm)", "fingers (inter-arm)",
          "arm links"]
models = [("pyroki", "pyroki capsule", C2), ("tight", "corrected capsule", C1),
          ("spheres", "180 spheres", C3)]

fig, ax = plt.subplots(figsize=(8.6, 3.6), constrained_layout=True)
x = np.arange(len(cats))
w = 0.26
for k, (col, label, color) in enumerate(models):
    vals = []
    for cat in cats:
        e = [abs(f(r, col) - f(r, "mesh")) * 1e3 for r in recs
             if r["cat"] == cat]
        vals.append(np.mean(e) if e else np.nan)
    bars = ax.bar(x + (k - 1) * w, vals, width=w - 0.02, color=color,
                  label=label)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center",
                va="bottom", fontsize=8.5, color=INK)
ax.set_xticks(x, labels)
ax.set_ylabel("mean |clearance error| vs mesh truth (mm)")
ax.set_title("Geometry accuracy by category — 70 mesh-ground-truth cases")
ax.legend(frameon=False)
ax.grid(axis="x", visible=False)
fig.savefig(OUT / "collision_geometry_accuracy.png", dpi=180)
plt.close(fig)

# ------------------------------------------------------------------ #
# 2. the weight threshold and the no-repulsion region (sweep)
# ------------------------------------------------------------------ #
rows = list(csv.DictReader(open(ROOT / "sweeps" / "collision_margin_weight.csv")))
soft = [r for r in rows if r["form"] == "soft"]
weights = [10, 30, 100, 300]
margins = [0.015, 0.020, 0.025, 0.030]
mcolors = {0.015: "#9ec4ec", 0.020: C1, 0.025: "#1f5eab", 0.030: "#12365f"}

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), constrained_layout=True)
ax = axes[0]
for m in margins:
    vals = [next(f(r, "clear_min_mm") for r in soft
                 if f(r, "margin") == m and f(r, "weight") == w
                 and r["trajectory"] == "arms_converge") for w in weights]
    ax.plot(range(len(weights)), vals, marker="o", ms=6, lw=2,
            color=mcolors[m], label=f"margin {m*1e3:.0f} mm")
ax.axhline(0, color=INK2, lw=1, ls="--")
ax.set_xticks(range(len(weights)), [str(w) for w in weights])
ax.set_xlabel("collision weight")
ax.set_ylabel("min model clearance (mm)")
ax.set_title("Designed crossing (arms_converge):\nweight ≥ 100 holds the boundary")
ax.legend(frameon=False, fontsize=8.5)

ax = axes[1]
for m in margins:
    vals = [max(next(f(r, "dev_max_mm") for r in soft
                     if f(r, "margin") == m and f(r, "weight") == w
                     and r["trajectory"] == t) for t in
                ("bimanual_parallel", "bimanual_handoff"))
            for w in weights]
    ax.plot(range(len(weights)), vals, marker="o", ms=6, lw=2,
            color=mcolors[m], label=f"margin {m*1e3:.0f} mm")
ax.set_xticks(range(len(weights)), [str(w) for w in weights])
ax.set_xlabel("collision weight")
ax.set_ylabel("max deviation from baseline path (mm)")
ax.set_title("Clean bimanual tests:\nfalse repulsion stays ≈ 0 up to w=100")
fig.suptitle("Margin × weight sweep on the collision subset (soft margin-hinge cost)",
             fontsize=12)
fig.savefig(OUT / "collision_sweep.png", dpi=180)
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("collision_*.png")))
