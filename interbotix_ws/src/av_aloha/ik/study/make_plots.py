"""Phase 8 — summary plots for the study report.

Uses the dataviz reference palette (validated instance): series-1 #2a78d6
(blue), series-2 #eb6834 (orange); text #0b0b0b / #52514e; surface #fcfcfb.
One axis per panel; thin marks; direct labels; recessive grid.

    JAX-free — reads only results/ CSVs.  Writes results/plots/*.png
"""

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

C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11,
})


def rows(path):
    return list(csv.DictReader(open(path)))


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return np.nan


# ------------------------------------------------------------------ #
# 1. smoothing knee
# ------------------------------------------------------------------ #
sw = rows(ROOT / "sweeps" / "smoothing.csv")
weights = sorted({f(r, "weight") for r in sw})
xs = np.arange(len(weights))

def series(traj, col):
    return [next(f(r, col) for r in sw
                 if r["trajectory"] == traj and f(r, "weight") == w)
            for w in weights]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
ax = axes[0]
ax.plot(xs, series("trans_y", "pos_mm_p95"), color=C1, lw=2, marker="o", ms=6,
        label="trans_y (pure lag)")
ax.plot(xs, series("teleop_grasp", "pos_mm_p95"), color=C2, lw=2, marker="o",
        ms=6, label="teleop_grasp (excursions)")
ax.legend(frameon=False, loc="upper center", fontsize=9)
ax.annotate("trans_y (lag)", (xs[-1], series("trans_y", "pos_mm_p95")[-1]),
            xytext=(-6, 8), textcoords="offset points", ha="right", color=INK)
ax.annotate("teleop_grasp\n(excursions)", (xs[1], series("teleop_grasp", "pos_mm_p95")[1]),
            xytext=(6, 10), textcoords="offset points", color=INK)
ax.axvspan(1.7, 3.3, color=C1, alpha=0.07, lw=0)
ax.text(2.5, ax.get_ylim()[1] * 0.92, "knee", ha="center", color=INK2)
ax.set_xticks(xs, [f"{w:g}" for w in weights])
ax.set_xlabel("smoothing weight w  (deployed = 0.5)")
ax.set_ylabel("position error p95 (mm)")
ax.set_title("Lag vs excursion suppression")

ax = axes[1]
jump_total = [sum(next(f(r, "config_jumps") for r in sw
                       if r["trajectory"] == t and f(r, "weight") == w)
                  for t in ("trans_y", "trans_lissajous", "rot_home",
                            "teleop_grasp", "jitter_teleop", "reach_limit"))
              for w in weights]
ax.plot(xs, jump_total, color=C1, lw=2, marker="o", ms=6)
ax.set_xticks(xs, [f"{w:g}" for w in weights])
ax.set_xlabel("smoothing weight w")
ax.set_ylabel("config jumps (sweep subset total)")
ax.set_title("Configuration flips vs weight")
fig.suptitle("Smoothing sweep: the knee sits at w ≈ 0.05–0.1", fontsize=12)
fig.savefig(OUT / "smoothing_knee.png", dpi=180)
plt.close(fig)

# ------------------------------------------------------------------ #
# 2. pos/ori tradeoff on the winning combo
# ------------------------------------------------------------------ #
po = rows(ROOT / "sweeps" / "posori_on_smooth_center.csv")
ows = sorted({f(r, "ori_w") for r in po})
grasp = ("teleop_grasp", "teleop_grasp_side", "teleop_grasp_yaw")
px = [np.mean([f(r, "pos_mm_p95") for r in po
               if f(r, "ori_w") == w and r["trajectory"] in grasp]) for w in ows]
py = [np.mean([f(r, "ori_deg_p95") for r in po
               if f(r, "ori_w") == w and r["trajectory"] in grasp]) for w in ows]

fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
ax.plot(px, py, color=C1, lw=2, marker="o", ms=7, zorder=3)
for x, y, w in zip(px, py, ows):
    star = "  ← chosen" if w == 10 else ""
    ax.annotate(f"ori={w:g}{star}", (x, y), xytext=(8, 4),
                textcoords="offset points", color=INK,
                fontweight="bold" if w == 10 else "normal")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_xlabel("position error p95 (mm), grasp-family mean")
ax.set_ylabel("orientation error p95 (°)")
ax.set_title("Pos/ori tradeoff on smoothing+centering\n(trap fixed → both errors sub-degree/sub-mm at ori=10)")
fig.savefig(OUT / "posori_tradeoff.png", dpi=180)
plt.close(fig)

# ------------------------------------------------------------------ #
# 3. master variant comparison
# ------------------------------------------------------------------ #
CLEAN = ("trans_x", "trans_y", "trans_z", "trans_diag", "trans_lissajous",
         "rot_home", "rot_low", "rot_forward", "teleop_grasp",
         "teleop_grasp_side", "teleop_grasp_yaw", "near_singular",
         "middle_gaze_sweep", "mid_trans_x", "mid_trans_y", "mid_trans_z",
         "mid_trans_diag", "mid_trans_lissajous", "mid_rot_home",
         "mid_rot_low", "mid_approach")
VARIANTS = [
    ("baseline", "baseline"),
    ("centering", "+ centering (0.05)"),
    ("manipulability", "+ manipulability (0.02)"),
    ("collision", "+ collision (5, 3 cm)"),
    ("smoothing", "+ smoothing (0.5)"),
    ("smooth_center", "smoothing 0.05\n+ centering 0.5"),
]
data = {v: {r["trajectory"]: r for r in rows(ROOT / v / "summary.csv")}
        for v, _ in VARIANTS}

def agg(v, col, how, subset=CLEAN):
    vals = np.asarray([f(data[v][t], col) for t in subset if t in data[v]])
    vals = vals[~np.isnan(vals)]
    return {"mean": np.mean, "sum": np.sum}[how](vals) if vals.size else np.nan

panels = [
    ("pos_mm_p95", "mean", CLEAN, "tracking error\npos p95 mm (clean mean)", None),
    ("solve_ms_mean", "mean", None, "solve time\nmean ms (log)", "log"),
    ("config_jumps", "sum", None, "config jumps\n(suite total)", None),
    ("pos_mm_p95", "mean", ("teleop_grasp", "teleop_grasp_side",
                            "teleop_grasp_yaw"), "grasp family\npos p95 mm (mean)", None),
]
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6), constrained_layout=True)
ylabels = [lab for _, lab in VARIANTS]
ypos = np.arange(len(VARIANTS))[::-1]
for ax, (col, how, subset, title, scale) in zip(axes, panels):
    vals = [agg(v, col, how, subset or tuple(data[v])) for v, _ in VARIANTS]
    ax.barh(ypos, vals, height=0.62, color=C1)
    for y, val in zip(ypos, vals):
        txt = f" {val:.0f}" if val >= 10 else f" {val:.2g}"
        ax.text(val, y, txt, va="center", color=INK, fontsize=9)
    ax.set_yticks(ypos, ylabels if ax is axes[0] else [""] * len(VARIANTS))
    if scale:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", visible=False)
fig.suptitle("One residual at a time vs the winning pair (suite rev 2.1, CPU, 50 Hz)",
             fontsize=12)
fig.savefig(OUT / "variants_compare.png", dpi=180)
plt.close(fig)

# ------------------------------------------------------------------ #
# 4. manipulability cost anatomy
# ------------------------------------------------------------------ #
labels = ["FK (all links)", "manip value — pyroki (full jacfwd)",
          "residual gradient — pyroki", "manip value — arm-restricted",
          "residual gradient — arm-restricted"]
ms = [0.015, 0.179, 1.063, 0.066, 0.477]
fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
yp = np.arange(len(labels))[::-1]
ax.barh(yp, ms, height=0.6, color=C1)
for y, v in zip(yp, ms):
    ax.text(v, y, f" {v:.3f} ms", va="center", color=INK, fontsize=9)
ax.set_yticks(yp, labels)
ax.set_xlabel("ms per call (jitted, CPU, n=200)")
ax.set_title("Where the manipulability cost goes: the gradient of the residual\n"
             "(what LM needs per arm per iteration) is ~70× an FK call")
ax.grid(axis="y", visible=False)
fig.savefig(OUT / "manip_anatomy.png", dpi=180)
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
