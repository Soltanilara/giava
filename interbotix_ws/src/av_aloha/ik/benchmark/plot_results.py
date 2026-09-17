"""Plots for the IK benchmark CSVs.

  python plot_results.py variants results/compare_per_workload.csv
  python plot_results.py pareto  results/sweep_sweep.csv

Produces one PNG per command next to the input CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed categorical order; never cycled, never reordered by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e0e0e0"


def _style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(row, key, default=np.nan) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def plot_variants(path: str) -> str:
    """Grouped bars: one panel per metric, one bar per variant, grouped by workload."""
    rows = read_rows(path)
    variants = sorted({r["variant"] for r in rows})
    workloads = list(dict.fromkeys(r["workload"] for r in rows))
    panels = [
        ("pos_err_p95_mm", "p95 position error (mm)"),
        ("ee_jerk_rms", "end-effector jerk RMS (m/s³)"),
        ("config_jumps", "configuration jumps (count)"),
        ("solve_ms_mean", "solve time (ms/step)"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.0 * len(panels)), sharex=True)
    x = np.arange(len(workloads))
    width = 0.8 / max(len(variants), 1)

    for ax, (key, label) in zip(axes, panels):
        _style(ax)
        for i, variant in enumerate(variants):
            vals = [
                next(
                    (_f(r, key) for r in rows if r["variant"] == variant and r["workload"] == w),
                    np.nan,
                )
                for w in workloads
            ]
            ax.bar(
                x + i * width - 0.4 + width / 2,
                vals,
                width * 0.88,  # 2px-equivalent gap between adjacent bars
                color=SERIES[i % len(SERIES)],
                label=variant if ax is axes[0] else None,
            )
        ax.set_ylabel(label, color=INK, fontsize=10)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(workloads, rotation=20, ha="right", color=INK)
    axes[0].legend(
        frameon=False, ncol=len(variants), fontsize=9, loc="upper left",
        bbox_to_anchor=(0, 1.28), labelcolor=INK,
    )
    fig.suptitle("IK cost variants across teleoperation workloads", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = os.path.splitext(path)[0] + "_variants.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")
    return out


def plot_pareto(path: str) -> str:
    """Accuracy vs smoothness for every sampled weight set, colored by score."""
    rows = read_rows(path)
    acc = np.array([_f(r, "pos_err_p95_mm") for r in rows])
    jerk = np.array([_f(r, "ee_jerk_rms") for r in rows])
    score = np.array([_f(r, "score") for r in rows])
    ok = np.isfinite(acc) & np.isfinite(jerk)
    acc, jerk, score = acc[ok], jerk[ok], score[ok]

    fig, ax = plt.subplots(figsize=(8, 6))
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    # Single-hue sequential ramp for magnitude (score), not a rainbow.  The pale
    # end is truncated so the worst samples stay visible on a white surface.
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "score", plt.get_cmap("Blues_r")(np.linspace(0.0, 0.72, 256))
    )
    sc = ax.scatter(
        acc, jerk, c=score, cmap=ramp, s=70, edgecolor="white", linewidth=1.2
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("teleoperation score (lower is better)", color=INK, fontsize=10)
    cb.ax.tick_params(colors=MUTED)

    # Pareto front: no other sample is better on both axes.
    order = np.argsort(acc)
    front_x, front_y, best = [], [], np.inf
    for i in order:
        if jerk[i] < best:
            best = jerk[i]
            front_x.append(acc[i])
            front_y.append(jerk[i])
    ax.plot(front_x, front_y, color=SERIES[1], linewidth=2, zorder=1, label="Pareto front")

    if score.size:
        b = int(np.argmin(score))
        ax.annotate(
            f"best score {score[b]:.1f}",
            (acc[b], jerk[b]),
            textcoords="offset points",
            xytext=(10, 8),
            color=INK,
            fontsize=9,
        )
    ax.set_xlabel("p95 position error (mm)", color=INK)
    ax.set_ylabel("end-effector jerk RMS (m/s³)", color=INK)
    ax.set_title("Accuracy vs smoothness across sampled weights", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    out = os.path.splitext(path)[0] + "_pareto.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["variants", "pareto"])
    p.add_argument("csv")
    a = p.parse_args()
    (plot_variants if a.kind == "variants" else plot_pareto)(a.csv)
