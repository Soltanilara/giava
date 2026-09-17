"""Plot replay tracking logs from teleop_debug_tool.

  python plot_tracking.py                          # all logs in trajectories/logs
  python plot_tracking.py trajectories/logs/log_yaw_test_*.npz

Per log      : expected-vs-achieved, one subplot per axis (x, y, z, rot-x/pitch,
               rot-y/roll, rot-z/yaw).
Per RATE     : all logs at that control rate overlaid -- one figure of signed
               percent differences, one of ABSOLUTE errors.  Percent panels
               mask steps where the expected value is tiny (< 2 mm / < 1 deg):
               a "300% yaw error" on an expected yaw of 1 deg is a 3 deg miss,
               so absolute panels are the honest companion.
Weights ride in every legend so runs with different tunings separate cleanly.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import glob
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e0e0e0"
AXES = ("x [m]", "y [m]", "z [m]", "rot-x pitch [deg]", "rot-y roll [deg]",
        "rot-z yaw [deg]")
## data_collection.py writes these next to ITSELF, not next to this
## plotter -- resolve against the scripts tree, not __file__.
LOGDIR = os.path.join(str(_giava_paths.SCRIPTS_DIR), "trajectories", "logs")


def style(ax):
    ax.set_facecolor("white")
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    for sd in ("left", "bottom"):
        ax.spines[sd].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def load(path):
    d = np.load(path, allow_pickle=True)
    rows = d["rows"]  # [t, k, e_p(3), a_p(3), e_r(3), a_r(3)]
    return {
        "name": os.path.splitext(os.path.basename(path))[0].replace("log_", ""),
        "t": rows[:, 0] - rows[0, 0],
        "e": np.concatenate([rows[:, 2:5], rows[:, 8:11]], axis=1),
        "a": np.concatenate([rows[:, 5:8], rows[:, 11:14]], axis=1),
        "rate": float(d["rate"]), "weights": str(d["weights"]),
        "arm": str(d["arm"]), "traj": str(d["traj"]),
    }


def per_log_figure(L, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        style(ax)
        ax.plot(L["t"], L["e"][:, i], color=SERIES[0], lw=1.8, label="expected")
        ax.plot(L["t"], L["a"][:, i], color=SERIES[1], lw=1.8, label="achieved")
        ax.set_title(AXES[i], color=INK, fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=INK)
    for ax in axes[1]:
        ax.set_xlabel("time [s]", color=MUTED, fontsize=9)
    fig.suptitle(f"{L['name']}  |  {L['rate']:g} Hz  |  {L['weights']}",
                 color=INK, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    out = os.path.join(outdir, f"track_{L['name']}.png")
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    return out


def per_rate_figures(rate, logs, outdir):
    outs = []
    for kind in ("pct", "abserr"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        for i, ax in enumerate(axes.ravel()):
            style(ax)
            for j, L in enumerate(logs):
                e, a = L["e"][:, i], L["a"][:, i]
                if kind == "pct":
                    tol = 0.002 if i < 3 else 1.0
                    y = np.where(np.abs(e) < tol, np.nan, (a - e) / np.abs(e) * 100)
                    ax.set_title(AXES[i] + "  % diff", color=INK, fontsize=10)
                else:
                    y = a - e
                    ax.set_title(AXES[i] + "  abs error", color=INK, fontsize=10)
                ax.plot(L["t"], y, color=SERIES[j % len(SERIES)], lw=1.5,
                        label=f"{L['name']} ({L['arm']})")
            if kind == "pct":
                ax.axhline(0, color=MUTED, lw=0.8)
        axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
        for ax in axes[1]:
            ax.set_xlabel("time [s]", color=MUTED, fontsize=9)
        wset = " / ".join(sorted({L["weights"] for L in logs}))
        fig.suptitle(f"{rate:g} Hz  |  {kind}  |  {wset[:120]}",
                     color=INK, x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        out = os.path.join(outdir, f"{kind}_{rate:g}hz.png")
        fig.savefig(out, dpi=140, facecolor="white")
        plt.close(fig)
        outs.append(out)
    return outs


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(LOGDIR, "*.npz")))
    if not paths:
        print(f"no logs found in {LOGDIR}")
        return
    logs = [load(p) for p in paths]
    outdir = os.path.dirname(paths[0]) or "."
    for L in logs:
        print("wrote", per_log_figure(L, outdir))
    by_rate = defaultdict(list)
    for L in logs:
        by_rate[L["rate"]].append(L)
    for rate, group in sorted(by_rate.items()):
        for out in per_rate_figures(rate, group, outdir):
            print("wrote", out)


if __name__ == "__main__":
    main()
