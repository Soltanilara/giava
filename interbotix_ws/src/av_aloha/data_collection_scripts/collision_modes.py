"""One switch for the three collision models, for A/B teleoperation.

    python data_collection.py --collision sphere
    python data_collection.py --collision capsule
    python data_collection.py --collision gjk

THE THREE MODELS DO NOT LIVE AT THE SAME LAYER
==============================================
This is the fact that makes a naive "swap the collision model" flag
misleading, so it is stated up front rather than hidden behind a preset.

There are two places collision geometry acts, and they are different in
kind:

  1. THE IK SOLVER'S SOFT COST.  A weighted residual the optimiser trades
     against pose tracking.  It shapes how the motion FEELS -- smooth,
     jerky, sluggish near obstacles -- and it must be DIFFERENTIABLE,
     because pyroki takes gradients through it.

  2. THE GATE.  A hard yes/no on the final clamped command, after the
     solver and after every clamp.  It decides WHERE YOU GET STOPPED and
     nothing else. It needs no gradients, only a conservative distance.

And so:

    sphere    solver only.  The 180-sphere model is INSCRIBED (optimistic
              by up to ~18 mm on true contact), which is fine for a soft
              cost and disqualifying for a gate: it cannot prove
              separation, so it can never be the last line of defence.

    capsule   both.  Circumscribed, differentiable, cheap -- but fat,
              padding the true surface by 28 mm on average and 61 mm on
              the gripper bases.

    GJK       gate only.  Exact, but it is an ITERATIVE algorithm with
              data-dependent branching and no useful gradient, so it
              cannot be an IK residual at all.  It answers "are these
              apart by at least X" and nothing more.

So the three presets below are not three settings of one knob.  Each is a
coherent configuration of both layers, and the banner prints exactly what
each one changed so a session's feel can be attributed correctly.


WHAT TO EXPECT, AND WHAT TO COMPARE
===================================
    sphere    the deployed baseline. The collision study's winner:
              0.60 mm / 0.24 deg clean p95, 5.2 ms mean solve.
    capsule   the study REJECTED this as a solver residual on simulation
              evidence -- 59-76 mm hand jerk, 33 % non-convergence, 43 ms
              solves. Measured 130 ms/solve on this machine's CPU jax.
              Feel it yourself; that is the point of the flag. Expect to
              need a lower control rate, and watch ik_overruns.
    gjk       same solver as `sphere`, but the gate is exact instead of
              fat: measured to stop the arms 27 mm closer to each other
              in real geometry while never touching.

The mode is recorded in the episode log, so which configuration produced
which recording is a fact about the data rather than something to
remember.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

## name -> (env overrides, one-line summary)
MODES: Dict[str, Tuple[Dict[str, str], str]] = {
    "sphere": (
        {"GIAVA_IK_COLLISION_MODEL": "sphere",
         "GIAVA_CAPSULE_GATE": "1",
         "GIAVA_CAPSULE_GATE_FINE": "0"},
        "deployed baseline: 180-sphere solver cost, coarse-capsule gate",
    ),
    "capsule": (
        {"GIAVA_IK_COLLISION_MODEL": "capsule",
         "GIAVA_CAPSULE_GATE": "1",
         "GIAVA_CAPSULE_GATE_FINE": "0"},
        "capsule solver cost (the study rejected this in sim -- feel it), "
        "coarse-capsule gate",
    ),
    "gjk": (
        {"GIAVA_IK_COLLISION_MODEL": "sphere",
         "GIAVA_CAPSULE_GATE": "1",
         "GIAVA_CAPSULE_GATE_FINE": "1"},
        "180-sphere solver cost, EXACT GJK gate -- stops ~27 mm closer",
    ),
}

DEFAULT = "gjk"


def take_option(name: str, argv: Optional[List[str]] = None,
                default: Optional[str] = None) -> Optional[str]:
    """Read `--name X` / `--name=X` and REMOVE it from argv.

    data_collection.py has no argparse and reads its episode index as a
    positional argument, so any flag bolted on must consume its own tokens
    or the positional parse sees them and fails.  Shared here so every
    such flag behaves identically instead of each one reinventing it."""
    in_place = argv is None
    argv = list(sys.argv if in_place else argv)
    value = default
    keep, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == f"--{name}" and i + 1 < len(argv):
            value = argv[i + 1]
            i += 2
            continue
        if a.startswith(f"--{name}="):
            value = a.split("=", 1)[1]
            i += 1
            continue
        keep.append(a)
        i += 1
    if in_place:
        sys.argv[:] = keep
    return value


def select(argv: Optional[List[str]] = None) -> str:
    """Read `--collision X` (or GIAVA_COLLISION) and apply it to os.environ.

    MUST be called before `study_ik` or `capsule_gate` are imported: both
    read their configuration at module import, so setting the environment
    afterwards would silently have no effect and the banner would describe
    a configuration that is not running."""
    mode = take_option("collision", argv,
                       os.environ.get("GIAVA_COLLISION", DEFAULT))
    mode = str(mode).strip().lower()
    if mode not in MODES:
        raise SystemExit(
            f"--collision must be one of {sorted(MODES)}, got '{mode}'")

    env, _ = MODES[mode]
    ## An explicit per-knob override wins over the preset, so a mode can be
    ## used as a starting point without being a straitjacket.
    for k, v in env.items():
        if k not in os.environ:
            os.environ[k] = v
    os.environ["GIAVA_COLLISION"] = mode
    return mode


def select_table(argv: Optional[List[str]] = None) -> bool:
    """Read `--table on|off` (or GIAVA_TABLE) and apply it to os.environ.

    Tabletop avoidance (ik/table_collision.py's soft cost + the
    table_gate.py hard floor gate) was hardware-validated 2026-09-09 on the
    real arms, though its margin has never been swept -- unlike the inter-arm
    collision modes above.  Both halves share ONE switch here so the whole
    feature can be killed in one flag if it misbehaves on hardware, instead
    of hunting two env vars mid-session.

    MUST be called before study_ik or table_gate are imported: both read
    their configuration from the environment at module import."""
    val = take_option("table", argv, os.environ.get("GIAVA_TABLE", "on"))
    val = str(val).strip().lower()
    if val not in ("on", "off"):
        raise SystemExit(f"--table must be on|off, got '{val}'")
    enabled = "1" if val == "on" else "0"
    os.environ.setdefault("GIAVA_IK_TABLE_ENABLE", enabled)
    os.environ.setdefault("GIAVA_TABLE_GATE", enabled)
    os.environ["GIAVA_TABLE"] = val
    return val == "on"


def banner(mode: str) -> str:
    _, summary = MODES[mode]
    g = os.environ
    fine = g.get("GIAVA_CAPSULE_GATE_FINE", "0") == "1"
    return "\n".join([
        "=" * 74,
        f"  COLLISION MODE: {mode.upper()}",
        f"  {summary}",
        "-" * 74,
        f"  IK solver soft cost : "
        f"{g.get('GIAVA_IK_COLLISION_MODEL', 'sphere')}"
        f"  (weight {g.get('GIAVA_IK_COLLISION_W', '100')}, "
        f"margin {float(g.get('GIAVA_IK_COLLISION_MARGIN', '0.020')) * 1e3:.0f} mm)"
        "   -- shapes how motion FEELS",
        f"  Hard gate           : "
        + ("OFF -- nothing prevents inter-arm contact"
           if g.get("GIAVA_CAPSULE_GATE", "1") != "1" else
           f"capsule"
           + (" -> EXACT GJK" if fine else " only")
           + f", margin "
           f"{float(g.get('GIAVA_CAPSULE_GATE_MARGIN', '0.025')) * 1e3:.0f} mm"
           "   -- decides WHERE YOU STOP"),
        "-" * 74,
        f"  Table (hw-validated) : "
        + ("OFF -- nothing prevents the arms pressing into the table"
           if g.get("GIAVA_TABLE", "on") != "on" else
           f"soft cost + hard z-floor gate, margin "
           f"{float(g.get('GIAVA_TABLE_GATE_MARGIN', '0.025')) * 1e3:.0f} mm"
           "   -- allows contact, resists penetration; see "
           "table_collision.py / table_gate.py"),
        "-" * 74,
        "  Switch with:  --collision sphere | capsule | gjk   --table on | off",
        "  The mode is recorded in the episode log.",
        "=" * 74,
    ])


def describe() -> str:
    lines = ["available collision modes:", ""]
    for name, (env, summary) in MODES.items():
        lines.append(f"  {name:<8} {summary}")
        lines.append(f"           {env}")
    return "\n".join(lines)


if __name__ == "__main__":
    m = select()
    print(banner(m))
    print()
    print(describe())
