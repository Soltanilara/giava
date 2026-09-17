"""Calibrate the middle arm's driver<->URDF joint offsets.

Procedure (once):
  1. SIM   : roslaunch av_aloha 3arms_sim.launch, run teleop_debug_tool, and
             JOG the sim middle arm until it visually matches the REAL arm's
             physical forward pose (compare with a photo or the real rig).
             Capture it as pose name  urdf_forward  (capture/goto arm: middle).
  2. Run   : python make_middle_offsets.py
             -> writes middle_joint_offsets.json (offset = M_FORWARD - captured)
  3. Restart anything using CoupledStudyIK; it loads the file automatically
             and corrects FK/commands on the REAL arm.  As of 2026-08-21 the
             2026-08 calibration has been FOLDED INTO giava.urdf instead, so
             this file is empty and the conversion is the identity; re-run
             with --check to confirm the URDF still matches the hardware.  Sim needs no offsets
             (the file only changes the driver<->URDF conversion, which is
             identity+waist in sim... the sim driver IS the URDF frame, so run
             sim WITHOUT the json present, real WITH it -- or pass --check to
             see the numbers before committing).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_config import ARM_CONFIG, POSES  # noqa: E402

POSES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "poses_custom.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "middle_joint_offsets.json")

book = json.load(open(POSES_FILE))
try:
    urdf_fwd = np.asarray(book["middle"]["urdf_forward"], float)
except KeyError:
    sys.exit("capture the sim pose as  middle:urdf_forward  first (see docstring)")

m_forward = np.asarray(POSES["middle"]["forward"], float)
names = ARM_CONFIG["middle"]["joint_names"]
off = m_forward - urdf_fwd

## Axis directions:  driver = sign * urdf + offset
##
## EMPTY since 2026-08-21.  The two flips measured here in 2026-08
## (middle_shoulder, middle_upper_arm) were not an assembly property at all --
## giava.urdf modelled both axes as `0 -1 0` where the vendor description
## wx250s_7dof.urdf.xacro and giava.urdf's own left/right arms all use `0 1 0`.
## The axes were negated in giava.urdf, so the URDF now turns the same way the
## driver does and no sign belongs here.  A re-run should therefore report
## sign +1 and ~0 offset on every joint; anything else means the URDF fold and
## the hardware have drifted apart.
SIGNS: dict = {}

print(f"{'joint':22s} {'sign':>4s} {'real fwd':>9s} {'urdf fwd':>9s} {'offset':>8s}")
out = {}
for j, n in enumerate(names):
    sg = SIGNS.get(n, 1)
    o = float(m_forward[j] - sg * urdf_fwd[j])
    if n == "middle_base":
        print(f"{n:22s} {'':>4s} {m_forward[j]:+9.3f} {urdf_fwd[j]:+9.3f} "
              f"(waist: handled separately; deviation from pi = "
              f"{np.degrees(abs(abs(o) - np.pi)):.1f} deg)")
        continue
    print(f"{n:22s} {sg:+4d} {m_forward[j]:+9.3f} {urdf_fwd[j]:+9.3f} {o:+8.3f}"
          + ("   << significant" if abs(o) > 0.03 else ""))
    ## Only genuinely nonzero corrections are emitted.  Post-fold this file
    ## is expected to come out EMPTY, and writing a dict of zeros would
    ## overwrite the documentation in middle_joint_offsets.json for no gain.
    if sg != 1 or abs(o) > 1e-3:
        out[n] = {"sign": sg, "offset": round(o, 5)}

if "--check" in sys.argv:
    print("\n--check: nothing written")
else:
    if not out:
        print("\nno correction needed -- the URDF already matches the driver. "
              f"Nothing written; {OUT} left as is.")
    else:
        json.dump(out, open(OUT, "w"), indent=2)
        print(f"\nwrote {OUT} -- restart data_collection / teleop_debug_tool.")
        print("NOTE: this file is a driver<->URDF patch that ONLY study_ik and")
        print("calibration/kinematics apply.  Anything loading giava.urdf")
        print("directly (RViz, pyroki collision, view_collision.py,")
        print("capsule_gate_view.py) will draw the UNCORRECTED arm.  Prefer")
        print("folding the correction into giava.urdf -- see the")
        print("_folded_into_urdf note in middle_joint_offsets.json for the")
        print("exact procedure and its FK verification.")
