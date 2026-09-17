# GIAVA world frame and arm placement — definitive reference

Everything below is read directly from `giava.urdf` (link `base` is the world
origin) plus the deployed teleop remap. Stand at the operator position and
check each row against the physical rig.

## The world frame

Origin: the `base` link. All three arm bases mount 20 mm above it (z = 0.020).

| axis | direction (operator terms) | evidence |
|---|---|---|
| **+z** | up | all bases at z=0.020, waists 79 mm above their base |
| **+x** | operator's **left** | validated by the working gripper-arm mapping (`R_arm_remap`): moving a hand left drives target +x, and that feels correct on the real arms |
| **+y** | operator's **backward** (toward you) | hand-forward drives target −y via the same validated remap |

Right-handed: x × y = z ✓.

## Arm base placements (from the URDF fixed joints)

| arm | base link position (x, y, z) | base yaw | reading |
|---|---|---|---|
| right | (**−0.520**, −0.019, 0.020) | 0 | 0.520 m to the operator's **right** |
| left | (**+0.535**, −0.021, **0.045**) | **π − 0.24°** | 0.535 m to the operator's **left**, 25 mm higher than the right base, base frame *facing the right arm*. Touch-calibrated 2026-09-03 against the right arm (`calibration/touch_calibrate.py`, 15 tip-to-tip touches, 15 mm rms residual); it was the symmetric assumption (+0.520, −0.019, 0.020, π) before. |
| middle | (0.000, **+0.400**, 0.020) | 0 | centered, 0.400 m **behind** the manipulator line (operator side) |

So: the two manipulators face each other 1.040 m apart along x (measured
centre-to-centre 2026-08-19; the URDF said 0.938 m before that), and the camera
arm sits 0.4 m behind their midpoint, 19 mm behind the manipulator base line
(they're at y = −0.019). To reach over the workspace, the camera arm extends
toward **−y**, the same direction as your hands going forward.

Two frame subtleties on top of this:

1. **The left arm's base is yawed 180°** — its local frame points opposite the
   right arm's. Anything reasoning in a manipulator's *local* frame must
   account for this; world-frame targets (what the IK uses) are unaffected.
2. **The middle waist joint frame is yawed −90°** relative to its base link
   (`middle_base` joint rpy = −π/2), *and* the driver's zero for that multiturn
   joint sits π away from the URDF zero (handled by
   `CoupledStudyIK.driver_to_urdf`, offset `WAIST_URDF_OFFSET = π`). Both are
   internal to the solver — but they are why "waist at 0" on the driver does
   not mean "pointing along +x" physically.

## Headset → robot mapping

`webrtc_headset.py` delivers the head pose (`HPosition`/`HRotation`) from the
**same runtime and same tracking frame** as the hand controllers. Therefore the
camera arm uses the **same remap** as the gripper arms (this was the bug: it
was identity before, feeding raw headset axes to the camera target).

```
R_remap = [[0, 1, 0],
           [-1, 0, 0],
           [0, 0, 1]]          # headset (fwd=+x, left=+y, up=+z) -> robot
```

- translation: `d_robot = R_remap @ d_headset`
- rotation: `dR_robot = R_remap @ dR_world @ R_remap.T` (world-frame
  composition, then frame change) — this sends head **yaw → robot yaw**
  (about +z), **pitch → pitch** (about x), **roll → roll** (about y).

Expected motions after the fix, from the operator seat:

| head motion | camera arm response |
|---|---|
| lean left / right | target moves +x / −x (your left / right) |
| lean forward / back | target moves −y / +y (away / toward you) |
| stand taller / crouch | target moves +z / −z |
| yaw left/right | camera yaws about vertical |
| nod up/down | camera pitches |

To verify empirically: run with `GIAVA_DEBUG_HEAD=1` and move your head along
one axis at a time; it prints `d_headset -> d_robot` at ~2 Hz.
