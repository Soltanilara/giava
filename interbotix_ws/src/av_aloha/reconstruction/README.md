# GIAVA reconstruction — stage two: pixels to metres

```
world coordinates -> robot pose -> camera pose -> calibrated/synchronised
images -> camera extrinsics   |   -> 3D reconstruction
                              |
                       calibration/  |  reconstruction/
```

`calibration/` establishes and measures the front half of that chain and
writes JSON. **This package consumes that JSON and does the geometry.** It
never writes or repairs a calibration: a missing extrinsic raises, naming
the command that would produce it.

## Running these

```bash
conda activate gym_av312
cd /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts
python reconstruction/selftest.py          # 33 checks, no hardware
```

The selftest is not a smoke test. Every check compares against ground
truth the code under test cannot see: synthetic cameras with known poses
project known 3D points, and the recovered numbers are compared with the
ones that generated them. A test that only checked self-consistency would
pass just as happily with the distortion direction inverted, which is the
single most likely bug here.

---

## 1. What one click actually buys you

**A single pixel in a single camera is a ray, not a point.** Every position
along that ray produces the same pixel. This is not a limitation of the
calibration; it is what a projection is.

There are exactly three honest ways out, and all three are implemented so
that a measurement can be *cross-checked* rather than trusted:

| | needs | exact when | use it for |
|---|---|---|---|
| **`triangulate`** | two views | the two clicks are the same feature | anything, this is the main path |
| **`ray_plane`** | one view + a known surface | the point really is on the plane | "where on the table is that" |
| **`depth_from_known_size`** | one view + a known length | the segment is fronto-parallel | **checking** the other two — it uses no extrinsics at all |

The third one is the important one for trusting the rig. It shares no
inputs with the extrinsics, so when it agrees, that agreement means
something.

### The epipolar curve is a curve

The ray from `top_scene` reappears in `low_scene` as the locus of every
place the feature could be. Everyone calls it the epipolar *line*; with a
78° D405 and real distortion it is a **curve**, and treating it as a line
is several pixels of error near the image border — measured at >0.5 px of
sagitta in the selftest, and much more at the corners.

So it is never approximated. The ray is sampled in 3D and every sample is
projected through the destination camera's full distortion model. Each
sample carries the depth that produced it, so the curve is not just a
locus but **a depth scale you can read off the image** — which is what the
Measure tab labels in millimetres.

`depth_range` is a real constraint, not a rendering detail. It is the prior
you are asserting about how far away the thing might be, and narrowing it
shortens the curve and makes the correspondence unambiguous.

---

## 2. The distortion trap this package exists to avoid

Two mutually inverse conventions are in play on this rig, and mixing them
is silent — the numbers stay plausible and are simply wrong by a few
pixels at the image edge.

| | polynomial runs | closed form | iterated |
|---|---|---|---|
| **OpenCV** `plumb_bob` (what `charuco_calibrate.py` writes) | ray → pixel | distorting | undistorting |
| **librealsense** `inverse_brown_conrady` (what the D405 reports, what `rs_intrinsics.py` recorded) | pixel → ray | undistorting | distorting |

Same five numbers, same symbols `k1 k2 p1 p2 k3`, opposite directions.
`compare_intrinsics.py` already refuses to difference them term by term for
exactly this reason.

So `PinholeCamera` never exposes "the coefficients". It exposes

```
unproject(uv)   pixel -> normalised ray
project(xyz)    camera-frame point -> pixel
```

and each model supplies whichever direction it has in closed form while the
other is obtained by damped fixed-point iteration to a stated tolerance.
Every consumer — triangulation, epipolar curves, reprojection error — is
then model-agnostic and correct for both.

**Consequence for OpenCV calls elsewhere:** never hand the factory
coefficients to `solvePnP` or `stereoCalibrate`. `scene_extrinsics.py`
converts corners to normalised rays first and calls OpenCV with `K = I`
and `dist = 0`, where the question does not arise.

### Points outside the model come back NaN, per element

A point at an absurd radius, or behind the camera, has no pixel. It
returns `NaN` rather than a plausible-looking number — and **per element**,
not by raising for the whole batch. That distinction is load-bearing: an
epipolar curve deliberately sweeps depths most of whose samples fall
outside the destination image, and raising would discard the valid samples
along with the invalid ones. `in_image` rejects NaN, so nothing silently
consumes one.

---

## 3. Every answer carries its own error bars

A triangulated point without them is not a measurement.

| | question it answers |
|---|---|
| `ray_gap_mm` | **are these two clicks the same thing?** The rays never quite meet; large means you clicked two different features, or the extrinsics are wrong |
| `reprojection_px` | where the answer lands back in each image versus where you clicked |
| `sigma_mm` / `sigma_axes_mm` | what a plausible click error costs, as a 3×3 covariance and its principal axes |
| `condition` | geometric conditioning — two cameras viewing along nearly the same direction cannot separate depth, and this says so numerically |
| `viewing_angle_deg` | 90° is ideal, which is why the rig is built perpendicular |

`ray_gap` and `sigma` answer *different* questions and both matter: the
first is "are these consistent", the second is "how much would being one
pixel off cost me". The covariance is computed by numeric Jacobian through
the real distortion model and the real solve, so it cannot silently
disagree with them the way a hand-derived formula would — it is checked
against a 4000-sample Monte Carlo in the selftest.

`quality()` returns a verdict and the reasons. It is deliberately
conservative: the failure it guards against is a confident millimetre
readout produced from two clicks on different objects, which looks
completely normal in the 3D view.

---

## 4. Which frame you are measuring in

```python
from reconstruction import load_rig
rig = load_rig(["top_scene", "low_scene"])   # frame="auto"
print(rig.ref_frame)
```

| `frame=` | reference | needs |
|---|---|---|
| `"base"` | robot world frame | the rig anchored to the robot |
| `"stereo"` | the reference camera's optical frame | only the camera-to-camera solve |
| `"auto"` | world if anchored, else stereo | — |

**Lengths, sizes and shapes are already fully correct in the stereo
frame.** The metric scale comes from the measured ChArUco square, not from
the robot. Only the *expression in robot coordinates* is missing. So you
can measure link lengths and check the rulers before ever touching the
arms — and those measurements are then **independent evidence about the
robot**, which is the entire point.

Verified on synthetic data with a deliberately imperfect anchor: absolute
positions inherited the anchor error (≈6–10 mm), while a measured 150 mm
length came out at **149.99 mm, bias −0.01 mm**. Distances do not inherit
the anchor error. Sizes are trustworthy long before world coordinates are.

Asking for `"base"` when no anchor exists **raises**. It does not quietly
fall back — a measurement labelled `base` that is actually in camera
coordinates is the worst failure mode available here.

---

## 5. Command line

```bash
python reconstruction/measure.py status       # what the rig is, where it is
python reconstruction/measure.py precision    # what it can actually resolve
python reconstruction/measure.py pair  --images top.png low.png
python reconstruction/measure.py known-size --camera top_scene \
    --a 210 288 --b 402 279 --size 0.30
```

`precision` is the one to run before relying on anything. A calibration
report tells you how well the solver fitted its own data; this tells you,
given where the cameras ended up, how many millimetres one pixel of click
error costs at the places you care about — and states plainly whether that
is enough for the question you are asking.

Interactive measurement normally happens in the viewer instead:

```bash
python calibration/world_view.py --from-robot --measure --depth
```

---

## 6. API

```python
from reconstruction import load_rig, distance_with_error

rig   = load_rig(["top_scene", "low_scene"])
curve = rig.epipolar_curve("top_scene", (312, 205), "low_scene",
                           depth_range=(0.15, 2.0))
tri   = rig.triangulate({"top_scene": (312, 205),
                         "low_scene":  (401,  88)}, sigma_px=1.0)
print(tri.describe(rig.ref_frame))

a = rig.triangulate({...});  b = rig.triangulate({...})
print(distance_with_error(a, b))         # mm, with a propagated sigma
```

| module | holds |
|---|---|
| `camera.py` | `PinholeCamera` — intrinsics, both distortion conventions, pose |
| `stereo.py` | `StereoRig` — epipolar curves, triangulation, plane/known-size |
| `rig.py` | assembling a rig from what `calibration/` wrote |
| `measure.py` | the CLI |
| `selftest.py` | 33 ground-truth checks |

---

## What this does NOT tell you

A propagated sigma covers **click noise only**. A scale error in the
extrinsics biases every length by the same factor and is invisible to
every number in this package — two calibrations agreeing means they are
consistent, not correct.

Settle it physically: the rulers on the table are in view of both cameras.
Measure a known span through the rig and compare. That is the only check
here that is not ultimately circular.
