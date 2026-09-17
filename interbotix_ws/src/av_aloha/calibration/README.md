# GIAVA calibration and validation

Stage one of

```
world coordinates -> robot pose -> camera pose -> calibrated/synchronised
images -> camera extrinsics -> 3D reconstruction
```

Everything here establishes and **measures** the front half of that chain.
There is no reconstruction code in this package and none should be added to
it; the JSON files it writes are the inputs the reconstruction stage
consumes.

That stage now exists, as a sibling package: **`../reconstruction/`**. It
turns pixels into metres (epipolar correspondence, triangulation, metric
measurement) and reads what this package writes. `world_view.py` imports
from it for the Measure tab; nothing here depends on it otherwise.

## Running these

Yes, `gym_av312` — it is the only env with pyrealsense2, OpenCV 5, viser,
pyroki and jax together. **Activate it, don't use `conda run`:**

```bash
conda activate gym_av312
cd /home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts
python calibration/<script>.py --help
```

`conda run` does **not forward stdin**, so every prompt in these scripts
dies with `EOFError` under it — and the interactive pauses are the whole
point of the ruler-measurement tools. Scripts that hit this now say so
instead of crashing. For scripted use, pass `--dry-run`, `--yes` or
`--no-prompt`.

Anything that touches the robot also needs ROS available. Sourcing ROS is
the normal way, but the scripts append the ROS `dist-packages` paths
themselves (as `data_collection.py` does), so `--from-robot` works whether
or not the shell sourced it first.

Scripts can be run from the repo scripts directory (`python
calibration/x.py`) or from inside `calibration/` (`python x.py`) — they fix
up `sys.path` either way.

Start here:

```bash
python calibration/selftest.py
```

64 checks, no hardware needed. It verifies the transform algebra, that this
package's forward kinematics agrees with the deployed `robot_control` path,
the driver↔URDF joint bridge, the never-overwrite guarantee, a full ChArUco
calibration against synthetic views with a known camera matrix, and the
scene-camera extrinsics stage end to end against a synthetic camera pair.

The reconstruction stage has its own:

```bash
python reconstruction/selftest.py       # 33 checks, no hardware needed
```

---

## 1. What coordinate frames exist

Read directly from `giava.urdf` (31 links, 30 joints, 23 actuated).

| frame | what it is |
|---|---|
| `base` | **the world frame** — root link of `giava.urdf` |
| `{left,right,middle}_base_link` | arm mounts, fixed to `base` |
| per-arm joint links | one per revolute joint (6 + 6 + 7) |
| `{left,right}_gripper_base` | what the code calls the gripper "end effector" |
| `{left,right}_{left,right}_finger_link` | prismatic finger links |
| `middle_camera` | camera-arm head; `middle_camera_body` and `middle_camera_cover` are **exact identity children of it** |

Arm bases, from the world frame:

| arm | position (m) | yaw |
|---|---|---|
| right | (−0.520, −0.019, 0.020) | 0 |
| left | (+0.535, −0.021, 0.045) | π − 0.24° | touch-calibrated 2026-09-03 (`touch_calibrate.py`); was the symmetric (+0.520, −0.019, 0.020) |
| middle | (0.000, +0.400, 0.020) | 0 |

World axes, per `../FRAMES.md` (validated against the deployed teleop
remap): **+x** operator's left, **+y** operator's backward, **+z** up.
Right-handed. Metres and radians throughout.

**Transform naming, used in every file and every JSON key:**

```
T_a_b   maps points from frame b into frame a:   p_a = T_a_b @ p_b
        and IS the pose of frame b expressed in frame a.
        T_b_a = inv(T_a_b).      T_world_camera = T_world_ee @ T_ee_camera
```

Quaternions are stored **wxyz** (scalar first) to match pyroki/jaxlie and
the IK solver. `scipy`'s `Rotation.as_quat()` is xyzw — convert at the
boundary.

---

## 2. What each frame means

Run the diagnostic:

```bash
python calibration/frame_report.py --tree
```

Options: `--pose forward` (a named pose from `arm_config.POSES`),
`--from-robot` (live joint states, needs ROS), `--link NAME ...` (extra
links), `--json out.json`.

---

## 3. What the end-effector / TCP point actually represents

**There is no TCP anywhere in GIAVA.** No `ee_link`, `ee_gripper_link`,
`tcp`, `tool` or `tip` frame exists in `giava.urdf`, and no code applies a
constant transform after forward kinematics. The links reported as
"end-effector pose" (from `arm_config.ARM_CONFIG`) are:

**`left_gripper_base` / `right_gripper_base` — the gripper mounting plate.**
Reached from `*gripper_link` by a fixed joint
`xyz=(0.035, 0, 0) rpy=(-1.570000, 0.000796, -1.570796)`, which also
re-orients the frame. Local **+z** is the approach axis (wrist →
fingertips), fingers open/close along local **±x**, **+y** is the palm
normal.

The upstream interbotix `vx300s.urdf.xacro` — which `giava.urdf` was
flattened from, dropping the whole gripper tail — puts the canonical grasp
frame `ee_gripper_link` at `0.042825 + 0.025875 + 0.0385 = 0.1072 m` along
`gripper_link`'s +x. Re-expressed in `gripper_base` coordinates that is

```
(0.00000, +0.00006, +0.07220) m      i.e. 72.2 mm along local +z
```

### Commanding the grasp point instead

`move_validation.py --tcp` makes every target and readout refer to the
**grasp point** rather than the flange. The target is converted before it
reaches the solver:

```
T_world_flange = T_world_tcp @ inv(T_flange_tcp)
```

Why it matters on real hardware: the 72.2 mm offset is a **lever**. A 10°
wrist tilt moves the fingertips 12.5 mm even if the flange tracked its
target perfectly, and a 20° tilt moves them 24.7 mm. Commanding in TCP
space forces the solver to place the flange wherever puts the *fingers*
where you asked, so orientation drift stops leaking into fingertip
position.

It also fixes the z confusion: the world-frame z difference between flange
and fingertips is the full 72.2 mm when the gripper points straight down
and zero when it points horizontally. Measuring the fingers while the
software reports the flange will look like a z error that isn't one.

Put a measured offset in `tcp_offsets.json`; the built-in default is
derived and labelled `measured: false`.

**This is a derived candidate, not a measurement.** It assumes GIAVA's
custom fingers (`vx300s_8_custom_finger_*.stl`) grasp where the stock ones
did. The finger STL origins in `giava.urdf` sit at the `gripper_base`
origin, so they give no independent fingertip evidence. **Measure it.**

**`middle_camera_cover` — a cosmetic link on the camera arm.** It and
`middle_camera_body` hang off `middle_camera` by fixed joints with *exactly*
identity origins, so all three are the same frame. It is the IK target only
because the target must be downstream of the `middle_pan` joint, or the
solver cannot see the camera-yaw motor (`arm_config.py:11-15`). Local +x is
the optical axis. It is a **mount** frame, not an optical frame.

---

## 4. Robot motion validation

```bash
# preview with no hardware
python calibration/move_validation.py \
    --arm right --dry-run

# the real thing: six axes at 50 mm, pausing for a ruler at each extreme
python calibration/move_validation.py \
    --arm right --distance 0.05 --pause
```

`--distance` is in metres and is the configurable displacement.
`--axes` chooses and orders the moves and is **comma separated** —
`--axes +x,-x,+z`, not `--axes +x -x` (argparse would read the bare `-x` as
a flag). For a single negative axis use the `=` form: `--axes=-y`.
`--frame ee` interprets the axes in the end-effector's own frame instead of
the world, `--no-return` stays at the displaced pose, `--yes` skips the
confirmation.

It reports four columns per axis:

| column | meaning |
|---|---|
| `commanded` | what was asked for |
| `ik_predicted` | FK of the IK solution — what the solver believes it can reach |
| `measured` | FK of the joint angles the servos report afterwards |
| `physical` | **yours**, with a ruler |

The JSON leaves `physical_measured_m` as `null` for each axis; fill it in
after measuring. Orientation drift over each move is reported too — a ruler
at the fingertips only measures the same translation while the orientation
holds.

The arm returns to its **starting joint configuration** (not a re-solved
pose) after every axis, so it cannot land in a different IK branch.

Expect `ik_predicted` to fall a little short of `commanded`: pose tracking
is a weighted soft cost, not a constraint. It is much more noticeable on
the middle arm, whose deployed weights favour orientation
(`pos 10 / ori 25`) — in a dry run it reaches roughly 46–49 mm of a 50 mm
command. That is the deployed configuration behaving as designed, not a
fault.

---

## 4b. Measuring the arm base placement

The base poses in `giava.urdf` are **assertions**, not measurements. A base
origin is a virtual point inside the base plate and a base yaw is a
direction, so neither takes a ruler directly. Each protocol turns one of
them into a distance you can measure, using the robot as the instrument.

```bash
python calibration/base_validation.py --list          # all three procedures
python calibration/base_validation.py --protocol waist-circle --arm right
python calibration/base_validation.py --protocol straight --arm right
python calibration/base_validation.py --protocol converge
```

| protocol | measures | how |
|---|---|---|
| `waist-circle` | one base's **(x, y)** | The waist axis is a vertical line through the base origin, so rotating *only* the waist sweeps a circle centred on it. Mark the gripper at several angles and fit a circle; its centre is the base (x, y). Uses no IK, no FK, and no other joint's calibration. |
| `straight` | one base's **yaw** | Sweep the end-effector along world +x past a clamped straightedge. A yaw error θ shows up as a sideways drift `d = L·sin θ`. Over 300 mm, 1 mm of drift is 0.19°. A *constant* offset is just where the straightedge sits and cancels. |
| `converge` | the two bases **relative to each other**, including relative yaw | Command both grippers to the same world point; the measurable gap is the relative translation error. Repeat at a second point displaced along y — the change in the x-gap over the y separation is the relative yaw. Needs no external reference frame at all. |

### Stopping the arm

**Ctrl-C on its own does not stop the arm.** `set_joint_positions(...,
blocking=False)` hands a goal to the interbotix driver and returns; the
driver then executes it independently. Killing Python ends the script and
leaves the arm travelling to wherever it was last told.

So every motion tool now installs a signal handler that **halts the arms
before unwinding**: Ctrl-C commands each arm to hold its current measured
position. Torque stays on throughout, so nothing drops. The tools print
`emergency stop ARMED for <arms>` once a real arm exists, before any motion
is possible.

`world_view.py --from-robot` also has a **STOP ALL ARMS** button, which has
no terminal-mode complications and works while a prompt is waiting.

There was a second, quieter problem underneath. `robot_control.stop_arm`
used a fixed `moving_time=0.05`, and interbotix rejects any command whose
implied speed exceeds the joint velocity limit — measured against the
driver's last *accepted command*, not the measured position. Mid-motion that
difference is the whole remaining travel, so cancelling 0.2 rad in 0.05 s
asks for 4 rad/s against a 3.14 rad/s limit and is refused. And
`set_joint_positions` returns `False` rather than raising, which nothing
checked. **The stop silently did nothing in exactly the case you would press
it.** Measured against a faithful reimplementation of the driver's check:

| remaining travel | old `stop_arm` | fixed |
|---|---|---|
| 0.05 rad | accepted | accepted (50 ms) |
| 0.20 rad | **refused** | accepted (91 ms) |
| 1.00 rad | **refused** | accepted (455 ms) |
| 2.00 rad | **refused** | accepted (909 ms) |

`stop_arm` now sizes `moving_time` from the actual distance, checks the
return value, and escalates. That fixes `data_collection.py`'s shutdown path
too, which calls it via `stop_robots`.

If a halt is ever refused four times over, the tools say so explicitly and
tell you to kill the roslaunch or use the physical switch — the one thing
that must never happen is a silent failure.

### Space, and what is actually checked

**The IK collision model is self-collision only.** It checks the robot
against itself and knows nothing about the table, the frame bars or the
workstation. A reported clearance of "+19 mm" means the arm is not
intersecting *itself* — it says nothing about whether it is about to swing
into a bar.

So before any large motion these tools print the **swept envelope**: the
world-frame bounding box of *every* arm link, across the configurations
passed through in between, not just the commanded stops. The envelope is
wider than the gripper's own radius — the elbow and forearm swing further
out — so don't size your free space off the fingertips.

Measure your rig's free space once into `workspace_limits.json` and every
motion tool checks against it automatically, naming the bound and the
overshoot. Until then the tools say plainly that the motion is
**unverified** and ask you to eyeball the envelope.

`waist-circle` sweeps **relative to the arm's current angle** and defaults
to a 90° arc — it does not do a full revolution unless asked. If your rig
has the headroom, `--tuck --arc 360` is both the most accurate option and
the most compact: standing the arm upright with the gripper ~150 mm off its
waist axis turns all the way round inside a ~450 mm square, versus the
better part of a metre untucked.

`--dry-run` previews everything with no hardware. Each run writes JSON with
`measured_*` fields left `null` for you to fill in.

On clearance numbers: the sphere model is **inscribed**, reading up to
~18 mm optimistic on true contacts, so ordinary safe poses report only
10–20 mm. The refusal threshold defaults to 5 mm for that reason; the
deployed IK's 20 mm is a soft *cost* activation, not a hard limit.

---

## 5. RealSense factory intrinsics

```bash
python calibration/rs_intrinsics.py --list
python calibration/rs_intrinsics.py --all
python calibration/rs_intrinsics.py \
    --cameras right_wrist --streams color depth --all-profiles
```

Saves per camera to
`data/cameras/<name>_<serial>/factory_intrinsics_<stamp>.json`: resolution,
fx, fy, ppx/cx, ppy/cy, distortion model and coefficients, FOV, stream and
device identity (serial, firmware, USB type), depth scale, and the
depth→color extrinsic when both streams are requested.

Camera names come from `camera_manager.CAMERA_SERIALS` — the same table
data collection uses. Defaults match production: 640×480 rgb8 @60.

---

## 6. Collecting ChArUco images

Board geometry lives in `charuco_board.json` and **every field is
configurable**. Nothing hard-codes a monitor size, a DPI, or a physical
dimension.

```bash
# 1. render a board to print or display
python calibration/charuco_capture.py \
    --generate-board /tmp/board.png

# 2. MEASURE the displayed/printed square, put it in charuco_board.json,
#    set "measured": true

# 3. collect views
python calibration/charuco_capture.py \
    --camera right_wrist --target 30 --min-tilt 12
```

`SPACE` saves a view, `q` finishes. Headless: `--auto --interval 1.5`.
Overrides: `--squares-x/--squares-y/--square-length/--marker-length/
--dictionary`.

Capture enforces the two things that actually determine calibration
quality: **pose diversity** (`--min-tilt`; a pile of fronto-parallel views
is nearly singular, because focal length and board distance trade off
against each other) and **image coverage** (a live grid; distortion is
unconstrained wherever no corners landed).

### Why the measured square length matters

`fx, fy, cx, cy` and the distortion coefficients are in pixels and are
**unaffected** by a scale error. Every metric quantity downstream — board
pose, hand-eye translation, reconstruction scale — is wrong by exactly the
same factor. A 2% error in `square_length_m` is a 2% error in every
distance you ever compute from that camera. Measure across several squares
and divide.

---

## 7. Running the OpenCV calibration

```bash
python calibration/charuco_calibrate.py \
    --images data/charuco/right_wrist_<stamp>/images --camera right_wrist
```

`--model plumb_bob|rational|no_k3`, `--reject-above 1.0` drops bad views and
re-solves once, `--annotate DIR` writes detection overlays.

Saves board definition, square/marker length, image resolution, camera
matrix, distortion coefficients, RMS, image count, coverage, and per-view
errors.

**RMS is an optimisation residual, not an accuracy figure.** A low RMS on a
poorly conditioned set of views means very little; the tool warns when
coverage is under 60%, when all views sit at nearly the same distance, and
when the board is not marked as measured.

---

## 8. Comparing factory vs OpenCV

```bash
python calibration/compare_intrinsics.py --camera right_wrist
```

Two independent sections, because they answer different questions.

**Directly comparable:** resolution, `fx`, `fy`, `cx`, `cy` — absolute and
relative differences, plus aspect ratio and principal-point offset from the
image centre.

**Distortion is NOT directly comparable.** librealsense reports the D405
colour stream as `inverse_brown_conrady`, whose coefficients are applied
going *pixel → ray*; OpenCV's are applied *ray → pixel*. They are different
parameterisations of mappings in opposite directions, so `k1` vs `k1` is
meaningless and no difference is computed. Instead the tool compares what
the two calibrations **do**: a grid of pixels is deprojected to normalised
rays under each model and the disagreement is reported back in pixels,
banded by distance from the principal point. That number is
parameterisation independent and is the one that matters downstream.

The tool does not declare a winner. Two calibrations agreeing means they
are consistent, not correct — a stale factory calibration or a mismeasured
board is invisible to any numerical comparison. Settle it physically:
image an object of known size at a known distance, or undistort a straight
edge near the image border with each and see which straightens it.

---

## 9. Two-camera synchronisation

```bash
python calibration/sync_capture.py \
    --cameras left_wrist top_scene --samples 300
```

`--interval` (keep it above the frame period for independent samples),
`--save-images`, `--no-global-time` to see the raw device clocks.

Writes `sync_summary.json`, `sync_raw.json` and `sync_raw.csv` (the CSV
drops straight into pandas). Per paired capture it records, for each
camera: name, serial, frame number, librealsense timestamp **and its
domain**, sensor/frame timestamps (device clock), backend timestamp and
time-of-arrival (host clock), host epoch, and image path. Statistics:
mean, median, std, min, max, range, p05/p95, frame-to-frame variation, and
linear drift across the run.

### These clocks are not interchangeable

| clock | what it is |
|---|---|
| `hardware_clock` domain | raw device counter; **arbitrary origin, unrelated between devices** |
| `global_time` domain | device clock mapped onto the host epoch by a **per-device linear fit** — an estimate, not sync |
| `sensor_timestamp` | device time at start of exposure; closest to the photons, still in the device's own base |
| `backend_timestamp` / `time_of_arrival` | **host** times after USB transfer; include transfer and scheduling jitter |

**Nothing here proves hardware synchronisation, and this rig provides
none:** the D405s have no external sync wiring and no `inter_cam_sync_mode`
is configured, so their shutters free-run and drift apart. The tool states
this in its output and in every file it writes. Usually the spread (std,
range, drift) matters more than the mean — a constant offset can be
calibrated out, jitter cannot.

Frame age at sample time and duplicate-frame rate are reported: if the
sampler outruns the cameras it re-reads the same frame, and those samples
are not independent measurements.

**Measured on this rig (2026-08-18, `left_wrist` vs `top_scene`, 150
samples):** global-time offset +7.30 ms, std 0.027 ms, range 0.076 ms,
drift +0.00043 ms/sample. Host-side clocks agree at ~7.3 ms with ~0.5–0.7 ms
jitter (1 ms quantisation).

### Frame alignment in the record loop

`camera_manager.select_synchronized_frames` picks, per camera, the frame
nearest a shared reference — the newest instant every camera has already
covered — instead of whatever is newest. `GIAVA_SYNC_FRAMES=0` restores
latest-frame-wins. Each timestep's achieved spread is recorded and reported
per episode as `cam_sync_spread_ms`.

What it buys: each chosen frame is within **half a frame period** (≤8.3 ms
at 60 fps) of a *recorded* reference, bounded and independent of when the
tick happened to poll. What it cannot buy: the spread *between* cameras.
Measured with three cameras all at a true 60.2 fps and dense histories, the
per-camera offsets were −6.16 / 0.00 / +5.41 ms — each well inside the
half-period bound, but 11.6 ms apart overall. **That spread is the cameras'
physical phase difference; frames do not exist at the same instants and no
selection rule can invent them.** Closing it needs hardware sync, which the
D405 does not offer.

The offsets are stable (std 0.027 ms over 150 samples), so they are a
calibratable constant rather than noise — which is what makes the faithful
per-camera timestamps worth recording.

### Stopwatch check — the independent test

```bash
python calibration/stopwatch_check.py --cameras left_wrist top_scene low_scene
```

Every other figure here is the software's claim about itself. Point all the
cameras at a running stopwatch, capture one aligned set, and read the
digits: the answer is written in the scene. Saves a labelled montage per
shot (camera, serial, frame number, Δt from reference) plus a JSON with
`stopwatch_reading` left null for what you read.

Use milliseconds if the stopwatch has them — hundredths can only prove
you're within ~10 ms, which is the same order as the spread you're trying
to see. Bright light helps too: at 1/60 s exposure the digits blur across
~16 ms regardless of alignment.

---

## 10. Launching the Viser viewers

### World view — robot, frames, cameras, jog (port 8094)

```bash
python calibration/world_view.py                      # offline
python calibration/world_view.py --from-robot --arms right   # live + jog
```

```bash
python calibration/world_view.py --cameras left_wrist top_scene   # + live images
```

Shows the world origin and its axes, the robot at its measured joint state,
and each camera drawn as a frustum at `T_world_ee @ T_ee_camera` — labelled
with its provenance, because none of those mounts is calibrated yet. Pass
`--cameras` to stream live RealSense images into the same view.

The panel is **tabbed** — Frames / Cameras / Pose readout / Poses / Jog — so
the camera images are one click away rather than a scroll past every arm
control. (Viser has only one control panel; there is no second/left panel in
its API. For a literal side-by-side, run two servers on two ports and tile
the browser windows.) **STOP ALL ARMS sits above the tab group**, never
inside it — an emergency control must not be one click behind a tab.

The **Poses** tab lists every named pose from `arm_config.POSES` plus
anything in `poses_custom.json` (shown as `custom:<name>`), moves the
selected arm there via `robot_control.interpolate_to_pose` (so the
wrapped-encoder guard and the middle waist's nearest-2π handling still
apply), and checks the whole interpolated path against the workspace box
first — a named pose can easily route through a bar.

**SAVE CURRENT JOINTS** captures the arm's measured joints under a name,
writes `poses_custom.json` (the same store `teleop_debug_tool.py` uses and
`make_middle_offsets.py` reads), and shows a line to paste straight into
`arm_config.py`:

```python
RIGHT_MY_POSE = np.array([0.11, -0.48, 0.33, -0.03, 1.35, 0.05], dtype=float)
# then add to POSES["right"] as  "my pose": RIGHT_MY_POSE
```

Per arm the dashboard reports, live:

- end-effector position in **m and mm**, rpy in degrees, quaternion wxyz
- **height above the z=0 plane** and **horizontal distance from its own
  waist axis** — the two numbers you can put a tape measure on directly
- every joint in **driver and URDF coordinates side by side**, with the
  differing ones flagged (they differ for the middle arm: flipped axes,
  mounting offsets, and the waist's π shift)

With `--from-robot` the jog panel moves one arm a fixed step along a world
axis (1 cm by default) and reports **commanded / IK-predicted / measured**
side by side, so a ruler reading has something to be compared against. Jogs
are saved to `data/robot/world_view_jog_*.json` with a
`physical_measured_mm` field for your reading. It uses the deployed
`CoupledStudyIK` and the same driver clamp as `data_collection.py`.

### Two-camera viewer (port 8093)

```bash
python calibration/viser_cameras.py \
    --cameras left_wrist top_scene --port 8093
```

Open <http://localhost:8093> (forward the port over SSH / remote VS Code).
Port 8093 continues the repo block: 8082 `teleop_debug_tool`, 8090
`view_trajectories`, 8091 `view_collision`, 8092 ik `playground`.

Two live images and a **CAPTURE BOTH CAMERAS** button. Each capture saves
both PNGs and records frame IDs, all timestamps, and Δt on every clock, to
`captures.json`.

The 3D scene is deliberately almost empty — drawing the cameras and robot
needs `T_world_camera = T_world_ee @ T_ee_camera`, and no calibrated
`T_ee_camera` exists yet.

---

## Camera mounting: what is and is not known

```bash
python calibration/camera_mount.py --world
```

| camera | hardware | rigid to | transform | provenance |
|---|---|---|---|---|
| `left_wrist` / `right_wrist` | D405 | `*gripper_base` | present | `mujoco_model` — **not validated** |
| `oak_left` / `oak_right` | OAK-D | `middle_camera` | **missing** | `unknown` |
| `top_scene` / `low_scene` | D405 | `base` (static) | via `scene_extrinsics.py` | `unknown` until solved |

**No hand-eye calibration exists for the WRIST cameras.** Their transform is
*derived*, not measured. The two **static scene cameras** are a different
problem with a different answer — see section 11.

- `giava.urdf` carries a `d405_solid.stl` **mesh** on each gripper base at
  `xyz=(0, −0.082475, −0.009595) rpy=(0.436333, 0, −3.141593)` — a mesh
  origin locates the camera *body*; its orientation is the STL's authoring
  frame, not the optical axis.
- The MuJoCo model's `wrist_cam_left` sits at the same position (agreeing
  to **0.0005 mm**) and supplies an optical axis, converted here from
  MuJoCo's convention (−z forward, +y up) to OpenCV's (+z forward, +y down)
  by a π rotation about x. The mesh and optical orientations differ by
  exactly 180°, as expected.

Record real calibrations in `ee_camera_transforms.json`; anything listed
there overrides the nominal value and is reported as `CALIBRATED`.
`T_world_camera()` **raises** rather than substituting a guess when a
transform is missing.

---

## 11. The static scene cameras: `top_scene` and `low_scene`

These two are bolted to the rig, not to an arm, so what they need is
`T_base_camera` **directly** — there is no `T_ee_camera` for them. Until
this existed, `camera_mount.py` listed both with `provenance: unknown` and
`T_world_camera()` raised for them, which was correct: nothing knew where
they were.

```bash
python calibration/scene_extrinsics.py status     # what is missing
python calibration/scene_extrinsics.py collect --cameras top_scene low_scene
python calibration/scene_extrinsics.py solve   --dir <session>
python calibration/scene_extrinsics.py anchor  --points <correspondences>
python calibration/scene_extrinsics.py write   --stereo <f> --anchor <f>
```

### Why it is two stages, and why that order

`solve` and `anchor` answer different questions and must not be merged.

**`solve`** uses only a ChArUco board held in the air. It learns where the
two cameras are *relative to each other*, and its metric scale comes from
the measured square — no robot, no URDF, no forward kinematics anywhere in
the chain. After this stage alone you can already measure lengths, sizes
and separations and check them against the rulers on the table.
**Everything measured at this stage is independent evidence about the
robot.**

**`anchor`** then finds where that rig sits in the robot's world frame, by
comparing points the robot *claims* to be at with where the cameras *see*
it. The fit residual is not a nuisance — **it is the measurement** of how
far forward kinematics disagrees with reality.

Doing it the other way round — calibrating the cameras against the robot
first — would make the cameras inherit whatever the URDF gets wrong, and
they could then never detect it. The measurement would be circular and
would look perfect. Worth being explicit about, because the circular
version is easier to run and gives lower residuals.

### Collecting views

Hold the board so **both** cameras see it, and vary the pose a lot. The
top camera looks down and the low one looks across, so the board wants to
sit at roughly 45° to satisfy both at once; `--min-tilt` (default 15°)
refuses views too fronto-parallel to carry information. Move it through
the whole shared volume — what is being solved is a rigid transform, and
views clustered in one place constrain it only there.

The shared volume on this rig is roughly **the central table region**: the
top camera sees the whole table from ~0.9 m, the low camera sits close and
horizontal and spends most of its frame on the background.

### What `solve` reports, and which number to believe

| | |
|---|---|
| per-view agreement | do independent views *agree*? Much harder to fake than a residual — a wrong board size, a mis-detection or a camera that moved all show up here and in nothing else |
| bundle RMS | how well the joint refinement fitted the data it was given |
| **closure check** | board corners re-triangulated by the solved rig, compared with the board's own geometry. A nonzero **bias** is a scale error; scatter is noise |

A large correction from the bundle refinement means the per-view estimates
were not really consistent, and the tool says so.

### Anchoring: the fit residual is the result

Correspondences come from `world_view.py --measure`: drive an arm to a
pose, click the same gripper feature in both camera images, record. Each
one pairs "where forward kinematics says the arm is" with "where the
cameras see it". Collect 8–15, spread across the workspace **and across
several heights** — near-coplanar points leave the fit poorly constrained
out of that plane, and the tool warns when they are.

The fit is run twice, deliberately:

* **rigid** (6 dof) — the residual is the honest FK disagreement. The fit
  cannot absorb it, which is exactly why it is the measurement.
* **similarity** (7 dof) — if a scale materially different from 1 fits
  better, something is wrong with a **length**: the ChArUco square (which
  sets the cameras' scale) or the URDF link lengths (which set the
  robot's). Reporting only the rigid fit would smear that into the
  residual and hide it. The similarity fit is *diagnostic only* — its
  scale is never written, since accepting it would silently rescale every
  future measurement.

The residual is then split into a **constant offset** and **scatter**,
because they have different causes and different fixes. A constant offset
surviving a 6-dof fit means a frame is defined in the wrong *place* — a
missing tool offset, a wrong link origin. That is precisely the shape of
"the gripper is 10 cm from where the software thinks". Scatter means joint
calibration, backlash, compliance or click noise.

A **per-arm** breakdown is printed when more than one arm contributed: a
systematic difference between arms is a *base placement* error, not a
joint error, and is invisible in the pooled number. This is the direct
measurement for the unresolved base-separation question.

### Distortion: never hand the factory coefficients to OpenCV

The D405 factory intrinsics are `inverse_brown_conrady` — coefficients
running **pixel → ray**. OpenCV's `solvePnP` and `stereoCalibrate` expect
**ray → pixel**. Passing the factory numbers straight in applies them
backwards, roughly a double distortion, a few pixels at the image edge,
and nothing would report an error.

So this script never passes distortion coefficients to OpenCV. Corners are
converted to normalised rays first (`reconstruction.PinholeCamera`, which
knows which direction each model runs) and OpenCV is called with `K = I`
and `dist = 0`, for which both conventions agree trivially. Consequence:
the reported RMS is in normalised units scaled back by the mean focal
length, not a raw OpenCV pixel residual, and says so.

---

## 12. Measuring: `world_view.py --measure`

```bash
python calibration/world_view.py --from-robot --measure --depth
```

Adds a **Measure** tab and, for every calibrated camera, a **clickable
image plane at that camera's real pose** in the 3D scene. Click a feature
in one camera, click the same feature in the other, and read its position
in the robot's world frame.

* **one click is a ray, not a point.** The first click draws that ray in
  3D and the epipolar **curve** on the other image, labelled in millimetres
  of depth. Every position the feature could be in lies on that curve.
* the second click picks which, and is snapped onto the curve — so the
  reported ray gap measures the *calibration* rather than the steadiness of
  your hand. Snapping can be turned off to see the raw disagreement.
* every readout carries its ray gap, reprojection error, viewing angle and
  the propagated cost of a 1 px click error.
* **plane mode** turns a single click into a position, for a point known
  to lie on the tabletop — exact if it really does, and the readout states
  how far a millimetre of height off the plane displaces it sideways.
* **ADD THIS POINT** / **DISTANCE** measures between two triangulated
  points, with a propagated sigma. This is the link-length measurement.
* **RECORD CORRESPONDENCE** captures an FK-vs-cameras pair for `anchor`.

`--cameras` defaults to `top_scene low_scene` when `--measure` is given.

### The click → pixel conversion is exact, not assumed

Viser reports a click as a *ray*, not a pixel. The conversion is derived
from the viser client's own frustum geometry rather than guessed, and the
displayed image is **undistorted to the ideal pinhole that frustum
actually is** — otherwise the ray implied by a click and the feature under
the cursor would disagree by the full distortion, worst exactly at the
edges. Pixel coordinates written to disk are converted back to the real
distorted image, because that is the frame the raw capture is in.

Verified headlessly by simulating clicks at known world points: the
recovered pixel and the recovered 3D point are exact to floating point.

### `--depth`: a cross-check, never a source

`--depth` streams depth aligned to colour and shows, per click, what each
D405's own stereo pair says — using **none** of the extrinsics, which is
what makes it an independent check.

It is the *weaker* number here. The D405 is short-range (best ~70–500 mm)
and the scene cameras sit around 0.9 m, so a disagreement is not
automatically the triangulation's fault. **The production record loop stays
RGB-only**; this is opt-in and lives in the measurement tooling.

Measured on this rig: `top_scene` returns depth for 95% of pixels (median
0.90 m), `low_scene` for 71% (median 0.38 m). No z16 saturation.

---

## Where results go

```
calibration/
    *.py                        source
    charuco_board.json          board geometry      (tracked, hand-edited)
    ee_camera_transforms.json   calibrated mounts   (tracked, hand-edited)
    data/                       results             (gitignored)
        robot/                  motion validation runs
        cameras/<name>_<serial>/  factory + charuco intrinsics
        charuco/<name>_<stamp>/   captured board images + manifest
        sync/<pair>_<stamp>/      timing summary, raw JSON, raw CSV
        comparisons/              factory-vs-opencv reports
        viser_captures/<pair>_<stamp>/  paired captures
        scene_views/<pair>_<stamp>/     shared ChArUco views for the
                                        scene-camera extrinsics
        extrinsics/                     stereo_*.json (camera-to-camera)
                                        anchor_*.json (rig -> robot)
        measurements/                   correspondences_*.json from the
                                        Measure tab
```

Every artifact carries a `metadata` block: method, date, host, user,
platform, python, **git sha and whether the tree was dirty**, the URDF path,
units, plus method-specific fields (camera, serial, resolution, board
config, image count, reprojection error, robot arm and ee_link).

**Results are never silently overwritten.** `save_json` refuses to clobber
and tells you to re-run for a fresh timestamped file or pass `--overwrite`
deliberately. Default output paths are timestamped, so collisions do not
arise in normal use.

The repo-wide `.gitignore` ignores `*.json` everywhere, so `data/` is local
by construction; `calibration/.gitignore` re-includes the two hand-edited
config files by name so `git add` does not silently skip them.

---

## What this reuses

Nothing here reimplements robot or camera infrastructure.

| from | used for |
|---|---|
| `arm_config.ARM_CONFIG`, `POSES`, `URDF_PATH` | arm definitions, ee links, named poses |
| `robot_control` | robot creation/configuration, waist Homing_Offset |
| `study_ik.CoupledStudyIK` | the deployed coupled IK solver, and its `middle_joint_offsets.json` driver↔URDF correction |
| `pyroki` + `yourdfpy` on `giava.urdf` | forward kinematics (self-tested against `robot_control.compute_fk_and_ee`) |
| `camera_manager.CAMERA_SERIALS` | camera identity, names and serials |
| `../FRAMES.md`, `ik/robot_model.py` | world-frame and gripper/camera frame conventions |
| `../reconstruction/` | the 3D geometry `world_view.py --measure` displays, and the distortion-direction-aware `PinholeCamera` that `scene_extrinsics.py` uses to keep raw coefficients away from OpenCV |

The one deliberate divergence: `rs_camera.py` opens its own RealSense
pipelines rather than using `camera_manager`'s threaded capture loop. That
loop is latest-frame-wins by design — correct for teleoperation, where the
control tick must never block, but wrong for calibration, which needs a
specific frame together with its own metadata. Camera identity and stream
settings are still shared.

---

## Hardware notes

### `right_wrist` — RESOLVED 2026-08-18

It previously enumerated, reported firmware 5.12.14.100, accepted a stream
configuration and started a pipeline, then produced nothing at every
resolution and frame rate. It was a cable/port problem, and all four D405s
now deliver. `sync_capture.py` and `viser_cameras.py` still detect a silent
camera and fail with a named diagnostic rather than collecting zero
samples.

**Re-verify before trusting anything calibrated while it was dead.**

### The D405s cannot be hardware-synchronised

Verified on all four units by enumerating every supported option: they do
**not** support `inter_cam_sync_mode` — only `output_trigger_enabled` is
present. The genlock approach used on D435/D455 is not available, so
cross-camera alignment has to be software and the residual spread is
physical (see section 9).

The wrist-pair offset **re-randomises between sessions** (free-running
phase), so it must be measured per session, not calibrated once.

### Only one process can hold a camera

A RealSense can be opened by exactly one process. If `data_collection.py`
is running it holds all four, and every tool here will fail to start with
`Device or resource busy`. `rs_camera.py` names the offending PID and
command rather than reporting it as a USB or bandwidth problem.
