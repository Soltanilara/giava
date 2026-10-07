# Teleoperation frames and math — the whole chain, with the actual numbers

Every pose in this system lives in one of four coordinate worlds, and every bug
we have chased for two weeks was a confusion between two of them. This is the
full pipeline, stage by stage, with the deployed constants.

## 1. The four worlds

**W_app — the tracking world.** Where the headset app's poses land *after
ingestion*. The wire format is Unity's left-handed y-up convention;
`webrtc_headset.py` converts every pose on arrival via
`headset_utils.convert_left_to_right_coordinates` (handedness flip: negate
y, qx, qz; then a fixed Euler(−90°, 0, −90°) rotation to z-up). What the rest
of the pipeline sees — and what the probe measured — is therefore right-handed,
**z = up**, but with **arbitrary horizontal orientation**: anchored wherever
the headset faced when tracking started (+43° in the probe session). This
ingestion step is also why the "left-handed quaternion" hypothesis during
debugging was a ghost: the mirror is real on the wire, but already fixed
upstream of everything we touched.

**F_head — the head's local frame.** The probe measured the *look direction* as
local **+x** (purity 0.90). Controllers are reported relative to this frame,
which is why the world hand pose must be composed:
`T_hand^W = T_head^W · T_hand^head`.

**W_robot — the robot world** (`giava.urdf` base). +x = operator's left, +y =
toward the operator, +z = up (FRAMES.md). The IK, the pose tables, and all
targets live here.

**EE frames** — `left_gripper_base`, `right_gripper_base`, `middle_camera_cover`
(one rigid body with the cameras, downstream of the pan joint).

## 2. Getting from W_app to W_robot: the calibrated remap

A *fixed* matrix cannot connect them, because W_app's yaw is random per
session. So at every anchor (button press) we **measure** the bridge:

```
f = horizontal( R_h · x̂ )          the operator's forward, from head gaze
u = ẑ                              up is up in both worlds (measured, z-up)
l = u × f                          left, by right-handedness
W = [f; l; u]  (rows)              W_app vector -> (fwd, left, up) components
M = [[0,1,0],[-1,0,0],[0,0,1]]     (fwd,left,up) -> W_robot
session_remap = M · W              the bridge, stamped at anchor time
```

Verified: with the probe's +43° session, fwd → robot −y, left → +x, up → +z.

Why yaw-only? Because z is shared (both worlds gravity-aligned) — only the
heading is unknown. If we calibrated full 3-DOF from the head pose, tilting
your head at anchor time would tilt the whole world mapping.

## 3. Deltas and anchors (the "clutch")

Nothing uses absolute poses. At anchor, each arm stores the controller pose
and its own commanded EE pose; afterwards only *changes* matter:

```
Δp = p_ctrl(t) − p_ctrl(anchor)          in W_app
Δp_robot = session_remap · Δp            in W_robot
```

For rotations the same idea, composed multiplicatively in the world frame:

```
ΔR = R_ctrl(t) · R_ctrl(anchor)⁻¹        world-frame delta in W_app
ΔR_robot = S · ΔR · Sᵀ                   S = session_remap  (change of basis)
R_target = ΔR_robot · R_EE(anchor)
```

The conjugation `S·ΔR·Sᵀ` is how a rotation *matrix* changes coordinate
systems (rotate into the frame, do the delta, rotate back). World-frame
composition (`ΔR·R₀`, not `R₀·ΔR`) makes "yaw your head" mean "yaw about the
world's vertical" regardless of how the camera link's own axes are oriented —
this is why the EE link's frame convention does not matter for control.

## 4. The POSITION pipeline for the camera arm — five stages

This is where your current symptom lives. After `Δp_robot`:

| stage | deployed value | purpose |
|---|---|---|
| 1. soft deadband | 1.5 cm | ignore the head's constant wander |
| 2. scale | × 0.6 | head cm → arm cm |
| 3. EMA filter | α = 0.3 | smooth tracker noise |
| 4. **step clamp** | **≤ 2 cm from the arm's CURRENT pose** | rate limit / anti-windup |
| 5. IK position cost | **weight 10** (vs orientation 25) | actually move the arm |

Stage 4 is subtle and is the culprit. It clamps the target to within 2 cm of
where the arm **is right now** — not of the previous target. That's a
deliberate anti-windup: if the arm is blocked, the target can never run far
ahead and cause a lunge later. But it creates a **feedback loop with stage 5**:

```
target ≤ current + 2 cm
   → position error the IK sees is at most 2 cm
   → cost = (10 · 0.02)² = 0.04   vs orientation terms (25 · error)²
   → solver barely moves the arm toward it
   → next tick, target re-clamped to 2 cm from the (unmoved) arm
   → the command never accumulates:  the x/y channel is starved
```

The gripper arms don't starve because their position weight is 50: a 2 cm
error costs (50·0.02)² = 1.0 — a loud signal. At weight 10 it's a whisper.
**My pos_mid=10 change fixed your rotation and simultaneously starved your
translation. Two individually-sensible stages, jointly broken.**

And the z overactivity is the same weakness seen from the other side: with
position held only at weight 10, the cheapest *joint-space* path to track your
pitch/yaw (weight 25) lets the EE origin swing — mostly vertically, since
shoulder/elbow/wrist are all pitch-axis joints. Plus nodding physically
translates your head (neck pivot, ~10 cm lever), which is a real z command.
So: z moves as a *byproduct of orientation tracking*, x/y commands die in the
clamp-weight loop.

## 5. The IK layer in one paragraph

The solver minimizes `Σ wᵢ‖rᵢ(q)‖²` — all soft costs, so behavior is set by
*ratios*. pos 10 / ori 25 means: 1 rad of orientation error hurts as much as
2.5 m of position error; equivalently the solver will trade ~4 cm of position
to fix 1° of orientation… it is not "inaccurate", it is doing exactly what the
weights say. Weights are the language; the clamp stages upstream decide what
error the solver is even allowed to see.

## 6. The fix

Give the camera arm a wider clamp window so the position cost sees a signal
worth acting on: `cam_max_ee_step = 0.05` (a 5 cm window ⇒ max cost signal
(10·0.05)² = 0.25, six times the starved value, while keeping anti-windup).
Verify with `GIAVA_DEBUG_HEAD=1`, which now prints each stage: the raw head
delta, the session-remapped robot delta, the net commanded offset, and what
the arm actually achieved — the starvation (command ≈ 0 while you lean) is
directly visible in those numbers, and so is its cure.
