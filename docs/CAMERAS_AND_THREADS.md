# Cameras, CV, and threads

The part of the system that is concurrent, and therefore the part where
failures look like something else.  This is the thread inventory, how frames
actually get from a sensor into an episode, and why quitting with `q` is not
the same as Ctrl-C.

---

## 1. The thread inventory

Eleven threads get spawned across the runtime.  Every one is a **daemon**,
which means the interpreter will not wait for them at exit — that fact is the
whole reason §4 exists.

| thread | spawned in | what it does | blocks on |
|---|---|---|---|
| **main** | — | the 50 Hz control loop | its own sleep-to-budget |
| `cam-<name>` ×N | `camera_manager.py:875` | one per RealSense D405 | `rs_pipeline.wait_for_frames()` |
| `oak-worker` | `camera_manager.py:983` | the OAK stereo pair + headset stream | `MessageQueue.get()` ×2 |
| `episode-saver` | `dataset.py:326` | writes/encodes finished episodes | `queue.Queue.get()` |
| headset | `gvlink_headset.py:369` | gvlink protocol I/O | socket |
| webrtc loop | `webrtc_headset.py:416` | asyncio event loop (older transport) | asyncio |
| keyboard | `data_collection.py:646` (also `teleop.py:573`, `rollout_policy.py:1861`) | reads operator keys | `stdin` |
| profile streamer | `robot_control.py:598` | streams a joint plan during named moves | its own pacing |
| policy driver | `rollout_policy.py:794` | runs the policy off the critical path | its own event |

Shared state, and the discipline around it:

```python
frame_lock      = threading.Lock()    # guards latest_frames / latest_timestamps / FRAME_HISTORY
camera_shutdown = threading.Event()   # "stop" — every camera worker checks it at loop top
FRAME_HISTORY   = {cam: deque(maxlen=8)}   # (timestamp_s, frame, frame_number)
CAMERA_WORKERS  = []                  # every spawned worker, so shutdown can join them
```

The design rule: **the control loop never waits on a device.**  Cameras
produce into shared dicts on their own schedule; the loop takes a snapshot.
Episodes are handed to a writer thread and the loop returns at once.  That is
why steps 10–11 of the tick are cheap.

---

## 2. How a frame gets from a sensor into an episode

### RealSense (`camera_worker`)

```python
while not camera_shutdown.is_set():
    frames = rs_pipeline.wait_for_frames()      # BLOCKS
    color  = np.asanyarray(color_frame.get_data()).copy()   # ← COPY, not view
    color_ts = color_frame.get_timestamp() * 1e-3           # host epoch
    with frame_lock:
        latest_frames[name]     = color
        latest_timestamps[name] = color_ts
        _record_history(name, color_ts, color, frame_number)
```

Two details that are not optional:

**The `.copy()` is load-bearing.**  `np.asanyarray(frame.get_data())` is a
*view* over librealsense's buffer, which is recycled the moment the frame is
released on the next iteration.  The history keeps several frames alive, so
aliasing would leave older entries pointing at memory since overwritten with
newer images — a bug that produces *plausible* wrong frames, which is the worst
kind.  One copy here is also the only copy: `latest_frames` points at the same
array.

**`global_time` puts every device on the host epoch.**  `get_timestamp()`
returns whatever domain `get_frame_timestamp_domain()` reports; with
`global_time` enabled (set in `setup_realsense_cameras`) that is host epoch
milliseconds, which is what makes it comparable with the OAK worker's
`time.time()` and across devices.

### OAK (`oak_worker`)

Same shape, plus three things:

1. **Left/right swap is applied here and nowhere else.**  `oak_camera_pipeline`
   binds CAM_B→`cam_left` and CAM_C→`cam_right`, which is the DepthAI
   convention for a factory OAK-D and an *assumption* on a hand-assembled rig.
   On this one it was wrong: measured 2026-08-27, the board sat 102 px further
   left in the frame called "left" across 25 of 25 pairs, and the installed
   calibration carried `T[0] = +62.63 mm` where a correctly ordered pair must
   be negative.  The right lens's image was going to the left eye — inverted
   depth, for months, **invisible to every calibration metric**, because a swap
   leaves rectified rows perfectly aligned.  Fixed at the single point where
   the two frames are *named*, so capture files, rectify maps, dataset streams
   and headset eyes agree by construction instead of needing four consistent
   edits.
2. **Self-healing.**  On 5 consecutive queue failures it stops, waits 3 s, and
   rebuilds the entire pipeline (`setup_oak_stereo()`).  X_LINK crashes and
   unplugs happen; a wedged worker that spams is worse than one that retries.
3. **Two views of the same frame.**  The dataset gets a crop-and-downscale to
   the historical 640×480 (centre-crop native 1280×800 to 4:3, then resize) —
   raw, unrectified.  The **headset** gets the rectified stream.  Dataset
   frames stay raw on purpose: policy training learns whatever lens the data
   was shot through, while geometry work needs undistortion.

Both OAK frames share one host timestamp taken *after* both queues returned.
The pair is self-consistent; its offset to the RealSense clock includes this
worker's scheduling, and that is stated rather than hidden.

---

## 3. Frame synchronisation — the honest version

The cameras free-run.  These D405s **do not support `inter_cam_sync_mode`**
(verified on all four at firmware 5.12.14.100 — only `output_trigger_enabled`
is exposed), so there is no hardware genlock and alignment must be software.

The historical behaviour was **latest-frame-wins**: each tick took whatever
frame each camera happened to have most recently produced.  How stale that was
depended on when the tick polled relative to each camera's phase, so skew was
bounded only by a full frame period (16.7 ms at 60 fps) and moved around tick
to tick.

`select_synchronized_frames()` instead keeps a short history per camera and
picks, for each, the frame nearest a **common reference instant**:

```python
reference = min(newest_timestamp_per_camera)   # the newest instant EVERY camera covers
for c in cameras:
    ts, frame, fn = min(history[c], key=lambda e: abs(e[0] - reference))
```

Using `min` not `max` is the point: `max` would ask slower cameras for a frame
they have not produced, and they would silently return their newest anyway.
Each chosen frame is then within **half a frame period** of the reference
(≤ 8.3 ms at 60 fps), bounded and independent of polling phase.

**What this cannot do**, stated in the code because it matters: it cannot
reduce the spread *between* cameras below their relative phase offset.
Measured on this rig — three cameras at a true 60.2 fps, dense histories — the
per-camera offsets were −6.16 / 0.00 / +5.41 ms: each well inside the
half-period bound, but an **11.6 ms spread overall**.  That spread is physical.
Frames do not exist at the same instants and no selection rule invents them.

What makes it useful anyway: the offsets are **stable** (std 0.027 ms over 150
samples), so they are a calibratable constant rather than noise — provided the
per-camera timestamp is recorded faithfully, which is why the dataset stores
them as float64.  `spread_s` is reported per timestep into
`robustness.jsonl:cam_sync_spread_s`, so alignment quality is *checked*, not
assumed.

`GIAVA_SYNC_FRAMES=0` restores latest-frame-wins.

---

## 4. Why `q` and not Ctrl-C — the shutdown order

This is the single most important operational rule in the system, and it has
two independent reasons.

### Reason one: the dataset

`episode_saver.close()` → `dataset.finalize()`, which flushes buffered episode
metadata and writes the parquet footers.  **Without it the dataset on disk
cannot be loaded back.**

### Reason two: the cameras

Stopping a librealsense pipeline or destroying the OAK device **while its
worker is still inside `wait_for_frames()` / `queue.get()` / a cv2 call** tears
the native library down under a live thread.  The observed result is glibc's
`FATAL: exception not rethrown` and a core dump: the forced-unwind exception
raised in the cancelled thread gets swallowed by a `catch(...)` inside OpenCV,
which glibc treats as fatal.  Harmless to the data (it happens after finalize)
but it masks real crashes and leaves cameras wedged for the next run.

### The order, and why each step is where it is

```
safe_shutdown():
  1. episode_saver.wait_until_idle()     ← idle FIRST. While the writer is
                                           draining, has_pending_frames() would
                                           see that episode and queue it TWICE.
  2. save_episode_async("unknown")       ← save whatever was in progress
  3. episode_saver.close()               ← finalize(): the data is now safe
  4. reset_arms(quit_pose)  /  hold      ← park the arms
  5. preview.close()                     ← BEFORE parking finishes: a window
                                           still showing live arms being stowed
                                           reads as "the session is running"
  6. shutdown_cameras(pipelines):
       a. camera_shutdown.set()          ← the flag
       b. _unblock_cameras(pipelines)    ← THE FLAG ALONE IS NOT ENOUGH
       c. join_camera_workers(timeout=2) ← now wait
       d. pipe.stop() / dev.wait()       ← only now destroy the devices
```

**Step 6b is the subtle one.**  `oak_worker` sits in a blocking
`MessageQueue.get()`; it only re-checks `camera_shutdown` at the *top* of its
loop, so setting the event does not wake it.  Closing the queues makes the
pending `get()` return, and stopping the pipeline shuts the device's own
pipeline down while its consumer is still alive to notice — the graceful order
depthai wants.

`join_camera_workers()` returns the stragglers still alive after 2 s.  That is
**information, not an error**: a worker blocked on a camera that has stopped
delivering frames cannot be interrupted from here, and stopping the pipeline
below is what will free it.  So it says so, rather than appearing to hang.

### Ctrl-C and tracebacks are covered too

`shutdown_cameras` is split out of `safe_shutdown` and registered with
`atexit`, so it runs on Ctrl-C and on a traceback as well.  It is idempotent
(`_CAMERAS_DOWN["done"]`), so the `q` path calls it directly and the atexit
hook then does nothing.  `atexit` is LIFO, and `episode_saver.close` is
registered *after*, so the data is finalised before the cameras come down.

One historical bug worth remembering: `setup_cameras()` returns
`{"oak": (device, q_left, q_right), "realsense": {name: pipeline}}` — neither
value has `.stop()`, so an earlier `getattr(p, "stop")` loop walked straight
past **both** and no camera was ever shut down.  `dai.Pipeline` has no
`close()` either, which is why the OAK only died in the process destructor and
printed `[depthai] Device ... has crashed` on every exit.  The current code
reaches into the two shapes it actually returns.

---

## 5. The episode writer thread

`dataset.py::EpisodeSaver` exists so recording a 60-second episode does not
mean waiting 20 seconds to start the next one.

```
control loop:  save_episode_async(outcome)
                 ├─ detach the buffer AND install a fresh one in one go
                 └─ queue.put(...)   → returns immediately
writer thread: encode video, write parquet, append episode_outcomes.jsonl
```

**Detach-and-replace is atomic with respect to the loop** — the worker owns
`buf` and `enc` from that moment and the control loop must never touch either
again.  Overlapped saving needs a *streaming encoder* to swap; without one it
falls back to the serialized path and says so rather than guessing.

Backpressure: `begin_episode()` blocks only if the writer has fallen
`MAX_PENDING` episodes behind, and prints why.  A disk that cannot keep up
should slow you down visibly, not fill RAM silently.

`task_override` relabels **every frame** of an episode before it is detached —
the last chance to fix a wrong target, since the task string is stamped per
frame while recording, and a mislabelled episode teaches the policy to fetch
the wrong object on command.  Applied here rather than at write time so the
buffer, the parquet and `episode_outcomes.jsonl` cannot disagree.

---

## 6. The CV that is actually in here

Not much is learned; most of it is classical and geometric.

| where | what |
|---|---|
| `camera_manager.py` | intrinsics capture (**with** distortion coefficients — the old path stored only `[fx, fy, ppx, ppy]`, so those datasets can never be undistorted), stereo rectification maps, crop/resize, digital zoom |
| `oak_stereo_calibrate.py`, `stereo_calib_live.py` | checkerboard stereo calibration for the OAK pair |
| `calibration/charuco_*.py` | ChArUco intrinsics + extrinsics for the scene cameras |
| `calibration/handeye.py` | hand-eye: where the camera is relative to the robot |
| `reconstruction/stereo.py`, `pointcloud.py` | triangulation, ray-plane intersection, dense tabletop point cloud, table/object segmentation |
| `scene_features.py` | object detection → centroid, for the object-centric env-state datasets |
| `build_mask_dataset.py` | object-mask camera stream |
| `place_grid.py`, `scene_snapshots.py` | placement lattice and ghost-alignment overlays |

### Two traps documented in-tree

**The distortion-convention trap** (`reconstruction/README.md`).  OpenCV's
`plumb_bob` runs its polynomial **ray → pixel** (closed form distorts, iterated
undistorts).  librealsense's `inverse_brown_conrady` runs it **pixel → ray**
(closed form *un*distorts).  Mixing them is silent: the numbers stay plausible
and are simply wrong by a few pixels at the edge.

**The epipolar line is a curve.**  With a 78° D405 and real distortion,
treating it as a line is several pixels of error near the border — measured at
> 0.5 px of sagitta in the selftest, much more at the corners.  So it is never
approximated: the ray is sampled in 3D and every sample projected through the
destination camera's full distortion model.  Each sample carries the depth that
produced it, so the curve is also a **depth scale you can read off the image**.

**The baseline check.**  A stereo `.npz` records image size, intrinsics and
extrinsics — but nothing identifying the *rig* it was shot on.  Re-casing the
cameras leaves a file that still loads, still matches on resolution, and is
silently wrong.  Wrong extrinsics do not blur the image; they break epipolar
alignment, leaving residual **vertical** disparity between the eyes — the one
stereo error an operator cannot fuse, which reads as eye strain rather than as
a picture fault.  So the measured baseline (62 mm, taken with a ruler
2026-08-26) is checked against the calibration at load.  It is the one number
both stored in the file and measurable by hand in ten seconds.

### Depth, briefly

The D405 depth residual is **99 % repeatable structure, not noise** — do the
two-capture test before blaming the sensor.

---

## 7. Console noise, and why it is suppressed the way it is

Three separate filters exist, and the principle behind all three is the same:
*"nothing was printed" and "nothing happened" are different statements.*

- **`camera_manager._info()`** — six cameras printed ~15 lines of
  normal-operation detail every run (device ids, USB link speed, which
  calibration loaded, four sets of intrinsics), sitting directly above the
  `[SAFETY]` and `[profile]` lines that need reading.  Routine lines go through
  `_info()`; anything describing a *failure*, a missing device, a fallback that
  changes behaviour, or a degraded link stays on a plain `print()`.  The
  intrinsics are not lost — `save_camera_intrinsics()` writes them into the
  dataset, which is where they are needed.  `GIAVA_CAM_VERBOSE=1` restores.
- **`robot_control._quiet_sdk()`** — the Interbotix constructors print six
  unconditional lines per arm plus two `rospy.logerr`s that are *expected on
  this rig* (velocity profile on purpose; gripper set to
  `current_based_position` immediately after).  Eighteen lines per start.
  Captured, replaced with one line per arm, and **reprinted in full if
  construction raises** — the noise is only noise when it works.
  `GIAVA_QUIET_SDK=0` restores.
- **`robot_control.quiet_solver_logs()`** — pyroki logs through loguru at INFO
  in colour to stderr.  Suppressed, not silenced: WARNING and above still print
  (reformatted), and everything below is **counted** so
  `solver_log_summary()` can say how much was hidden.
  `GIAVA_QUIET_SOLVER=0` restores.
