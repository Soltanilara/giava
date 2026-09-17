"""Replay a recorded episode's ACTION stream back onto the arms.

WHAT IS BEING REPLAYED
======================
`action` holds the joint vector this rig actually COMMANDED at each tick, in
DRIVER coordinates -- what `set_joint_positions` received after IK, after both
clamps and after the collision gates.  So replay is a straight re-send: no IK,
no solver, no frames decoded.  `observation.state` (the measured joints) is
recorded alongside and is only ever used here to report tracking error.

THE MODE IS A PROPERTY OF THE DATASET, NOT OF THE COMMAND LINE
==============================================================
The action vector's layout depends on which arms were active when it was
recorded (ACTION_LAYOUTS in data_col_config.py).  Getting that wrong is not a
crash -- a three-arm recording read as `--mode right` slices indices 0..5,
which is the LEFT arm's joints, and sends them to the right arm.  So the mode
now comes from the dataset's own `meta.json`, and an explicit `--mode` that
disagrees with the recorded action width is refused rather than obeyed.

TIMING: THE EPISODE IS READ UP FRONT
=====================================
`dataset[i]` decodes that frame's video for every camera.  Doing that inside a
50 Hz send loop cannot hold the rate, so the whole episode's actions and
timestamps are pulled from the non-video columns before any motion starts and
the send loop touches nothing but numpy.  (The previous version also slept a
fixed 30 ms per iteration ON TOP of pacing to the recorded timestamps, which
alone put a 50 Hz recording at roughly two-thirds speed and printed a
lag warning on every tick.)

STARTING POSITION
=================
The arms are reset to `forward`, which is nowhere near where an arbitrary
episode begins.  interbotix validates every command as
|goal - last_command| / moving_time against the joint velocity limit and
REFUSES the whole group command if any joint fails, so jumping straight into
the recording's first action gets silently dropped and the arm sits still
while replay runs away from it.  The first action is therefore approached with
the same step-limited interpolation the startup poses use.
"""

from pathlib import Path
import argparse
import json
import os
import time

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
try:
    import rospy
except ImportError:
    rospy = None

if __package__:
    from .data_col_config import ARM_MODES, ACTION_LAYOUTS, DATASET_ROOT
    from .arm_config import ARM_CONFIG
    from .dataset import (
        BackgroundEpisodeSaver,
        quiet_libav,
        add_camera_features,
        build_action_names,
        build_state_names,
    )
    from .robot_control import (
        create_and_configure_robot,
        stop_robots,
        replay_arm_command,
        reset_arm,
        interpolate_to_pose,
    )
    from .gripper import command_gripper
else:
    from data_col_config import ARM_MODES, ACTION_LAYOUTS, DATASET_ROOT
    from arm_config import ARM_CONFIG
    from dataset import (
        BackgroundEpisodeSaver,
        quiet_libav,
        add_camera_features,
        build_action_names,
        build_state_names,
    )
    from robot_control import (
        create_and_configure_robot,
        stop_robots,
        replay_arm_command,
        reset_arm,
        interpolate_to_pose,
    )
    from gripper import command_gripper


## How close an arm must get to the episode's first commanded pose before the
## timed replay is allowed to start.  Generous: the point is to catch "did not
## move at all", not to grade the approach.
START_POSE_TOLERANCE = float(os.environ.get("GIAVA_REPLAY_START_TOL", "0.15"))

## Replay datasets go under dataset/replays/<task>/<run>_ep<N>_<timestamp>,
## a sibling of dataset/lerobot rather than a child of it -- see the comment at
## the default in main().
REPLAY_ROOT = Path(DATASET_ROOT).parent / "replays"


## ---------------------------------------------------------------------------
## RECORDING A REPLAY
##
## The point is a CONTROLLED REPEAT: the same commanded trajectory sent at the
## same rate, N times, with everything measurable recorded each time.  What that
## buys is two different numbers, and they answer different questions:
##
##   command -> measured, within one run   how well the arm follows an order
##   measured -> measured, across runs     how REPEATABLE the hardware is
##
## The second is the interesting one and it cannot be obtained from a single
## recording session.  Backlash, gravity sag, thermal drift in the servos and
## the driver's own velocity clamping all show up as run-to-run spread on an
## identical command stream.  That spread is the floor: no policy trained on
## this rig can be asked to reproduce a trajectory more precisely than the rig
## reproduces its own.
##
## The recorded columns are DELIBERATELY the same ones data collection writes --
## build_state_names / build_action_names / add_camera_features are imported
## from dataset.py rather than restated -- so a replay dataset can be loaded,
## diffed and plotted by exactly the same code as the recording it came from.
##
## ONE COLUMN IS OMITTED: observation.ee_pose.  It is FK of the commanded
## joints, i.e. derived data, and computing it here would mean importing pyroki
## and JAX and duplicating study_ik's driver->URDF calibration for a number that
## can be recomputed offline from observation.state whenever it is wanted.  It
## would also put a JAX context on a GPU that a live collection session is
## probably using.  analyze_replays.py works entirely in joint space.
class ReplayRecorder:
    """Writes replay runs into a lerobot dataset, one episode per run."""

    def __init__(self, root, mode, active_cameras, fps, provenance):
        from lerobot.datasets import LeRobotDataset

        self.mode = mode
        self.active_cameras = list(active_cameras)
        self.root = Path(root)
        self.arms = ARM_MODES[mode]

        features = {}
        add_camera_features(features, self.active_cameras)
        state_names = build_state_names(self.arms)
        action_names = build_action_names(self.arms)
        features["observation.state"] = {
            "dtype": "float32", "shape": (len(state_names),),
            "names": state_names}
        features["action"] = {
            "dtype": "float32", "shape": (len(action_names),),
            "names": action_names}
        for arm in self.arms:
            features[f"observation.timestamps.{arm}"] = {
                "dtype": "float64", "shape": (1,), "names": None}

        self.dataset = LeRobotDataset.create(
            repo_id=f"deviamar/replay_{provenance['task'] or 'unknown'}",
            root=str(self.root),
            fps=int(fps),
            features=features,
            streaming_encoding=True,
            encoder_queue_maxsize=120,
        )
        ## Same background writer the collection loop uses: a save must not
        ## stall the send loop, and each run gets its own encoder so runs
        ## cannot end up in one video.  See dataset.py.
        self.saver = BackgroundEpisodeSaver(self.dataset)
        self.task_string = f"replay of {provenance['source_root']} " \
                           f"episode {provenance['source_episode']}"

        ## PROVENANCE IS THE WHOLE VALUE OF THESE FILES.  A replay dataset that
        ## does not say what it replayed is indistinguishable from a recording,
        ## and comparing it against the wrong source is a silent wrong answer.
        (self.root / "replay_meta.json").write_text(
            json.dumps(dict(provenance,
                            mode=mode,
                            arms=self.arms,
                            active_cameras=self.active_cameras,
                            fps=int(fps),
                            state_names=state_names,
                            action_names=action_names), indent=2))
        print(f"[record] writing replay runs to {self.root}")

    def begin_run(self, run_idx):
        idx = self.saver.begin_episode()
        print(f"[record] run {run_idx} -> episode_{idx:04d}")
        return idx

    def add(self, robots, commanded, latest_frames, latest_timestamps,
            frame_lock, stamp):
        """One recorded step: measured joints + the command that produced them."""
        frame = {}

        if self.active_cameras:
            with frame_lock:
                for camera in self.active_cameras:
                    img = latest_frames.get(camera)
                    ts = latest_timestamps.get(camera)
                    if img is None or ts is None:
                        return False          # camera not up yet; skip the row
                    frame[f"observation.images.{camera}"] = torch.from_numpy(
                        img.copy())
                    frame[f"observation.timestamps.{camera}"] = torch.tensor(
                        [ts], dtype=torch.float64)

        state, action = [], []
        for arm in self.arms:
            bot = robots[arm]
            n = ARM_CONFIG[arm]["num_joints"]
            js = bot.dxl.joint_states
            state.extend(np.asarray(js.position[:n], dtype=np.float32))
            if ARM_CONFIG[arm]["has_gripper"]:
                state.append(float(js.position[6]))
            action.extend(np.asarray(commanded[f"{arm}_arm"], dtype=np.float32))
            if ARM_CONFIG[arm]["has_gripper"]:
                action.append(float(commanded.get(f"{arm}_gripper", 0.0)))
            frame[f"observation.timestamps.{arm}"] = torch.tensor(
                [stamp], dtype=torch.float64)

        frame["observation.state"] = torch.tensor(state, dtype=torch.float32)
        frame["action"] = torch.tensor(action, dtype=torch.float32)
        self.dataset.add_frame(frame, self.task_string)
        return True

    def end_run(self, ok=True):
        try:
            return self.saver.save_episode_async("success" if ok else "failure")
        except Exception as exc:
            print(f"[record] nothing to save for this run ({exc})")
            return None

    def close(self):
        self.saver.close()
        ## A replay that never got past the start-pose check leaves a valid but
        ## EMPTY dataset behind.  Say so: an empty run folder sitting next to
        ## real ones is the kind of thing that gets loaded later and quietly
        ## contributes nothing, or worse, gets counted.
        if self.saver.saved_count == 0:
            print(f"[record] no runs were recorded -- {self.root} is empty "
                  f"and safe to delete.")
        else:
            print(f"[record] {self.saver.saved_count} run(s) -> {self.root}")
            print(f"[record] compare them:  python analyze_replays.py "
                  f"{self.root} --per-joint")


def start_replay_cameras(mode, want_cameras):
    """Bring up the same cameras collection uses.  Returns (names, dicts...).

    Returns an EMPTY camera list rather than raising when the devices cannot be
    opened -- which is the expected case while a collection session has them.
    An arm-only replay dataset is still worth having; a crashed replay with the
    arms energised is not.
    """
    import threading as _th
    if not want_cameras:
        return [], {}, {}, _th.Lock(), None
    try:
        if __package__:
            from .camera_manager import (CameraConfig, get_active_cameras,
                                         setup_cameras)
        else:
            from camera_manager import (CameraConfig, get_active_cameras,
                                        setup_cameras)
    except Exception as exc:
        print(f"[record] camera_manager unavailable ({exc}) -- arms only")
        return [], {}, {}, _th.Lock(), None

    names = get_active_cameras(mode, CameraConfig(top_active=True,
                                                  low_active=True))
    latest_frames, latest_timestamps = {}, {}
    frame_lock = _th.Lock()
    shutdown = _th.Event()
    try:
        setup_cameras(names, shutdown, frame_lock, latest_frames,
                      latest_timestamps)
    except Exception as exc:
        print(f"[record] could not open cameras ({exc}).")
        print("[record] Recording arm data only -- a collection session "
              "holding the devices is the usual cause.")
        shutdown.set()
        return [], {}, {}, frame_lock, None
    print(f"[record] cameras: {names}")
    return names, latest_frames, latest_timestamps, frame_lock, shutdown


def open_local_dataset(root, video_backend="pyav"):
    """Open a dataset that lives on disk, without asking the hub about it.

    LeRobotDataset's first positional argument is a REPO ID, not a path.
    Passing the path there sends it to huggingface_hub, which rejects it as a
    malformed repo id -- so a purely local dataset could not be opened at all.
    The repo id in meta/info.json is the dataset's own record of what it would
    be called on the hub; `root` is what actually gets read.
    """
    from lerobot.datasets import LeRobotDataset

    root = Path(root)
    if not (root / "meta" / "episodes").is_dir():
        raise SystemExit(
            f"{root} has no meta/episodes -- it was never finalized, so its "
            "episode boundaries were never written and nothing can read it "
            "back.\n  Episode metadata is buffered (10 episodes) and flushed "
            "by finalize(), which runs when data collection exits with 'q' or "
            "through atexit.\n  A session killed with SIGKILL never gets "
            "there.  A session still RUNNING has not got there yet -- quit it "
            "first, then replay.")
    repo_id = "local/dataset"
    info = root / "meta" / "info.json"
    if info.is_file():
        try:
            repo_id = json.load(open(info)).get("repo_id") or repo_id
        except Exception:
            pass
    return LeRobotDataset(repo_id, root=str(root), video_backend=video_backend)


def to_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x


def action_width(mode):
    """Number of values ACTION_LAYOUTS[mode] describes."""
    return max(
        idx.stop if isinstance(idx, slice) else idx + 1
        for idx in ACTION_LAYOUTS[mode].values()
    )


def read_dataset_meta(dataset_root):
    """The meta.json save_dataset_metadata() wrote next to the episodes.

    Absent for datasets recorded before it existed, so every caller treats a
    missing file as "unknown", not as an error."""
    path = Path(dataset_root) / "meta.json"
    if not path.is_file():
        return {}
    try:
        return json.load(open(path))
    except Exception as exc:
        print(f"[meta] could not read {path}: {exc}")
        return {}


def frame_table(dataset):
    """The episode/action/timestamp columns, WITHOUT decoding any video.

    `dataset[i]` pulls every camera's frame for that index.  The columns this
    replay needs all live in the parquet, which `hf_dataset` exposes directly;
    the fallback exists only for a LeRobotDataset that does not expose it."""
    hf = getattr(dataset, "hf_dataset", None)
    if hf is None:
        print("[replay] no hf_dataset on this LeRobotDataset -- falling back to "
              "indexed access (decodes video; slower to load)")
    return hf


def episode_range(dataset, episode_idx):
    """[start, end) row indices of one episode.

    `episode_data_index` is the documented accessor; when it is absent the same
    ranges are recovered from the `episode_index` column, which every version
    writes."""
    edi = getattr(dataset, "episode_data_index", None)
    if edi is not None:
        n_eps = len(edi["from"])
        if episode_idx < 0 or episode_idx >= n_eps:
            raise ValueError(
                f"episode_idx {episode_idx} out of range (dataset has {n_eps})")
        return int(edi["from"][episode_idx]), int(edi["to"][episode_idx]), n_eps

    hf = frame_table(dataset)
    if hf is None:
        raise RuntimeError(
            "cannot determine episode boundaries: this LeRobotDataset exposes "
            "neither episode_data_index nor hf_dataset")
    eps = np.asarray(hf["episode_index"])
    present = np.unique(eps)
    if episode_idx not in present:
        raise ValueError(
            f"episode_idx {episode_idx} not in dataset (has {present.tolist()})")
    rows = np.nonzero(eps == episode_idx)[0]
    return int(rows[0]), int(rows[-1]) + 1, len(present)


def load_episode(dataset, mode, start, end):
    """(actions [N, D], timestamps [N]) for one episode, no video decoded."""
    arms = ARM_MODES[mode]
    ts_key = f"observation.timestamps.{arms[0]}"

    hf = frame_table(dataset)
    if hf is not None:
        rows = hf.select(range(start, end))
        ## lerobot may have put the table in torch format, in which case a
        ## column is a list of tensors rather than a list of lists -- go
        ## through to_numpy_1d row by row so either shape lands the same.
        actions = np.asarray([to_numpy_1d(a) for a in rows["action"]],
                             dtype=np.float64)
        if ts_key in rows.column_names:
            stamps = np.asarray([float(to_numpy_1d(t)[0]) for t in rows[ts_key]],
                                dtype=np.float64)
        else:
            # Pre-timestamp datasets, and any version that stores only
            # lerobot's own per-episode `timestamp` column.
            print(f"[replay] {ts_key} absent; pacing from lerobot's "
                  "`timestamp` column")
            stamps = np.asarray([float(np.asarray(t).reshape(-1)[0])
                                 for t in rows["timestamp"]], dtype=np.float64)
    else:
        actions, stamps = [], []
        for i in range(start, end):
            sample = dataset[i]
            actions.append(to_numpy_1d(sample["action"]))
            stamps.append(float(to_numpy_1d(sample[ts_key])[0]))
        actions = np.asarray(actions, dtype=np.float64)
        stamps = np.asarray(stamps, dtype=np.float64)

    return actions, stamps


def parse_action(action, mode):
    layout = ACTION_LAYOUTS[mode]
    return {key: action[idx] for key, idx in layout.items()}


def align_middle_waist(cmd, measured_waist):
    """Put the recorded middle-waist angle in the frame the servo booted into.

    The waist is a multiturn joint and `urdf_to_driver` recorded it relative to
    whatever 2pi-equivalent frame the driver happened to be in THAT session.
    A later session can boot a full turn away (encoder wrap; Homing_Offset is
    inert in extended-position mode), and re-sending the recorded number
    verbatim would then command a whole physical revolution and wind the
    cables.  Shift by whole turns to the equivalent nearest the current
    reading -- the physical pose is identical, the winding is not."""
    k = np.round((measured_waist - cmd[0]) / (2 * np.pi))
    if k == 0:
        return cmd, 0.0
    out = np.asarray(cmd, dtype=float).copy()
    out[0] = cmd[0] + 2 * np.pi * k
    return out, float(2 * np.pi * k)


def resolve_mode(args, meta, action_dim):
    """Decide the action layout, preferring the dataset's own record of it."""
    recorded = meta.get("mode")
    mode = args.mode or recorded

    if mode is None:
        raise SystemExit(
            "cannot tell which arms this dataset recorded: it has no meta.json "
            f"and no --mode was given. Its action width is {action_dim}; pass "
            f"the matching --mode from {sorted(ARM_MODES)}.")

    if mode not in ARM_MODES or mode not in ACTION_LAYOUTS:
        raise SystemExit(f"--mode must be one of "
                         f"{sorted(set(ARM_MODES) & set(ACTION_LAYOUTS))}, "
                         f"got '{mode}'")

    if args.mode and recorded and args.mode != recorded:
        print(f"[mode] dataset recorded mode '{recorded}', replaying as "
              f"'{args.mode}' because --mode said so.")

    ## The layout is the whole safety argument here: a wrong one sends one
    ## arm's joint angles to a different arm without any error.  The action
    ## width is the one fact on disk that can refute it, so it is checked.
    want = action_width(mode)
    if want != action_dim:
        raise SystemExit(
            f"mode '{mode}' describes a {want}-value action, but this dataset's "
            f"actions are {action_dim} values wide -- the layout does not "
            f"belong to this recording. Recorded mode: {recorded or 'unknown'}.")

    return mode


def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded episode's commanded joint trajectory.")
    parser.add_argument(
        "--mode",
        choices=sorted(set(ARM_MODES) & set(ACTION_LAYOUTS)),
        default=None,
        help="Action layout to use. Default: whatever the dataset's meta.json "
             "recorded.",
    )
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument(
        "--base-root",
        type=str,
        default=str(DATASET_ROOT),
        help="Where task folders live. Defaults to data_col_config.DATASET_ROOT, "
             "the same constant data collection writes to.",
    )
    parser.add_argument("--task", type=str, default=None, help="Task name, e.g. screwdriver_insertion")
    parser.add_argument("--run", type=str, default="latest", help="Run folder name or 'latest'")
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None,
                        help="Override the replay rate. Default: pace from the "
                             "recorded timestamps.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Load and validate the episode; move nothing.")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--verbose", action="store_true",
                        help="Print every step instead of a ~1 Hz summary.")
    parser.add_argument("--record", action="store_true",
                        help="Record each replay run as an episode of a new "
                             "lerobot dataset: the same state/action/timestamp "
                             "columns collection writes, plus every camera. "
                             "Compare the runs with analyze_replays.py.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Replay the episode this many times. Run-to-run "
                             "spread on an identical command stream is the "
                             "hardware's repeatability floor.")
    parser.add_argument("--record-root", type=str, default=None,
                        help="Where to write the replay dataset. Default: "
                             "dataset/replays/<task>/<run>_ep<N>_<timestamp>.")
    parser.add_argument("--no-cameras", action="store_true",
                        help="Record arm data only. Use this when a collection "
                             "session already holds the cameras.")
    parser.add_argument("--settle", type=float, default=1.0,
                        help="Seconds to wait between runs before re-homing.")
    parser.add_argument("--note", type=str, default=None,
                        help="Free-text note stored in replay_meta.json, e.g. "
                             "'cold servos' or 'after regrease'.")
    args = parser.parse_args()

    ## Same reason as data_collection: keep libav/x264 off the console, so the
    ## per-run [record] and [move] lines stay readable.
    quiet_libav()

    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if args.record and args.dry_run:
        raise SystemExit("--record and --dry-run are mutually exclusive: "
                         "there is nothing to record without motion.")

    if args.dataset_root is not None:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
    else:
        if args.task is None:
            raise SystemExit("Either --dataset-root OR (--task and optionally --run) must be provided")

        base_root = Path(args.base_root).expanduser().resolve()
        task_dir = base_root / args.task

        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {task_dir}")

        if args.run == "latest":
            run_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
            if not run_dirs:
                raise FileNotFoundError(f"No runs found under task directory: {task_dir}")
            # Run folders are named YYYYmmdd_HHMMSS, so lexical order is
            # chronological order.
            dataset_root = sorted(run_dirs)[-1]
        else:
            dataset_root = task_dir / args.run

        dataset_root = dataset_root.resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    print(f"Using dataset_root={dataset_root}")

    meta = read_dataset_meta(dataset_root)
    if meta:
        print(f"[meta] task={meta.get('task')} mode={meta.get('mode')} "
              f"arms={meta.get('active_arms')} fps={meta.get('fps')} "
              f"action_dim={meta.get('action_dim')} recorded {meta.get('created')}")
    else:
        print("[meta] no meta.json in this dataset -- mode must come from --mode")

    dataset = open_local_dataset(dataset_root)

    start, end, n_eps = episode_range(dataset, args.episode_idx)
    start = min(start + args.start_offset, end)
    if end - start <= 0:
        raise SystemExit("No steps to replay (check --start-offset)")

    ## Probe the width from row 0 so the mode can be validated BEFORE any arm
    ## is created, let alone energised.
    hf = frame_table(dataset)
    if hf is not None:
        action_dim = len(np.asarray(hf.select(range(start, start + 1))["action"])[0])
    else:
        action_dim = len(to_numpy_1d(dataset[start]["action"]))

    mode = resolve_mode(args, meta, action_dim)
    arm_names = ARM_MODES[mode]
    print(f"[mode] {mode} -> arms {arm_names}, action width {action_dim}")

    actions, stamps = load_episode(dataset, mode, start, end)

    total_steps = len(actions)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
        actions, stamps = actions[:total_steps], stamps[:total_steps]

    ## Recorded timestamps are time.monotonic() values, so only DIFFERENCES
    ## mean anything -- and a monotonic clock does not survive a reboot, which
    ## is exactly when a stale absolute value would look plausible.  Rebase on
    ## the first sample and sanity-check the spacing.
    rel = stamps - stamps[0]
    if total_steps > 1:
        dts = np.diff(rel)
        bad = (dts <= 0) | (dts > 1.0)
        if bad.any():
            print(f"[timing] {int(bad.sum())} of {len(dts)} recorded intervals "
                  "are non-monotonic or over 1 s -- pacing from the dataset fps "
                  "instead")
            rel = np.arange(total_steps) / float(dataset.fps)
        else:
            print(f"[timing] recorded interval: median {np.median(dts) * 1e3:.1f} ms "
                  f"(~{1.0 / max(np.median(dts), 1e-6):.1f} Hz), "
                  f"max {dts.max() * 1e3:.1f} ms")
    if args.fps is not None:
        rel = np.arange(total_steps) / float(args.fps)
        print(f"[timing] --fps {args.fps} overrides the recorded pacing")

    parsed_first = parse_action(actions[0], mode)
    print(f"episode {args.episode_idx} of {n_eps}: rows [{start}, {end}), "
          f"replaying {total_steps} steps, "
          f"{rel[-1] if total_steps else 0:.1f} s")
    for arm in arm_names:
        key = f"{arm}_arm"
        if key in parsed_first:
            print(f"  first {arm} command: {np.round(parsed_first[key], 3)}")

    ## Every command is length-checked here, once, rather than per step in the
    ## timed loop where a bad row would only be discovered mid-motion.
    for arm in arm_names:
        n = ARM_CONFIG[arm]["num_joints"]
        got = len(parsed_first[f"{arm}_arm"])
        if got != n:
            raise SystemExit(
                f"{arm}: layout yields {got} joints, arm has {n}")
    if not np.isfinite(actions).all():
        n_bad = int((~np.isfinite(actions)).any(axis=1).sum())
        raise SystemExit(
            f"{n_bad} of {total_steps} recorded actions contain NaN/inf -- "
            "refusing to replay a trajectory with holes in it")

    if args.dry_run:
        print("Dry run: episode loads, layout matches, no NaNs. "
              "Exiting before robot motion.")
        return

    ## Checked HERE rather than at the top of main(): everything above this
    ## line is a dataset question, and --dry-run answering it off the robot
    ## (no ROS, no arms) is the point of having the flag.
    if rospy is None:
        raise ImportError("rospy is required to replay episodes on hardware "
                          "(--dry-run works without it).")

    rospy.init_node("replay_episode", anonymous=True)

    robots = {arm_name: create_and_configure_robot(arm_name)
              for arm_name in arm_names}

    stop_requested = False

    def request_stop():
        nonlocal stop_requested
        stop_requested = True
        stop_robots(robots)

    rospy.on_shutdown(request_stop)

    ## Waist frame alignment, decided once from the pose measured after reset
    ## and applied to every middle command, so the whole trajectory stays in
    ## one frame instead of each command being re-judged independently.
    waist_turn_shift = 0.0

    ## Cameras come up BEFORE the arms move so the first run is not recorded
    ## with half the streams missing.  An empty list here means "arms only";
    ## start_replay_cameras() prefers that to failing while the arms are live.
    rec_cameras, latest_frames, latest_timestamps, frame_lock, cam_shutdown = (
        start_replay_cameras(mode, args.record and not args.no_cameras))

    recorder = None
    if args.record:
        if args.record_root:
            record_root = Path(args.record_root).expanduser().resolve()
        else:
            ## Replays live in their own tree beside the recordings, NOT inside
            ## the run they replay.  A replay is not an episode of that dataset
            ## -- it has different columns, a different provenance and no
            ## teleop -- and nesting it under dataset_root puts it in the path
            ## of anything that globs for runs, which is how a replay ends up
            ## in a training set.  The name carries what it replayed so the
            ## folder is identifiable without opening replay_meta.json.
            record_root = (REPLAY_ROOT / (meta.get("task") or "unknown")
                           / f"{dataset_root.name}_ep{args.episode_idx:04d}"
                           f"_{time.strftime('%Y%m%d_%H%M%S')}")
        recorder = ReplayRecorder(
            record_root, mode, rec_cameras,
            fps=(args.fps or dataset.fps),
            provenance={
                "source_root": str(dataset_root),
                "source_episode": int(args.episode_idx),
                "source_rows": [int(start), int(end)],
                "task": meta.get("task"),
                "steps": int(total_steps),
                "runs_requested": int(args.repeat),
                "note": args.note,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

    def run_pass(run_idx):
        """One full replay of the episode.  Returns (steps_done, per-arm error)."""
        waist_turn_shift = 0.0

        for arm_name, bot in robots.items():
            reset_arm(bot, arm_name)
            n = ARM_CONFIG[arm_name]["num_joints"]
            measured_q = np.asarray(bot.dxl.joint_states.position[:n], dtype=float)
            print(f"Measured {arm_name} joints after reset:", np.round(measured_q, 4))

            first_cmd = np.asarray(parsed_first[f"{arm_name}_arm"], dtype=float)
            if arm_name == "middle":
                first_cmd, waist_turn_shift = align_middle_waist(
                    first_cmd, measured_q[0])
                if waist_turn_shift:
                    print(f"[frame] middle waist recorded at "
                          f"{parsed_first['middle_arm'][0]:+.3f}; replaying at "
                          f"{first_cmd[0]:+.3f} (nearest 2pi-equivalent to the "
                          f"measured {measured_q[0]:+.3f})")

            ## Walk to the episode's first commanded pose at the same
            ## step-limited rate the startup poses use.  Sending it in one
            ## jump would fail the driver's velocity check, which rejects the
            ## whole group command rather than clipping it.
            gap = float(np.max(np.abs(first_cmd - measured_q)))
            print(f"[start] {arm_name}: {gap:.3f} rad from reset pose to the "
                  "episode's first command; interpolating.")
            interpolate_to_pose(bot, arm_name, first_cmd)

            ## interpolate_to_pose RETURNS WITHOUT MOVING when its wrapped-
            ## encoder guard fires (a multiturn joint reading outside the
            ## driver's own limits), and it says so on stdout rather than
            ## raising.  Streaming a whole episode at an arm that never left
            ## its reset pose is exactly the situation that guard exists to
            ## prevent, so confirm the arm is actually where the trajectory
            ## starts before sending anything at it.
            rospy.sleep(0.1)
            landed = np.asarray(bot.dxl.joint_states.position[:n], dtype=float)
            residual = float(np.max(np.abs(first_cmd - landed)))
            if residual > START_POSE_TOLERANCE:
                raise SystemExit(
                    f"{arm_name}: still {residual:.3f} rad from the episode's "
                    f"first command after interpolating (tolerance "
                    f"{START_POSE_TOLERANCE:.3f}). Look above for a [SAFETY] "
                    "or driver-limit message -- replaying from here would "
                    "stream the whole trajectory at an arm standing "
                    "somewhere else.")
            print(f"[start] {arm_name}: in position "
                  f"({residual:.3f} rad residual).")

        if recorder is not None:
            recorder.begin_run(run_idx)

        print(f"Starting replay of {total_steps} steps"
              f"{f' (run {run_idx + 1}/{args.repeat})' if args.repeat > 1 else ''}"
              ". Press Ctrl+C to stop.")

        t0_wall = time.monotonic()
        lag_max = 0.0
        lagging = 0
        err_sum = {arm: 0.0 for arm in arm_names}
        err_max = {arm: 0.0 for arm in arm_names}
        err_n = 0
        recorded = 0
        next_report = t0_wall + 1.0
        i = -1

        for i in range(total_steps):
            if stop_requested or rospy.is_shutdown():
                print(f"\nStopped at step {i} of {total_steps}.")
                break

            sleep_time = (t0_wall + rel[i]) - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                lag = -sleep_time
                if lag > 0.01:
                    lagging += 1
                    lag_max = max(lag_max, lag)

            parsed = parse_action(actions[i], mode)
            ## The waist shift is part of the command that was actually SENT,
            ## so fold it in once here -- the recorded action column and the
            ## reporting below then both describe the same numbers the servos
            ## received, not the ones on disk in the source episode.
            if waist_turn_shift and "middle_arm" in parsed:
                parsed = dict(parsed)
                parsed["middle_arm"] = np.asarray(
                    parsed["middle_arm"], dtype=float).copy()
                parsed["middle_arm"][0] += waist_turn_shift

            for arm_name, bot in robots.items():
                cmd = np.asarray(parsed[f"{arm_name}_arm"], dtype=float)
                replay_arm_command(bot, cmd)

                gripper_key = f"{arm_name}_gripper"
                if gripper_key in parsed:
                    command_gripper(bot, float(parsed[gripper_key]))

            ## Recorded IMMEDIATELY after the send, so the measured joints in a
            ## row are the arm's response to the command in the row before it --
            ## the same one-tick-stale relationship data collection records, so
            ## the two datasets can be compared without a phase correction.
            if recorder is not None:
                if recorder.add(robots, parsed, latest_frames,
                                latest_timestamps, frame_lock,
                                time.monotonic()):
                    recorded += 1

            ## Reporting reads the joint states the driver publishes
            ## asynchronously -- no extra wait, so it costs the loop nothing
            ## and the error it shows is one tick stale by construction.
            now = time.monotonic()
            if args.verbose or now >= next_report or i == total_steps - 1:
                for arm_name, bot in robots.items():
                    n = ARM_CONFIG[arm_name]["num_joints"]
                    measured_q = np.asarray(
                        bot.dxl.joint_states.position[:n], dtype=float)
                    cmd = np.asarray(parsed[f"{arm_name}_arm"], dtype=float)
                    err = float(np.linalg.norm(measured_q - cmd))
                    err_sum[arm_name] += err
                    err_max[arm_name] = max(err_max[arm_name], err)
                    if args.verbose:
                        print(f"{arm_name} target={np.round(cmd, 3)}")
                        print(f"{arm_name} measured={np.round(measured_q, 3)}")
                    print(f"[{i + 1}/{total_steps}] {arm_name} "
                          f"tracking_error={err:.4f}")
                err_n += 1
                next_report = now + 1.0

        elapsed = time.monotonic() - t0_wall
        done = i + 1
        print(f"\nReplayed {done}/{total_steps} steps in {elapsed:.1f} s "
              f"(recorded {rel[total_steps - 1]:.1f} s)")
        if lagging:
            print(f"[timing] {lagging} steps ran late, worst "
                  f"{lag_max * 1e3:.0f} ms behind the recording")
        if err_n:
            for arm_name in arm_names:
                print(f"[tracking] {arm_name}: mean "
                      f"{err_sum[arm_name] / err_n:.4f} rad, max "
                      f"{err_max[arm_name]:.4f} rad (sampled {err_n}x)")

        if recorder is not None:
            print(f"[record] run {run_idx}: {recorded} frames recorded")
            recorder.end_run(ok=(done == total_steps and not stop_requested))
        return done

    try:
        for run_idx in range(args.repeat):
            if args.repeat > 1:
                print("\n" + "=" * 60)
                print(f"RUN {run_idx + 1} of {args.repeat}")
                print("=" * 60)
            run_pass(run_idx)
            if stop_requested or rospy.is_shutdown():
                break
            if run_idx + 1 < args.repeat:
                ## Let the arms settle before the next reset: a run started
                ## while the previous one is still coasting begins from a
                ## different pose, which is the one thing a repeatability
                ## measurement must not vary.
                time.sleep(args.settle)

    except KeyboardInterrupt:
        request_stop()
    finally:
        stop_robots(robots)
        if recorder is not None:
            recorder.close()
        if cam_shutdown is not None:
            cam_shutdown.set()


if __name__ == "__main__":
    main()
