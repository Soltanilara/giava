"""Measure the latencies baked into a recorded GIAVA dataset.

The control loop records, per tick: the commanded joints (action), the
measured joints (observation.state), a per-camera capture timestamp and a
per-arm observation timestamp.  Those four streams contain every latency that
matters for policy training, and this script extracts them OFFLINE -- no
robot, no cameras, just a recorded run directory:

  1. TICK TIMING      how regular was the control loop actually?  (dt mean /
                      p95 / max from the recorded arm timestamps, overrun
                      fraction)
  2. CAMERA SYNC      residual spread between the frames chosen for each tick,
                      each camera's stable phase offset, and -- when the arm
                      timestamps are wall-clock (post 2026-08 fix) -- image
                      STALENESS: how old each camera's pixels were at the
                      moment the observation was assembled.  Staleness is the
                      number inference must reproduce.
  3. ACTUATION LAG    cross-correlation of d(action)/dt against d(state)/dt
                      per joint: how many ticks the arm trails its command.
                      This is the mechanical+driver latency a policy's actions
                      will experience at deployment.
  4. VISUAL LAG       (--visual CAM, slow: decodes video) cross-correlation of
                      commanded joint speed against pixel motion energy in the
                      tick-aligned frames: end-to-end command -> photons ->
                      recorded-frame delay, as the policy will see it.

Usage (from data_collection_scripts/, in the gym_av312 env):

  python measure_latency.py --root dataset/lerobot/<task>/<run>            # newest episode
  python measure_latency.py --root dataset/lerobot/<task>/<run> --episode 3
  python measure_latency.py --root ... --episode 3 --visual left_wrist
  python measure_latency.py --root ... --all --json latency_report.json

Interpretation guide (printed numbers):
  - tick dt p95 should sit within ~10% of 1/fps; a fat tail means the loop
    stalls (IK spikes, camera polls) and the dataset's nominal fps lies.
  - camera offsets are the free-running cameras' phase; they are STABLE on
    this rig (see camera_manager.py) -- a drifting offset means a camera
    dropped to a lower rate mid-episode.
  - actuation lag is normally a small constant (moving_time + bus latency).
    A lag that differs strongly between joints, or between episodes, points
    at driver clamping -- check meta/robustness.jsonl for clamp counters.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_dataset(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    meta_json = root / "meta.json"
    task = "unknown"
    if meta_json.exists():
        with open(meta_json) as f:
            task = json.load(f).get("task", "unknown")
    return LeRobotDataset(repo_id=f"deviamar/{task}", root=str(root))


def _episode_slice(ds, episode):
    idx = ds.episode_data_index
    ep_from = int(idx["from"][episode])
    ep_to = int(idx["to"][episode])
    return ep_from, ep_to


## READ THE PARQUET DIRECTLY, never through ds.hf_dataset.
##
## The recorded timestamps are float64 for a specific reason (dataset.py):
## they are absolute epoch values around 1.79e9, and float32 resolution there
## is 128 SECONDS.  lerobot's hf_dataset view hands columns back as float32 --
## fine for training, fatal here, because every timestamp in a 15 s episode
## then rounds into a single bucket.  The symptom is not an error: it is a
## report of dt = 0.000 ms, spread = 0.000 ms and staleness = 0.000 ms, i.e.
## a PERFECT-looking rig.  Measured 2026-09-03 on a dataset whose real rate
## was 48.7 Hz.  A validation tool that fails by printing perfection is worse
## than one that crashes, so the parquet is read straight.
_PARQUET_CACHE = {}


def _episode_table(root, episode):
    """All columns for one episode, as a pyarrow Table (float64 preserved)."""
    import pyarrow.parquet as pq

    key = str(root)
    if key not in _PARQUET_CACHE:
        files = sorted(Path(root).glob("data/**/*.parquet"))
        if not files:
            raise SystemExit(f"no parquet files under {root}/data")
        tables = [pq.read_table(f) for f in files]
        if len(tables) == 1:
            _PARQUET_CACHE[key] = tables[0]
        else:
            import pyarrow as pa

            _PARQUET_CACHE[key] = pa.concat_tables(tables)
    table = _PARQUET_CACHE[key]
    ep = np.asarray(table["episode_index"])
    mask = ep == episode
    if not mask.any():
        raise SystemExit(f"episode {episode} not present in {root}")
    return table, mask


def _columns(root, episode, names):
    """Per-episode float64 columns straight out of the parquet."""
    table, mask = _episode_table(root, episode)
    out = {}
    for name in names:
        if name not in table.column_names:
            continue
        col = table[name].to_pylist()
        arr = np.asarray(
            [c[0] if isinstance(c, (list, tuple)) and len(c) == 1 else c
             for c in col], dtype=np.float64)
        out[name] = arr[mask] if arr.ndim == 1 else arr[mask, :]
    return out


def _xcorr_lag(a, b, max_lag):
    """Lag (in samples) at which b best follows a, with parabolic refinement.

    Both inputs are derivative-like signals; correlation is normalized per
    lag so amplitude differences do not bias the argmax.  Returns (lag,
    peak_correlation) or (None, 0.0) when there is not enough signal.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9 or len(a) < 4 * max_lag:
        return None, 0.0
    corrs = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        x = a[: len(a) - k] if k else a
        y = b[k:]
        xs, ys = x.std(), y.std()
        if xs < 1e-12 or ys < 1e-12:
            continue
        corrs[k] = float(np.mean((x - x.mean()) * (y - y.mean())) / (xs * ys))
    k = int(np.argmax(corrs))
    lag = float(k)
    ## Parabolic sub-sample refinement on the peak's neighbours.
    if 0 < k < max_lag:
        y0, y1, y2 = corrs[k - 1], corrs[k], corrs[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            lag = k + 0.5 * (y0 - y2) / denom
    return lag, float(corrs[k])


def analyze_timing(cols, arm_keys, fps):
    """Tick regularity from the recorded per-arm observation timestamps."""
    ts = cols[arm_keys[0]]
    dt = np.diff(ts)
    nominal = 1.0 / fps
    report = {
        "n_ticks": int(len(ts)),
        "nominal_dt_ms": 1000 * nominal,
        "dt_mean_ms": float(1000 * dt.mean()),
        "dt_std_ms": float(1000 * dt.std()),
        "dt_p95_ms": float(1000 * np.percentile(dt, 95)),
        "dt_max_ms": float(1000 * dt.max()),
        "late_tick_fraction": float(np.mean(dt > 1.5 * nominal)),
    }
    ## Wall-clock timestamps can only be told from monotonic ones by their
    ## magnitude: epoch seconds are ~1.7e9, boot-relative ones are small.
    report["clock_domain"] = "epoch" if ts[0] > 1e9 else "monotonic (pre-fix dataset)"
    return report


def analyze_camera_sync(cols, cam_keys, arm_keys):
    """Per-tick spread, per-camera phase offset, and (if possible) staleness."""
    cams = {k: cols[k] for k in cam_keys}
    stack = np.stack(list(cams.values()))          # (n_cams, n_ticks)
    ref = np.median(stack, axis=0)
    spread = stack.max(axis=0) - stack.min(axis=0)
    report = {
        "spread_ms": {
            "mean": float(1000 * spread.mean()),
            "p95": float(1000 * np.percentile(spread, 95)),
            "max": float(1000 * spread.max()),
        },
        "per_camera_offset_ms": {},
        "staleness_ms": {},
    }
    for key, ts in cams.items():
        off = ts - ref
        report["per_camera_offset_ms"][key] = {
            "mean": float(1000 * off.mean()),
            "std": float(1000 * off.std()),
        }
    arm_ts = cols[arm_keys[0]]
    ## Staleness needs both clocks in one domain (see data_collection.py's
    ## obs_ts comment).  A pre-fix dataset mixes epoch and monotonic; detect
    ## and say so rather than printing a nonsense number.
    if abs(float(np.median(arm_ts - ref))) < 10.0:
        for key, ts in cams.items():
            age = arm_ts - ts
            report["staleness_ms"][key] = {
                "mean": float(1000 * age.mean()),
                "p95": float(1000 * np.percentile(age, 95)),
                "max": float(1000 * age.max()),
            }
    else:
        report["staleness_ms"] = (
            "unavailable: arm timestamps are monotonic-clock (recorded before "
            "the epoch-domain fix), camera timestamps are epoch")
    return report


def analyze_actuation_lag(cols, joint_names, fps, max_lag_ticks):
    """Per-joint command->measurement lag by derivative cross-correlation."""
    action = np.atleast_2d(cols["action"])
    state = np.atleast_2d(cols["observation.state"])
    report = {"per_joint": {}}
    lags = []
    for j, name in enumerate(joint_names):
        cmd = np.diff(action[:, j])
        meas = np.diff(state[:, j])
        lag, peak = _xcorr_lag(cmd, meas, max_lag_ticks)
        if lag is None or peak < 0.2:
            report["per_joint"][name] = {"lag_ms": None, "peak_corr": peak,
                                         "note": "not enough motion"}
            continue
        err = action[:, j] - state[:, j]
        entry = {
            "lag_ticks": round(lag, 2),
            "lag_ms": round(1000 * lag / fps, 1),
            "peak_corr": round(peak, 3),
            "tracking_rmse": float(np.sqrt(np.mean(err ** 2))),
        }
        report["per_joint"][name] = entry
        if "gripper" not in name:
            lags.append(1000 * lag / fps)
    if lags:
        report["arm_joints_median_lag_ms"] = float(np.median(lags))
        report["arm_joints_max_lag_ms"] = float(np.max(lags))
    return report


def analyze_visual_lag(ds, ep_from, ep_to, camera_key, cols, fps,
                       max_lag_ticks, max_frames=1200):
    """Command speed vs pixel-motion energy in the tick-aligned frames.

    Decodes video, so it is opt-in and windowed: at most `max_frames`
    contiguous ticks starting where commanded motion first exceeds its
    median (so the window is not the operator standing still).
    """
    action = np.atleast_2d(cols["action"])
    cmd_speed = np.linalg.norm(np.diff(action, axis=0), axis=1)
    ## Pick the busiest contiguous window.
    n = min(max_frames, ep_to - ep_from - 1)
    if len(cmd_speed) > n:
        smooth = np.convolve(cmd_speed, np.ones(n), mode="valid")
        start = int(np.argmax(smooth))
    else:
        start = 0
        n = len(cmd_speed)
    full_key = f"observation.images.{camera_key}"
    prev = None
    energy = np.zeros(n)
    for i in range(n):
        item = ds[ep_from + start + i]
        img = item[full_key]
        arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
        arr = arr.astype(np.float32)
        if prev is not None:
            energy[i] = float(np.mean(np.abs(arr - prev)))
        prev = arr
        if i and i % 200 == 0:
            print(f"  ... decoded {i}/{n} frames", file=sys.stderr)
    lag, peak = _xcorr_lag(cmd_speed[start:start + n][1:], energy[1:],
                           max_lag_ticks)
    if lag is None or peak < 0.15:
        return {"camera": camera_key, "lag_ms": None, "peak_corr": peak,
                "note": "correlation too weak -- try a window with more motion"}
    return {
        "camera": camera_key,
        "window_ticks": n,
        "lag_ticks": round(lag, 2),
        "lag_ms": round(1000 * lag / fps, 1),
        "peak_corr": round(peak, 3),
    }


def _fmt(d, indent=2):
    return json.dumps(d, indent=indent, default=str)


def analyze_episode(ds, root, episode, visual_camera=None):
    fps = ds.meta.fps
    ep_from, ep_to = _episode_slice(ds, episode)
    features = ds.meta.features

    ## Timestamp schema has evolved: current datasets carry one column per
    ## camera plus one per arm; mid-2026 datasets carried a single
    ## observation.timestamps.robot; older ones none at all.  Analyze whatever
    ## is there and SAY what is missing rather than crashing on it.
    image_names = {k.removeprefix("observation.images.")
                   for k in features if k.startswith("observation.images.")}
    ts_keys = [k for k in features if k.startswith("observation.timestamps.")]
    cam_keys = sorted(k for k in ts_keys
                      if k.removeprefix("observation.timestamps.") in image_names)
    arm_keys = sorted(k for k in ts_keys if k not in cam_keys)
    joint_names = features["action"]["names"]

    names = cam_keys + arm_keys + ["action", "observation.state"]
    cols = _columns(root, episode, names)

    max_lag = max(4, int(0.5 * fps))   # search up to 500 ms of lag

    report = {
        "episode": episode,
        "frames": ep_to - ep_from,
        "fps": fps,
        "tick_timing": (analyze_timing(cols, arm_keys, fps) if arm_keys else
                        "unavailable: no per-arm timestamps in this dataset"),
        "camera_sync": (analyze_camera_sync(cols, cam_keys, arm_keys)
                        if len(cam_keys) > 1 and arm_keys else
                        "unavailable: no per-camera timestamps in this dataset"),
        "actuation_lag": analyze_actuation_lag(cols, joint_names, fps, max_lag),
    }
    if visual_camera:
        print(f"[visual] decoding {visual_camera} frames "
              f"(slow, one-time)...", file=sys.stderr)
        report["visual_lag"] = analyze_visual_lag(
            ds, ep_from, ep_to, visual_camera, cols, fps, max_lag)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True,
                    help="dataset run directory (the one holding meta.json)")
    ap.add_argument("--episode", type=int, default=None,
                    help="episode index (default: last)")
    ap.add_argument("--all", action="store_true",
                    help="analyze every episode (no --visual)")
    ap.add_argument("--visual", metavar="CAMERA", default=None,
                    help="also measure end-to-end visual lag for this camera "
                         "(e.g. left_wrist); decodes video, slow")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="write the full report as JSON here")
    args = ap.parse_args()

    ds = _load_dataset(args.root)
    n_eps = ds.meta.total_episodes
    print(f"dataset: {args.root}  ({n_eps} episodes @ {ds.meta.fps} fps)")

    if args.all:
        episodes = list(range(n_eps))
    else:
        episodes = [args.episode if args.episode is not None else n_eps - 1]

    reports = []
    for ep in episodes:
        rep = analyze_episode(ds, args.root, ep, visual_camera=None if args.all else args.visual)
        reports.append(rep)
        print(f"\n===== episode {ep} ({rep['frames']} frames) =====")
        print(_fmt({k: rep[k] for k in
                    ("tick_timing", "camera_sync", "actuation_lag")}))
        if "visual_lag" in rep:
            print("visual_lag:", _fmt(rep["visual_lag"]))

    if len(reports) > 1:
        med = [r["actuation_lag"].get("arm_joints_median_lag_ms") for r in reports]
        med = [m for m in med if m is not None]
        if med:
            print(f"\nacross {len(med)} episodes: actuation lag median "
                  f"{np.median(med):.1f} ms  (min {min(med):.1f} / max {max(med):.1f})")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(reports, f, indent=2, default=str)
        print(f"\nreport -> {args.json}")


if __name__ == "__main__":
    main()
