"""Are the cameras' frames fresh, and does synchronization make them stale?

    python probe_frame_age.py                 # right-arm cameras + top/low
    python probe_frame_age.py --seconds 8

Opens the cameras exactly the way rollout_policy.py does (camera_manager's
worker threads), waits for them to settle, then samples for a few seconds and
reports, per camera: how old its NEWEST frame is versus wall-clock, and how
old the frame select_synchronized_frames() would hand the policy is.  A
camera whose newest frame is consistently far behind the others drags the
synchronized set back to its lagging instant -- the policy then acts on old
images with a deceptively small "spread".

No arm is touched.  Cannot run while anything else holds the cameras.
"""
from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="right")
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    from camera_manager import (CameraConfig, get_active_cameras, setup_cameras,
                                select_synchronized_frames, FRAME_HISTORY,
                                join_camera_workers)
    cams = get_active_cameras(args.mode, CameraConfig(top_active=True, low_active=True))
    shutdown, lock = threading.Event(), threading.Lock()
    latest, stamps = {c: None for c in cams}, {c: None for c in cams}
    pipes = setup_cameras(cams, shutdown, lock, latest, stamps)
    print(f"[probe] cameras: {cams}; settling 3 s ...")
    time.sleep(3.0)

    newest_age = {c: [] for c in cams}
    chosen_age = {c: [] for c in cams}
    ref_age, spread = [], []
    t_end = time.time() + args.seconds
    n = 0
    while time.time() < t_end:
        with lock:
            now = time.time()
            for c in cams:
                h = FRAME_HISTORY.get(c)
                if h:
                    newest_age[c].append((now - h[-1][0]) * 1e3)
            _f, ts, info = select_synchronized_frames(cams)
            if info.get("reference_s") is not None:
                ref_age.append((now - info["reference_s"]) * 1e3)
                spread.append((info.get("spread_s") or 0) * 1e3)
                for c, t in ts.items():
                    chosen_age[c].append((now - t) * 1e3)
        n += 1
        time.sleep(0.02)

    def q(v):
        return (f"median {statistics.median(v):6.1f}  p95 {sorted(v)[int(0.95 * (len(v) - 1))]:6.1f}"
                if v else "   (no data)")
    print(f"\n[probe] {n} samples over {args.seconds:.0f} s   (all values in ms behind wall-clock)")
    print(f"{'camera':14s} {'NEWEST frame age':>34s}   {'frame the POLICY gets (synced)':>34s}")
    for c in cams:
        print(f"{c:14s} {q(newest_age[c]):>34s}   {q(chosen_age[c]):>34s}")
    print(f"\nsynchronized reference age: {q(ref_age)}      spread: {q(spread)}")
    print("\nreading: a healthy camera's newest frame is ~1 frame old (17-33 ms).  If one camera's\n"
          "newest is far larger, it is lagging, and the synced column will show EVERY camera\n"
          "dragged to that age -- that is the policy acting on old images.")

    shutdown.set()
    join_camera_workers(timeout=2.0)
    for name, p in (pipes or {}).items():
        if isinstance(p, dict):
            for pipe in p.values():
                try:
                    pipe.stop()
                except Exception:
                    pass
        elif isinstance(p, (tuple, list)) and p:
            for qq in p[1:]:
                try:
                    qq.close()
                except Exception:
                    pass
            try:
                p[0].stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
