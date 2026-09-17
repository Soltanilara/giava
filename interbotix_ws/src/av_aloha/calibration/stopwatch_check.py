"""End-to-end synchronisation check against a physical stopwatch.

Every other timing figure in this package is a claim the software makes
about itself.  This is the one test where the answer is written in the
scene: point all the cameras at a running stopwatch (a phone works), grab
one aligned set of frames, and tile them into a single image.  Read the
digits.  If the frames were captured at the same instant, they show the
same time.

That closes the loop the software cannot close on its own -- it validates
the clocks, the alignment, AND the assumption that a timestamp has anything
to do with when the photons landed.

WHAT TO EXPECT
    These cameras free-run; the D405 has no inter_cam_sync_mode, so there
    is no genlock.  With nearest-timestamp alignment the residual spread
    measured on this rig is ~1.7 ms.  A stopwatch showing hundredths (10 ms
    per digit step) therefore should USUALLY read the same on every camera,
    and occasionally differ by one step when a tick lands mid-exposure.
    Reading two or more steps apart, repeatably, means something is wrong.

    Use a stopwatch with MILLISECOND digits if you can. Hundredths can only
    prove you are within ~10 ms; milliseconds actually resolve the ~2 ms
    you are trying to see.

    Exposure time matters too: at 1/60 s the digits blur across ~16 ms of
    change regardless of how well aligned the cameras are. Bright light
    (short exposure) makes the digits crisp and the test meaningful.

Run

    python calibration/stopwatch_check.py --cameras left_wrist top_scene low_scene
    python calibration/stopwatch_check.py --all --shots 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    DIR_SYNC,
    wait_for_enter,
    ensure_dirs,
    provenance,
    save_json,
    timestamp,
)
from rs_camera import (  # noqa: E402
    PRODUCTION_COLOR,
    connected_devices,
    name_for,
    require_rs,
    resolve,
)
from sync_capture import CameraWorker, _check_delivering  # noqa: E402


def aligned_pick(workers: List[CameraWorker]) -> Optional[Dict[str, Any]]:
    """Nearest-timestamp selection across the workers' current frames.

    Same rule as camera_manager.select_synchronized_frames, so the montage
    validates the alignment production actually uses: the reference is the
    newest instant EVERY camera has already covered, and each camera
    contributes the buffered frame nearest it."""
    newest = [w.newest_timestamp_s() for w in workers]
    if any(t is None for t in newest):
        return None
    # The newest instant EVERY camera has already covered.
    ref = min(newest)
    snaps = [(w, w.nearest(ref)) for w in workers]
    if any(r is None for _, r in snaps):
        return None
    ts = {w.cam.name: r.timestamp_ms * 1e-3 for w, r in snaps}
    return {
        "reference_s": ref,
        "spread_s": max(ts.values()) - min(ts.values()),
        "frames": {w.cam.name: r for w, r in snaps},
        "timestamps_s": ts,
    }


def montage(pick: Dict[str, Any], order: List[str], cols: int = 2):
    """Tile the frames into one labelled image, one panel per camera."""
    import cv2

    ref = pick["reference_s"]
    panels = []
    for name in order:
        rec = pick["frames"][name]
        img = cv2.cvtColor(rec.image, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bar = np.zeros((58, w, 3), dtype=np.uint8)
        dt_ms = (pick["timestamps_s"][name] - ref) * 1e3
        cv2.putText(bar, f"{name}  [{rec.serial}]", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bar, f"frame {rec.frame_number}   "
                         f"dt {dt_ms:+.2f} ms from reference", (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        panels.append(np.vstack([bar, img]))

    rows = []
    for i in range(0, len(panels), cols):
        row = panels[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)

    foot = np.zeros((44, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(foot, f"spread {pick['spread_s']*1e3:.2f} ms   "
                      f"reference epoch {ref:.6f}   "
                      f"software alignment only - cameras free-run, no genlock",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return np.vstack([grid, foot])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="camera names or serials to include")
    ap.add_argument("--all", action="store_true",
                    help="every connected RealSense")
    ap.add_argument("--shots", type=int, default=5,
                    help="how many aligned sets to capture (default 5)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between shots")
    ap.add_argument("--cols", type=int, default=2,
                    help="montage columns")
    ap.add_argument("--width", type=int, default=PRODUCTION_COLOR[0])
    ap.add_argument("--height", type=int, default=PRODUCTION_COLOR[1])
    ap.add_argument("--fps", type=int, default=PRODUCTION_COLOR[2])
    ap.add_argument("--no-prompt", action="store_true",
                    help="skip the 'press ENTER' setup pause (for scripted "
                         "runs; conda run cannot answer prompts at all)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import cv2

    require_rs()
    ensure_dirs()

    if args.all:
        cams = [name_for(d["serial"]) or d["serial"]
                for d in connected_devices()]
    elif args.cameras:
        cams = args.cameras
    else:
        ap.error("give --cameras or --all")

    resolved = resolve(cams)
    names = [n for n, _ in resolved]
    outdir = (Path(args.outdir) if args.outdir
              else DIR_SYNC / f"stopwatch_{timestamp()}")
    outdir.mkdir(parents=True, exist_ok=True)

    print()
    print("#" * 74)
    print("#  Stopwatch synchronisation check")
    print("#" * 74)
    print(f"""
  cameras : {', '.join(names)}
  shots   : {args.shots}, one every {args.interval:.1f} s
  output  : {outdir}

  SET UP:
    1. Start a stopwatch showing at least hundredths -- milliseconds if you
       can get them. A phone works.
    2. Aim every camera at it. It must be readable in all of them at once,
       so a big display and a bright room help (bright light also shortens
       the exposure, which stops the digits blurring).
    3. Let it run and press ENTER.

  Then open the montage images and compare the digits panel to panel.
""")
    if not args.no_prompt:
        wait_for_enter("press ENTER to begin capturing... ")

    workers = [CameraWorker(n, args.width, args.height, args.fps,
                            global_time=True, keep_images=True)
               for n in names]
    shots = []
    try:
        for w in workers:
            w.start()
        time.sleep(1.5)
        _check_delivering(workers)
        print()
        for i in range(args.shots):
            pick = aligned_pick(workers)
            if pick is None:
                print(f"  shot {i}: a camera had no frame -- skipped")
                continue
            img = montage(pick, names, args.cols)
            fn = f"stopwatch_{i:03d}.png"
            cv2.imwrite(str(outdir / fn), img)
            print(f"  shot {i}: spread {pick['spread_s']*1e3:6.2f} ms   "
                  f"frames "
                  + " ".join(f"{n}={pick['frames'][n].frame_number}"
                             for n in names)
                  + f"   -> {fn}")
            shots.append({
                "index": i, "image": fn,
                "reference_s": pick["reference_s"],
                "spread_s": pick["spread_s"],
                "per_camera": {
                    n: {"timestamp_s": pick["timestamps_s"][n],
                        "frame_number": pick["frames"][n].frame_number,
                        "serial": pick["frames"][n].serial,
                        "offset_from_reference_s":
                            pick["timestamps_s"][n] - pick["reference_s"],
                        "stopwatch_reading": None}
                    for n in names},
            })
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        for w in workers:
            w.stop()

    if shots:
        sp = np.array([s["spread_s"] for s in shots]) * 1e3
        print(f"""
  software spread across {len(shots)} shots:
      mean {sp.mean():.2f} ms   max {sp.max():.2f} ms

  NOW READ THE IMAGES.  Enter what each camera's panel shows into
  'stopwatch_reading' in the JSON. The software says the frames are
  {sp.mean():.2f} ms apart; the digits are the independent check on whether
  that is true.
""")

    save_json({
        "metadata": provenance("stopwatch_sync_check",
                               cameras=[{"name": n, "serial": s}
                                        for n, s in resolved],
                               width=args.width, height=args.height,
                               fps=args.fps),
        "hardware_sync": {
            "external_sync_wired": False,
            "inter_cam_sync_mode_supported": False,
            "statement": ("D405 does not expose inter_cam_sync_mode; these "
                          "cameras free-run and alignment is software only."),
        },
        "how_to_complete": (
            "For each shot, open the montage image, read the stopwatch in "
            "each camera's panel, and record it in stopwatch_reading. "
            "Disagreement between panels is the REAL inter-camera offset; "
            "compare it against spread_s, which is what the software "
            "believed."),
        "n_shots": len(shots),
        "shots": shots,
    }, outdir / "stopwatch_readings.json", overwrite=args.overwrite)
    print(f"  readings template: {outdir / 'stopwatch_readings.json'}\n")


if __name__ == "__main__":
    main()
