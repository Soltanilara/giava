"""Two-camera Viser viewer with paired capture  (PHASE 8).

Live view of two cameras side by side, and a CAPTURE BOTH CAMERAS button
that snapshots both, shows their timestamps / frame IDs / delta t, and
saves the pair to disk.

    JAX_PLATFORMS=cpu python calibration/viser_cameras.py \\
        --cameras left_wrist top_scene [--port 8093]

Then open http://localhost:8093 (forward the port if you are working over
SSH or in a remote VS Code session).

Port 8093 continues the repo's block: 8082 teleop_debug_tool, 8090
view_trajectories, 8091 view_collision, 8092 ik playground.

The 3D scene is intentionally almost empty -- just a world grid with the
frame convention labelled.  Placing the cameras and the robot in it needs
calibrated extrinsics, which do not exist yet (see camera_mount.py); this
viewer exists to collect the data that will produce them.

Capture semantics: each camera runs its own thread keeping its newest
frame, and the button snapshots both under their locks back to back.  The
two frames were therefore exposed at slightly different times -- the
displayed delta t is that difference, on the clocks named next to it.  It
is NOT evidence of hardware synchronisation; see sync_capture.py.
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
    DIR_VISER,
    ensure_dirs,
    provenance,
    save_json,
    timestamp,
)
from rs_camera import PRODUCTION_COLOR, require_rs, resolve  # noqa: E402
from sync_capture import CameraWorker, _check_delivering  # noqa: E402


def fmt_ms(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", nargs=2, required=True,
                    metavar=("CAM1", "CAM2"),
                    help="the two camera names or serials to show")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--width", type=int, default=PRODUCTION_COLOR[0])
    ap.add_argument("--height", type=int, default=PRODUCTION_COLOR[1])
    ap.add_argument("--fps", type=int, default=PRODUCTION_COLOR[2])
    ap.add_argument("--outdir", default=None,
                    help="where captures are saved (default: a timestamped "
                         "session dir under calibration/data/viser_captures/)")
    ap.add_argument("--preview-hz", type=float, default=15.0,
                    help="live preview refresh rate (default: 15)")
    args = ap.parse_args()

    import viser

    require_rs()
    ensure_dirs()
    resolved = resolve(args.cameras)
    names = [n for n, _ in resolved]

    outdir = (Path(args.outdir) if args.outdir
              else DIR_VISER / f"{'_'.join(names)}_{timestamp()}")
    (outdir / "images").mkdir(parents=True, exist_ok=True)

    print()
    print("#" * 74)
    print("#  Two-camera Viser viewer  (PHASE 8)")
    print("#" * 74)
    print(f"\n  cameras: {', '.join(f'{n} ({s})' for n, s in resolved)}")
    print(f"  saving to {outdir}\n")

    workers = [CameraWorker(n, args.width, args.height, args.fps,
                            global_time=True, keep_images=True)
               for n in names]
    for w in workers:
        w.start()
        print(f"  started {w.cam.name}")
    time.sleep(1.2)
    _check_delivering(workers)

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)

    captures: List[Dict[str, Any]] = []
    blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)

    with server.gui.add_folder("Live view"):
        images = {w.cam.name: server.gui.add_image(
            blank, label=f"{w.cam.name}  ({w.cam.serial})", format="jpeg",
            jpeg_quality=70) for w in workers}
        live_status = server.gui.add_text("status", "starting...",
                                          disabled=True)

    with server.gui.add_folder("Capture"):
        capture_btn = server.gui.add_button("CAPTURE BOTH CAMERAS")
        n_captures = server.gui.add_number("captures saved", 0, disabled=True)
        dt_box = server.gui.add_text("delta t (ms)", "--", disabled=True)
        detail = server.gui.add_markdown("_no capture yet_")

    with server.gui.add_folder("Frame convention", expand_by_default=False):
        server.gui.add_markdown(f"""
World frame is `base`, the root link of `giava.urdf`:
**+x** operator's left, **+y** operator's backward, **+z** up.

The grid is that world frame's z=0 plane. The cameras and robot are **not**
drawn: doing so needs `T_world_camera = T_world_ee @ T_ee_camera`, and no
calibrated `T_ee_camera` exists yet. Run
`python calibration/camera_mount.py` to see what is and is not known.

Delta t below is the difference between the two frames' timestamps on the
clock named with it. These cameras free-run with no hardware sync, so it
measures software clocks only -- see `sync_capture.py`.
""")

    @capture_btn.on_click
    def _(_) -> None:
        entry = _do_capture(workers, outdir, len(captures))
        if entry is None:
            dt_box.value = "capture failed"
            return
        captures.append(entry)
        n_captures.value = len(captures)
        dt_box.value = fmt_ms(entry["delta_t_ms"]["timestamp_ms"])
        detail.content = _markdown(entry)
        _save_manifest(captures, outdir, names, resolved, args)
        print(f"  captured #{entry['index']}  "
              f"dt={fmt_ms(entry['delta_t_ms']['timestamp_ms'])} ms")

    print(f"\n  Two-camera viewer:  http://localhost:{args.port}\n",
          flush=True)

    period = 1.0 / max(args.preview_hz, 1.0)
    try:
        while True:
            for w in workers:
                rec = w.snapshot()
                if rec is not None and rec.image is not None:
                    images[w.cam.name].image = rec.image
            live_status.value = "  ".join(
                f"{w.cam.name}: {w.count} frames" for w in workers)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n  shutting down")
    finally:
        for w in workers:
            w.stop()
        if captures:
            print(f"  {len(captures)} capture(s) in {outdir}")


def _do_capture(workers: List[CameraWorker], outdir: Path,
                idx: int) -> Optional[Dict[str, Any]]:
    """Snapshot both cameras back to back and write the pair to disk."""
    import cv2

    t0 = time.perf_counter()
    epoch = time.time()
    snaps = [(w, w.snapshot()) for w in workers]
    window_ms = (time.perf_counter() - t0) * 1e3

    if any(rec is None for _, rec in snaps):
        return None

    frames: Dict[str, Any] = {}
    for w, rec in snaps:
        d = rec.timing_dict()
        fn = f"images/capture{idx:04d}_{w.cam.name}.png"
        cv2.imwrite(str(outdir / fn),
                    cv2.cvtColor(rec.image, cv2.COLOR_RGB2BGR))
        d["image_path"] = fn
        frames[w.cam.name] = d

    a, b = workers[0].cam.name, workers[1].cam.name
    ## Delta on every clock that both cameras reported, so the viewer does
    ## not silently pick one and imply it is authoritative.
    deltas: Dict[str, Optional[float]] = {}
    for field, scale in (("timestamp_ms", 1.0),
                         ("sensor_timestamp_us", 1e-3),
                         ("frame_timestamp_us", 1e-3),
                         ("backend_timestamp_ms", 1.0),
                         ("time_of_arrival_ms", 1.0),
                         ("host_epoch_s", 1e3)):
        va, vb = frames[a].get(field), frames[b].get(field)
        deltas[field] = ((va - vb) * scale
                         if va is not None and vb is not None else None)

    return {
        "index": idx,
        "capture_epoch_s": epoch,
        "sampling_window_ms": window_ms,
        "delta_definition": f"{a} minus {b}",
        "delta_t_ms": deltas,
        "frames": frames,
    }


def _markdown(entry: Dict[str, Any]) -> str:
    lines = [f"**capture #{entry['index']}** &mdash; "
             f"`delta = {entry['delta_definition']}`", ""]
    for cam, f in entry["frames"].items():
        lines += [
            f"**{cam}**",
            f"- frame id: `{f.get('frame_number')}`",
            f"- timestamp: `{fmt_ms(f.get('timestamp_ms'))}` ms "
            f"(`{f.get('timestamp_domain')}`)",
            f"- arrival (host): `{fmt_ms(f.get('time_of_arrival_ms'))}` ms",
            f"- image: `{f.get('image_path')}`",
            "",
        ]
    lines += ["**delta t by clock (ms)**", ""]
    for k, v in entry["delta_t_ms"].items():
        lines.append(f"- `{k}`: {fmt_ms(v)}")
    lines += ["",
              f"_snapshot window: {entry['sampling_window_ms']:.3f} ms_",
              "",
              "_Software clocks only. These cameras free-run; this is not "
              "hardware synchronisation._"]
    return "\n".join(lines)


def _save_manifest(captures, outdir: Path, names, resolved, args) -> None:
    save_json({
        "metadata": provenance(
            "viser_paired_capture",
            cameras=[{"name": n, "serial": s} for n, s in resolved],
            width=args.width, height=args.height, fps=args.fps),
        "hardware_sync": {
            "external_sync_wired": False,
            "statement": ("Cameras free-run. Delta t values are differences "
                          "between software clocks, not shutter alignment."),
        },
        "delta_definition": f"{names[0]} minus {names[1]}",
        "n_captures": len(captures),
        "captures": captures,
    }, outdir / "captures.json", overwrite=True)


if __name__ == "__main__":
    main()
