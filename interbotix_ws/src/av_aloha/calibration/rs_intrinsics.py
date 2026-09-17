"""RealSense factory intrinsics dump  (PHASE 4).

Queries the on-device factory calibration through librealsense and saves it
as JSON, one file per camera, with enough metadata to say exactly which
device and which stream each number belongs to.

These values come from Intel's factory calibration stored in the camera's
own memory.  They are a reference to compare against, not ground truth:
factory calibration ages, and a camera that has been knocked or thermally
cycled can drift from it.  Comparing them with an independent ChArUco
calibration (phases 5 and 6) is the point of the exercise.

Examples

    # every connected camera, at the resolution data collection uses
    python calibration/rs_intrinsics.py --all

    # one camera, colour + depth, plus every profile the device offers
    python calibration/rs_intrinsics.py --cameras right_wrist \\
        --streams color depth --all-profiles

    # just list what is plugged in
    python calibration/rs_intrinsics.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    DIR_CAMERAS,
    ensure_dirs,
    provenance,
    save_json,
    timestamp,
)
from rs_camera import (  # noqa: E402
    PRODUCTION_COLOR,
    RealSenseCamera,
    connected_devices,
    extrinsics_dict,
    intrinsics_dict,
    name_for,
    require_rs,
    resolve,
    rs,
)
from camera_manager import CAMERA_SERIALS  # noqa: E402


def list_devices() -> None:
    devs = connected_devices()
    print()
    print(f"  {len(devs)} RealSense device(s) connected:")
    print()
    print(f"    {'serial':16s} {'giava name':14s} {'model':28s} "
          f"{'firmware':12s} {'usb':6s}")
    print("    " + "-" * 82)
    for d in devs:
        print(f"    {d['serial']:16s} {name_for(d['serial']) or '(unmapped)':14s} "
              f"{d['name']:28s} {d['firmware']:12s} {d['usb_type']:6s}")
    print()
    unmapped = [d["serial"] for d in devs if name_for(d["serial"]) is None]
    if unmapped:
        print(f"    NOTE: {unmapped} are not in camera_manager.CAMERA_SERIALS,")
        print("    so they have no GIAVA name. Address them by serial.")
        print()
    missing = {n: s for n, s in CAMERA_SERIALS.items()
               if s not in {d["serial"] for d in devs}}
    if missing:
        print(f"    NOT CONNECTED: {missing}")
        print()


def all_profiles(serial: str) -> List[Dict[str, Any]]:
    """Intrinsics for every video profile the device offers.

    Read straight off the sensors -- no pipeline start, so this does not
    disturb anything and covers resolutions we never stream."""
    out = []
    for dev in rs.context().query_devices():
        if dev.get_info(rs.camera_info.serial_number) != serial:
            continue
        for sensor in dev.query_sensors():
            sname = sensor.get_info(rs.camera_info.name)
            for p in sensor.get_stream_profiles():
                if not p.is_video_stream_profile():
                    continue
                vp = p.as_video_stream_profile()
                try:
                    intr = vp.get_intrinsics()
                except Exception:
                    continue
                out.append({
                    "sensor": sname,
                    "stream": str(vp.stream_type()),
                    "stream_index": int(vp.stream_index()),
                    "format": str(vp.format()),
                    "fps": int(vp.fps()),
                    "intrinsics": intrinsics_dict(intr),
                })
    return out


def dump_camera(camera: str, streams: List[str], width: int, height: int,
                fps: int, want_all_profiles: bool,
                global_time: bool | None) -> Dict[str, Any]:
    """Open the camera, record the active streams' factory intrinsics."""
    doc: Dict[str, Any] = {
        "metadata": provenance(
            "realsense_factory_intrinsics",
            source="librealsense on-device factory calibration",
            note=("Factory values read from the camera's own memory. They "
                  "are a reference, NOT a measurement made in this lab, and "
                  "they can drift from the physical optics over time."),
        ),
        "streams": {},
        "extrinsics": {},
    }

    for stream in streams:
        with RealSenseCamera(camera, width=width, height=height, fps=fps,
                             stream=stream, global_time=global_time,
                             warmup_frames=0) as cam:
            meta = cam.stream_metadata()
            doc["streams"][stream] = {
                "identity": meta,
                "intrinsics": cam.intrinsics,
            }
            ## Serial/device info is per camera, not per stream; lift the
            ## first one so the file is identifiable at a glance.
            doc.setdefault("camera", meta["camera"])
            doc.setdefault("serial", meta["serial"])
            doc.setdefault("device", meta["device"])

            if stream == "depth":
                try:
                    ds = cam.profile.get_device().first_depth_sensor()
                    doc["streams"][stream]["depth_scale_m_per_unit"] = float(
                        ds.get_depth_scale())
                except Exception:
                    pass

    ## Depth->colour extrinsic is factory data too and is needed the moment
    ## anyone aligns the two streams, so record it while we are here.
    if "color" in streams and "depth" in streams:
        try:
            doc["extrinsics"]["depth_to_color"] = _stream_extrinsics(
                camera, width, height, fps)
        except Exception as e:
            doc["extrinsics"]["depth_to_color"] = {"error": str(e)}

    if want_all_profiles:
        doc["all_available_profiles"] = all_profiles(doc["serial"])

    return doc


def _stream_extrinsics(camera: str, width: int, height: int,
                       fps: int) -> Dict[str, Any]:
    from rs_camera import serial_for
    cfg = rs.config()
    cfg.enable_device(serial_for(camera))
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    try:
        c = prof.get_stream(rs.stream.color)
        d = prof.get_stream(rs.stream.depth)
        return extrinsics_dict(d.get_extrinsics_to(c), "depth", "color")
    finally:
        pipe.stop()


def print_summary(camera: str, doc: Dict[str, Any]) -> None:
    print()
    print("=" * 74)
    print(f"  {camera}   serial {doc.get('serial')}   "
          f"{doc.get('device', {}).get('name', '')}")
    print("=" * 74)
    for stream, s in doc["streams"].items():
        i = s["intrinsics"]
        print(f"\n  stream '{stream}'  {i['width']}x{i['height']} "
              f"@ {s['identity']['fps']} fps  format {s['identity']['format']}")
        print(f"      fx = {i['fx']:12.5f}     fy = {i['fy']:12.5f}")
        print(f"      cx = {i['cx']:12.5f}     cy = {i['cy']:12.5f}"
              "        (= ppx / ppy)")
        print(f"      distortion model : {i['model']}")
        print(f"      coeffs           : "
              + ", ".join(f"{c:+.6f}" for c in i["coeffs"]))
        print(f"      fov (h, v) deg   : "
              f"{i['fov_deg'][0]:.2f}, {i['fov_deg'][1]:.2f}")
        if "depth_scale_m_per_unit" in s:
            print(f"      depth scale      : "
                  f"{s['depth_scale_m_per_unit']} m per unit")
    if doc.get("all_available_profiles"):
        print(f"\n  + {len(doc['all_available_profiles'])} additional stream "
              f"profiles recorded in the JSON")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="camera names (camera_manager.CAMERA_SERIALS) or "
                         "raw serial numbers")
    ap.add_argument("--all", action="store_true",
                    help="every connected RealSense")
    ap.add_argument("--list", action="store_true",
                    help="list connected devices and exit")
    ap.add_argument("--streams", nargs="+", default=["color"],
                    choices=("color", "depth", "infrared"),
                    help="streams to query (default: color)")
    ap.add_argument("--width", type=int, default=PRODUCTION_COLOR[0])
    ap.add_argument("--height", type=int, default=PRODUCTION_COLOR[1])
    ap.add_argument("--fps", type=int, default=PRODUCTION_COLOR[2])
    ap.add_argument("--all-profiles", action="store_true",
                    help="also record intrinsics for every stream profile "
                         "the device offers, at every resolution")
    ap.add_argument("--global-time", dest="global_time", action="store_true",
                    default=None,
                    help="enable librealsense global time before reading")
    ap.add_argument("--no-global-time", dest="global_time",
                    action="store_false",
                    help="disable it (frame timestamps stay on the device "
                         "clock)")
    ap.add_argument("--outdir", default=str(DIR_CAMERAS),
                    help="output directory")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing file of the same name")
    args = ap.parse_args()

    require_rs()
    ensure_dirs()

    print()
    print("#" * 74)
    print("#  RealSense factory intrinsics  (PHASE 4)")
    print("#" * 74)

    if args.list:
        list_devices()
        return

    if args.all:
        cams = [name_for(d["serial"]) or d["serial"]
                for d in connected_devices()]
    elif args.cameras:
        cams = args.cameras
    else:
        ap.error("give --cameras, or --all, or --list")

    resolved = resolve(cams)
    list_devices()

    stamp = timestamp()
    outdir = Path(args.outdir)
    written = []
    for name, serial in resolved:
        doc = dump_camera(name, args.streams, args.width, args.height,
                          args.fps, args.all_profiles, args.global_time)
        print_summary(name, doc)
        path = outdir / f"{name}_{serial}" / f"factory_intrinsics_{stamp}.json"
        save_json(doc, path, overwrite=args.overwrite)
        written.append(path)
        print(f"\n  -> {path}")

    print()
    print("-" * 74)
    print(f"  {len(written)} calibration file(s) written under {outdir}")
    print("  Compare against a ChArUco calibration with:")
    print("      python calibration/compare_intrinsics.py --camera <name>")
    print("-" * 74)
    print()


if __name__ == "__main__":
    main()
