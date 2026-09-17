"""Clear wedged RealSense devices with a hardware reset.

Symptom this fixes: the pipeline starts, every ioctl succeeds, and no
frames ever arrive.  Nothing is holding the device -- `fuser /dev/video*`
comes back empty -- it is simply stuck, usually because a previous process
died without closing its pipeline.

A device-level reset clears it without root and without unplugging
anything.  Verified on this rig: all four D405s went from 0 fps to 44 fps.

    python calibration/reset_cameras.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rs_camera import (connected_devices, name_for, require_rs,  # noqa: E402
                       reset_all_devices, rs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait", type=float, default=12.0,
                    help="seconds to wait for re-enumeration")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the post-reset frame check")
    args = ap.parse_args()

    require_rs()
    print()
    print("#" * 66)
    print("#  RealSense device reset")
    print("#" * 66)
    print()
    for d in connected_devices():
        print(f"  {d['serial']}  {name_for(d['serial']) or '(unmapped)'}")
    print()

    reset_all_devices(wait_s=args.wait)
    if args.no_verify:
        return

    print("\n  verifying each camera delivers frames:")
    ok = True
    for d in connected_devices():
        serial = d["serial"]
        name = name_for(serial) or serial
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
        pipe = rs.pipeline()
        try:
            pipe.start(cfg)
        except Exception as e:
            print(f"    {name:14s} START FAILED: {str(e)[:50]}")
            ok = False
            continue
        got, t0 = 0, time.time()
        while time.time() - t0 < 2.0:
            try:
                if pipe.wait_for_frames(1200).get_color_frame():
                    got += 1
            except Exception:
                break
        print(f"    {name:14s} {got / 2.0:5.1f} fps  "
              f"{'OK' if got else '*** STILL NO FRAMES ***'}")
        ok = ok and bool(got)
        try:
            pipe.stop()
        except Exception:
            pass
        time.sleep(0.4)

    print()
    if ok:
        print("  all cameras delivering. Re-run whatever failed.")
    else:
        print("  some cameras still dead -- now suspect the hardware:")
        print("    a different USB3 cable, a different port, `lsusb -t`.")
    print()


if __name__ == "__main__":
    main()
