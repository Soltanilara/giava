"""Standalone OAK -> headset streaming test.

Minimal end-to-end check of the exact camera + headset path data_collection
uses: same WebRTCHeadset (local webrtc_headset.py, bgr24 color tracks), same
OAK pipeline (OV9782 COLOR fix, RGB888p 640x480 @ 25 fps).

Requires depthai==3.5.0 (environment.yml pin): depthai >= 3.6 ships RVC2
firmware that heap-crashes this device on any stream configuration.

(Previously imported gym_av_aloha.vr.headset, which fails on python 3.12 —
mutable-default dataclasses — and is a different implementation from the one
data_collection uses anyway.)

    python oak_to_headset.py

Prints the device list and USB hints before starting, then a once-per-second
FPS line while streaming.  On X_LINK errors it reports and rebuilds the
pipeline instead of exiting.  Ctrl+C to quit.
"""

from __future__ import annotations

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cv2  # noqa: E402
import depthai as dai  # noqa: E402

from headset_link import make_headset  # noqa: E402
import camera_manager as cm  # noqa: E402
from camera_manager import compose_eye_view, oak_stream_settings  # noqa: E402

WIDTH, HEIGHT, FPS = oak_stream_settings()


def tuning_thread() -> None:
    """Live stereo-comfort tuning from stdin while streaming.

    a/z = image bigger/smaller, k/m = images closer together / farther
    apart (each +Enter).  Or set directly: 's 0.55', 'i 0.12'.
    Prints paste-ready lines for camera_manager.py after every change."""
    import threading

    def _loop() -> None:
        while True:
            try:
                line = input().strip().lower()
            except EOFError:
                return
            parts = line.split()
            if not parts:
                continue
            cmd = parts[0]
            if cmd == "a":
                cm.EYE_VIEW_SCALE = min(1.0, cm.EYE_VIEW_SCALE + 0.05)
            elif cmd == "z":
                cm.EYE_VIEW_SCALE = max(0.1, cm.EYE_VIEW_SCALE - 0.05)
            elif cmd == "k":
                cm.EYE_VIEW_INWARD_FRAC = min(0.4, cm.EYE_VIEW_INWARD_FRAC + 0.02)
            elif cmd == "m":
                cm.EYE_VIEW_INWARD_FRAC = max(-0.4, cm.EYE_VIEW_INWARD_FRAC - 0.02)
            elif cmd == "s" and len(parts) == 2:
                cm.EYE_VIEW_SCALE = float(parts[1])
            elif cmd == "i" and len(parts) == 2:
                cm.EYE_VIEW_INWARD_FRAC = float(parts[1])
            else:
                print("\n  a/z size, k/m spacing, or 's 0.55' / 'i 0.12'")
                continue
            print(f"\nEYE_VIEW_SCALE = {cm.EYE_VIEW_SCALE:.2f}\n"
                  f"EYE_VIEW_INWARD_FRAC = {cm.EYE_VIEW_INWARD_FRAC:.2f}\n"
                  "(paste into camera_manager.py when it looks right)")

    threading.Thread(target=_loop, daemon=True).start()


def build_pipeline():
    """The exact production pipeline (camera_manager), which also loads the
    EEPROM calibration into cm.OAK_RECTIFY for undistort + rectify."""
    res = cm.setup_oak_stereo()
    if res is None:
        raise RuntimeError("no OAK device found")
    return res


def main() -> None:
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("NO OAK DEVICE FOUND.")
        print("  check:  lsusb | grep 03e7")
        print("  then :  lsusb -t   (its bus root hub must say 5000M for")
        print("          full-res color; 480M = USB2 cable/port)")
        sys.exit(1)
    for d in devices:
        print(f"OAK found: {d.deviceId}  state={d.state}")
    print("If streaming crash-loops (X_LINK errors ~every 7 s): check "
          "depthai version — must be 3.5.0 (>=3.6 firmware crashes on this "
          "device: RTEMS heap corruption on any stream config).")
    print(f"depthai version: {dai.__version__}")

    headset = make_headset(src_size=(WIDTH, HEIGHT), fps=FPS)
    headset.run_in_thread()
    print("Headset link started — put the headset on / open the app to "
          "connect.")

    pipeline, q_left, q_right = build_pipeline()
    ## build_pipeline() is what reads the OAK's calibration, so the measured
    ## geometry only exists now -- after the headset was constructed.
    cm._push_camera_params_to_headset(headset)
    print(f"Streaming {WIDTH}x{HEIGHT} color @ {FPS} fps ... Ctrl+C to stop")
    print(f"Stereo comfort: scale={cm.EYE_VIEW_SCALE:.2f} "
          f"inward={cm.EYE_VIEW_INWARD_FRAC:.2f} — tune live: a/z = "
          "bigger/smaller, k/m = closer/farther (then Enter)")
    tuning_thread()

    frames = 0
    t_last = time.time()
    while True:
        try:
            if not pipeline.isRunning():
                print("pipeline stopped — rebuilding in 3 s")
                time.sleep(3.0)
                pipeline, q_left, q_right = build_pipeline()
                continue

            left_bgr = q_left.get().getCvFrame()   # RGB888p -> BGR (OpenCV)
            right_bgr = q_right.get().getCvFrame()
            maps = cm.OAK_RECTIFY["maps"]
            if maps is not None:
                left_bgr = cv2.remap(left_bgr, *maps["left"], cv2.INTER_LINEAR)
                right_bgr = cv2.remap(right_bgr, *maps["right"], cv2.INTER_LINEAR)
            headset.send_images(
                compose_eye_view(left_bgr, "left"),
                compose_eye_view(right_bgr, "right"),
            )

            frames += 1
            now = time.time()
            if now - t_last >= 1.0:
                print(f"\r{frames / (now - t_last):5.1f} fps   "
                      f"frame {left_bgr.shape}", end="", flush=True)
                frames = 0
                t_last = now

        except KeyboardInterrupt:
            print("\nstopping.")
            break
        except Exception as e:
            print(f"\nstream error: {e} — rebuilding pipeline in 3 s")
            time.sleep(3.0)
            try:
                pipeline.stop()
            except Exception:
                pass
            try:
                pipeline, q_left, q_right = build_pipeline()
            except Exception as e2:
                print(f"rebuild failed ({e2}); retrying in 5 s")
                time.sleep(5.0)


if __name__ == "__main__":
    main()
