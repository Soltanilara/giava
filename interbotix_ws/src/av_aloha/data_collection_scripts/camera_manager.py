import os

import numpy as np
import threading
import time
import cv2
from dataclasses import dataclass

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import depthai as dai
except ImportError:
    dai = None

from data_col_config import ARM_MODES

CAMERAS = [
    "left_wrist",
    "right_wrist",
    "oak_left",
    "oak_right",
    "top_scene",
    "low_scene",
]

CAMERA_SERIALS = {
    "left_wrist": "230322272239",
    "right_wrist": "230322270105",
    "top_scene": "230322270396",
    "low_scene": "230322271312",
}

# RealSense colour-stream settings
RS_COLOR_WIDTH = 640
RS_COLOR_HEIGHT = 480
RS_COLOR_FPS = 60

# Factory intrinsics of every RealSense camera used for undistortion
CAMERA_INTRINSICS = {}

## ------------------------------------------------------------------ ##
## The D405s have no hardware genlock, so each tick picks, per camera, the
## frame nearest the newest instant every camera has already covered -- within
## half a frame period of a common reference, where latest-frame-wins was
## bounded only by a full period.  The residual offset between cameras
## (~11.6 ms at 60 fps, from the recorded timestamps) cannot be closed in
## software but is stable, so it is calibratable.  GIAVA_SYNC_FRAMES=0
## restores latest-frame-wins.
SYNC_FRAMES = os.environ.get("GIAVA_SYNC_FRAMES", "1") == "1"

## Frames kept per camera.  At 60 fps this is ~0.13 s of history, plenty to
## reach back to a reference set by the slowest camera (the OAK at 25 fps).
FRAME_HISTORY_LEN = int(os.environ.get("GIAVA_FRAME_HISTORY", "8"))

## {camera: deque[(timestamp_s, frame, frame_number)]}.  Written by the
## capture workers under the caller's frame_lock, same as latest_frames.
FRAME_HISTORY = {}


def _record_history(name, timestamp, frame, frame_number=None):
    """Append one frame to a camera's history. Call under frame_lock."""
    from collections import deque
    h = FRAME_HISTORY.get(name)
    if h is None:
        h = FRAME_HISTORY[name] = deque(maxlen=FRAME_HISTORY_LEN)
    h.append((timestamp, frame, frame_number))


def select_synchronized_frames(cameras):
    """Pick each camera's frame nearest a shared reference instant.

    CALL UNDER frame_lock -- it reads the histories the workers write.

    Returns (frames, timestamps, info).  `info` carries the alignment
    quality so a recording can be judged afterwards instead of trusted:
        reference_s   the instant frames were matched to
        spread_s      max minus min of the CHOSEN timestamps -- the residual
                      misalignment of this timestep
        per_camera    chosen timestamp, frame number, and offset from the
                      reference, per camera

    All timestamps are on one clock: librealsense global_time maps each
    device onto the host epoch, and the OAK worker stamps host time
    directly, so they are comparable.  That is a software correspondence,
    not evidence the shutters fired together."""
    newest = {}
    for c in cameras:
        h = FRAME_HISTORY.get(c)
        if h:
            newest[c] = h[-1][0]
    if not newest:
        return {}, {}, {"reference_s": None, "spread_s": None,
                        "per_camera": {}, "cameras_missing": list(cameras)}

    # The newest instant EVERY camera has already covered.  Using the max
    # instead would ask the slower cameras for a frame they have not
    # produced yet, and they would silently return their newest anyway.
    reference = min(newest.values())

    frames, timestamps, per_camera = {}, {}, {}
    for c in cameras:
        h = FRAME_HISTORY.get(c)
        if not h:
            continue
        ts, frame, fn = min(h, key=lambda e: abs(e[0] - reference))
        frames[c] = frame
        timestamps[c] = ts
        per_camera[c] = {
            "timestamp_s": ts,
            "frame_number": fn,
            "offset_from_reference_s": ts - reference,
            "history_depth": len(h),
        }
    spread = (max(timestamps.values()) - min(timestamps.values())
              if len(timestamps) > 1 else 0.0)
    return frames, timestamps, {
        "reference_s": reference,
        "spread_s": spread,
        "per_camera": per_camera,
        "cameras_missing": [c for c in cameras if c not in frames],
    }

@dataclass
class CameraConfig:
    top_active: bool = True
    low_active: bool = False

def get_active_cameras(arm_mode: str, camera_config: CameraConfig):
    cameras = []
    arms = ARM_MODES[arm_mode]

    if "left" in arms:
        cameras.append("left_wrist")

    if "right" in arms:
        cameras.append("right_wrist")

    if "middle" in arms:
        cameras.append("oak_left")
        cameras.append("oak_right")

    if camera_config.top_active:
        cameras.append("top_scene")

    if camera_config.low_active:
        cameras.append("low_scene")

    return cameras

def intrinsics_to_dict(intr, serial=None, name=None, fps=None):
    """rs.intrinsics -> plain dict, with an OpenCV-shaped camera matrix.

    ppx/ppy and cx/cy are both present on purpose: librealsense says ppx/ppy,
    OpenCV says cx/cy, and they are the same numbers.  Conflating them is a
    classic silent bug, so neither name is dropped.

    NOTE the distortion model.  The D405 colour stream reports
    'inverse_brown_conrady', whose coefficients undistort (pixel -> ray),
    which is the OPPOSITE direction from OpenCV's (ray -> pixel).  Use
    rs2_deproject_pixel_to_point, or convert, but do not feed these straight
    into cv2.undistort as if they were OpenCV coefficients."""
    return {
        "camera": name,
        "serial": serial,
        "width": int(intr.width),
        "height": int(intr.height),
        "fps": fps,
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "camera_matrix": [[float(intr.fx), 0.0, float(intr.ppx)],
                          [0.0, float(intr.fy), float(intr.ppy)],
                          [0.0, 0.0, 1.0]],
        "distortion_model": str(intr.model),
        "distortion_coefficients": [float(c) for c in intr.coeffs],
        "distortion_direction": (
            "librealsense convention: for inverse_brown_conrady these "
            "coefficients map PIXEL -> RAY (rs2_deproject_pixel_to_point). "
            "OpenCV's map RAY -> PIXEL. Not interchangeable."
        ),
        "source": "realsense_factory",
    }


def save_camera_intrinsics(path):
    """Write this session's intrinsics next to the dataset it belongs to.

    Called by data_collection so every recording carries the calibration of
    the exact cameras and resolution that produced it."""
    import json
    import time as _time

    payload = {
        "source": "realsense_factory",
        "recorded": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "color_stream": {"width": RS_COLOR_WIDTH, "height": RS_COLOR_HEIGHT,
                         "fps": RS_COLOR_FPS, "format": "rgb8"},
        "cameras": CAMERA_INTRINSICS,
        "note": (
            "Factory intrinsics read from each device at session start. "
            "Intrinsics are resolution specific and belong to the "
            "color_stream above. For an independent check against an OpenCV "
            "ChArUco calibration see calibration/README.md."
        ),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path



## The ROM bootloader is USB2-only by design: it enumerates at 480M on a good
## USB3 port, then re-enumerates at 5000M once depthai uploads the firmware.
OAK_BOOTLOADER_PIDS = {"2485", "2486", "f63c"}


def oak_usb_speed_mbps():
    """Negotiated USB speed from sysfs (480/5000), or None if not yet knowable.

    None includes "only an unbooted bootloader is present" -- its 480M says
    nothing about the real link, so the caller must not warn on None."""
    import glob
    for vend in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            if open(vend).read().strip() != "03e7":
                continue
            dev_dir = vend.rsplit("/", 1)[0]
            try:
                pid = open(dev_dir + "/idProduct").read().strip().lower()
            except OSError:
                pid = ""
            if pid in OAK_BOOTLOADER_PIDS:
                continue  # not booted yet; its speed says nothing about the link
            return int(float(open(dev_dir + "/speed").read().strip()))
        except (OSError, ValueError):
            continue
    return None


def oak_stream_settings():
    """(width, height, fps) for the OAK color streams.

    1280x800 is the OV9782's NATIVE mode: no ISP crop/rescale (the 640x480
    mode crops to 4:3 and downscales, which is visibly soft), and it matches
    the resolution of Devi's checkerboard calibration (foveated_world_model
    stereo.npz), so the rectify maps apply pixel-for-pixel.  Needs the USB3
    (5000M) link; the dataset copies are downscaled in oak_worker.

    Sidenote: the env pins depthai==3.5.0 (environment.yml) — >=3.6 ships
    RVC2 firmware that heap-crashes this device on any stream config."""
    speed = oak_usb_speed_mbps()
    if speed is None:
        ## Almost always "the device has not booted yet" on a cold start, since
        ## this runs before dai.Device().  Say nothing rather than guess.
        _info("[OAK] USB link speed unknown (device not booted yet) — "
              "native 1280x800@25 color x2.")
    else:
        _info(f"[OAK] USB link is {speed}M — native 1280x800@25 color x2.")
        if speed < 5000:
            print("[OAK] WARNING: link is not USB3; native res may stall — "
                  "move to a USB3 port/cable, or drop to (640, 480, 25).")
    return 1280, 800, 25

## Dataset frames keep the historical 640x480 shape: center-crop the native
## 1280x800 to 4:3 (1066x800) then resize — the same view the on-device ISP
## produced when we streamed 640x480 directly.
OAK_DATASET_SIZE = (640, 480)

def oak_dataset_view(bgr):
    h, w = bgr.shape[:2]
    tw, th = OAK_DATASET_SIZE
    if (w, h) == (tw, th):
        return bgr
    crop_w = min(w, int(round(h * tw / th)))
    x0 = (w - crop_w) // 2
    return cv2.resize(bgr[:, x0:x0 + crop_w], (tw, th),
                      interpolation=cv2.INTER_AREA)


## Undistort + rectify maps for the OAK pair.  Preferred source:
## checkerboard calibration (stereo.npz from the foveated_world_model repo,
## branch oak-cameras — 46 pairs @1280x800, rms 1.3 px, verified with
## rect_*.png samples).  Fallback: the device EEPROM.  Applied to the
## HEADSET stream only — dataset frames stay raw (just downscaled).
## {"maps": None} means no usable calibration: stream raw.
OAK_RECTIFY = {"maps": None}

# The left and right camera modules were swapped when switching the camera case cover
OAK_SWAP_EYES = {"on": os.environ.get("GIAVA_OAK_SWAP_EYES", "0") == "1"}

def oak_swap_eyes(on=None) -> bool:
    """Get, or set, whether the two OAK frames are exchanged at the source."""
    if on is not None:
        OAK_SWAP_EYES["on"] = bool(on)
    return OAK_SWAP_EYES["on"]

OAK_CALIB_STEREO_NPZ = "/home/devi/foveated_world_model/stereo.npz"

# Baseline (distance between camera modules) as measured by ruler and verified
# with stereo calibration. Getting this value wrong breaks epipolar alignment, 
# leaving residual VERTICAL disparity between the eyes, which is the one stereo 
# error the operator cannot fuse and reads as eye strain rather than as a 
# picture fault. Warping with the wrong value is worse than not rectifying at all,
# because the failure does not look like a failure.
OAK_EXPECTED_BASELINE_M = 0.062
OAK_BASELINE_TOLERANCE_M = 0.002


def _npz_rectified_baseline(d):
    """SIGNED rectified baseline in metres in case camera modules are reversed"""
    fx = float(d["P1"][0, 0])
    return (-float(d["P2"][0, 3]) / fx) if fx else 0.0


def load_oak_rectify_maps_from_file(w, h, path=None):
    """Rectify maps from the checkerboard stereo calibration npz.

    The npz stores the cv2.initUndistortRectifyMap outputs directly
    (map1x/map1y = left/CAM_B, map2x/map2y = right/CAM_C).

    Raises if the calibration's baseline disagrees with the measured one
    which is stored as OAK_EXPECTED_BASELINE_M."""
    d = np.load(path or OAK_CALIB_STEREO_NPZ)
    cw, ch = (int(v) for v in d["image_size"])
    if (cw, ch) != (w, h):
        raise ValueError(f"calibration is {cw}x{ch} but stream is {w}x{h}")

    if "P1" in d and "P2" in d:
        signed = _npz_rectified_baseline(d)
        if signed < 0.0:
            raise ValueError(
                f"Based on calibration results, eye cam modules seem to be reversed. "
                f"Re-fit the capture with `oak_stereo_calibrate.py --swap`, and fix "
                f"the labels at the source (GIAVA_OAK_SWAP_EYES=1, or the `swap` "
                f"command in teleop).")

    if OAK_EXPECTED_BASELINE_M is not None and "P1" in d and "P2" in d:
        got = abs(_npz_rectified_baseline(d))
        if abs(got - OAK_EXPECTED_BASELINE_M) > OAK_BASELINE_TOLERANCE_M:
            raise ValueError(
                f"calibration baseline is {got * 1000:.1f} mm but this rig "
                f"measures {OAK_EXPECTED_BASELINE_M * 1000:.1f} mm "
                f"(tolerance {OAK_BASELINE_TOLERANCE_M * 1000:.1f} mm) -- "
                f"{path or OAK_CALIB_STEREO_NPZ} was shot on a DIFFERENT "
                "physical arrangement and its extrinsics do not describe this "
                "one. Re-run the stereo calibration, then update "
                "OAK_EXPECTED_BASELINE_M if the rig itself changed.")

    return {
        "left": (d["map1x"], d["map1y"]),
        "right": (d["map2x"], d["map2y"]),
    }


def install_rectify_from_npz(path, w, h):
    """Measure new rectification and directly implement into live view

    Used by teleop's `cal` command so a new fit can be SEEN rather than read
    as an RMS number.  Goes through `load_oak_rectify_maps_from_file`, so a fit 
    with reversed handedness or the wrong baseline is refused here exactly 
    as it would be at startup."""
    maps = load_oak_rectify_maps_from_file(w, h, path=path)
    OAK_RECTIFY["maps"] = maps
    try:
        OAK_GVLINK_CAMERA["params"] = oak_params_from_npz(w, h, True, path=path)
    except Exception as exc:
        print(f"[OAK] rectify maps installed, but the viewer intrinsics could "
              f"not be refreshed ({exc}) -- the images are rectified and the "
              f"headset is still placing them at the previous field of view.")
    return maps


def build_oak_rectify_maps(calib, w, h):
    """Code to undistort AND stereo-rectify the OAK pair such that only 
    horizontal disparity remains, making stereo pair fuse comfortably."""
    left, right = dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
    K1 = np.array(calib.getCameraIntrinsics(left, w, h), dtype=np.float64)
    K2 = np.array(calib.getCameraIntrinsics(right, w, h), dtype=np.float64)
    D1 = np.array(calib.getDistortionCoefficients(left), dtype=np.float64)
    D2 = np.array(calib.getDistortionCoefficients(right), dtype=np.float64)
    ext = np.array(calib.getCameraExtrinsics(left, right), dtype=np.float64)
    R, T = ext[:3, :3], ext[:3, 3]
    if K1[0, 0] < 10 or K2[0, 0] < 10 or not np.any(T):
        raise ValueError("EEPROM has no real calibration (zero intrinsics "
                         "or zero baseline)")
    size = (w, h)
    R1, R2, P1, P2 = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)[:4]
    return {
        "left": cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2),
        "right": cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2),
    }


def oak_gvlink_camera_params(calib, w, h, fisheye=None):
    """
    Rectified intrinsics + baseline for the headset viewer, in gvlink wire form.

    The viewer places each image from these numbers instead of the operator guessing a
    field of view, so they must describe the frames actually sent -- i.e. the RECTIFIED
    ones. That means P1/P2 out of cv2.stereoRectify, not the raw K: after rectification
    fx = P1[0,0], cx = P1[0,2], and the baseline is -P2[0,3] / fx.

    Deliberately runs the same stereoRectify call as build_oak_rectify_maps() with the
    same flags, so the geometry the viewer is told matches the remap the sender applies.
    """
    left, right = dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
    K1 = np.array(calib.getCameraIntrinsics(left, w, h), dtype=np.float64)
    K2 = np.array(calib.getCameraIntrinsics(right, w, h), dtype=np.float64)
    D1 = np.array(calib.getDistortionCoefficients(left), dtype=np.float64)
    D2 = np.array(calib.getDistortionCoefficients(right), dtype=np.float64)
    ext = np.array(calib.getCameraExtrinsics(left, right), dtype=np.float64)
    R, T = ext[:3, :3], ext[:3, 3]
    if K1[0, 0] < 10 or K2[0, 0] < 10 or not np.any(T):
        raise ValueError("EEPROM has no real calibration")

    if fisheye is None:
        try:
            fisheye = calib.getDistortionModel(left) == dai.CameraModel.Fisheye
        except Exception:
            fisheye = False

    size = (w, h)
    if fisheye:
        d1, d2 = D1[:4].reshape(4, 1), D2[:4].reshape(4, 1)
        _R1, _R2, P1, P2, _Q = cv2.fisheye.stereoRectify(
            K1, d1, K2, d2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0)
    else:
        _R1, _R2, P1, P2 = cv2.stereoRectify(
            K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)[:4]

    fx = float(P1[0, 0])
    baseline = abs(float(P2[0, 3]) / fx) if fx else 0.0
    return {
        "w": int(w), "h": int(h),
        "b": baseline,
        "rect": True,
        "l": {"fx": fx, "fy": float(P1[1, 1]),
              "cx": float(P1[0, 2]), "cy": float(P1[1, 2])},
        "r": {"fx": float(P2[0, 0]), "fy": float(P2[1, 1]),
              "cx": float(P2[0, 2]), "cy": float(P2[1, 2])},
    }


def oak_params_from_npz(w, h, rectified, path=None, baseline_m=None):
    """Per-eye intrinsics + baseline for the headset viewer, from the npz.

    ONE function for two cases that differ only in which matrices describe the
    frames actually being sent:

      rectified=True   P1/P2 -- the geometry of the same cv2.stereoRectify
                       call that produced the remap tables, so it is by
                       construction the geometry of the transmitted frames.
                       Baseline comes from P2[0,3]/fx.
      rectified=False  K1/K2 -- the raw per-lens intrinsics, for when the
                       rectification was rejected (re-casing invalidates the
                       stereo EXTRINSICS but not the per-lens intrinsics).
                       Baseline is the hand-measured one, because the npz's
                       extrinsics are exactly the part that went stale.

    Why it matters that this is populated at all: without it the viewer
    synthesises a 90-degree pinhole, while these lenses measure ~95-97 deg
    horizontally -- everything renders ~13% too small, disparities included,
    which reads as the images sitting too far apart to fuse.

    `rect` goes on the wire as given and must match reality: the operator's
    residual-misalignment trim lives in the viewer, not here.
    """
    d = np.load(path or OAK_CALIB_STEREO_NPZ)
    cw, ch = (int(v) for v in d["image_size"])
    if (cw, ch) != (w, h):
        raise ValueError(f"calibration is {cw}x{ch} but stream is {w}x{h}")

    if rectified:
        L, Rm = d["P1"], d["P2"]
        fx = float(L[0, 0])
        b = abs(float(Rm[0, 3]) / fx) if fx else 0.0
    else:
        L, Rm = d["K1"], d["K2"]
        b = float(baseline_m if baseline_m is not None
                  else (OAK_EXPECTED_BASELINE_M or 0.0))
    return {
        "w": int(w), "h": int(h), "b": b, "rect": bool(rectified),
        "l": {"fx": float(L[0, 0]), "fy": float(L[1, 1]),
              "cx": float(L[0, 2]), "cy": float(L[1, 2])},
        "r": {"fx": float(Rm[0, 0]), "fy": float(Rm[1, 1]),
              "cx": float(Rm[0, 2]), "cy": float(Rm[1, 2])},
    }


## Filled in by setup_oak_stereo() so the headset link can describe the camera without
## re-reading the device. None means "no calibration"; the link then synthesises a
## symmetric pinhole from GIAVA_HEADSET_HFOV, which is a guess, not the truth.
OAK_GVLINK_CAMERA = {"params": None}


## ------------------------------------------------------------------ ##
## Single-shot RAW stereo capture for checkerboard calibration: taken inside
## the worker, the only place holding both eyes untouched at native resolution
## (latest_frames has the 640x480 crop, the headset copy is remapped).  Armed
## rather than polled, so idle cost is one boolean read per frame.
OAK_CAPTURE = {"want": False, "left": None, "right": None, "ts": None}


def request_oak_capture():
    """Ask the OAK worker for the next raw, native, synchronized pair."""
    OAK_CAPTURE["left"] = None
    OAK_CAPTURE["right"] = None
    OAK_CAPTURE["ts"] = None
    OAK_CAPTURE["want"] = True


def take_oak_capture(frame_lock):
    """(left_bgr, right_bgr, ts) once the request has been served, else None.

    Consumes the pair: a second call returns None until the next request."""
    with frame_lock:
        if OAK_CAPTURE["left"] is None or OAK_CAPTURE["right"] is None:
            return None
        out = (OAK_CAPTURE["left"], OAK_CAPTURE["right"], OAK_CAPTURE["ts"])
        OAK_CAPTURE["left"] = None
        OAK_CAPTURE["right"] = None
        OAK_CAPTURE["ts"] = None
    return out


def setup_oak_stereo():
    if dai is None:
        raise ImportError("depthai is required for OAK stereo cameras.")
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print(
            "=" * 60 + "\n"
            "[OAK] NO DEVICE FOUND.  Check:\n"
            "  lsusb | grep -i 03e7      (Movidius/Luxonis VID)\n"
            "  If it was connected: unplug/replug -- an earlier X_LINK crash\n"
            "  can wedge the device.  Prefer a USB3 port and cable.\n"
            "[OAK] Continuing without the OAKs: no headset video, oak_* frames\n"
            "      will be missing from the dataset.\n" + "=" * 60)
        return None
    _info(f"[OAK] found: {[(d.deviceId, str(d.state)) for d in devices]}")
    _w, _h, _fps = oak_stream_settings()

    device = dai.Device()
    try:
        OAK_RECTIFY["maps"] = load_oak_rectify_maps_from_file(_w, _h)
        _info("[OAK] rectify maps loaded from checkerboard calibration "
              f"({OAK_CALIB_STEREO_NPZ}) — headset stream undistorted + "
              "rectified.")
        try:
            OAK_GVLINK_CAMERA["params"] = oak_params_from_npz(_w, _h)
            p = OAK_GVLINK_CAMERA["params"]
            _info(f"[OAK] headset camera params (npz, RECTIFIED): "
                  f"{p['w']}x{p['h']} fx={p['l']['fx']:.0f} "
                  f"baseline={p['b'] * 1000:.1f} mm")
        except Exception as e_cam:
            print(f"[OAK] could not read headset camera params from the npz "
                  f"({e_cam}); the viewer will guess a field of view.")
    except Exception as e_file:
        try:
            _calib = device.readCalibration()
            OAK_RECTIFY["maps"] = build_oak_rectify_maps(_calib, _w, _h)
            try:
                OAK_GVLINK_CAMERA["params"] = oak_gvlink_camera_params(_calib, _w, _h)
                _info(f"[OAK] headset camera params: {OAK_GVLINK_CAMERA['params']['w']}x"
                      f"{OAK_GVLINK_CAMERA['params']['h']} "
                      f"fx={OAK_GVLINK_CAMERA['params']['l']['fx']:.0f} "
                      f"baseline={OAK_GVLINK_CAMERA['params']['b'] * 1000:.0f} mm")
            except Exception as e_cam:
                print(f"[OAK] could not derive headset camera params ({e_cam})")
            print(f"[OAK] checkerboard npz unavailable ({e_file}); using "
                  "EEPROM calibration — headset stream undistorted + "
                  "rectified.")
        except Exception as e_eeprom:
            OAK_RECTIFY["maps"] = None
            print(f"[OAK] no usable calibration (npz: {e_file}; EEPROM: "
                  f"{e_eeprom}).")
            print("      Headset stream stays raw/distorted.")
            # The stream is raw, but the LENSES are still measured: hand the viewer
            # the npz intrinsics (rect=False, ruler baseline) so it places the raw
            # images at their true angular size instead of a 90-degree guess.
            try:
                OAK_GVLINK_CAMERA["params"] = oak_params_from_npz(_w, _h)
                p = OAK_GVLINK_CAMERA["params"]
                _info(f"[OAK] headset camera params (UNRECTIFIED, npz intrinsics): "
                      f"{p['w']}x{p['h']} fx={p['l']['fx']:.0f}/{p['r']['fx']:.0f} "
                      f"baseline={p['b'] * 1000:.0f} mm")
            except Exception as e_k:
                print(f"[OAK] npz intrinsics also unavailable ({e_k}); viewer "
                      "will guess a field of view.")
    oak_pipeline = dai.Pipeline(device)

    cam_left = oak_pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    cam_right = oak_pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)

    # This device misreports OV9782 colour sensors as OV9282 (mono); without
    # forcing COLOR the frames come out mono with grid artifacts (depthai 3.5+).
    cam_left.setSensorType(dai.CameraSensorType.COLOR)
    cam_right.setSensorType(dai.CameraSensorType.COLOR)

    left_out = cam_left.requestOutput(
        (_w, _h),
        type=dai.ImgFrame.Type.RGB888p,
        fps=_fps,
    )
    right_out = cam_right.requestOutput(
        (_w, _h),
        type=dai.ImgFrame.Type.RGB888p,
        fps=_fps,
    )

    q_left = left_out.createOutputQueue()
    q_right = right_out.createOutputQueue()

    oak_pipeline.start()
    time.sleep(0.5)
    return oak_pipeline, q_left, q_right

def setup_realsense_cameras(active_cameras, camera_shutdown, frame_lock, latest_frames, latest_timestamps):
    if rs is None:
        raise ImportError("pyrealsense2 is required for RealSense cameras.")
    rs_pipelines = {}

    for name in active_cameras:
        if name.startswith("oak"):
            continue

        serial = CAMERA_SERIALS[name]

        rs_pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(serial)
        config.enable_stream(rs.stream.color, RS_COLOR_WIDTH, RS_COLOR_HEIGHT,
                             rs.format.rgb8, RS_COLOR_FPS)

        try:
            profile = rs_pipeline.start(config)
        except RuntimeError as e:
            print("=" * 60)
            print(f"[RealSense] '{name}' (serial {serial}) failed to start: {e}")
            print("  'Couldn't resolve requests' usually means the camera is")
            print("  on a USB2 port/cable (640x480 rgb8 @60fps needs USB3).")
            print("  Check: lsusb -t   (the camera's bus must say 5000M)")
            print(f"[RealSense] Continuing WITHOUT '{name}' — its dataset")
            print("  frames will be missing until reconnected properly.")
            print("=" * 60)
            continue

        # Global time maps this device's clock onto the host epoch via a
        # per-device linear fit, which is what makes timestamps comparable
        # BETWEEN cameras.
        # (This is an estimate against the host, NOT hardware sync.)
        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                try:
                    sensor.set_option(rs.option.global_time_enabled, 1.0)
                except Exception as e:
                    print(f"[RealSense] '{name}': could not enable "
                          f"global_time ({e}); cross-camera timestamps will "
                          f"not be comparable.")

        color_profile = (
            profile.get_stream(rs.stream.color).as_video_stream_profile()
        )

        ## Record the factory intrinsics of the stream we just opened.  The
        ## profile was already being fetched and thrown away here; keeping it
        ## is what makes these recordings undistortable later.
        try:
            CAMERA_INTRINSICS[name] = intrinsics_to_dict(
                color_profile.get_intrinsics(), serial=serial, name=name,
                fps=RS_COLOR_FPS)
            i = CAMERA_INTRINSICS[name]
            _info(f"[RealSense] '{name}' intrinsics: "
                  f"{i['width']}x{i['height']} "
                  f"fx={i['fx']:.3f} fy={i['fy']:.3f} "
                  f"cx={i['cx']:.3f} cy={i['cy']:.3f} "
                  f"model={i['distortion_model']}")
        except Exception as e:
            print(f"[RealSense] '{name}': could not read intrinsics: {e}")

        rs_pipelines[name] = rs_pipeline

        ## Registered in CAMERA_WORKERS (defined below) so shutdown can join
        ## it before this pipeline is stopped -- see join_camera_workers.
        _rs_t = threading.Thread(
            target=camera_worker,
            args=(name, rs_pipeline, camera_shutdown, frame_lock, latest_frames, latest_timestamps),
            daemon=True,
            name=f"cam-{name}",
        )
        CAMERA_WORKERS.append(_rs_t)
        _rs_t.start()

    return rs_pipelines

def _push_camera_params_to_headset(headset):
    """Give the headset the OAK's measured geometry, if it can take it.

    Best-effort by design: the WebRTC transport has no equivalent (the viewer
    guesses a field of view there), and a headset that cannot be told is not a
    reason to fail camera setup."""
    if headset is None:
        return
    setter = getattr(headset, "set_camera_params", None)
    if setter is None:
        return
    params = OAK_GVLINK_CAMERA.get("params")
    if not params:
        print("[OAK] no rectified calibration to send the viewer -- it will "
              "place the images from a synthesised field of view.")
        return
    try:
        if setter(params):
            _info("[OAK] viewer camera geometry updated from the OAK "
                  "calibration.")
    except Exception as exc:
        print(f"[OAK] could not send camera geometry to the viewer: {exc}")


## Worker threads spawned by setup_cameras(), so shutdown can WAIT for them.
## They are daemon threads: nothing joins them automatically, and stopping a
## librealsense pipeline or destroying the OAK device while its worker is still
## inside wait_for_frames() / queue.get() tears the native library down under a
## live thread.  The observed result is glibc's "FATAL: exception not rethrown"
## and a core dump at exit -- the forced-unwind exception raised in the
## cancelled thread gets swallowed by a catch(...) inside OpenCV, which glibc
## treats as fatal.  Harmless to the data (it happens after finalize) but it
## masks real crashes and leaves cameras in a bad state for the next run.
## ROUTINE CAMERA CHATTER vs THINGS THAT NEED READING.
##
## Bringing six cameras up printed ~15 lines of normal-operation detail --
## device ids, USB link speed, which calibration file was loaded, four sets of
## intrinsics -- every run.  All of it is true and none of it needs acting on,
## and it sat directly above the [SAFETY] / [profile] lines that do.
##
## _info() carries the routine lines; anything describing a FAILURE, a missing
## device, a fallback that changes behaviour, or a degraded link stays on a
## plain print() and is always shown.  The intrinsics in particular are not
## lost by hiding them: save_camera_intrinsics() writes them into the dataset
## itself, which is where they are actually needed.
##
## GIAVA_CAM_VERBOSE=1 brings the detail back.
CAM_VERBOSE = os.environ.get("GIAVA_CAM_VERBOSE", "0").strip() not in ("0", "false", "no")


def _info(*args, **kwargs):
    """A camera line worth printing only when something looks wrong."""
    if CAM_VERBOSE:
        print(*args, **kwargs)


CAMERA_WORKERS = []


def join_camera_workers(timeout=2.0):
    """Stop the camera worker threads before their devices are destroyed.

    Callers must have set the camera_shutdown event first -- this only waits.
    Returns the workers still alive after `timeout`, which is information, not
    an error: a worker blocked in wait_for_frames() on a dead camera cannot be
    interrupted from here, and stopping the pipeline is what will free it.
    """
    stragglers = []
    for t in CAMERA_WORKERS:
        t.join(timeout=timeout)
        if t.is_alive():
            stragglers.append(t.name)
    CAMERA_WORKERS.clear()
    return stragglers


def setup_cameras(
    active_cameras,
    camera_shutdown,
    frame_lock,
    latest_frames,
    latest_timestamps,
    headset=None,
):
    cameras = {}

    if any(cam.startswith("oak") for cam in active_cameras):
        cameras["oak"] = setup_oak_stereo()
        if cameras["oak"] is not None:
            ## setup_oak_stereo() is what FILLS OAK_GVLINK_CAMERA -- the
            ## rectified intrinsics and baseline only exist once the device
            ## has been opened and its calibration read.  Callers build the
            ## headset first (so the viewer can connect while the cameras come
            ## up), which means the constructor saw None and the viewer would
            ## otherwise place both images from a synthesised 90-degree
            ## pinhole.  Hand the measured geometry over now.
            _push_camera_params_to_headset(headset)
            _oak_t = threading.Thread(
                target=oak_worker,
                args=(cameras, headset, camera_shutdown, frame_lock,
                      latest_frames, latest_timestamps),
                daemon=True,
                name="oak-worker",
            )
            CAMERA_WORKERS.append(_oak_t)
            _oak_t.start()

    ## Only touch librealsense if a RealSense camera was actually asked for.
    ## setup_realsense_cameras() raises ImportError when the module is absent,
    ## before it looks at the camera list -- so an OAK-only caller (teleop.py's
    ## `--cameras oak`) used to need pyrealsense2 installed to open a camera it
    ## had not requested.
    if any(not cam.startswith("oak") for cam in active_cameras):
        cameras["realsense"] = setup_realsense_cameras(
            active_cameras,
            camera_shutdown,
            frame_lock,
            latest_frames,
            latest_timestamps,
        )
    else:
        cameras["realsense"] = {}

    ## ONE LINE FOR THE NORMAL CASE, and a named list when anything asked for
    ## did not come up.  A missing camera is not an error here -- setup
    ## deliberately continues without it -- so it has to be visible, or a
    ## session records five streams while the operator believes it recorded
    ## six.
    up = set(cameras.get("realsense") or {})
    if cameras.get("oak") is not None:
        up |= {c for c in active_cameras if c.startswith("oak")}
    missing = [c for c in active_cameras if c not in up]
    print(f"[cameras] {len(up)}/{len(active_cameras)} up: "
          f"{', '.join(c for c in active_cameras if c in up)}")
    if missing:
        print(f"[cameras] NOT AVAILABLE: {', '.join(missing)} -- recording "
              f"will continue without them.")

    return cameras


## Headset stereo comfort tuning (does NOT affect dataset frames).
## The headset shows each eye's frame full-view, which makes the raw OAK
## images look zoomed-in and too far apart to fuse.  Each frame is shrunk
## by EYE_VIEW_SCALE onto a black canvas and shifted toward the nose by
## EYE_VIEW_INWARD_FRAC of the width (left image moves right, right image
## moves left), which zooms the view out and pulls the pair together.
##
## DISABLED BY DEFAULT (scale 1.0, inward 0.0): the Unity viewer now does
## both jobs as quad geometry.videoVFOV/videoScale replace the scale, 
## stereoSeparationDeg replaces the inward shift (opposite sign: 
## inward positive == stereoSeparationDeg negative).  Doing it here instead 
## cost real image quality: at scale 0.825 about a third of every transmitted 
## frame was black border that the encoder still had to spend bitrate on, 
## and the resize + canvas copy ran twice per frame on the capture thread 
## that feeds the encoder.
##
## compose_eye_view() short-circuits and returns the frame untouched at these
## values, so leaving them here costs nothing.  Set them again only if you
## deliberately want the sender to do the framing -- and if you do, zero the
## Unity side or the two will collide.
EYE_VIEW_SCALE = 1.0
EYE_VIEW_INWARD_FRAC = 0.0

## Runtime-tunable copies (the constants above are the defaults).  The values
## went stale when checkerboard rectification was added to the headset path.
## rectification rotates each view and re-centers the principal points, which
## changes the disparity geometry the old inward shift was tuned against.
## Retune LIVE from the data-collection keyboard ( , . adjust convergence,
## - = adjust scale ) and paste the printed values back here.
EYE_VIEW = {
    "scale": float(os.environ.get("GIAVA_EYE_SCALE", EYE_VIEW_SCALE)),
    "inward": float(os.environ.get("GIAVA_EYE_INWARD", EYE_VIEW_INWARD_FRAC)),
}

## Optional downscale of the headset stream only (dataset frames untouched).
## With the VP8 bitrate cap raised (webrtc_headset.py) try 1.0 first; if encode
## latency is the bottleneck, 0.5 quarters the pixels per frame.
HEADSET_SEND_SCALE = float(os.environ.get("GIAVA_HEADSET_SCALE", "1.0"))


def adjust_eye_view(dscale=0.0, dinward=0.0):
    """Nudge the live stereo-comfort parameters; returns the new values."""
    EYE_VIEW["scale"] = float(np.clip(EYE_VIEW["scale"] + dscale, 0.3, 1.0))
    EYE_VIEW["inward"] = float(np.clip(EYE_VIEW["inward"] + dinward, -0.05, 0.25))
    w = 1280  # native stream width; only used for the printout
    px = 2 * EYE_VIEW["inward"] * w * EYE_VIEW["scale"]
    print(f"[eye-view] scale={EYE_VIEW['scale']:.3f} "
          f"inward_frac={EYE_VIEW['inward']:.3f} "
          f"(~{px:.0f} px total convergence)  "
          f"paste into camera_manager.py when it feels right")
    return dict(EYE_VIEW)


def compose_eye_view(img, side, scale=None, inward_frac=None):
    """Shrink img onto a same-size black canvas, shifted toward the nose.

    side: 'left' shifts the image right, 'right' shifts it left."""
    scale = EYE_VIEW["scale"] if scale is None else scale
    inward_frac = EYE_VIEW["inward"] if inward_frac is None else inward_frac
    if scale >= 1.0 and inward_frac == 0.0:
        return img
    h, w = img.shape[:2]
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros_like(img)
    shift = int(round(w * inward_frac))
    x0 = (w - sw) // 2 + (shift if side == "left" else -shift)
    x0 = max(0, min(w - sw, x0))
    y0 = (h - sh) // 2
    canvas[y0:y0 + sh, x0:x0 + sw] = small
    return canvas


def oak_worker(cameras, headset, camera_shutdown, frame_lock, latest_frames, latest_timestamps):
    """Pump the OAK stereo queues: store frames for the dataset and stream
    them to the headset.  Resilient: when the device drops (X_LINK crash,
    unplug), it stops spamming, waits, and rebuilds the whole pipeline."""
    import time as _time

    _, q_left, q_right = cameras["oak"]
    fail_count = 0
    while not camera_shutdown.is_set():
        try:
            in_left = q_left.get()
            in_right = q_right.get()
            fail_count = 0
        except Exception as e:
            fail_count += 1
            if fail_count == 1:
                print(f"[OAK] frame wait failed ({e}) — will retry, then "
                      "rebuild the pipeline")
            if fail_count >= 5:
                print("[OAK] rebuilding pipeline in 3 s...")
                _time.sleep(3.0)
                try:
                    old = cameras.get("oak")
                    if old is not None:
                        try:
                            old[0].stop()
                        except Exception:
                            pass
                    rebuilt = setup_oak_stereo()
                    if rebuilt is None:
                        continue  # device still absent; loop and retry
                    cameras["oak"] = rebuilt
                    _, q_left, q_right = rebuilt
                    fail_count = 0
                    print("[OAK] pipeline rebuilt")
                except Exception as e2:
                    print(f"[OAK] rebuild failed ({e2}); retrying...")
            else:
                _time.sleep(0.5)
            continue

        left_bgr = in_left.getCvFrame()   # RGB888p -> getCvFrame gives BGR
        right_bgr = in_right.getCvFrame()
        ## THE one place handedness is decided -- see OAK_SWAP_EYES.  Applied
        ## before anything else touches these frames, so the capture files,
        ## the rectify maps, the dataset streams and the headset eyes all
        ## agree by construction instead of needing four consistent edits.
        if OAK_SWAP_EYES["on"]:
            left_bgr, right_bgr = right_bgr, left_bgr
        ts = _time.time()

        with frame_lock:
            # dataset stores RGB at the historical 640x480 (crop+downscale
            # of the native 1280x800), consistent with the RealSense streams
            l_rgb = oak_dataset_view(left_bgr)[..., ::-1].copy()
            r_rgb = oak_dataset_view(right_bgr)[..., ::-1].copy()
            latest_frames["oak_left"] = l_rgb
            latest_timestamps["oak_left"] = ts
            latest_frames["oak_right"] = r_rgb
            latest_timestamps["oak_right"] = ts
            # Host time, stamped after both queues returned -- NOT a device
            # timestamp.  The OAK's own getTimestamp() is never read here
            # (unchanged from before); these two share one host stamp, so
            # the pair is self-consistent but its offset to the RealSense
            # global-time clock includes this worker's scheduling.
            _record_history("oak_left", ts, l_rgb, None)
            _record_history("oak_right", ts, r_rgb, None)

            ## Armed single-shot: raw BGR, native resolution, both eyes from
            ## this one iteration.  Copied because these arrays are handed
            ## to cv2.remap and compose_eye_view below and outlive this
            ## block only here.
            if OAK_CAPTURE["want"]:
                OAK_CAPTURE["left"] = left_bgr.copy()
                OAK_CAPTURE["right"] = right_bgr.copy()
                OAK_CAPTURE["ts"] = ts
                OAK_CAPTURE["want"] = False

        if headset is not None:
            # headset video tracks are configured bgr24 (webrtc_headset.py)
            maps = OAK_RECTIFY["maps"]
            if maps is not None:
                left_view = cv2.remap(left_bgr, *maps["left"], cv2.INTER_LINEAR)
                right_view = cv2.remap(right_bgr, *maps["right"], cv2.INTER_LINEAR)
            else:
                left_view, right_view = left_bgr, right_bgr
            left_send = compose_eye_view(left_view, "left")
            right_send = compose_eye_view(right_view, "right")
            if HEADSET_SEND_SCALE != 1.0:
                hs, ws = left_send.shape[:2]
                sz = (int(ws * HEADSET_SEND_SCALE), int(hs * HEADSET_SEND_SCALE))
                left_send = cv2.resize(left_send, sz, interpolation=cv2.INTER_AREA)
                right_send = cv2.resize(right_send, sz, interpolation=cv2.INTER_AREA)
            headset.send_images(left_send, right_send)

def camera_worker(name, rs_pipeline, camera_shutdown, frame_lock, latest_frames, latest_timestamps):

    while not camera_shutdown.is_set():
        try:
            frames = rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
        except Exception as e:
            print(f"Warning: Waiting for camera frames failed: {e}")
            continue

        if not color_frame:
            continue

        # COPY, do not alias.  np.asanyarray(frame.get_data()) is a VIEW over
        # librealsense's buffer, which is recycled as soon as this frame is
        # released at the next loop iteration.  The history keeps several
        # frames alive, so aliasing would leave older entries pointing at
        # memory that has since been overwritten with newer images.  One
        # copy here is also the only copy: latest_frames points at this same
        # array.
        color = np.asanyarray(color_frame.get_data()).copy()
        # get_timestamp() is in the domain get_frame_timestamp_domain()
        # reports.  With global_time enabled (set in setup_realsense_cameras)
        # that is the HOST EPOCH in milliseconds, which is what makes this
        # comparable with the OAK worker's time.time() and across devices.
        color_ts = color_frame.get_timestamp() * 1e-3
        try:
            frame_number = int(color_frame.get_frame_number())
        except Exception:
            frame_number = None

        with frame_lock:
            latest_frames[name] = color
            latest_timestamps[name] = color_ts
            _record_history(name, color_ts, color, frame_number)

## Previously zoomed the top camera to crop out background distractors.
## Not used anymore.
def digital_zoom(frame, zoom=1.6):
    h, w = frame.shape[:2]
    new_w = int(w / zoom)
    new_h = int(h / zoom)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
