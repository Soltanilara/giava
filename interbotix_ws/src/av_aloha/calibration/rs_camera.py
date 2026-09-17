"""RealSense access for the calibration tools.

Camera identity comes from ``camera_manager.CAMERA_SERIALS`` -- the same
table data collection uses -- so a camera is always addressed by the name
the rest of GIAVA knows it by, and the serial is carried into every
artifact.

This module deliberately does NOT touch ``camera_manager``'s threaded
latest-frame-wins capture loop.  That loop is right for teleoperation
(never block the control tick) and wrong for calibration, where we need a
specific frame together with its own metadata rather than whatever is
newest.  What is shared is the identity table and the stream settings.


TIMESTAMPS -- read this before interpreting any timing result
=============================================================
librealsense exposes several clocks and they are NOT interchangeable:

``get_timestamp()``            The frame's timestamp, in whatever domain
                               ``get_frame_timestamp_domain()`` reports.
                               Units: milliseconds.
``timestamp_domain.hardware_clock``
                               Device clock.  Monotonic per device, with
                               an arbitrary origin.  Two DIFFERENT devices'
                               hardware clocks are unrelated -- comparing
                               them directly is meaningless.
``timestamp_domain.system_time``
                               Host clock, stamped when the frame arrived.
                               Includes USB transfer and driver latency.
``timestamp_domain.global_time``
                               Device clock LINEARLY MAPPED onto the host
                               epoch by librealsense's global time sync.
                               This makes cross-device comparison possible,
                               but it is an ESTIMATE from a running fit --
                               not hardware synchronisation.
``sensor_timestamp`` (metadata)
                               Device time of the start of exposure, in
                               microseconds.  The closest thing to "when
                               the photons landed".
``frame_timestamp`` (metadata) Device time the frame was assembled, µs.
``backend_timestamp`` (metadata)
                               Host time the kernel driver got it, ms.
``time_of_arrival`` (metadata) Host time librealsense got it, ms.

None of these proves hardware synchronisation.  The D405 has no external
sync wiring in this rig, so any cross-camera agreement is a property of
the software clocks, not of the shutters.  Phase 7 records all of them
separately and never collapses them into one number.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import pyrealsense2 as rs
except ImportError:  # keep import-time failures readable
    rs = None

## The production camera identity table -- single source of truth.
from camera_manager import CAMERA_SERIALS  # noqa: E402

## What data collection actually streams (camera_manager.py:268).  Matching
## it matters: intrinsics are resolution-specific, and a calibration taken
## at another resolution does not transfer without rescaling.
PRODUCTION_COLOR = (640, 480, 60, "rgb8")


def require_rs() -> None:
    if rs is None:
        raise ImportError(
            "pyrealsense2 is not available in this environment.\n"
            "  The validated env is gym_av312:\n"
            "    conda run -n gym_av312 python calibration/<script>.py")


## ------------------------------------------------------------------ ##
## Device discovery
## ------------------------------------------------------------------ ##

def connected_devices() -> List[Dict[str, str]]:
    """Every RealSense currently attached, with its identifying info."""
    require_rs()
    out = []
    for d in rs.context().query_devices():
        def info(key, default=""):
            try:
                return d.get_info(key)
            except Exception:
                return default
        out.append({
            "name": info(rs.camera_info.name),
            "serial": info(rs.camera_info.serial_number),
            "firmware": info(rs.camera_info.firmware_version),
            "product_line": info(rs.camera_info.product_line),
            "physical_port": info(rs.camera_info.physical_port),
            "usb_type": info(rs.camera_info.usb_type_descriptor),
        })
    return sorted(out, key=lambda r: r["serial"])


def serial_for(camera: str) -> str:
    """Map a GIAVA camera name to its serial; pass a serial through."""
    if camera in CAMERA_SERIALS:
        return CAMERA_SERIALS[camera]
    if camera.isdigit():
        return camera
    raise KeyError(
        f"unknown camera '{camera}'. Known names: "
        f"{sorted(CAMERA_SERIALS)}; or give a serial number.")


def name_for(serial: str) -> Optional[str]:
    for n, s in CAMERA_SERIALS.items():
        if s == serial:
            return n
    return None


def resolve(cameras: Sequence[str]) -> List[Tuple[str, str]]:
    """[(name, serial)] for the requested cameras, validated as connected."""
    present = {d["serial"] for d in connected_devices()}
    out = []
    for c in cameras:
        s = serial_for(c)
        if s not in present:
            raise RuntimeError(
                f"camera '{c}' (serial {s}) is not connected.\n"
                f"  Connected: "
                f"{[(d['serial'], name_for(d['serial'])) for d in connected_devices()]}")
        out.append((name_for(s) or c, s))
    return out


## ------------------------------------------------------------------ ##
## Frame records
## ------------------------------------------------------------------ ##

@dataclass
class FrameRecord:
    """One captured frame plus every clock we can attach to it."""
    camera: str
    serial: str
    stream: str
    image: Optional[np.ndarray] = None
    frame_number: Optional[int] = None
    ## The librealsense timestamp and, crucially, WHICH CLOCK it is on.
    timestamp_ms: Optional[float] = None
    timestamp_domain: Optional[str] = None
    ## Device clocks (microseconds, device origin).
    sensor_timestamp_us: Optional[float] = None
    frame_timestamp_us: Optional[float] = None
    ## Host clocks (milliseconds).
    backend_timestamp_ms: Optional[float] = None
    time_of_arrival_ms: Optional[float] = None
    frame_counter: Optional[int] = None
    ## Host clocks stamped by US, immediately after the frame was handed over.
    host_epoch_s: Optional[float] = None
    host_monotonic_s: Optional[float] = None
    ## Depth ALIGNED TO THE COLOUR FRAME, in metres, same HxW as `image`.
    ## Only populated when the camera was opened with with_depth=True.
    ## Zero means "no return", which is a real and common answer -- it is
    ## NOT a distance and must never be averaged in as one.
    depth_m: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def depth_at(self, uv, window: int = 2) -> Optional[float]:
        """Median valid depth in a small window about a pixel, in metres.

        A window rather than the single pixel: D405 depth is per-pixel
        noisy and drops out entirely on specular or dark surfaces, so one
        sample is often 0. The median over valid returns only is robust to
        both. Returns None when nothing in the window came back -- which is
        the honest answer, and is why this is a cross-check on
        triangulation rather than a replacement for it."""
        if self.depth_m is None:
            return None
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        h, w = self.depth_m.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None
        patch = self.depth_m[max(v - window, 0):v + window + 1,
                             max(u - window, 0):u + window + 1]
        valid = patch[patch > 0]
        return float(np.median(valid)) if valid.size else None

    def timing_dict(self) -> Dict[str, Any]:
        """Everything except the image -- what the sync log stores."""
        return {
            "camera": self.camera,
            "serial": self.serial,
            "stream": self.stream,
            "frame_number": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "timestamp_domain": self.timestamp_domain,
            "sensor_timestamp_us": self.sensor_timestamp_us,
            "frame_timestamp_us": self.frame_timestamp_us,
            "backend_timestamp_ms": self.backend_timestamp_ms,
            "time_of_arrival_ms": self.time_of_arrival_ms,
            "frame_counter": self.frame_counter,
            "host_epoch_s": self.host_epoch_s,
            "host_monotonic_s": self.host_monotonic_s,
        }


def _metadata(frame, key) -> Optional[float]:
    """Read one metadata field, or None when the device does not supply it.

    Availability varies by kernel/driver build, so every read is guarded."""
    try:
        if frame.supports_frame_metadata(key):
            return float(frame.get_frame_metadata(key))
    except Exception:
        pass
    return None


def frame_to_record(frame, camera: str, serial: str, stream: str,
                    with_image: bool = True) -> FrameRecord:
    host_epoch = time.time()
    host_mono = time.perf_counter()
    rec = FrameRecord(camera=camera, serial=serial, stream=stream,
                      host_epoch_s=host_epoch, host_monotonic_s=host_mono)
    try:
        rec.frame_number = int(frame.get_frame_number())
    except Exception:
        pass
    try:
        rec.timestamp_ms = float(frame.get_timestamp())
        rec.timestamp_domain = str(frame.get_frame_timestamp_domain())
    except Exception:
        pass
    rec.sensor_timestamp_us = _metadata(frame, rs.frame_metadata_value.sensor_timestamp)
    rec.frame_timestamp_us = _metadata(frame, rs.frame_metadata_value.frame_timestamp)
    rec.backend_timestamp_ms = _metadata(frame, rs.frame_metadata_value.backend_timestamp)
    rec.time_of_arrival_ms = _metadata(frame, rs.frame_metadata_value.time_of_arrival)
    fc = _metadata(frame, rs.frame_metadata_value.frame_counter)
    rec.frame_counter = int(fc) if fc is not None else None
    if with_image:
        rec.image = np.asanyarray(frame.get_data()).copy()
    return rec


## ------------------------------------------------------------------ ##
## Intrinsics
## ------------------------------------------------------------------ ##

def intrinsics_dict(intr) -> Dict[str, Any]:
    """rs.intrinsics -> plain dict, with an OpenCV-shaped K alongside.

    The distortion coefficient count is model dependent; librealsense
    always hands back 5 slots, so the trailing zeros for e.g. a
    brown_conrady model are real values, not padding to be trimmed."""
    K = [[intr.fx, 0.0, intr.ppx],
         [0.0, intr.fy, intr.ppy],
         [0.0, 0.0, 1.0]]
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        ## cx/cy are aliases for ppx/ppy -- named both ways because the
        ## OpenCV side of this pipeline says cx/cy and the RealSense side
        ## says ppx/ppy, and confusing them is a classic silent bug.
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(c) for c in intr.coeffs],
        "camera_matrix": K,
        "fov_deg": [float(v) for v in rs.rs2_fov(intr)],
    }


def extrinsics_dict(ext, frm: str, to: str) -> Dict[str, Any]:
    """rs.extrinsics -> dict, with the direction stated explicitly.

    librealsense's rotation is COLUMN-major; it is transposed here into a
    row-major 3x3 so the stored matrix follows this package's convention."""
    R = np.asarray(ext.rotation, dtype=float).reshape(3, 3).T
    t = np.asarray(ext.translation, dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return {
        "name": f"T_{to}_{frm}",
        "convention": (
            f"T_{to}_{frm} maps points from the '{frm}' stream frame into "
            f"the '{to}' stream frame: p_{to} = T @ p_{frm}. Source: "
            f"librealsense rs2_get_extrinsics(from={frm}, to={to}), whose "
            f"rotation is column-major and has been transposed here."
        ),
        "matrix_4x4_row_major": T.tolist(),
        "translation_m": t.tolist(),
    }


def who_holds_the_cameras():
    """PIDs and command lines of processes holding any /dev/video* node.

    A RealSense can only be opened by one process at a time.  When a second
    one tries, librealsense reports a bare 'Device or resource busy' with no
    hint as to who has it -- which is the single most common way these
    tools fail, and the least self-explanatory."""
    import glob
    import subprocess

    holders = {}
    for dev in sorted(glob.glob("/dev/video*")):
        try:
            out = subprocess.run(["fuser", dev], capture_output=True,
                                 text=True, timeout=5)
        except Exception:
            return {}
        for pid in out.stdout.split():
            if pid.isdigit():
                holders.setdefault(pid, set()).add(dev)
    info = []
    for pid, devs in holders.items():
        try:
            cmd = subprocess.run(["ps", "-p", pid, "-o", "cmd="],
                                 capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        except Exception:
            cmd = "(unknown)"
        info.append({"pid": pid, "cmd": cmd, "n_devices": len(devs)})
    return info


def reset_all_devices(wait_s: float = 10.0) -> int:
    """Issue a hardware reset to every connected RealSense and wait.

    The recovery for a wedged device: the pipeline starts, the ioctls
    succeed, and no frames ever arrive.  That state is usually left behind
    by a process that died without closing its pipeline cleanly, and it
    survives closing every handle -- nothing is holding the device, it is
    simply stuck.  A device-level reset clears it without root, unlike
    unplugging or a USB-level reset.

    Returns the number of devices that came back."""
    require_rs()
    before = [d.get_info(rs.camera_info.serial_number)
              for d in rs.context().query_devices()]
    print(f"  resetting {len(before)} RealSense device(s)...")
    for d in rs.context().query_devices():
        try:
            d.hardware_reset()
        except Exception as e:
            print(f"    {d.get_info(rs.camera_info.serial_number)}: "
                  f"reset failed ({e})")
    deadline = time.time() + wait_s
    n = 0
    while time.time() < deadline:
        time.sleep(1.0)
        n = len(list(rs.context().query_devices()))
        if n >= len(before):
            break
    print(f"  {n}/{len(before)} device(s) re-enumerated")
    return n


def _start_failure_hint(err: str) -> str:
    """Turn librealsense's terse start errors into the actual next step."""
    low = err.lower()
    if "busy" in low or "errno=16" in low:
        lines = [
            "  ANOTHER PROCESS ALREADY HAS THIS CAMERA.",
            "  A RealSense can only be opened by one process at a time.",
        ]
        holders = who_holds_the_cameras()
        if holders:
            lines.append("  Currently holding /dev/video* nodes:")
            for h in holders:
                lines.append(f"      PID {h['pid']}  ({h['n_devices']} "
                             f"device nodes)")
                lines.append(f"          {h['cmd'][:110]}")
            lines.append("  Stop that process (Ctrl-C in its terminal so it "
                         "shuts down cleanly)")
            lines.append("  and re-run. This is NOT a USB or bandwidth "
                         "problem.")
        else:
            lines.append("  Could not identify the holder (is `fuser` "
                         "installed?). Look for a")
            lines.append("  running world_view.py / capture_session.py / "
                         "data_collection.py.")
        return "\n".join(lines)
    if "resolve requests" in low:
        return ("  'Couldn't resolve requests' means the requested "
                "resolution/format/fps\n"
                "  is not available -- usually a USB2 link. Check `lsusb -t` "
                "shows 5000M\n  for this device.")
    return ("  Check the camera is connected and not held by another "
            "process\n  (`fuser /dev/video*`).")


## ------------------------------------------------------------------ ##
## A calibration-oriented camera handle
## ------------------------------------------------------------------ ##

class RealSenseCamera:
    """One RealSense opened for deterministic single-frame capture.

    Use as a context manager:

        with RealSenseCamera("right_wrist") as cam:
            rec = cam.capture()
    """

    def __init__(self, camera: str,
                 width: int = PRODUCTION_COLOR[0],
                 height: int = PRODUCTION_COLOR[1],
                 fps: int = PRODUCTION_COLOR[2],
                 stream: str = "color",
                 global_time: Optional[bool] = None,
                 warmup_frames: int = 15,
                 with_depth: bool = False):
        require_rs()
        self.requested = camera
        self.serial = serial_for(camera)
        self.name = name_for(self.serial) or camera
        self.width, self.height, self.fps = width, height, fps
        self.stream_name = stream
        self.global_time = global_time
        self.warmup_frames = warmup_frames
        ## Depth alongside colour, aligned to it. Off by default and
        ## deliberately so: the production record loop is RGB-only, and
        ## enabling depth here must not quietly change what that path
        ## does. This exists for measurement cross-checks.
        self.with_depth = bool(with_depth) and stream == "color"
        self.align = None
        self.depth_scale = None
        self.depth_intrinsics = None
        self.pipeline = None
        self.profile = None
        self.intrinsics: Optional[Dict[str, Any]] = None
        self.device_info: Dict[str, str] = {}

    # -------------------------------------------------------------- #
    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> "RealSenseCamera":
        cfg = rs.config()
        cfg.enable_device(self.serial)
        if self.stream_name == "color":
            cfg.enable_stream(rs.stream.color, self.width, self.height,
                              rs.format.rgb8, self.fps)
            if self.with_depth:
                cfg.enable_stream(rs.stream.depth, self.width, self.height,
                                  rs.format.z16, self.fps)
        elif self.stream_name == "depth":
            cfg.enable_stream(rs.stream.depth, self.width, self.height,
                              rs.format.z16, self.fps)
        elif self.stream_name in ("infrared", "infrared1"):
            cfg.enable_stream(rs.stream.infrared, 1, self.width, self.height,
                              rs.format.y8, self.fps)
        else:
            raise ValueError(f"unsupported stream '{self.stream_name}'")

        self.pipeline = rs.pipeline()
        try:
            self.profile = self.pipeline.start(cfg)
        except RuntimeError as e:
            raise RuntimeError(
                f"could not start '{self.name}' (serial {self.serial}) at "
                f"{self.width}x{self.height}@{self.fps} "
                f"{self.stream_name}:\n  {e}\n" + _start_failure_hint(str(e))
            ) from e

        dev = self.profile.get_device()
        self.device_info = {
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version),
            "product_line": dev.get_info(rs.camera_info.product_line),
            "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor),
        }

        ## Global time maps the device clock onto the host epoch. It changes
        ## what get_timestamp() MEANS, so it is set explicitly (when asked)
        ## and always reported.
        if self.global_time is not None:
            for sensor in dev.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    try:
                        sensor.set_option(rs.option.global_time_enabled,
                                          1.0 if self.global_time else 0.0)
                    except Exception as e:
                        print(f"[{self.name}] could not set global_time: {e}")

        sp = self.profile.get_stream(
            getattr(rs.stream, "infrared" if "infrared" in self.stream_name
                    else self.stream_name)).as_video_stream_profile()
        self.active_profile = sp
        self.intrinsics = intrinsics_dict(sp.get_intrinsics())

        if self.with_depth:
            ## Align depth INTO the colour frame, so depth_m[v, u] belongs
            ## to colour pixel (u, v) with no further bookkeeping. Without
            ## this the two streams have different intrinsics and a
            ## different origin, and indexing one with the other's pixel is
            ## wrong by tens of millimetres at the image edge.
            self.align = rs.align(rs.stream.color)
            for sensor in dev.query_sensors():
                if sensor.is_depth_sensor():
                    self.depth_scale = float(
                        sensor.as_depth_sensor().get_depth_scale())
                    break
            try:
                self.depth_intrinsics = intrinsics_dict(
                    self.profile.get_stream(rs.stream.depth)
                    .as_video_stream_profile().get_intrinsics())
            except Exception:
                self.depth_intrinsics = None

        ## Auto-exposure needs a few frames to settle; a calibration image
        ## taken during the ramp is darker and noisier than the rest.
        for _ in range(self.warmup_frames):
            try:
                self.pipeline.wait_for_frames(5000)
            except Exception:
                break
        return self

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

    # -------------------------------------------------------------- #
    def capture(self, timeout_ms: int = 5000,
                with_image: bool = True) -> FrameRecord:
        """Block for the next frame and return it with all its clocks."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if self.align is not None:
            frames = self.align.process(frames)
        frame = (frames.get_color_frame() if self.stream_name == "color"
                 else frames.get_depth_frame() if self.stream_name == "depth"
                 else frames.get_infrared_frame(1))
        rec = frame_to_record(frame, self.name, self.serial,
                              self.stream_name, with_image)
        rec.extra["global_time_setting"] = self.global_time
        if self.align is not None and with_image:
            d = frames.get_depth_frame()
            if d:
                ## Metres. The scale is per-device, not a constant.
                rec.depth_m = (np.asanyarray(d.get_data()).astype(np.float32)
                               * (self.depth_scale or 0.0))
                rec.extra["depth_scale"] = self.depth_scale
                rec.extra["depth_aligned_to"] = "color"
        return rec

    def global_time_state(self) -> Optional[bool]:
        """What global_time_enabled actually is on the streaming sensor."""
        if self.profile is None:
            return None
        for sensor in self.profile.get_device().query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                try:
                    return bool(sensor.get_option(rs.option.global_time_enabled))
                except Exception:
                    return None
        return None

    def stream_metadata(self) -> Dict[str, Any]:
        """Everything needed to identify what this calibration belongs to."""
        return {
            "camera": self.name,
            "requested_as": self.requested,
            "serial": self.serial,
            "device": self.device_info,
            "stream": self.stream_name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "format": ("rgb8" if self.stream_name == "color"
                       else "z16" if self.stream_name == "depth" else "y8"),
            "global_time_enabled": self.global_time_state(),
            "depth_aligned_to_color": self.with_depth,
            "depth_scale": self.depth_scale,
        }
