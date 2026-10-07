"""
Headset link over the gvlink UDP stack -- a drop-in replacement for WebRTCHeadset.

The Unity viewer moved off WebRTC/Firestore (av-aloha-unity branch v2) onto a direct
UDP transport: fragmented H.264 video, a TCP control channel, and a fixed-size input
uplink at display rate. Nothing is negotiated through the cloud and there is no
signalling handshake -- the headset finds this process by LAN beacon, or is given its
address, opens the control channel, and video starts flowing to wherever that
connection came from.

This class keeps WebRTCHeadset's public surface exactly:

    headset = GvLinkHeadset()
    headset.run_in_thread()
    while not headset.data_channel_open: ...
    headset.send_images(left_bgr, right_bgr)
    data = headset.receive_data()          # HeadsetData, or None
    headset.send_feedback(feedback)
    headset.close()

so data_collection.py, camera_manager.py and oak_to_headset.py only need the import
swapped. Everything below that line is different.

Two behaviours worth knowing, because they differ from the WebRTC version:

*   send_images() does not encode on the calling thread. It drops the pair into a
    one-slot mailbox and returns; an encoder thread picks up the newest pair and sends
    it. The OAK capture loop therefore never waits on H.264, and if capture outruns the
    encoder the intermediate pairs are discarded rather than queued -- a frame from two
    captures ago is worth less than the current one.

*   The viewer is told the real camera geometry (rectified intrinsics + baseline) over
    the control channel, and places each image from it. That replaces the operator
    guessing a field of view in the headset settings. Pass a calibration JSON if you
    have one; otherwise a symmetric pinhole is synthesised from `hfov`, which is a
    starting point rather than the truth.

gvlink itself lives in the Unity repo next to the viewer that has to agree with it
byte for byte, so it is imported rather than vendored -- one copy, no drift. Point
GVLINK_PATH at it if it is not in the default place.
"""

from __future__ import annotations

import dataclasses
import os
import socket
import sys
import threading
import time

import numpy as np

# The live checkout: ~/av-aloha-unity is where the v2 branch with the OAK colour fix
# and the v3 input protocol actually lives.  (~/dev/av-aloha-unity is a stale
# WebRTC-era clone that predates gvlink entirely -- importing from there fails.)
_DEFAULT_GVLINK = os.path.expanduser("~/av-aloha-unity/Guided-Vision/python")
_GVLINK_PATH = os.environ.get("GVLINK_PATH", _DEFAULT_GVLINK)
if _GVLINK_PATH not in sys.path:
    sys.path.insert(0, _GVLINK_PATH)

try:
    from gvlink.beacon import Beacon, build_payload
    from gvlink.camera import CameraParams
    from gvlink.foveal import MIN_CANVAS, AtlasLayout, SaccadeWidener
    from gvlink.protocol import (BUTTON_ONE, BUTTON_STICK, BUTTON_TWO, CODEC_H264,
                                 CODEC_MJPEG, DEFAULT_PORTS, EYE_LEFT, EYE_RIGHT,
                                 INPUT_LEFT_VALID, INPUT_RIGHT_VALID, MTU_PAYLOAD,
                                 MTU_PAYLOAD_TUNNEL, HeadsetInput)
    from gvlink.ratecontrol import BitrateController
    from gvlink.robotlink import RobotLink
    from gvlink.stream import EyeStreamSender, make_udp_socket
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        f"could not import gvlink from {_GVLINK_PATH!r}: {exc}\n"
        "gvlink ships with the Unity viewer (av-aloha-unity, branch v2) at "
        "Guided-Vision/python. Set GVLINK_PATH to that directory."
    ) from exc

from headset_utils import HeadsetData, HeadsetFeedback, convert_left_to_right_coordinates


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _tighten(layout: "AtlasLayout") -> "AtlasLayout":
    try:
        return layout.tightened()
    except Exception:
        return layout


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


## How old the newest input packet may be before receive_data() calls it nothing.
##
## Deliberately separate from the 0.5 s default _fresh_input() keeps for the gaze
## path: there, a brief hiccup falling back to a centred fovea is a visible quality
## flicker for no gain, so leniency is right.  On the CONTROL path leniency is the
## opposite of right -- 0.5 s at the headset's 90 Hz uplink is ~45 missed packets
## driven on stale poses before anything notices.  0.15 s is ~13.
CONTROL_STALE_S = _env_float("GIAVA_CONTROL_STALE_S", 0.15)


class GvLinkHeadset:
    """Robot-side end of the gvlink connection to the Unity viewer."""

    def __init__(self,
                 # --- accepted and ignored, so existing call sites still construct ---
                 serviceAccountKeyFile: str | None = None,
                 signalingSettingsFile: str | None = None,
                 video_buffer_size: int = 1,
                 data_buffer_size: int = 1,
                 send_data_freq: int = 10,
                 # --- gvlink configuration ------------------------------------------
                 name: str | None = None,
                 fps: int | None = None,
                 bitrate_kbps: int | None = None,
                 min_bitrate_kbps: int | None = None,
                 canvas: tuple[int, int] | None = None,
                 coarse_scale: float | None = None,
                 fovea_scale: float | None = None,
                 foveation: bool | None = None,
                 codec: str | None = None,
                 calib: str | None = None,
                 camera_params: dict | None = None,
                 hfov_deg: float | None = None,
                 baseline_m: float | None = None,
                 rectified: bool = True,
                 src_size: tuple[int, int] | None = None,
                 saccade_zoom: float | None = None,
                 beacon: bool = True,
                 tunnel: bool | None = None,
                 adapt_bitrate: bool = True,
                 ports: dict | None = None) -> None:

        if serviceAccountKeyFile or signalingSettingsFile:
            print("[headset] gvlink needs no Firestore credentials; "
                  "ignoring serviceAccountKey/signalingSettings.")

        self.name = name or os.environ.get("GIAVA_ROBOT_NAME", socket.gethostname())
        self.fps = fps if fps is not None else _env_int("GIAVA_HEADSET_FPS", 30)
        self.bitrate_kbps = (bitrate_kbps if bitrate_kbps is not None
                             else _env_int("GIAVA_HEADSET_BITRATE_KBPS", 12000))
        self.min_bitrate_kbps = (min_bitrate_kbps if min_bitrate_kbps is not None
                                 else _env_int("GIAVA_HEADSET_MIN_KBPS", 800))
        ## GIAVA_HEADSET_ADAPT=0 pins the bitrate at bitrate_kbps and ignores
        ## viewer reports entirely.  Adaptation earns its place on a link whose
        ## capacity is unknown or shared; on a quiet LAN it is solving a
        ## problem that does not exist, and its delay heuristic misfires (see
        ## GIAVA_RATE_DELAY_MS below).  There is no safety argument for keeping
        ## it on here -- the failure mode of too much bitrate is dropped
        ## fragments, which the viewer already reports and you can see.
        if adapt_bitrate and os.environ.get("GIAVA_HEADSET_ADAPT", "1") in ("0", "false", "False"):
            adapt_bitrate = False
        self.adapt_bitrate = adapt_bitrate

        canvas = canvas or (_env_int("GIAVA_CANVAS_W", 1024), _env_int("GIAVA_CANVAS_H", 1024))
        self.foveation = (foveation if foveation is not None
                          else os.environ.get("GIAVA_FOVEATION", "1") not in ("0", "false", "False"))
        _layout = AtlasLayout(
            canvas_w=canvas[0], canvas_h=canvas[1],
            coarse_scale=(coarse_scale if coarse_scale is not None
                          else _env_float("GIAVA_COARSE_SCALE", 0.35)),
            fovea_scale=(fovea_scale if fovea_scale is not None
                         else _env_float("GIAVA_FOVEA_SCALE", 0.5)))
        # tightened() trims the atlas to what the layers actually occupy; the reference
        # sender does the same, and it is worth roughly 2x encode and 14% bitrate.
        #
        # ONLY ON THE FOVEATED PATH.  Tightening is sound there because the two layers
        # keep their exact pixel sizes -- it scales the canvas by
        # max(coarse_scale, fovea_scale) and divides both scales by the same factor, so
        # every scale*dimension product is unchanged and the sender simply declines to
        # encode black padding.
        #
        # With foveation OFF there are no layers.  build_atlas takes its `gaze is None`
        # branch and returns `_downscale(src, canvas_w, canvas_h)` -- the whole image
        # IS the whole canvas, and the scales are never read.  Tightening then shrinks
        # the picture itself: at the shipping defaults (0.35 / 0.5) it halved a
        # 1024x1024 canvas to 512x512, so `--fovea off` downscaled the OAK's native
        # 1280x800 into 512x512 and looked far worse than the foveated stream it
        # replaced.  That is not a quality setting anyone chose; it is the two-band
        # assumption applied where it does not hold.
        self.layout = _tighten(_layout) if self.foveation else _layout

        codec = (codec or os.environ.get("GIAVA_HEADSET_CODEC", "h264")).lower()
        self.codec = CODEC_MJPEG if codec == "mjpeg" else CODEC_H264

        # Tailscale/WireGuard's 1280-byte tunnel would IP-fragment every datagram.
        use_tunnel = (tunnel if tunnel is not None
                      else os.environ.get("GIAVA_TUNNEL", "0") not in ("0", "false", "False"))
        self.mtu_payload = MTU_PAYLOAD_TUNNEL if use_tunnel else MTU_PAYLOAD

        self.ports = dict(DEFAULT_PORTS)
        if ports:
            self.ports.update(ports)

        self._calib = calib or os.environ.get("GIAVA_HEADSET_CALIB") or None
        # Already-rectified intrinsics, e.g. camera_manager.OAK_GVLINK_CAMERA["params"].
        # Preferred over everything else: it describes the frames actually being sent.
        self._camera_wire = camera_params
        self._hfov = hfov_deg if hfov_deg is not None else _env_float("GIAVA_HEADSET_HFOV", 90.0)
        # Fallback only -- used when no measured calibration reaches the viewer
        # (no OAK, or its params could not be derived).  0.062 m is the OAK's
        # nominal stereo baseline; the checkerboard calibration measures the
        # RECTIFIED baseline at 64.3 mm, and when those params are available
        # they take precedence over this number entirely (see camera()).
        self._baseline = (baseline_m if baseline_m is not None
                          else _env_float("GIAVA_HEADSET_BASELINE", 0.062))
        self._rectified = rectified
        # The viewer connects before the camera has produced anything, and it needs the
        # geometry to place the very first frame -- so this is configuration, not
        # something discovered from a frame. If the first frame disagrees, it is
        # corrected and republished.
        self._src_wh = src_size or (_env_int("GIAVA_SRC_W", 1280), _env_int("GIAVA_SRC_H", 800))
        self.camera: CameraParams | None = None

        self._widener = SaccadeWidener(
            max_zoom=(saccade_zoom if saccade_zoom is not None
                      else _env_float("GIAVA_SACCADE_ZOOM", 2.5)))
        ## GIAVA_HEADSET_STATS=<seconds>: periodic link telemetry.  Off by
        ## default because the terminal is also carrying episode feedback, but
        ## it is the only way to tell an ENCODER-limited picture from a
        ## LINK-limited one.  adapt_bitrate cuts the target by 25% on 2%
        ## fragment loss or 40 ms of queueing and only raises it 8% a second,
        ## so a link pushed past what it can carry sits far below the
        ## requested bitrate almost all the time -- and looks exactly like a
        ## bitrate that was never raised.
        self._stats_every = _env_float("GIAVA_HEADSET_STATS", 0.0)
        ## Congestion threshold, in ms of queueing above the rolling minimum.
        ##
        ## THE DEFAULT IS DANGEROUSLY CLOSE TO ONE FRAME PERIOD.  40 ms IS one
        ## frame at 25 fps, and `latency_ms` is (viewer clock - robot capture
        ## stamp) sampled per report, so ordinary frame pacing and decode
        ## jitter land right on the threshold.  Combined with a baseline that
        ## is a rolling MINIMUM decaying only 2 ms/s, one lucky low sample
        ## pins the floor and every normal sample afterwards reads as ~60 ms
        ## of "queue" -- congestion that is not there.
        ##
        ## The controller then loses by construction: cuts are multiplicative
        ## (x0.75, as often as every 0.35 s) and raises are gentler (x1.08, at
        ## most once a second), so a false positive every few seconds walks the
        ## target down to min_kbps and pins it there.  Observed on this rig at
        ## 25 fps over LAN: 43 cuts / 33 raises, target stuck at the 800 kbps
        ## floor against a requested 25000.
        _delay_ms = _env_float("GIAVA_RATE_DELAY_MS", 40.0)
        self._rate = BitrateController(start_kbps=self.bitrate_kbps,
                                       min_kbps=self.min_bitrate_kbps,
                                       max_kbps=max(self.bitrate_kbps, self.min_bitrate_kbps),
                                       delay_rise_ms=_delay_ms)

        self._want_beacon = beacon
        self._beacon: Beacon | None = None
        self._link: RobotLink | None = None
        self._sock = None
        self._senders: dict = {}
        self._active = None          # (dest, codec) the senders were built for

        # One slot, newest wins. See the class docstring.
        self._pending: tuple | None = None
        self._pending_lock = threading.Lock()
        self._frame_event = threading.Event()

        self._input_sock = None
        self._input_lock = threading.Lock()
        self._input_pkt: HeadsetInput | None = None
        self._input_at = 0.0
        self.input_rate_hz = 0.0

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        self.frames_sent = 0
        self.frames_dropped = 0

    # ---------------------------------------------------------------- lifecycle

    @property
    def data_channel_open(self) -> bool:
        """True once the viewer has opened the control channel."""
        return bool(self._link is not None and self._link.connected)

    def run_in_thread(self) -> "GvLinkHeadset":
        self._sock = make_udp_socket()

        self._link = RobotLink(self.ports["control"], self.name)
        self._link.start()
        self._link.subscribe("viewer/stats", self._on_viewer_stats)

        # The viewer needs the camera geometry before it can place the first frame, so
        # answer on request and push it the moment a session appears.
        @self._link.handler("camera/params")
        def _params(_data):
            cam = self._ensure_camera()
            return cam.to_wire() if cam else None

        self._link.on_session(self._on_session)

        self._input_sock = make_udp_socket(("0.0.0.0", self.ports["input"]))
        self._input_sock.settimeout(0.2)

        self._spawn(self._input_loop, "gv-input")
        self._spawn(self._encode_loop, "gv-encode")

        if self._want_beacon:
            self._start_beacon()

        if os.environ.get("GIAVA_MEMDEBUG") == "1":
            self._spawn(self._memdebug_loop, "gv-memdebug")
        if self._stats_every > 0:
            self._spawn(self._stats_loop, "gv-stats")

        print(f"[headset] gvlink up as '{self.name}': control :{self.ports['control']}, "
              f"video :{self.ports['video']}, input :{self.ports['input']}"
              + (f", beacon :{self.ports['beacon']}" if self._beacon else ""))
        print(f"[headset] {self.fps} fps cap, {self.bitrate_kbps} kbps/eye, "
              f"canvas {self.layout.canvas_w}x{self.layout.canvas_h}, "
              f"fovea={'on' if self.foveation else 'off'}, "
              f"codec={'mjpeg' if self.codec == CODEC_MJPEG else 'h264'}, "
              f"rate={'adaptive' if self.adapt_bitrate else 'FIXED'}"
              + (f" (floor {self.min_bitrate_kbps}, congestion at "
                 f"{self._rate.delay_rise_ms:.0f} ms)" if self.adapt_bitrate else ""))
        return self

    def _stats_loop(self) -> None:
        """GIAVA_HEADSET_STATS=<seconds>: one line of link telemetry per tick.

        Reports what the rate controller has actually settled on, not what was
        requested.  A target sitting well under `bitrate_kbps` with a rising
        `cuts` count means the link is dropping fragments and the picture is
        link-limited -- lowering the requested bitrate will IMPROVE it, because
        the controller stops thrashing.  A target pinned at the requested value
        with a grainy picture means the opposite: the link is fine and the
        encoder is the limit (see GIAVA_X264_PRESET)."""
        while not self._stop.is_set():
            if self._stop.wait(self._stats_every):
                break
            print(f"[headset] {self.describe()}")

    def _memdebug_loop(self) -> None:
        """GIAVA_MEMDEBUG=1: periodic tracemalloc snapshot diff, to find what is
        actually growing rather than guessing from code review."""
        import tracemalloc
        tracemalloc.start(15)
        prev = tracemalloc.take_snapshot()
        while not self._stop.is_set():
            if self._stop.wait(15.0):
                break
            snap = tracemalloc.take_snapshot()
            top = snap.compare_to(prev, "lineno")
            print("[memdebug] top growth since last sample:")
            for stat in top[:12]:
                print(f"  {stat}")
            prev = snap

    def _spawn(self, fn, name: str) -> None:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _start_beacon(self) -> None:
        w, h = self._src_wh
        payload = build_payload(
            self.name,
            [{"id": "oak", "w": w, "h": h, "fps": self.fps,
              "canvasW": self.layout.canvas_w, "canvasH": self.layout.canvas_h,
              "codec": "mjpeg" if self.codec == CODEC_MJPEG else "h264"}],
            ports={k: self.ports[k] for k in ("control", "video", "input")},
            foveation=self.foveation)
        try:
            self._beacon = Beacon(payload, port=self.ports["beacon"]).start()
        except OSError as exc:
            print(f"[headset] beacon could not start ({exc}); "
                  "type the robot address in the headset instead")

    def close(self) -> None:
        self._stop.set()
        self._frame_event.set()
        for obj in (self._beacon, self._link):
            try:
                if obj is not None:
                    obj.stop()
            except Exception:
                pass
        for sock in (self._input_sock, self._sock):
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=1.0)

    # ------------------------------------------------------------------- camera

    def _ensure_camera(self) -> CameraParams | None:
        """Built once, from the calibration if given, else synthesised from hfov."""
        if self.camera is not None:
            return self.camera
        w, h = self._src_wh
        if self._camera_wire:
            try:
                self.camera = CameraParams.from_wire(self._camera_wire)
            except Exception as exc:
                print(f"[headset] camera_params did not parse ({exc}); falling back")
        if self.camera is None and self._calib:
            try:
                self.camera = CameraParams.load(self._calib)
            except Exception as exc:
                print(f"[headset] could not load calibration {self._calib!r} ({exc}); "
                      f"synthesising from hfov={self._hfov}")
        if self.camera is None:
            self.camera = CameraParams.from_hfov(w, h, self._hfov, self._baseline)
        # Measured params carry their own honest "rect" flag (an unrectified stream
        # says so); only the synthesised fallback takes it from the constructor,
        # because a guess has no opinion of its own.
        if not self._camera_wire and self.camera.rectified != self._rectified:
            # CameraParams is frozen, so this is a copy rather than a mutation.
            self.camera = dataclasses.replace(self.camera, rectified=self._rectified)
        print(f"[headset] camera: {self.camera.describe()}")
        return self.camera

    def set_camera_params(self, wire: dict | None) -> bool:
        """Adopt measured camera geometry AFTER construction.

        The OAK's rectified intrinsics only exist once the device has been
        opened and its calibration read, which happens well after this object
        is built -- data_collection.py starts the headset thread first so the
        viewer can connect while the cameras come up.  Passing camera_params
        to __init__ therefore hands over whatever was known at construction
        time, which for the real capture path is None, and the viewer silently
        places its images from a synthesised 90-degree pinhole instead of the
        calibration sitting one function call away.

        Returns True if the geometry changed, in which case a connected viewer
        is told immediately."""
        if not wire:
            return False
        if wire == self._camera_wire and self.camera is not None:
            return False
        self._camera_wire = wire
        self.camera = None                       # force _ensure_camera to rebuild
        cam = self._ensure_camera()
        if cam is None:
            return False
        if self._link is not None and self._link.connected:
            self._link.publish("camera/params", cam.to_wire())
        return True

    def _on_session(self, session) -> None:
        if session is None:
            print("[headset] viewer disconnected")
            self._active = None
            return
        print(f"[headset] viewer connected: {session}")
        ## THE ATLAS IS NEGOTIATED, NOT CHOSEN.  _layout_for() honours a
        ## canvas the viewer asks for, and the viewer tightens its own request
        ## before making it -- so the robot's GIAVA_CANVAS_W/H can be silently
        ## discarded and every quality knob upstream of it becomes a no-op.
        ## That is invisible without this line: the startup banner prints the
        ## robot's PREFERENCE, and "streaming to ..." only appears once the
        ## first frame goes out, by which point it has scrolled past whatever
        ## the operator was reading.  Say it here, at the moment it is
        ## decided, and say WHOSE number won.
        _mine, _used = self.layout, self._layout_for(session)
        if (_used.canvas_w, _used.canvas_h) != (_mine.canvas_w, _mine.canvas_h):
            _src_px = self._src_wh or ("?", "?")
            print(f"[headset] ATLAS OVERRIDDEN BY THE VIEWER: it asked for "
                  f"{_used.canvas_w}x{_used.canvas_h}, robot wanted "
                  f"{_mine.canvas_w}x{_mine.canvas_h}. The camera is "
                  f"{_src_px[0]}x{_src_px[1]}, so frames are being resampled "
                  f"to the viewer's size. THE ROBOT CANNOT OVERRIDE THIS: "
                  f"GvVideoSource allocates its decode texture ONCE from the "
                  f"size it requested (CreateExternalTexture(Width, Height)), "
                  f"so a differently-shaped atlas is squashed into that "
                  f"texture and then aspect-corrected again by the quad -- it "
                  f"arrives stretched. Fix it at the viewer: raise its canvas "
                  f"in the start menu (its sizes are SQUARE, so pick the one "
                  f"matching the camera's LONG side).")
        else:
            print(f"[headset] atlas {_used.canvas_w}x{_used.canvas_h} "
                  f"(robot's own; the viewer requested none)")
        cam = self._ensure_camera()
        if cam is not None:
            self._link.publish("camera/params", cam.to_wire())

    def _on_viewer_stats(self, data) -> None:
        if not self.adapt_bitrate or not isinstance(data, dict):
            return
        loss = float(data.get("loss", 0.0))
        lat = data.get("lat")
        want = self._rate.update(time.monotonic(), loss,
                                 float(lat) if lat is not None else None)
        for snd in self._senders.values():
            snd.set_bitrate(want)

    # -------------------------------------------------------------------- video

    def send_images(self, left_image: np.ndarray, right_image: np.ndarray) -> None:
        """Hand over the newest stereo pair. Returns immediately; never blocks."""
        if left_image is None or right_image is None:
            return
        h, w = left_image.shape[:2]
        if (w, h) != self._src_wh:
            print(f"[headset] source is {w}x{h}, not {self._src_wh[0]}x{self._src_wh[1]}; "
                  "rebuilding camera params")
            self._src_wh = (w, h)
            self.camera = None
            cam = self._ensure_camera()
            if cam is not None and self._link is not None and self._link.connected:
                self._link.publish("camera/params", cam.to_wire())
        with self._pending_lock:
            if self._pending is not None:
                self.frames_dropped += 1
            self._pending = (left_image, right_image)
        self._frame_event.set()

    def _encode_loop(self) -> None:
        period = 1.0 / max(1, self.fps)
        next_at = time.monotonic()
        while not self._stop.is_set():
            if not self._frame_event.wait(0.2):
                continue
            self._frame_event.clear()

            with self._pending_lock:
                pair, self._pending = self._pending, None
            if pair is None:
                continue

            # Hold the configured frame rate even if capture is faster; the extra
            # frames were already dropped in the mailbox above.
            now = time.monotonic()
            if now < next_at:
                time.sleep(next_at - now)
                now = time.monotonic()
            next_at = max(now, next_at) + period

            dest = self._destination()
            if dest is None:
                continue
            try:
                self._send_pair(dest, pair, now)
            except Exception as exc:
                print(f"[headset] send failed: {exc}")

    def _destination(self):
        sess = self._link.session if self._link else None
        if sess is None:
            return None
        return (sess.addr, getattr(sess, "video_port", self.ports["video"])), sess

    def _layout_for(self, sess) -> "AtlasLayout":
        """
        The viewer may ask for its own atlas shape -- it is the end that knows its own
        decoder and its own link. A requested canvas is honoured AS GIVEN, not
        re-tightened: the viewer allocates its decode texture at the size it asked for
        and divides every layer span by that number, so changing it underneath would
        leave it sampling the wrong part of the atlas. The clamp is a guard against a
        broken client, not a negotiation.
        """
        if sess is None or not getattr(sess, "canvas", None):
            return self.layout
        try:
            w, h = sess.canvas
            return AtlasLayout(
                max(MIN_CANVAS, min(2048, int(w))), max(MIN_CANVAS, min(2048, int(h))),
                coarse_scale=(sess.coarse_scale if getattr(sess, "coarse_scale", None)
                              else self.layout.coarse_scale),
                fovea_scale=(sess.fovea_scale if getattr(sess, "fovea_scale", None)
                             else self.layout.fovea_scale))
        except (ValueError, TypeError) as exc:
            print(f"[headset] ignoring requested stream shape: {exc}")
            return self.layout

    def _send_pair(self, dest_and_session, pair, now: float) -> None:
        (dest, sess) = dest_and_session
        codec = getattr(sess, "codec", self.codec)
        want_fovea = self.foveation and getattr(sess, "foveation", True)

        layout = self._layout_for(sess)
        shape = (layout.canvas_w, layout.canvas_h, layout.coarse_scale, layout.fovea_scale)
        # Keyed on the session id as well as the settings. A reconnect that lands
        # between two frames never shows up as a None session here, so an identical
        # address and codec looked like nothing had changed -- leaving the encoders
        # running mid-GOP while the headset's freshly built decoder waited for a
        # keyframe that never came. That is a reliable black screen on reconnect.
        key = (getattr(sess, "id", None), dest, codec, shape)
        if self._active != key:
            self._active = key
            # Each EyeStreamSender owns a real libx264 encoder context (EyeEncoder,
            # gvlink/video.py) that only frees cleanly through its own close() --
            # dropping the reference here without draining it leaks one x264 context
            # per abandoned sender.  A flaky link reconnects often (observed: 180
            # cycles inside one session), so this reliably OOMs the process rather
            # than leaking slowly -- it's how a 3 GB process becomes ~60 GB and gets
            # killed mid-teleop.
            for old in self._senders.values():
                enc = getattr(old, "enc", None)
                if enc is not None:
                    enc.close()
            self._senders = {
                eye: EyeStreamSender(self._sock, dest, eye, layout, self.fps,
                                     self._rate.target_kbps, True, codec=codec,
                                     jpeg_quality=85, mtu_payload=self.mtu_payload)
                for eye in (EYE_LEFT, EYE_RIGHT)
            }
            print(f"[headset] streaming to {dest[0]}:{dest[1]} "
                  f"canvas {layout.canvas_w}x{layout.canvas_h} "
                  f"fovea={'on' if want_fovea else 'off'}")

        gl = gr = None
        if want_fovea:
            pkt = self._fresh_input()
            if pkt is not None and pkt.gaze_valid:
                gl, gr = tuple(pkt.gaze_l), tuple(pkt.gaze_r)
            else:
                gl = gr = (0.5, 0.5)

        zoom = self._widener.update(gl, now)
        left, right = pair
        self._senders[EYE_LEFT].send(left, gl, zoom)
        self._senders[EYE_RIGHT].send(right, gr, zoom)
        self.frames_sent += 1

    # -------------------------------------------------------------------- input

    def _input_loop(self) -> None:
        rate_at, rate_n = time.monotonic(), 0
        dropped = 0
        seen_reasons: set = set()
        while not self._stop.is_set():
            try:
                data, addr = self._input_sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            pkt = HeadsetInput.unpack(data)
            if pkt is None:
                # NEVER in silence.  A packet that arrives and cannot be parsed
                # is indistinguishable, downstream, from no packet at all: both
                # show up as "0 Hz, last=never", which points at the network or
                # the address and is wrong on both counts.  It cost a session
                # here -- the headset was running an older build still speaking
                # input protocol v2, HeadsetInput.unpack rejected every packet
                # on the version check, and this `continue` threw away the one
                # fact that would have explained it.  Ian hit the same thing
                # twice in one session on gvlink's own InputListener.
                #
                # Reported once per distinct reason, so a mismatched sender says
                # so exactly once instead of at 90 Hz.
                dropped += 1
                reason = self._why_unparseable(data)
                key = (addr[0], reason)
                if key not in seen_reasons:
                    seen_reasons.add(key)
                    print(f"[headset] DROPPING input packets from {addr[0]}: "
                          f"{reason} -- the uplink will read as 0 Hz until this "
                          f"is fixed (dropped {dropped} so far)")
                continue
            now = time.monotonic()
            with self._input_lock:
                self._input_pkt, self._input_at = pkt, now
            rate_n += 1
            if now - rate_at >= 1.0:
                self.input_rate_hz = rate_n / (now - rate_at)
                rate_at, rate_n = now, 0

    @staticmethod
    def _why_unparseable(data: bytes) -> str:
        """Name the reason a datagram was rejected, in the sender's terms."""
        from gvlink.protocol import INPUT_MAGIC, INPUT_SIZE, INPUT_VERSION
        if len(data) < 6:
            return f"runt packet ({len(data)} bytes, need at least {INPUT_SIZE})"
        magic = data[:4]
        if magic != INPUT_MAGIC:
            return (f"not a gvlink input packet (magic {magic!r}, want "
                    f"{INPUT_MAGIC!r}) -- something else is sending to this port")
        version = data[4]
        if version != INPUT_VERSION:
            return (f"input protocol v{version}, this robot speaks "
                    f"v{INPUT_VERSION} -- REBUILD THE HEADSET APP")
        if len(data) < INPUT_SIZE:
            return f"truncated ({len(data)} bytes, need {INPUT_SIZE})"
        return f"unpack failed ({len(data)} bytes, magic and version both look right)"

    def _fresh_input(self, stale_s: float = 0.5) -> HeadsetInput | None:
        with self._input_lock:
            if self._input_pkt is None:
                return None
            if time.monotonic() - self._input_at > stale_s:
                return None
            return self._input_pkt

    def receive_data(self) -> HeadsetData | None:
        """
        Newest headset state as a HeadsetData, or None if nothing fresh has arrived.

        Poses go through the same left-to-right handedness conversion the WebRTC path
        used, so the arm control maths downstream is unchanged.

        "Fresh" here is CONTROL_STALE_S, tighter than the gaze path's tolerance --
        this is the data the arms are driven from.
        """
        pkt = self._fresh_input(CONTROL_STALE_S)
        if pkt is None:
            return None

        d = HeadsetData()
        d.h_pos, d.h_quat = convert_left_to_right_coordinates(
            np.asarray(pkt.head_pos, dtype=float), np.asarray(pkt.head_rot, dtype=float))

        # Controller pose vs. hand-tracking wrist pose: the operator uses exactly
        # one at a time (Quest falls back to hand tracking the instant it can't
        # see a controller), and ControllerState defaults to (0,0,0) when it was
        # never populated -- so on a hand-tracking session, using pkt.left/right
        # unconditionally sends BOTH "controllers" to the same point (world
        # origin composed with the head), which reads as the hands overlapping
        # and the arms refusing to move (self-collision at zero separation).
        # Resolved per side, independently, because one hand can drop tracking
        # while the other keeps a controller. Hand wrist poses ride the same
        # uplink frame as controller poses (HEADSET_API.md), so the same
        # handedness conversion applies unchanged.
        # OFF by default (2026-08-26).  Resolving the POSE from a wrist is only
        # half of hand control and the other half does not exist: the trigger
        # and button fields below are read from ControllerState, which the
        # viewer zeroes out wholesale on a hand session (GvInputUplink.Update).
        # So with hands live the grippers never actuate AND the arms never
        # activate -- l_button_one/r_button_one are what data_collection.py
        # gates arm_active on.  Enabling the pose half alone buys a correctly
        # tracked hand that cannot drive anything, while making the arms move
        # in response to hands the operator did not mean to control with.
        # Controllers-only is the honest configuration until pinch -> trigger
        # and a hand gesture for arm-activate are wired up.
        #     GIAVA_HEADSET_HAND_TRACKING=1   re-enable the pose fallback
        hand_fallback = os.environ.get("GIAVA_HEADSET_HAND_TRACKING") == "1"

        l_tracked = bool(hand_fallback and pkt.hand_l is not None and pkt.hand_l.tracked)
        r_tracked = bool(hand_fallback and pkt.hand_r is not None and pkt.hand_r.tracked)
        l_ctrl_valid = bool(pkt.flags & INPUT_LEFT_VALID)
        r_ctrl_valid = bool(pkt.flags & INPUT_RIGHT_VALID)

        if l_tracked and not l_ctrl_valid:
            l_pos_raw, l_rot_raw = pkt.hand_l.wrist_pos, pkt.hand_l.wrist_rot
        else:
            l_pos_raw, l_rot_raw = pkt.left.pos, pkt.left.rot
        if r_tracked and not r_ctrl_valid:
            r_pos_raw, r_rot_raw = pkt.hand_r.wrist_pos, pkt.hand_r.wrist_rot
        else:
            r_pos_raw, r_rot_raw = pkt.right.pos, pkt.right.rot

        d.l_pos, d.l_quat = convert_left_to_right_coordinates(
            np.asarray(l_pos_raw, dtype=float), np.asarray(l_rot_raw, dtype=float))
        d.r_pos, d.r_quat = convert_left_to_right_coordinates(
            np.asarray(r_pos_raw, dtype=float), np.asarray(r_rot_raw, dtype=float))
        d.l_using_hand_track = l_tracked and not l_ctrl_valid
        d.r_using_hand_track = r_tracked and not r_ctrl_valid

        d.l_thumbstick_x, d.l_thumbstick_y = float(pkt.left.stick[0]), float(pkt.left.stick[1])
        d.r_thumbstick_x, d.r_thumbstick_y = float(pkt.right.stick[0]), float(pkt.right.stick[1])
        d.l_index_trigger, d.l_hand_trigger = float(pkt.left.trigger), float(pkt.left.grip)
        d.r_index_trigger, d.r_hand_trigger = float(pkt.right.trigger), float(pkt.right.grip)

        d.l_button_one = pkt.left.held(BUTTON_ONE)
        d.l_button_two = pkt.left.held(BUTTON_TWO)
        d.l_button_thumbstick = pkt.left.held(BUTTON_STICK)
        d.r_button_one = pkt.right.held(BUTTON_ONE)
        d.r_button_two = pkt.right.held(BUTTON_TWO)
        d.r_button_thumbstick = pkt.right.held(BUTTON_STICK)

        # Additive: gaze in normalised source-image coordinates (x right, y down).
        # The WebRTC HeadsetData had no equivalent, so nothing downstream reads these
        # yet -- they are here because the uplink now carries them.
        d.gaze_l = tuple(pkt.gaze_l)
        d.gaze_r = tuple(pkt.gaze_r)
        d.gaze_confidence = float(pkt.gaze_confidence)
        d.gaze_valid = bool(pkt.gaze_valid)

        # NOTE: the uplink still carries INPUT_DEADMAN (bit 6) and gvlink still
        # decodes it as `pkt.deadman`; it is deliberately not copied onto
        # HeadsetData any more.  Nothing in this repo gates on it as of
        # 2026-09-01 -- see LinkGuard in data_col_config.py for why.
        d.hands_valid = bool(getattr(pkt, "hands_valid", False))
        d.hand_l = getattr(pkt, "hand_l", None)
        d.hand_r = getattr(pkt, "hand_r", None)
        return d

    # ----------------------------------------------------------------- feedback

    def send_feedback(self, data: HeadsetFeedback) -> None:
        """
        Publish robot state on the control channel.

        NOTE: the v2 viewer subscribes to "arm/state" but only renders it as a debug
        string; it has no equivalent of the WebRTC build's out-of-sync ghost arms or
        info banner. This is sent so the data is there when the viewer grows a use for
        it, not because anything draws it today.
        """
        if self._link is None or not self._link.connected:
            return

        def vec(v):
            return [float(x) for x in np.asarray(v, dtype=float).ravel()]

        self._link.publish("arm/state", {
            "info": str(getattr(data, "info", "") or ""),
            "headOutOfSync": bool(getattr(data, "head_out_of_sync", False)),
            "leftOutOfSync": bool(getattr(data, "left_out_of_sync", False)),
            "rightOutOfSync": bool(getattr(data, "right_out_of_sync", False)),
            "leftArmPosition": vec(getattr(data, "left_arm_position", np.zeros(3))),
            "leftArmRotation": vec(getattr(data, "left_arm_rotation", np.zeros(4))),
            "rightArmPosition": vec(getattr(data, "right_arm_position", np.zeros(3))),
            "rightArmRotation": vec(getattr(data, "right_arm_rotation", np.zeros(4))),
            "middleArmPosition": vec(getattr(data, "middle_arm_position", np.zeros(3))),
            "middleArmRotation": vec(getattr(data, "middle_arm_rotation", np.zeros(4))),
        })

    # -------------------------------------------------------------------- stats

    def describe(self) -> str:
        return (f"tx {self.frames_sent} frames, {self.frames_dropped} dropped in mailbox, "
                f"uplink {self.input_rate_hz:.0f} Hz, {self._rate.describe()}, "
                f"{self._link.session if self._link else 'no link'}")


# Same name as the WebRTC class so `from gvlink_headset import GvLinkHeadset as
# WebRTCHeadset` reads honestly at the call sites that want a one-line swap.
WebRTCHeadset = GvLinkHeadset
