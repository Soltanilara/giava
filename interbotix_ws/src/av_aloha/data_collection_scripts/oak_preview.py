"""On-screen OAK preview: watch the stereo feed on the monitor while the
headset operator teleoperates.

WHY THIS IS NOT A MODE
======================
Recording and previewing are two consumers of the SAME frames, not two
modes.  `oak_worker` already publishes every OAK pair into `latest_frames`
and pushes it to the headset; the episode writer picks frames out of there
only while an episode is being recorded.  This adds a third reader that
never stops: recording or not, the window keeps showing what the OAK sees.
"Record and stream" during an episode and "just stream" between episodes are
therefore the same code path with the recorder either running or not --
there is no preview-only path that could drift out of step with the
recording one.  The window says which of the two is happening (see the
badge in `_viewer`), because that is the one thing the frames themselves
cannot tell the operator.

IT SHOWS THE DATASET VIEW, DELIBERATELY
=======================================
The frames drawn are `latest_frames["oak_*"]` -- the 640x480 crop and
downscale of the native 1280x800 that `oak_dataset_view` produces, which is
byte for byte what lands in the episode.  So the window answers the question
an operator actually has ("is what I am recording any good?") rather than
the prettier one the headset answers (rectified, full resolution).  It is
NOT the rectified pair: eye alignment is a headset concern, and rectifying
here would cost a remap per frame for a view nobody trains on.

WHY A SUBPROCESS AND NOT A WINDOW IN THE LOOP
=============================================
Drawing costs far more than it looks like it should.  Measured on this rig
with the Qt5 HighGUI this OpenCV is built against:

    cvtColor into a preallocated buffer     0.07 ms
    cv2.imshow + cv2.pollKey                3.34 ms   <-- the Qt repaint
    a whole naive draw (hstack, waitKey(1)) ~12 ms

The control loop's period is 20 ms and the IK solve already spends a median
27.5 ms on the ticks it runs, so ~4.5 ms of Qt repaint 15 times a second
lands squarely on top of a budget that is already over.  A preview must not
make teleop worse during exactly the sessions somebody is watching it.

The two cheaper-looking options were both rejected on evidence:

  * a HighGUI thread inside this process -- tested, and Qt reports
    "QObject::killTimer: Timers cannot be stopped from another thread".  It
    happens to run, but a QApplication outside the main thread is
    unsupported, and the failure mode is aborting the process that is
    holding a live recording.
  * drawing on the loop thread anyway -- rejected for the numbers above.

So the viewer is a separate process.  What the loop pays is one memcpy of
the frame into shared memory at the preview rate (~0.1 ms, no GIL contention
and no Qt), and a viewer that crashes or is closed takes nothing with it.

OFF BY DEFAULT
==============
    GIAVA_OAK_PREVIEW=0        (default) nothing is started at all
    GIAVA_OAK_PREVIEW=1        the left eye (same as `left`)
    GIAVA_OAK_PREVIEW=left|right|both

or per run:  python data_collection.py --preview          (left eye)
             python data_collection.py --preview both

    GIAVA_OAK_PREVIEW_FPS=15   redraw cap.  The loop runs at 50 Hz and the
                               OAK at 25, so publishing every tick would
                               copy the same frame twice.
    GIAVA_OAK_PREVIEW_SCALE=1  resize factor applied in the viewer.

DEMOS, WITH NO RECORDING AT ALL
===============================
A demo that records nothing does not need the arms, ROS or a dataset, only
the camera on the screen:

    python oak_preview.py both

`run_standalone` below opens the OAK through camera_manager's own
`setup_oak_stereo` and draws `oak_dataset_view`, so it is the same pipeline
and the same framing a session would record -- not a second opinion from a
private pipeline like the ad-hoc scripts around it.  The OAK is claimed
exclusively by whichever process opens it first, so this and a running
data_collection.py are alternatives, never both at once.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional

import numpy as np

MODES = ("off", "left", "right", "both")

## GIAVA_OAK_PREVIEW takes the switch spellings as well as the eye names, so
## `=1` does the obvious thing.
_ALIASES = {
    "0": "off", "": "off", "no": "off", "false": "off", "none": "off",
    "1": "left", "yes": "left", "true": "left", "on": "left",
    "l": "left", "r": "right", "stereo": "both", "lr": "both",
}

DEFAULT = "off"

WINDOW = "OAK -- data collection"

## Shared-memory layout.  A fixed header, a fixed-size text slot, then the
## frame.  Deliberately flat and fixed: the viewer maps it once at startup
## and never has to negotiate a size change, because the OAK's dataset view
## is a fixed 640x480 for the life of a session.
_HDR_SLOTS = 8                     # int64s
_HDR_BYTES = _HDR_SLOTS * 8
_LABEL_BYTES = 128
_IMG_OFFSET = _HDR_BYTES + _LABEL_BYTES

## Header slots, by index.
_SEQ = 0        # seqlock: odd while the writer is mid-update
_STOP = 1       # publisher -> viewer: shut down
_GONE = 2       # viewer -> publisher: window closed, stop publishing
_RECORDING = 3  # is this frame being written into an episode?
_EPISODE = 4    # episode index, or -1
_LABEL_LEN = 5  # valid bytes in the label slot


def take_flag(argv: Optional[List[str]] = None) -> Optional[str]:
    """Read `--preview [mode]` and REMOVE it from argv.

    collision_modes.take_option cannot be reused: it only understands
    `--name VALUE`, so a bare `--preview` would survive into
    data_collection.py's positional parse, which rejects anything
    flag-shaped -- aborting the session over the preview switch.  This
    accepts the bare form (meaning `left`) as well as `--preview both` and
    `--preview=both`."""
    in_place = argv is None
    argv = list(sys.argv if in_place else argv)
    value = None
    keep, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--preview":
            ## The next token is the mode only if it IS one.  Otherwise it
            ## belongs to somebody else -- the episode index, another flag --
            ## and swallowing it would silently change the run.
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is not None and _normalize(nxt) is not None:
                value = nxt
                i += 2
            else:
                value = "left"
                i += 1
            continue
        if a.startswith("--preview="):
            value = a.split("=", 1)[1]
            i += 1
            continue
        keep.append(a)
        i += 1
    if in_place:
        sys.argv[:] = keep
    return value


def _normalize(name) -> Optional[str]:
    """Resolve one spelling to a mode name, or None if it is not one."""
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    return key if key in MODES else None


def select(argv: Optional[List[str]] = None) -> str:
    """Resolve `--preview` / GIAVA_OAK_PREVIEW to a mode name.

    Call from module scope like the collision and jax selectors: the flag has
    to be consumed before data_collection.py's positional parse sees it."""
    raw = take_flag(argv)
    if raw is None:
        raw = os.environ.get("GIAVA_OAK_PREVIEW", DEFAULT)
    mode = _normalize(raw)
    if mode is None:
        raise SystemExit(
            f"--preview / GIAVA_OAK_PREVIEW must be one of {sorted(MODES)}, "
            f"got '{raw}'")
    os.environ["GIAVA_OAK_PREVIEW"] = mode
    return mode


## ---------------------------------------------------------------------- ##
## The viewer -- runs in the child process.  Nothing above cv2 is imported
## here on purpose: this module is what `spawn` re-imports in the child, so
## anything it pulls in is paid for again in a second interpreter.
## ---------------------------------------------------------------------- ##

def _viewer(shm_name, height, width, mode, fps, scale):
    import cv2
    from multiprocessing import shared_memory

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return  # publisher already gone
    ## Attaching registers the block with THIS process's resource tracker
    ## too, which then reports a spurious "leaked shared_memory" on exit and,
    ## worse, may unlink a block the publisher still owns.  The publisher
    ## created it and the publisher unlinks it.
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass

    hdr = np.ndarray((_HDR_SLOTS,), np.int64, buffer=shm.buf)
    label_buf = np.ndarray((_LABEL_BYTES,), np.uint8, buffer=shm.buf,
                           offset=_HDR_BYTES)
    img = np.ndarray((height, width, 3), np.uint8, buffer=shm.buf,
                     offset=_IMG_OFFSET)

    ## Reused across frames so the steady state allocates nothing: cvtColor
    ## writes straight into it (0.07 ms) instead of numpy building a new
    ## reversed-stride copy (1.85 ms).
    bgr = np.empty((height, width, 3), np.uint8)
    period = 1.0 / fps if fps > 0 else 0.0
    opened = False
    last_seq = -1
    ## ORPHAN CHECK.  This is a plain subprocess, so nothing reaps it if the
    ## session dies without running its shutdown path -- a SIGKILL, an OOM
    ## kill -- and the window would sit there showing a frozen frame from a
    ## session that no longer exists.
    ##
    ## The check is the parent's pid, NOT "no frame for N seconds".  A
    ## silence timeout cannot tell death from a pause, and the pauses here
    ## are long and legitimate: the preview is constructed as soon as the
    ## cameras exist, but the loop that publishes does not start until the
    ## URDF is parsed, the solver is jitted (tens of seconds on a cold
    ## compilation cache) and the arms have parked.  Any timeout short enough
    ## to be useful would kill the window during startup.  On Linux an
    ## orphan is reparented, so getppid() changing is exact and costs a
    ## syscall.
    parent_pid = os.getppid()

    try:
        while True:
            if hdr[_STOP]:
                break
            seq = int(hdr[_SEQ])
            ## Seqlock: an odd counter means the publisher is mid-write, and
            ## an unchanged one means no new frame.  A torn preview frame
            ## would be harmless, but skipping is free.
            if os.getppid() != parent_pid:
                print("[preview] session gone; closing")
                break
            if seq == last_seq or seq & 1:
                time.sleep(period * 0.25 if period else 0.005)
                continue
            recording = bool(hdr[_RECORDING])
            episode = int(hdr[_EPISODE])
            n = int(hdr[_LABEL_LEN])
            label = bytes(label_buf[:n]).decode("utf-8", "replace") if n else ""
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, dst=bgr)
            if int(hdr[_SEQ]) != seq:
                continue  # publisher overwrote it mid-read; take the next one
            last_seq = seq

            canvas = bgr
            if scale != 1.0:
                canvas = cv2.resize(
                    bgr, None, fx=scale, fy=scale,
                    interpolation=(cv2.INTER_AREA if scale < 1.0
                                   else cv2.INTER_LINEAR))
            _annotate(cv2, canvas, recording, episode, label, mode)

            if not opened:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW, canvas.shape[1], canvas.shape[0])
                opened = True
            cv2.imshow(WINDOW, canvas)
            ## pollKey, not waitKey(1): non-blocking, ~1 ms cheaper, and the
            ## key is ignored on purpose.  Keystrokes that land in this
            ## window instead of the terminal must never be able to start,
            ## save or discard an episode.
            cv2.pollKey()
            ## Closing the window is how the operator says "stop previewing".
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                hdr[_GONE] = 1
                break
            if period:
                time.sleep(period)
    except KeyboardInterrupt:
        ## Ctrl-C in the terminal reaches this process group too.  The
        ## publisher's shutdown path is what should end the viewer, but
        ## exiting quietly here beats a traceback racing the session's.
        pass
    except Exception as exc:
        print(f"[preview] viewer stopped: {exc}")
    finally:
        try:
            hdr[_GONE] = 1
        except Exception:
            pass
        if opened:
            try:
                cv2.destroyWindow(WINDOW)
                cv2.pollKey()
            except Exception:
                pass
        shm.close()


def _annotate(cv2, canvas, recording, episode, label, mode):
    """Recording state, big enough to read from the operator's chair.

    This is the whole reason the window carries text: the frames look
    identical whether or not they are being written, and "I thought it was
    recording" is the one preview failure that costs a demonstration."""
    if recording:
        text = "REC" + (f"  ep {episode}" if episode >= 0 else "")
        colour = (0, 0, 255)          # BGR red
        cv2.circle(canvas, (22, 24), 9, colour, -1)
    else:
        text = "TELEOP - not recording"
        colour = (0, 215, 255)        # BGR amber
    ## Drawn twice, dark then coloured: the OAK view is a bright tabletop and
    ## thin red text on it is unreadable without an outline.
    for col, thick in ((0, 0, 0), 4), (colour, 2):
        cv2.putText(canvas, text, (38, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    col, thick, cv2.LINE_AA)
    if mode == "both":
        cv2.line(canvas, (canvas.shape[1] // 2, 0),
                 (canvas.shape[1] // 2, canvas.shape[0]), (60, 60, 60), 1)
    if label:
        for col, thick in ((0, 0, 0), 3), ((255, 255, 255), 1):
            cv2.putText(canvas, label, (12, canvas.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, thick, cv2.LINE_AA)


## ---------------------------------------------------------------------- ##
## The publisher -- lives in the data-collection process.
## ---------------------------------------------------------------------- ##

class OakPreview:
    """Publishes OAK frames to a viewer subprocess.

    Constructed even when the preview is off, so the loop can call `publish()`
    unconditionally and the switch stays in one place.  Self-disabling: a
    closed window, a dead viewer, a missing display or a repeated failure
    turns it off for the rest of the session instead of raising into the
    control loop."""

    ## Consecutive publish failures tolerated before giving up.  One is a
    ## transient; a run of them means the transport is broken and the loop
    ## should stop paying for it.
    MAX_FAILURES = 5

    ## The OAK dataset view, i.e. the shape oak_worker publishes.  Checked
    ## against every frame in publish() rather than assumed: the viewer maps
    ## a fixed-size block, so a frame of another size must turn the preview
    ## off, not be written past the end of it.
    FRAME_H, FRAME_W = 480, 640

    def __init__(self, mode, active_cameras=(), fps=None, scale=None):
        self.mode = _normalize(mode) or "off"
        self.eyes = {"left": ["oak_left"], "right": ["oak_right"],
                     "both": ["oak_left", "oak_right"]}.get(self.mode, [])
        self.enabled = False
        self._proc = None
        self._shm = None
        self._failures = 0
        self.published = 0

        if not self.eyes:
            return
        ## Nothing to show if the OAK is not among the cameras this mode
        ## opened -- say so once, here, rather than opening a dead window.
        missing = [c for c in self.eyes if c not in set(active_cameras)]
        if missing:
            print(f"[preview] --preview {self.mode} needs "
                  f"{', '.join(missing)}, which this mode does not open; "
                  f"preview off")
            return
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("[preview] no DISPLAY/WAYLAND_DISPLAY -- preview off "
                  "(run from a desktop session, or forward X)")
            return

        self.fps = float(fps if fps is not None
                         else os.environ.get("GIAVA_OAK_PREVIEW_FPS", 15.0))
        self.scale = float(scale if scale is not None
                           else os.environ.get("GIAVA_OAK_PREVIEW_SCALE", 1.0))
        self._period = 1.0 / self.fps if self.fps > 0 else 0.0
        self._next = 0.0
        self._width = self.FRAME_W * len(self.eyes)

        try:
            self._start()
        except Exception as exc:
            print(f"[preview] could not start the viewer ({exc}); preview off")
            self._teardown()
            return

        self.enabled = True
        print(f"[preview] OAK preview ON ({self.mode}, {self.fps:g} fps"
              + (f", scale {self.scale:g}" if self.scale != 1.0 else "")
              + ") -- close the window to stop it")

    def _start(self):
        import subprocess
        from multiprocessing import shared_memory

        nbytes = _IMG_OFFSET + self.FRAME_H * self._width * 3
        self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
        self._hdr = np.ndarray((_HDR_SLOTS,), np.int64, buffer=self._shm.buf)
        self._hdr[:] = 0
        self._hdr[_EPISODE] = -1
        self._label = np.ndarray((_LABEL_BYTES,), np.uint8,
                                 buffer=self._shm.buf, offset=_HDR_BYTES)
        self._img = np.ndarray((self.FRAME_H, self._width, 3), np.uint8,
                               buffer=self._shm.buf, offset=_IMG_OFFSET)

        ## A PLAIN SUBPROCESS, not multiprocessing.Process.  `spawn` re-imports
        ## the parent's __main__ in the child, and here that is
        ## data_collection.py -- whose module scope selects the collision
        ## model, pins the jax platform, imports torch and pushes the ROS
        ## dist-packages onto sys.path.  A window would have paid for all of
        ## it, twice, and reprinted the startup banner.  `fork` is worse
        ## still: this process holds ROS threads, the camera workers, CUDA
        ## contexts and depthai's XLink threads, and forking a
        ## multi-threaded process is how a child ends up deadlocked in an
        ## allocator lock it inherited held.  Running THIS file as a script
        ## imports cv2 and numpy and nothing else.
        ##
        ## The shared block is found by name, so the two processes need no
        ## relationship beyond it.
        env = dict(os.environ)
        ## This OpenCV's Qt plugin ships no fonts and says so, once per
        ## window, in a terminal the operator is reading for episode
        ## feedback.  Silences that qWarning; Python tracebacks still reach
        ## the inherited stderr.
        env.setdefault("QT_LOGGING_RULES", "default.warning=false")
        self._proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--viewer",
             self._shm.name, str(self.FRAME_H), str(self._width), self.mode,
             str(self.fps), str(self.scale)],
            env=env,
        )

    ## ------------------------------------------------------------------ ##

    def publish(self, frame_lock, latest_frames, recording=False,
                episode=None, label=None):
        """Copy the current OAK frame(s) to the viewer if one is due.

        Never raises -- a preview must not be able to end a recording
        session.

        `recording` is the tick's REAL recording state: pass the same
        `collecting_episode_now` the frame writer keys off, which is False
        both between episodes and, under GIAVA_RECORD_GATE=teleop, before the
        first teleop enable inside one.  Then the badge says what the
        recorder is actually doing rather than what the operator pressed."""
        if not self.enabled:
            return
        t = time.monotonic()
        if t < self._next:
            return
        self._next = t + self._period
        try:
            if self._hdr[_GONE] or self._proc.poll() is not None:
                self.close()
                return
            ## Under the lock only long enough to take references.  The
            ## worker publishes a NEW array per iteration and never writes
            ## into one it has published, so the references stay valid after
            ## the lock is dropped -- and the memcpy below, the expensive
            ## part, happens outside it rather than stalling the camera
            ## worker.
            with frame_lock:
                views = [latest_frames.get(cam) for cam in self.eyes]
            if any(v is None for v in views):
                return  # camera still starting, or a dropped pair
            if any(v.shape[:2] != (self.FRAME_H, self.FRAME_W) for v in views):
                print(f"[preview] unexpected OAK frame size "
                      f"{views[0].shape[:2]}, expected "
                      f"{(self.FRAME_H, self.FRAME_W)}; preview off")
                self.close()
                return

            ## Seqlock: odd while writing, even when the frame is whole.
            self._hdr[_SEQ] += 1
            for i, v in enumerate(views):
                self._img[:, i * self.FRAME_W:(i + 1) * self.FRAME_W] = v
            self._hdr[_RECORDING] = 1 if recording else 0
            self._hdr[_EPISODE] = -1 if episode is None else int(episode)
            raw = str(label or "").encode("utf-8")[:_LABEL_BYTES]
            self._label[:len(raw)] = np.frombuffer(raw, np.uint8)
            self._hdr[_LABEL_LEN] = len(raw)
            self._hdr[_SEQ] += 1

            self._failures = 0
            self.published += 1
        except Exception as exc:
            self._failures += 1
            if self._failures == 1:
                print(f"[preview] publish failed ({exc}); will retry")
            if self._failures >= self.MAX_FAILURES:
                print(f"[preview] giving up after {self._failures} failures")
                self.close()

    def close(self):
        """Stop the viewer and release the shared block.  Idempotent, and
        safe to call from a shutdown path or an atexit hook."""
        if not (self._proc or self._shm):
            self.enabled = False
            return
        self.enabled = False
        try:
            if self._shm is not None:
                self._hdr[_STOP] = 1
        except Exception:
            pass
        self._teardown()

    def _teardown(self):
        proc, self._proc = self._proc, None
        if proc is not None:
            ## _STOP is already set, so the viewer should be on its way out.
            ## terminate() is the backstop for one wedged in a Qt call.
            try:
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
        shm, self._shm = self._shm, None
        if shm is not None:
            ## Drop the numpy views FIRST: SharedMemory.close() raises
            ## BufferError while any memoryview over the block is alive, and
            ## the block would then never be unlinked.
            self._hdr = self._label = self._img = None
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


## ---------------------------------------------------------------------- ##
## Standalone: the OAK on the screen with nothing else running.
## ---------------------------------------------------------------------- ##

def run_standalone(mode="both", fps=None, scale=None):
    """Open the OAK and show it, with no ROS, no arms and no dataset.

    For demos and for checking the camera: `python oak_preview.py both`.
    It builds the camera through camera_manager.setup_oak_stereo and draws
    camera_manager.oak_dataset_view, the SAME pipeline and the same framing
    data collection records -- so what this window shows is what an episode
    would contain, rather than a second opinion from a private pipeline.

    Runs the window on this process's main thread, which is where Qt wants
    it: there is no control loop here to protect, so the subprocess the
    publisher needs would buy nothing.

    NOT usable alongside data_collection.py.  The OAK is claimed exclusively
    by whichever process opens it first, so a session already running owns
    it -- use --preview there instead."""
    import cv2

    if __package__:
        from .camera_manager import setup_oak_stereo, oak_dataset_view, OAK_SWAP_EYES
    else:
        from camera_manager import setup_oak_stereo, oak_dataset_view, OAK_SWAP_EYES

    mode = _normalize(mode) or "both"
    if mode == "off":
        return 0
    fps = float(fps if fps is not None
                else os.environ.get("GIAVA_OAK_PREVIEW_FPS", 30.0))
    scale = float(scale if scale is not None
                  else os.environ.get("GIAVA_OAK_PREVIEW_SCALE", 1.0))

    opened = setup_oak_stereo()
    if opened is None:
        print("[preview] no OAK device found")
        return 1
    pipeline, q_left, q_right = opened
    period = 1.0 / fps if fps > 0 else 0.0
    print(f"[preview] standalone OAK preview ({mode}) -- q or closing the "
          f"window quits")

    try:
        while True:
            in_left = q_left.get()
            in_right = q_right.get()
            left_bgr = in_left.getCvFrame()
            right_bgr = in_right.getCvFrame()
            ## The same handedness decision the worker makes, for the same
            ## reason: a preview that disagrees with the recording about
            ## which eye is which is worse than no preview.
            if OAK_SWAP_EYES["on"]:
                left_bgr, right_bgr = right_bgr, left_bgr
            views = {"left": [left_bgr], "right": [right_bgr],
                     "both": [left_bgr, right_bgr]}[mode]
            views = [oak_dataset_view(v) for v in views]
            canvas = np.hstack(views) if len(views) > 1 else views[0]
            canvas = np.ascontiguousarray(canvas)
            if scale != 1.0:
                canvas = cv2.resize(
                    canvas, None, fx=scale, fy=scale,
                    interpolation=(cv2.INTER_AREA if scale < 1.0
                                   else cv2.INTER_LINEAR))
            ## recording=False is the literal truth here: nothing in this
            ## process can write an episode.
            _annotate(cv2, canvas, False, -1, "standalone -- not recording",
                      mode)
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(max(1, int(period * 1000))) in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass
    return 0


def _main(argv):
    ## `--viewer` is the internal entry point OakPreview._start launches; it
    ## is positional-only and undocumented on purpose.  Everything else is
    ## the standalone viewer.
    if len(argv) > 1 and argv[1] == "--viewer":
        name, h, w, mode, fps, scale = argv[2:8]
        _viewer(name, int(h), int(w), mode, float(fps), float(scale))
        return 0
    if any(a in ("-h", "--help") for a in argv[1:]):
        print(__doc__)
        print("standalone:  python oak_preview.py [left|right|both]")
        return 0
    rest = [a for a in argv[1:] if not a.startswith("-")]
    return run_standalone(rest[0] if rest else "both")


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
