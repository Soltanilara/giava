"""Two-camera synchronisation measurement  (PHASE 7).

Runs two (or more) RealSense cameras concurrently and records, for every
paired capture, every clock the hardware and driver expose.  Then reports
the offset between them over many samples so the timing behaviour of this
rig can be characterised rather than assumed.


WHAT THIS CAN AND CANNOT ESTABLISH
==================================
It measures SOFTWARE-OBSERVABLE timing.  It cannot prove hardware
synchronisation, and nothing in this rig provides any:

  * The D405s have no external sync cabling here, and no
    ``inter_cam_sync_mode`` is configured, so their shutters free-run
    independently at 60 fps.  Two cameras will drift relative to each other
    at whatever their crystal difference is.
  * ``timestamp_domain = global_time`` does NOT mean the cameras are
    synchronised.  It means librealsense is fitting a linear map from each
    device's own clock onto the host epoch, independently per device.  The
    fit has its own error and re-converges over time.
  * ``time_of_arrival`` and ``backend_timestamp`` are HOST timestamps taken
    after USB transfer.  They include transfer and scheduling jitter, which
    is unrelated to when the exposure happened.

The clock that comes closest to "when the photons landed" is the metadata
``sensor_timestamp`` (device clock, microseconds, start of exposure) -- but
it lives in each device's OWN time base, so a raw difference between two
cameras' sensor timestamps is an arbitrary constant plus drift, not a
latency.  Read the per-clock sections separately; they are reported
separately for exactly this reason.

The number that matters operationally is usually the spread (std, p95,
min-max range) rather than the mean: a constant offset can be calibrated
out, jitter cannot.


CAPTURE MODEL
=============
One thread per camera blocks on ``wait_for_frames`` and keeps the newest
frame -- the same latest-frame-wins model ``camera_manager.py`` uses in
production, so the numbers describe what data collection actually sees.
The sampler then takes both cameras' newest frames under one lock.

That means a sample can pick up a frame that has been sitting there a
while.  Frame ages at sample time are recorded, and repeated frame numbers
are counted and reported: a high duplicate rate means the sample rate is
outrunning the cameras and those samples are not independent measurements.

Examples

    python calibration/sync_capture.py --cameras left_wrist right_wrist \\
        --samples 300

    # save the paired images too, and turn global time off to see the raw
    # device clocks
    python calibration/sync_capture.py --cameras left_wrist right_wrist \\
        --samples 100 --save-images --no-global-time
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    ensure_dirs,
    provenance,
    save_json,
    timestamp,
)
from rs_camera import (  # noqa: E402
    PRODUCTION_COLOR,
    FrameRecord,
    RealSenseCamera,
    require_rs,
    resolve,
)

## The clocks compared, and what each one actually is.  Keys are the
## FrameRecord fields; the scale converts each to SECONDS so offsets are
## reported in one unit.
CLOCKS: List[Tuple[str, float, str, str]] = [
    ("timestamp_ms", 1e-3, "librealsense frame timestamp",
     "domain-dependent -- see timestamp_domain; global_time maps the device "
     "clock onto the host epoch via a per-device linear fit"),
    ("sensor_timestamp_us", 1e-6, "sensor timestamp (device clock)",
     "closest to start of exposure, but in each DEVICE's own time base -- "
     "a cross-camera difference is an arbitrary constant plus drift"),
    ("frame_timestamp_us", 1e-6, "frame timestamp (device clock)",
     "device time the frame was assembled; same time-base caveat"),
    ("backend_timestamp_ms", 1e-3, "backend timestamp (host clock)",
     "host time the kernel driver received it; includes USB transfer"),
    ("time_of_arrival_ms", 1e-3, "time of arrival (host clock)",
     "host time librealsense received it; includes transfer and scheduling"),
    ("host_epoch_s", 1.0, "host wall clock at hand-off",
     "stamped by this script when the frame was taken from the queue; "
     "includes everything above plus Python scheduling"),
]


## ------------------------------------------------------------------ ##
## Capture threads
## ------------------------------------------------------------------ ##

class CameraWorker:
    """Keeps the newest frame from one camera, with all its clocks."""

    def __init__(self, camera: str, width: int, height: int, fps: int,
                 global_time: Optional[bool], keep_images: bool,
                 history_len: int = 8, with_depth: bool = False):
        self.camera = camera
        self.cam = RealSenseCamera(camera, width=width, height=height,
                                   fps=fps, stream="color",
                                   global_time=global_time,
                                   with_depth=with_depth)
        self.keep_images = keep_images
        self.lock = threading.Lock()
        self.latest: Optional[FrameRecord] = None
        ## Short history, so a consumer can pick the frame nearest a shared
        ## reference instant instead of only the newest one -- the same
        ## nearest-timestamp alignment camera_manager uses in production.
        from collections import deque
        self.history: "deque[FrameRecord]" = deque(maxlen=history_len)
        self.count = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.cam.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rec = self.cam.capture(timeout_ms=2000,
                                       with_image=self.keep_images)
            except Exception:
                self.errors += 1
                continue
            with self.lock:
                self.latest = rec
                self.history.append(rec)
                self.count += 1

    def snapshot(self) -> Optional[FrameRecord]:
        with self.lock:
            return self.latest

    def newest_timestamp_s(self) -> Optional[float]:
        with self.lock:
            if not self.history or self.history[-1].timestamp_ms is None:
                return None
            return self.history[-1].timestamp_ms * 1e-3

    def nearest(self, reference_s: float) -> Optional[FrameRecord]:
        """The buffered frame whose timestamp is closest to reference_s."""
        with self.lock:
            cand = [r for r in self.history if r.timestamp_ms is not None]
        if not cand:
            return None
        return min(cand, key=lambda r: abs(r.timestamp_ms * 1e-3 - reference_s))

    def stop(self) -> None:
        """Stop the capture thread, THEN the pipeline -- never the reverse.

        Calling pipeline.stop() while the capture thread is still inside
        wait_for_frames makes librealsense throw from that thread.  Nothing
        can catch a C++ exception raised on another thread, so the runtime
        calls std::terminate and the whole process aborts with a core dump
        -- which is exactly what happened when a camera stopped delivering
        and the tool tried to shut down.

        If the thread will not exit, leaking the pipeline is strictly
        better than aborting: the process is ending anyway, and the device
        is released when it does."""
        self._stop.set()
        if self._thread is not None:
            # Comfortably longer than the 2 s wait_for_frames timeout the
            # loop can be sitting in.
            self._thread.join(timeout=6.0)
            if self._thread.is_alive():
                print(f"  [{self.cam.name}] capture thread did not exit; "
                      f"leaving the pipeline open rather than risking an "
                      f"abort. The device is released on process exit.")
                return
        self.cam.stop()


## ------------------------------------------------------------------ ##
## Statistics
## ------------------------------------------------------------------ ##

def summarize(values: np.ndarray) -> Dict[str, Any]:
    """Mean/median/std/min/max plus frame-to-frame variation."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    out = {
        "n": int(v.size),
        "mean_ms": float(v.mean() * 1e3),
        "median_ms": float(np.median(v) * 1e3),
        "std_ms": float(v.std(ddof=1) * 1e3) if v.size > 1 else 0.0,
        "min_ms": float(v.min() * 1e3),
        "max_ms": float(v.max() * 1e3),
        "range_ms": float((v.max() - v.min()) * 1e3),
        "p05_ms": float(np.percentile(v, 5) * 1e3),
        "p95_ms": float(np.percentile(v, 95) * 1e3),
    }
    if v.size > 1:
        d = np.diff(v)
        out["frame_to_frame"] = {
            "mean_abs_change_ms": float(np.abs(d).mean() * 1e3),
            "std_ms": float(d.std(ddof=1) * 1e3) if d.size > 1 else 0.0,
            "max_abs_change_ms": float(np.abs(d).max() * 1e3),
        }
        ## A steady linear trend across the run is clock DRIFT between the
        ## two devices -- qualitatively different from jitter, and the
        ## reason a one-off offset measurement does not stay valid.
        x = np.arange(v.size, dtype=float)
        slope = float(np.polyfit(x, v, 1)[0])
        out["drift_ms_per_sample"] = slope * 1e3
    return out


def compare_clocks(samples: List[Dict[str, Any]], cam_a: str,
                   cam_b: str) -> Dict[str, Any]:
    """Delta t = clock(cam_a) - clock(cam_b), per clock."""
    out: Dict[str, Any] = {}
    for field, scale, label, caveat in CLOCKS:
        deltas, a_vals, b_vals = [], [], []
        for s in samples:
            a = s["frames"].get(cam_a, {}).get(field)
            b = s["frames"].get(cam_b, {}).get(field)
            if a is None or b is None:
                continue
            a_vals.append(a * scale)
            b_vals.append(b * scale)
            deltas.append((a - b) * scale)
        if not deltas:
            out[field] = {"available": False, "label": label,
                          "caveat": caveat}
            continue
        entry = {"available": True, "label": label, "caveat": caveat,
                 "delta_definition": f"{cam_a} minus {cam_b}",
                 "statistics": summarize(np.asarray(deltas))}
        ## Same-device-clock fields have arbitrary origins; say so with the
        ## data rather than only in prose.
        if field in ("sensor_timestamp_us", "frame_timestamp_us"):
            entry["note"] = (
                "Both values are on their OWN device clocks. The mean is an "
                "arbitrary constant; only the spread and the drift are "
                "physically meaningful.")
        out[field] = entry
    return out


def freshness(samples: List[Dict[str, Any]],
              cameras: List[str]) -> Dict[str, Any]:
    """How stale were the frames, and how often were they reused?"""
    out = {}
    for cam in cameras:
        ages, dup, total, last_fn = [], 0, 0, None
        for s in samples:
            f = s["frames"].get(cam)
            if not f:
                continue
            total += 1
            if f.get("host_epoch_s") is not None:
                ages.append(s["sample_epoch_s"] - f["host_epoch_s"])
            fn = f.get("frame_number")
            if fn is not None and fn == last_fn:
                dup += 1
            last_fn = fn
        a = np.asarray(ages, dtype=float)
        out[cam] = {
            "n_samples": total,
            "duplicate_frame_samples": dup,
            "duplicate_fraction": (dup / total) if total else None,
            "frame_age_at_sample_ms": (
                {"mean": float(a.mean() * 1e3),
                 "median": float(np.median(a) * 1e3),
                 "max": float(a.max() * 1e3)} if a.size else None),
        }
    return out


## ------------------------------------------------------------------ ##

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", nargs="+", required=True,
                    help="two or more camera names or serials")
    ap.add_argument("--samples", type=int, default=200,
                    help="paired captures to record (default: 200)")
    ap.add_argument("--interval", type=float, default=0.05,
                    help="seconds between samples (default: 0.05). Below the "
                         "frame period this mostly re-reads the same frames "
                         "-- see the duplicate-frame report.")
    ap.add_argument("--width", type=int, default=PRODUCTION_COLOR[0])
    ap.add_argument("--height", type=int, default=PRODUCTION_COLOR[1])
    ap.add_argument("--fps", type=int, default=PRODUCTION_COLOR[2])
    ap.add_argument("--save-images", action="store_true",
                    help="also save both images for every sample")
    ap.add_argument("--global-time", dest="global_time", action="store_true",
                    default=True,
                    help="enable librealsense global time (default)")
    ap.add_argument("--no-global-time", dest="global_time",
                    action="store_false",
                    help="leave frame timestamps on the raw device clock")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if len(args.cameras) < 2:
        ap.error("give at least two cameras")

    require_rs()
    ensure_dirs()
    resolved = resolve(args.cameras)
    names = [n for n, _ in resolved]

    outdir = (Path(args.outdir) if args.outdir
              else DIR_SYNC / f"{'_'.join(names)}_{timestamp()}")
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (outdir / "images").mkdir(exist_ok=True)

    print()
    print("#" * 78)
    print("#  Two-camera synchronisation measurement  (PHASE 7)")
    print("#" * 78)
    print(f"""
  cameras     {', '.join(f'{n} ({s})' for n, s in resolved)}
  stream      {args.width}x{args.height} @ {args.fps} fps, color
  samples     {args.samples} every {args.interval * 1000:.0f} ms
  global time {'ENABLED' if args.global_time else 'DISABLED'}
  output      {outdir}

  NOTE: these cameras free-run. There is no hardware sync wiring in this
  rig and no inter_cam_sync_mode is set, so anything measured here
  describes software clocks, not shutter alignment.
""")

    workers = [CameraWorker(n, args.width, args.height, args.fps,
                            args.global_time, args.save_images)
               for n in names]
    samples: List[Dict[str, Any]] = []
    try:
        for w in workers:
            w.start()
            print(f"  started {w.cam.name} "
                  f"(global_time now {w.cam.global_time_state()})")
        print("\n  warming up...")
        time.sleep(1.5)
        _check_delivering(workers)
        print(f"  capturing {args.samples} samples...\n")

        for i in range(args.samples):
            sample = _take_sample(i, workers, outdir, args.save_images)
            if sample is not None:
                samples.append(sample)
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{args.samples}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n  interrupted -- analysing what was collected")
    finally:
        for w in workers:
            w.stop()

    if len(samples) < 2:
        raise SystemExit("  too few samples collected to analyse")

    _analyse_and_save(samples, names, resolved, workers, outdir, args)


def _check_delivering(workers: List[CameraWorker],
                      timeout_s: float = 6.0) -> None:
    """Fail fast, and by name, when a camera starts but sends no frames.

    A D405 can enumerate, report firmware, accept a stream configuration
    and start a pipeline while never delivering a single frame -- a faulty
    cable, port or unit looks exactly like a healthy one until you wait for
    data.  Without this check the run would just collect zero samples and
    blame itself."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all(w.snapshot() is not None for w in workers):
            print("  all cameras delivering frames")
            return
        time.sleep(0.2)

    dead = [w for w in workers if w.snapshot() is None]
    lines = [
        "",
        "  *** NO FRAMES from: "
        + ", ".join(f"{w.cam.name} (serial {w.cam.serial})" for w in dead),
        "",
        "  Frames received so far:",
    ]
    for w in workers:
        lines.append(f"      {w.cam.name:14s} serial {w.cam.serial}  "
                     f"{w.count:5d} frames, {w.errors} errors")
    lines += [
        "",
        "  The pipeline started, so the camera is enumerated and accepted",
        "  the stream configuration -- the fault is downstream of that.",
        "",
        "  MOST LIKELY: the device is wedged. This happens when a previous",
        "  process died without closing its pipeline cleanly, and it",
        "  survives closing every handle -- nothing is holding the device,",
        "  it is simply stuck. Clear it with a device reset (no root, no",
        "  unplugging):",
        "",
        "      python calibration/reset_cameras.py",
        "",
        "  If that does not fix it, then check the hardware:",
        "    * a different USB3 cable",
        "    * a different USB3 port, ideally on another controller",
        "    * `lsusb -t` shows 5000M for this device",
        "    * try the camera on its own:",
        "        python calibration/rs_intrinsics.py --cameras <name>",
        "",
    ]
    raise SystemExit("\n".join(lines))


def _take_sample(idx: int, workers: List[CameraWorker], outdir: Path,
                 save_images: bool) -> Optional[Dict[str, Any]]:
    """Snapshot every camera's newest frame under its own lock.

    The snapshots are taken back to back and bracketed by host timestamps,
    so the sampling window itself is measured rather than assumed zero."""
    t0 = time.perf_counter()
    epoch = time.time()
    snaps = [(w, w.snapshot()) for w in workers]
    t1 = time.perf_counter()

    if any(rec is None for _, rec in snaps):
        return None

    frames: Dict[str, Any] = {}
    for w, rec in snaps:
        d = rec.timing_dict()
        if save_images and rec.image is not None:
            import cv2
            fn = f"images/sample{idx:05d}_{w.cam.name}.png"
            cv2.imwrite(str(outdir / fn),
                        cv2.cvtColor(rec.image, cv2.COLOR_RGB2BGR))
            d["image_path"] = fn
        frames[w.cam.name] = d

    return {
        "index": idx,
        "sample_epoch_s": epoch,
        "sample_monotonic_s": t0,
        ## How long the snapshot itself took -- an upper bound on the error
        ## this sampling method introduces.
        "sampling_window_ms": (t1 - t0) * 1e3,
        "frames": frames,
    }


def _analyse_and_save(samples, names, resolved, workers, outdir, args) -> None:
    cam_a, cam_b = names[0], names[1]
    clocks = compare_clocks(samples, cam_a, cam_b)
    fresh = freshness(samples, names)
    windows = np.array([s["sampling_window_ms"] for s in samples])

    print()
    print("=" * 78)
    print(f"  RESULTS   ({len(samples)} samples)")
    print("=" * 78)
    print(f"\n  sampling window (time to snapshot all cameras): "
          f"mean {windows.mean():.3f} ms, max {windows.max():.3f} ms")

    print("\n  frame freshness and reuse:")
    print(f"      {'camera':14s} {'frames rx':>10s} {'dup samples':>12s} "
          f"{'age mean ms':>12s} {'age max ms':>11s}")
    print("      " + "-" * 62)
    for w in workers:
        f = fresh[w.cam.name]
        age = f["frame_age_at_sample_ms"] or {}
        print(f"      {w.cam.name:14s} {w.count:>10d} "
              f"{f['duplicate_frame_samples']:>5d} "
              f"({f['duplicate_fraction'] * 100:4.1f}%) "
              f"{age.get('mean', float('nan')):>12.2f} "
              f"{age.get('max', float('nan')):>11.2f}")
    worst_dup = max(fresh[n]["duplicate_fraction"] or 0 for n in names)
    if worst_dup > 0.25:
        print(f"\n      NOTE: {worst_dup * 100:.0f}% of samples reused a "
              f"frame already seen.")
        print("      Those samples are not independent measurements. Raise")
        print("      --interval above the frame period "
              f"({1000.0 / args.fps:.1f} ms) for independent samples.")

    print()
    print("=" * 78)
    print(f"  OFFSET:  delta t = {cam_a}  minus  {cam_b}")
    print("=" * 78)
    for field, _, label, _ in CLOCKS:
        entry = clocks[field]
        print(f"\n  {label}   [{field}]")
        if not entry["available"]:
            print("      not reported by this device/driver")
            continue
        st = entry["statistics"]
        print(f"      mean   {st['mean_ms']:+12.4f} ms       "
              f"median {st['median_ms']:+12.4f} ms")
        print(f"      std    {st['std_ms']:12.4f} ms       "
              f"range  {st['range_ms']:12.4f} ms")
        print(f"      min    {st['min_ms']:+12.4f} ms       "
              f"max    {st['max_ms']:+12.4f} ms")
        if "frame_to_frame" in st:
            f2f = st["frame_to_frame"]
            print(f"      frame-to-frame: mean |change| "
                  f"{f2f['mean_abs_change_ms']:.4f} ms, "
                  f"max {f2f['max_abs_change_ms']:.4f} ms")
            print(f"      drift over the run: "
                  f"{st['drift_ms_per_sample']:+.6f} ms/sample")
        if entry.get("note"):
            print(f"      NOTE: {entry['note']}")

    domains = {n: samples[0]["frames"][n].get("timestamp_domain")
               for n in names}
    print()
    print("=" * 78)
    print("  TIMESTAMP DOMAINS IN EFFECT")
    print("=" * 78)
    for n, d in domains.items():
        print(f"      {n:14s} {d}")
    print(f"""
  Read this together with the numbers above.  'global_time' means
  librealsense mapped each DEVICE's clock onto the host epoch with its own
  independent linear fit -- the cameras are not synchronised, they are
  merely both estimated against the host.  'hardware_clock' means the raw
  device counter, whose origin is arbitrary and differs per device.

  Neither proves the shutters fired together. Only the spread (std, range)
  and the drift bound how well any software alignment can do here.
""")

    _write_outputs(samples, names, resolved, clocks, fresh, windows, outdir,
                   args, domains, workers)


def _write_outputs(samples, names, resolved, clocks, fresh, windows, outdir,
                   args, domains, workers) -> None:
    doc = {
        "metadata": provenance(
            "two_camera_sync",
            cameras=[{"name": n, "serial": s} for n, s in resolved],
            width=args.width, height=args.height, fps=args.fps,
            samples_requested=args.samples,
            sample_interval_s=args.interval,
            global_time_requested=args.global_time,
        ),
        "hardware_sync": {
            "external_sync_wired": False,
            "inter_cam_sync_mode_configured": False,
            "statement": (
                "These cameras free-run. Nothing measured here demonstrates "
                "hardware synchronisation; all figures describe software "
                "clocks (device counters, driver-side host stamps, and "
                "librealsense's per-device global-time fit)."),
        },
        "timestamp_domains": domains,
        "clock_definitions": {f: {"label": l, "caveat": c, "unit_scale_to_s": s}
                              for f, s, l, c in CLOCKS},
        "n_samples": len(samples),
        "frames_received": {w.cam.name: w.count for w in workers},
        "sampling_window_ms": {
            "mean": float(windows.mean()), "max": float(windows.max()),
            "std": float(windows.std(ddof=1)) if len(windows) > 1 else 0.0},
        "freshness": fresh,
        "offsets": clocks,
        "delta_definition": f"{names[0]} minus {names[1]}",
    }
    save_json(doc, outdir / "sync_summary.json", overwrite=args.overwrite)

    ## Raw per-sample data, both as JSON and as a CSV that drops straight
    ## into a spreadsheet or pandas for independent analysis.
    save_json({"metadata": doc["metadata"], "samples": samples},
              outdir / "sync_raw.json", overwrite=args.overwrite)
    _write_csv(samples, names, outdir / "sync_raw.csv")

    print(f"  summary : {outdir / 'sync_summary.json'}")
    print(f"  raw JSON: {outdir / 'sync_raw.json'}")
    print(f"  raw CSV : {outdir / 'sync_raw.csv'}")
    print()


def _write_csv(samples, names, path: Path) -> None:
    fields = ["frame_number", "timestamp_ms", "timestamp_domain",
              "sensor_timestamp_us", "frame_timestamp_us",
              "backend_timestamp_ms", "time_of_arrival_ms", "frame_counter",
              "host_epoch_s", "image_path"]
    header = ["index", "sample_epoch_s", "sampling_window_ms"]
    for n in names:
        header += [f"{n}.{f}" for f in fields]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for s in samples:
            row = [s["index"], s["sample_epoch_s"], s["sampling_window_ms"]]
            for n in names:
                fr = s["frames"].get(n, {})
                row += [fr.get(f) for f in fields]
            w.writerow(row)


if __name__ == "__main__":
    main()
