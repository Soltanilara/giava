import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_writer import DatasetWriter
from lerobot.datasets.video_utils import StreamingVideoEncoder

if __package__:
    from .arm_config import ARM_CONFIG
    from .data_col_config import ARM_MODES, DATASET_ROOT
else:
    from arm_config import ARM_CONFIG
    from data_col_config import (
        ARM_MODES,
        DATASET_ROOT,
    )

ARM_DATASET_JOINTS = {
    arm: cfg["joint_names"]
    for arm, cfg in ARM_CONFIG.items()
}


def append_save_debug_line(root: Path, message: str) -> None:
    log_dir = root / "save_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "save_episode.log", "a") as f:
        f.write(f"{message}\n")


## WHY A VERIFY PASS EXISTS AT ALL
#
# finalize() writing without raising is NOT the same as the dataset being
# loadable.  The two failures that actually happen here are silent:
#
#   * a video file whose trailing frames never reached the muxer, because the
#     process ended while an encoder thread was still draining;
#   * episode metadata that disagrees with the videos it points into, so the
#     dataset loads but hands the wrong frames back at training time.
#
# Both look exactly like a healthy run in the log -- the operator only finds
# out when a policy trains on garbage.  So at the end of every session, count
# what is actually in the files and compare it against what the metadata
# claims.  Cheap (one ffprobe per video file, seconds) and it runs once.
#
# NOTE ON MULTIPLE .mp4 PER CAMERA: that is NORMAL and not a fault.  lerobot
# v3 rolls a new video file every `video_files_size_in_mb` (200 MB here), so a
# long session leaves several files per camera and the episode metadata says
# which file and which timestamp range each episode lives in.  This check
# verifies exactly that mapping; it does not expect one file per camera.

def _probe_video_frames(path, timeout=180):
    """Frames actually muxed into `path`, or None if it cannot be counted."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None          # no ffprobe on this box
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def verify_finalized_dataset(root, probe_videos=True):
    """Cross-check a finalized dataset against its own metadata.

    Returns (ok, lines).  `lines` is the human-readable report; it is printed
    and appended to save_logs/save_episode.log by the caller.  Never raises --
    a broken verifier must not be able to take down a session that recorded
    fine.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    problems, notes = [], []

    try:
        info = json.loads((root / "meta" / "info.json").read_text())
    except Exception as exc:
        return False, [f"meta/info.json unreadable: {exc}"]

    want_eps = int(info.get("total_episodes", 0))
    want_frames = int(info.get("total_frames", 0))

    ## v2.1 datasets keep per-episode metadata in meta/episodes.jsonl and one
    ## mp4 PER EPISODE -- neither of which this check knows how to read, and
    ## reporting them as broken would be a lie.  Say what they are instead.
    cbv = str(info.get("codebase_version", "")).lstrip("v")
    if not cbv.startswith("3"):
        return True, [f"SKIPPED: codebase_version={info.get('codebase_version')!r}"
                      f" -- this check only understands v3.0 layouts "
                      f"({want_eps} episode(s), {want_frames} frame(s) claimed)."]

    n_data = len(list((root / "data").rglob("*.parquet")))
    n_mp4 = len(list((root / "videos").rglob("*.mp4")))

    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        ## Nothing recorded at all: a session that started and was quit before
        ## the first episode.  Not a fault, and not worth an alarm.
        if want_eps == 0 and n_data == 0 and n_mp4 == 0:
            return True, ["EMPTY RUN: no episodes were recorded in this "
                          "session -- nothing to verify."]
        return False, [f"meta/episodes/ has no parquet, but {n_data} data "
                       f"file(s) and {n_mp4} video(s) were written -- "
                       f"finalize() never ran, so info.json reads "
                       f"{want_eps} episodes / {want_frames} frames while the "
                       f"recorded data is still on disk. NOT LOADABLE as-is."]
    try:
        table = pq.read_table(ep_files)
        ep = table.to_pydict()
    except Exception as exc:
        return False, [f"episode metadata unreadable: {exc}"]

    n_ep = len(ep.get("episode_index", []))
    lengths = [int(x) for x in ep.get("length", [])]
    total_len = sum(lengths)

    if n_ep != want_eps:
        problems.append(f"info.json says {want_eps} episodes, episode "
                        f"metadata has {n_ep}")
    if total_len != want_frames:
        problems.append(f"info.json says {want_frames} frames, episode "
                        f"lengths sum to {total_len}")
    if want_eps and not want_frames:
        problems.append("info.json reports 0 frames for a non-empty dataset "
                        "-- finalize() ran against an empty buffer")

    ## ---- data parquet: rows on disk vs. episode lengths ----------------- ##
    data_rows = 0
    for f in sorted((root / "data").rglob("*.parquet")):
        try:
            data_rows += pq.ParquetFile(f).metadata.num_rows
        except Exception as exc:
            problems.append(f"data/{f.name} unreadable ({exc}) -- its footer "
                            "may never have been written")
    if data_rows != total_len:
        problems.append(f"data parquet holds {data_rows} rows, episode "
                        f"metadata accounts for {total_len}")
    notes.append(f"data: {data_rows} rows across "
                 f"{len(list((root / 'data').rglob('*.parquet')))} file(s)")

    ## ---- videos: frames muxed vs. frames claimed, per FILE -------------- ##
    video_keys = [k for k, v in info.get("features", {}).items()
                  if v.get("dtype") == "video"]
    vpath = info.get("video_path",
                     "videos/{video_key}/chunk-{chunk_index:03d}/"
                     "file-{file_index:03d}.mp4")
    for key in video_keys:
        ci = ep.get(f"videos/{key}/chunk_index")
        fi = ep.get(f"videos/{key}/file_index")
        if ci is None or fi is None:
            problems.append(f"{key}: episode metadata carries no file index "
                            f"-- episodes cannot be located in the videos")
            continue
        # frames each (chunk, file) is supposed to contain
        want = {}
        for c, f_, n in zip(ci, fi, lengths):
            want[(int(c), int(f_))] = want.get((int(c), int(f_)), 0) + int(n)
        on_disk = sorted((root / "videos" / key).rglob("*.mp4"))
        if len(on_disk) != len(want):
            problems.append(
                f"{key}: metadata references {len(want)} video file(s), "
                f"{len(on_disk)} on disk")
        got_total = 0
        for (c, f_), n in sorted(want.items()):
            p = root / vpath.format(video_key=key, chunk_index=c, file_index=f_)
            if not p.exists():
                problems.append(f"{key}: {p.relative_to(root)} is referenced "
                                f"by {n} frames of metadata but MISSING")
                continue
            if not probe_videos:
                continue
            have = _probe_video_frames(p)
            if have is None:
                notes.append(f"{key}: {p.name} not counted (ffprobe "
                             f"unavailable or timed out)")
                continue
            got_total += have
            if have != n:
                problems.append(
                    f"{key}: {p.relative_to(root)} holds {have} frames, "
                    f"metadata claims {n}"
                    + (f" -- {n - have} frames LOST, the encoder was still "
                       f"draining when the process ended"
                       if have < n else " -- more frames than expected"))
        if probe_videos and got_total:
            notes.append(f"{key}: {got_total} frames across "
                         f"{len(want)} file(s)")

    ok = not problems
    lines = []
    if ok:
        lines.append(f"VERIFY OK: {n_ep} episode(s), {total_len} frame(s); "
                     f"videos and metadata agree.")
    else:
        lines.append(f"VERIFY FAILED: {len(problems)} problem(s) in "
                     f"{n_ep} episode(s) / {total_len} frame(s).")
        lines += [f"  - {p}" for p in problems]
    lines += [f"  . {n}" for n in notes]
    return ok, lines


# HOW SAVING WORKS HERE, AND WHY IT LOOKS LIKE THIS
#
# lerobot v0.6.0 streams frames to per-camera encoder threads as they are
# recorded (`streaming_encoding=True`), so save_episode() only has to drain the
# encoders and write metadata.  That drain still took seconds with six cameras,
# and it used to run ON THE CONTROL LOOP: the loop stopped ticking, the headset
# stalled and the keyboard went dead until it finished.
#
# Saving now runs on ONE worker thread fed by a FIFO queue.  Saves stay
# serialized with respect to each other -- they share the parquet writer and the
# dataset metadata, and two at once would corrupt both -- but they overlap with
# RECORDING, which is the part that costs operator time.
#
# The overlap needs one thing lerobot does not offer directly.  Its writer owns
# a SINGLE streaming encoder; add_frame() feeds it and save_episode() closes it
# out.  Keep recording while a background save_episode() drains and the next
# episode's frames land in the buffer and the encoder being closed -- which is
# exactly how three separate recordings came out as ONE long episode with a jump
# cut at every seam.
#
# So the handoff detaches BOTH pieces of per-episode state at once and installs
# fresh ones for the next episode:
#
#     detach   writer.episode_buffer      -> the worker saves this dict
#              writer._streaming_encoder  -> the worker drains this encoder
#     install  a new buffer (next index) + a NEW StreamingVideoEncoder
#
# The worker then calls the stock DatasetWriter.save_episode() through
# _DetachedWriterView, whose only job is to answer `_streaming_encoder` with the
# DETACHED encoder while forwarding everything else to the real writer.  Nothing
# in lerobot is patched or forked; the two threads touch disjoint per-episode
# state, and the shared state (parquet writer, metadata) is only ever touched by
# the single worker.
#
# validate_episode_buffer() requires episode_index == meta.total_episodes.  That
# holds because saves are FIFO-serialized and each one increments
# total_episodes, so the episode reaching the worker is always the next one due.
#
# MEMORY is the cost of the overlap: each in-flight encoder holds up to
# encoder_queue_maxsize frames per camera (120 x 640x480x3 x 6 cameras ~ 660 MB
# worst case), and with overlap two sets can exist at once.  MAX_PENDING caps
# how far the writer may fall behind before 'r' blocks.
#
# GIAVA_OVERLAP_SAVE=0 falls back to the serialized behaviour (recording waits
# for the save), which is the thing to try first if a dataset ever looks wrong.
OVERLAP_SAVE = os.environ.get("GIAVA_OVERLAP_SAVE", "1").strip() not in ("0", "false", "no")
MAX_PENDING = int(os.environ.get("GIAVA_MAX_PENDING_SAVES", "2"))


class _DetachedWriterView:
    """A DatasetWriter that reports the DETACHED streaming encoder.

    save_episode() reads `self._streaming_encoder` to drain the episode it is
    closing out.  By the time the worker runs, the real writer's attribute has
    already been replaced with the encoder for the episode being recorded NOW,
    so the worker gets this view instead.  Every other attribute -- parquet
    writer, metadata, root -- forwards to the real writer, reads and writes
    alike, because those are genuinely shared and the worker is the only thread
    that touches them.
    """

    def __init__(self, writer, encoder):
        object.__setattr__(self, "_real_writer", writer)
        object.__setattr__(self, "_detached_encoder", encoder)

    def __getattr__(self, name):
        if name == "_streaming_encoder":
            return object.__getattribute__(self, "_detached_encoder")
        return getattr(object.__getattribute__(self, "_real_writer"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_real_writer"), name, value)


class BackgroundEpisodeSaver:
    def __init__(self, dataset: LeRobotDataset):
        self.dataset = dataset
        self._writer = dataset.writer
        self.next_episode_index = int(dataset.episode_buffer["episode_index"])
        self._finalized = False

        ## Overlap needs a streaming encoder to swap.  Without one (streaming
        ## disabled, or a lerobot that no longer works this way) fall back to
        ## the serialized path rather than guessing.
        enc = getattr(self._writer, "_streaming_encoder", None)
        self.overlap = bool(OVERLAP_SAVE and enc is not None)
        if OVERLAP_SAVE and enc is None:
            print("[save] no streaming encoder -- overlapped saving disabled; "
                  "recording will wait for each save.")
        self._enc_kwargs = None
        if enc is not None:
            ## Cloned once from the encoder the dataset was built with, so
            ## every per-episode encoder is configured identically to it.
            self._enc_kwargs = dict(
                fps=enc.fps,
                rgb_encoder=enc._rgb_encoder,
                depth_encoder=enc._depth_encoder,
                queue_maxsize=enc.queue_maxsize,
                encoder_threads=enc._encoder_threads,
            )

        self._queue: "queue.Queue" = queue.Queue()
        self._cv = threading.Condition()
        self._inflight = 0
        self.failures = 0
        self.saved_count = 0
        self._worker = threading.Thread(
            target=self._run, name="episode-saver", daemon=True)
        self._worker.start()

    # ---- called from the control loop -------------------------------------

    @property
    def busy(self) -> bool:
        """True while an episode is queued or being written."""
        with self._cv:
            return self._inflight > 0

    @property
    def pending(self) -> int:
        with self._cv:
            return self._inflight

    def begin_episode(self) -> int:
        """Prepare a fresh buffer + encoder for the episode about to start.

        Returns the episode index that will be recorded.  Blocks only if the
        writer has fallen MAX_PENDING episodes behind -- see the memory note
        above.  In the serialized fallback it waits for the writer to be idle.
        """
        if not self.overlap:
            self.wait_until_idle(announce=True)
        else:
            with self._cv:
                if self._inflight >= MAX_PENDING:
                    print(f"[save] writer is {self._inflight} episodes behind "
                          f"-- waiting before recording again...", flush=True)
                    while self._inflight >= MAX_PENDING:
                        self._cv.wait()
        self._install_fresh_episode(self.next_episode_index)
        return self.next_episode_index

    def save_episode_async(self, outcome: str = "unknown",
                           task_override: str | None = None) -> int:
        """Hand the current episode to the writer thread and return at once.

        `outcome` is the operator's success/failure verdict; it is recorded in
        episode_outcomes.jsonl beside the dataset (see _record_outcome).

        `task_override` relabels EVERY frame of this episode before it is
        detached -- the last chance to fix a wrong target, since the task
        string is stamped per frame while recording and a mislabelled episode
        teaches the policy to fetch the wrong object on command.  Applied here
        rather than at write time so the buffer, the parquet and
        episode_outcomes.jsonl cannot disagree.

        Raises ValueError on an empty buffer, exactly as before.
        """
        buf = self._writer.episode_buffer
        if buf is None or buf["size"] == 0:
            raise ValueError("No frames available to save.")

        episode_index = int(buf["episode_index"])
        num_frames = int(buf["size"])
        if task_override is not None and buf.get("task"):
            buf["task"] = [task_override] * len(buf["task"])
        tasks = buf.get("task") or []
        task = tasks[0] if tasks else ""

        ## DETACH AND REPLACE IN ONE GO, before anything can add a frame: the
        ## worker owns `buf` and `enc` from here on, and the control loop must
        ## never touch either again.
        enc = getattr(self._writer, "_streaming_encoder", None)
        if self.overlap:
            self._install_fresh_episode(episode_index + 1)

        with self._cv:
            self._inflight += 1
        self._queue.put((episode_index, num_frames, outcome, task,
                         buf if self.overlap else None,
                         enc if self.overlap else None))

        self.next_episode_index = episode_index + 1
        print(f"[save] episode_{episode_index:04d} saving "
              f"({num_frames} frames, {outcome})...", flush=True)
        return episode_index

    def discard_current_episode(self) -> None:
        """Drop the episode being recorded and reuse its index."""
        idx = int(self._writer.episode_buffer["episode_index"])
        if self.overlap:
            enc = getattr(self._writer, "_streaming_encoder", None)
            ## Kills this episode's encoder threads and deletes its temp mp4s.
            ## Episodes already queued are untouched -- their encoders are
            ## different objects, which is the whole point of the swap.
            if enc is not None:
                try:
                    enc.cancel_episode()
                    enc.close()
                except Exception as exc:
                    print(f"[save] encoder cancel failed: {exc}")
            self._install_fresh_episode(idx)
        else:
            self.wait_until_idle()
            self.dataset.clear_episode_buffer()
        self.next_episode_index = idx

    def status_line(self) -> str:
        """One line for the operator: is anything still being written?"""
        with self._cv:
            pending = self._inflight
        if pending:
            return (f"[save] {pending} episode(s) still being written "
                    f"-- {self.saved_count} done. Quitting now WILL wait for "
                    f"them; do not kill the process.")
        return (f"[save] idle -- {self.saved_count} episode(s) written"
                f"{f', {self.failures} FAILED' if self.failures else ''}. "
                f"Safe to quit with q.")

    def wait_until_idle(self, announce: bool = False) -> None:
        """Block until every queued episode has been written."""
        with self._cv:
            if self._inflight == 0:
                return
            if announce:
                print(f"[save] waiting for {self._inflight} episode(s) to "
                      f"finish saving...", flush=True)
            while self._inflight > 0:
                self._cv.wait()

    def close(self) -> None:
        # finalize() flushes buffered episode metadata and writes the parquet
        # footers. Without it the dataset on disk cannot be loaded back.
        #
        # IDEMPOTENT on purpose: this is registered with atexit as well as
        # being called from the normal 'q' shutdown, because a session that
        # ends any other way (exception, Ctrl-C, rospy shutdown) would
        # otherwise leave every episode it recorded unreadable.  Whichever
        # path gets here first does the work; the other returns.
        if self._finalized:
            return
        self._finalized = True
        # Queued episodes first: finalizing while the writer thread is still
        # mid-episode loses that episode and can leave the parquet half-written.
        try:
            self.wait_until_idle(announce=True)
        except Exception:
            pass
        self._queue.put(None)
        self._worker.join(timeout=60)
        try:
            self.dataset.finalize()
        except Exception as exc:
            # Reported, not re-raised: this also runs from atexit, where an
            # exception would bury the one line that says what went wrong
            # under an interpreter-shutdown traceback.
            append_save_debug_line(self.dataset.root, f"FINALIZE FAILED: {exc}")
            print(f"[dataset] FINALIZE FAILED: {exc}\n"
                  f"[dataset] {self.dataset.root} may not load back -- "
                  "check it with replay_episode.py --dry-run before recording "
                  "more.")
            return
        append_save_debug_line(self.dataset.root, "DATASET FINALIZED")
        print(f"[dataset] finalized -> {self.dataset.root}")

        ## VERIFY, don't assume.  finalize() returning cleanly says the writer
        ## did not raise; it does not say the videos on disk hold the frames
        ## the metadata promises.  Counting them here is the difference
        ## between finding a truncated session now -- while the operator is
        ## still at the robot and can re-record -- and finding it weeks later
        ## in a policy that trained on frames that were never there.
        ##
        ## Advisory only: it reports, it does not repair or raise.  Set
        ## GIAVA_VERIFY_DATASET=0 to skip it, or =meta to skip the ffprobe
        ## pass and check metadata consistency alone (instant).
        _v = os.environ.get("GIAVA_VERIFY_DATASET", "1").strip().lower()
        if _v not in ("0", "no", "off", "false"):
            print("[dataset] verifying videos against metadata "
                  "(GIAVA_VERIFY_DATASET=0 to skip)...", flush=True)
            try:
                ok, lines = verify_finalized_dataset(
                    self.dataset.root, probe_videos=(_v != "meta"))
            except Exception as exc:
                ok, lines = False, [f"verify pass itself failed: {exc}"]
            for line in lines:
                append_save_debug_line(self.dataset.root, line)
            prefix = "[dataset]" if ok else "[dataset] *** "
            for line in lines:
                print(f"{prefix} {line}")
            if not ok:
                print("[dataset] *** This dataset is NOT safe to train on "
                      "as-is. Re-check with replay_episode.py --dry-run.")

    # ---- internals --------------------------------------------------------

    def _install_fresh_episode(self, episode_index: int) -> None:
        """Give the control loop a private buffer + encoder for `episode_index`."""
        self._writer.episode_buffer = self._writer._create_episode_buffer(
            episode_index=episode_index)
        if self._enc_kwargs is not None:
            ## Unstarted: add_frame() calls start_episode() on the first frame,
            ## which is what allocates this episode's queues, threads and temp
            ## directory (tempfile.mkdtemp, so two live encoders cannot collide
            ## on a path).
            self._writer._streaming_encoder = StreamingVideoEncoder(
                **self._enc_kwargs)
        self.next_episode_index = episode_index

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            episode_index, num_frames, outcome, task, buf, enc = job
            try:
                t0 = time.time()
                if buf is None:
                    self.dataset.save_episode()
                else:
                    ## Stock save_episode(), pointed at the detached encoder.
                    DatasetWriter.save_episode(
                        _DetachedWriterView(self._writer, enc),
                        episode_data=buf)
                    ## episode_data is not None, so save_episode() does not
                    ## clear the live buffer -- correct here, that buffer
                    ## belongs to the episode being recorded now.
                    try:
                        enc.close()
                    except Exception:
                        pass
                dt = time.time() - t0
                self._record_outcome(episode_index, num_frames, outcome, task)
                append_save_debug_line(
                    self.dataset.root,
                    f"EPISODE SAVED: episode_{episode_index:04d} "
                    f"frames={num_frames} outcome={outcome} secs={dt:.2f}")
                self.saved_count += 1
                print(f"[save] episode_{episode_index:04d} saved "
                      f"{num_frames} frames in {dt:.1f}s ({outcome})",
                      flush=True)
            except Exception as exc:
                # A failed save must not kill the session or the thread: the
                # arms are live and the operator is mid-run.  Report loudly,
                # keep serving the queue.
                self.failures += 1
                append_save_debug_line(
                    self.dataset.root,
                    f"EPISODE SAVE FAILED: episode_{episode_index:04d}: {exc}")
                print(f"[save] episode_{episode_index:04d} FAILED TO SAVE: "
                      f"{exc}", flush=True)
            finally:
                with self._cv:
                    self._inflight -= 1
                    drained = self._inflight == 0
                    self._cv.notify_all()
                ## THE LINE THE OPERATOR IS WAITING FOR.  Saving is off the
                ## control loop now, so "the last thing I pressed was ss" no
                ## longer means "the episode is on disk" -- without this there
                ## is no way to tell from the terminal whether quitting is safe
                ## yet.  Printed only on the transition to empty, so a run of
                ## queued saves produces one line, not one per episode.
                if drained:
                    print(f"[save] ALL EPISODES WRITTEN "
                          f"({self.saved_count} this session"
                          f"{f', {self.failures} FAILED' if self.failures else ''})"
                          f" -- safe to quit with q.", flush=True)
                self._queue.task_done()

    def _record_outcome(self, episode_index, num_frames, outcome, task) -> None:
        """Append the operator's verdict to episode_outcomes.jsonl.

        A SIDECAR rather than a dataset feature on purpose: the verdict is only
        known after the last frame is recorded, so writing it as a per-frame
        column would mean rewriting the buffer at save time, and adding a
        column changes the schema of every dataset this rig has produced.
        Training filters read the jsonl and join on episode_index; failures are
        kept, not discarded -- see the collection notes.
        """
        record = {
            "episode_index": episode_index,
            "outcome": outcome,
            "num_frames": num_frames,
            "task": task,
            "wall_time": time.time(),
        }
        path = Path(self.dataset.root) / "episode_outcomes.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


## LIBAV / x264 CONSOLE SPAM
##
## Encoding an episode used to fill the terminal with hundreds of lines --
##     [libx264 @ 0x...] using cpu capabilities: MMX2 SSE2Fast ...
##     [libx264 @ 0x...] frame I:2  Avg QP:25.59  size: 7027
## -- interleaved with the collection output, which buries the [save],
## [tracking] and [SAFETY] lines that actually need reading.
##
## WHERE IT COMES FROM, because it is not obvious: PyAV's DEFAULT state routes
## libav messages into the Python `libav` logger, which has no handler, so
## nothing prints.  lerobot's encoders call av.logging.restore_default_callback()
## when each encode finishes (video_utils.py:864), which switches libav back to
## writing on the C stderr directly.  From that moment every encoder context
## that closes anywhere in the process dumps its stats -- including the
## HEADSET's per-segment x264 encoders, which close every ~38 frames.  That is
## why the spam is tied to saving but keeps going afterwards.
##
## Measured: default = 0 lines; after restore_default_callback() = 16 lines per
## encoder close; restore then set_level(ERROR) = 0 again.  And setting the
## level at startup is NOT enough -- restore_default_callback() resets it, so a
## one-shot call before the first save does nothing.
##
## So the level is re-applied AFTER every restore, by wrapping the function
## itself: that covers lerobot, aiortc and anything else that calls it, rather
## than only the save path this module happens to own.  Errors still print.
def quiet_libav(level=None):
    """Silence libav/x264 console output, and keep it silenced."""
    if os.environ.get("GIAVA_QUIET_LIBAV", "1").strip() in ("0", "false", "no"):
        return False
    try:
        import av
        import av.logging
    except Exception:
        return False

    lvl = av.logging.ERROR if level is None else level
    original = av.logging.restore_default_callback
    if getattr(original, "_giava_requiets", False):
        av.logging.set_level(lvl)
        return True

    def restore_default_callback_quiet(*args, **kwargs):
        result = original(*args, **kwargs)
        ## The restore is what un-silences libav, so re-apply immediately.
        ## Anything that genuinely wants libav chatter can set
        ## GIAVA_QUIET_LIBAV=0 and get the unpatched behaviour.
        av.logging.set_level(lvl)
        return result

    restore_default_callback_quiet._giava_requiets = True
    av.logging.restore_default_callback = restore_default_callback_quiet
    av.logging.set_level(lvl)
    return True


def load_outcomes(dataset_root):
    """episode_index -> the CURRENT outcome record, from episode_outcomes.jsonl.

    The file is append-only, so relabelling an episode appends a new record
    rather than rewriting the file: a rewrite can be interrupted halfway and
    take the whole labelling history with it, and the history is worth keeping
    ("this was called a success during the session and downgraded on review" is
    a real thing to want to know).  LAST RECORD WINS -- every reader must use
    this function rather than parsing the file itself, or a relabelled episode
    will read differently depending on who looked.
    """
    path = Path(dataset_root) / "episode_outcomes.jsonl"
    outcomes = {}
    if not path.is_file():
        return outcomes
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                outcomes[int(rec["episode_index"])] = rec
            except Exception:
                ## A truncated final line is what a killed session leaves;
                ## skip it rather than losing every label before it.
                continue
    return outcomes


def build_state_names(active_arms):
    names = []

    for arm in active_arms:
        names.extend(ARM_DATASET_JOINTS[arm])

        if ARM_CONFIG[arm]["has_gripper"]:
            names.append(f"{arm}_gripper")

    return names

def build_action_names(active_arms):
    names = []

    for arm in active_arms:
        names.extend(f"{joint}_cmd" for joint in ARM_DATASET_JOINTS[arm])

        if ARM_CONFIG[arm]["has_gripper"]:
            names.append(f"{arm}_gripper_cmd")

    return names

def add_camera_features(features, active_cameras):

    for camera in active_cameras:

        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        }

        # float64, NOT float32.  These are absolute epoch timestamps (~1.79e9
        # right now).  float32 has 24 bits of mantissa, so near that magnitude
        # consecutive representable values are 128 SECONDS apart -- every
        # camera timestamp was being quantised into 128 s buckets, destroying
        # exactly the sub-millisecond information they exist to carry.
        # float64 keeps ~0.2 us at this magnitude.
        features[f"observation.timestamps.{camera}"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": None,
        }

def add_ee_features(features, active_arms):

    for arm in active_arms:

        features[f"observation.ee_pose.{arm}"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        }
    
def build_dataset_features(mode, active_cameras):
    active_arms = ARM_MODES[mode]

    state_names = build_state_names(active_arms)
    action_names = build_action_names(active_arms)

    features = {}

    add_camera_features(features, active_cameras)

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(state_names),),
        "names": state_names,
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (len(action_names),),
        "names": action_names,
    }

    ## MEASURED VELOCITY AND EFFORT, same layout and same names as
    ## observation.state so the three line up index for index.
    ##
    ## Both already arrive on every joint_states message and were being
    ## discarded -- data_collection.py read `effort` for the servo watchdog
    ## and threw it away, and velocity was never read at all.  Recording
    ## them costs no extra bus traffic and no extra latency, and neither can
    ## be recovered afterwards:
    ##
    ##   velocity  the servo's OWN measurement (Present_Velocity), which is
    ##             not the same thing as differencing recorded positions --
    ##             a finite difference inherits every bit of timestamp
    ##             jitter and quantisation in the position stream.
    ##   effort    per-joint current.  This is contact sensing the rig
    ##             already has: a spike as the gripper closes is a grasp, a
    ##             spike on an arm joint is a collision, and a sustained
    ##             gripper current while transporting means the object is
    ##             still held.  Getting those labels from vision instead is
    ##             a research project; getting them from this column is a
    ##             threshold.
    for kind in ("velocity", "effort"):
        features[f"observation.{kind}"] = {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        }

    for arm in active_arms:
        # float64 for the same reason as the camera timestamps above: these
        # are absolute epoch values, where float32 quantises to 128 s.
        features[f"observation.timestamps.{arm}"] = {
            "dtype": "float64",
            "shape": (1,),
            "names": None,
        }

    add_ee_features(features, active_arms)

    return features

def _check_fps_against_previous_runs(task_dir, fps):
    """Warn LOUDLY when this session's rate differs from earlier runs.

    The recorded fps comes straight from GIAVA_CONTROL_HZ, so a forgotten env
    var silently produces a run at a different rate.  Runs at different rates
    cannot be trained on together as one dataset (temporal models take fps as
    ground truth), and nothing else ever compares the two numbers.
    """
    seen = {}
    try:
        for meta_path in sorted(task_dir.glob("*/meta.json")):
            with open(meta_path) as f:
                prev = json.load(f).get("fps")
            if prev is not None and prev != fps:
                seen[meta_path.parent.name] = prev
    except Exception:
        return
    if seen:
        runs = ", ".join(f"{name} @ {prev} fps" for name, prev in seen.items())
        print("=" * 70)
        print(f"[dataset] WARNING: this session records at {fps} fps but "
              f"earlier runs of this task were recorded at a DIFFERENT rate: "
              f"{runs}.")
        print("[dataset] Mixed-rate runs cannot be combined for training. "
              "If this rate change is intentional, carry on; otherwise check "
              "GIAVA_CONTROL_HZ before recording.")
        print("=" * 70)


def _rgb_encoder_from_env():
    """GIAVA_VCODEC / GIAVA_VIDEO_GOP -> RGBEncoderConfig, or None to leave
    lerobot's defaults alone (see the note in create_dataset)."""
    _vcodec = os.environ.get("GIAVA_VCODEC", "").strip()
    _gop = os.environ.get("GIAVA_VIDEO_GOP", "").strip()
    if not (_vcodec or _gop):
        return None
    from lerobot.configs.video import RGBEncoderConfig
    kwargs = {}
    if _vcodec:
        kwargs["vcodec"] = _vcodec
    if _gop:
        kwargs["g"] = int(_gop)
    enc = RGBEncoderConfig(**kwargs)
    print(f"[dataset] rgb encoder: vcodec={enc.vcodec} g={enc.g} crf={enc.crf}")
    return enc


def resolve_resume_run(task_name, run):
    """Absolute path of the run to append to.  `run` is a timestamp directory
    name or 'latest'.  Raises rather than guessing: appending to the wrong run
    silently mixes two sessions' episodes into one dataset."""
    task_dir = Path(DATASET_ROOT) / task_name
    if run == "latest":
        runs = sorted(d for d in task_dir.glob("*/") if (d / "meta.json").exists())
        if not runs:
            raise SystemExit(f"--resume latest: no run with a meta.json under "
                             f"{task_dir}")
        return runs[-1]
    ## A path is accepted too -- the run directory as printed at startup or
    ## as tab-completed from the shell.  Passing one used to be silently
    ## joined onto task_dir and fail with a doubled path (2026-09-15).
    cand = [task_dir / run]
    if "/" in run:
        cand = [Path(run).expanduser().resolve(), Path(DATASET_ROOT).parent.parent / run] + cand
    for root in cand:
        if (root / "meta.json").exists():
            return root
    raise SystemExit(f"--resume {run}: no meta.json at any of "
                     f"{[str(c) for c in cand]}")


def create_dataset(task_name, mode, active_cameras, control_dt,
                   resume_run=None):
    repo_id = f"deviamar/{task_name}"

    if resume_run is not None:
        ## APPEND to an existing run.  Every frame of a LeRobot dataset shares
        ## one feature schema, so a session that resumes with a different arm
        ## mode, camera set or rate would write frames the earlier episodes
        ## cannot be loaded beside.  Check before opening anything.
        dataset_root = Path(resume_run)
        prev = json.loads((dataset_root / "meta.json").read_text())
        want = {"mode": mode, "active_cameras": list(active_cameras),
                "fps": round(1.0 / control_dt)}
        got = {"mode": prev.get("mode"),
               "active_cameras": list(prev.get("active_cameras") or []),
               "fps": prev.get("fps")}
        if got != want:
            raise SystemExit(
                f"--resume: this session does not match the run it would "
                f"append to.\n  run  {dataset_root}\n"
                f"  existing: {got}\n  this run: {want}\n"
                f"Start a new run, or relaunch with matching flags.")
        dataset = LeRobotDataset(
            repo_id,
            root=str(dataset_root),
            streaming_encoding=True,
            encoder_queue_maxsize=120,
            rgb_encoder=_rgb_encoder_from_env(),
        )
        n = int(dataset.meta.total_episodes)
        print(f"[dataset] RESUMING {dataset_root}\n"
              f"[dataset] {n} episode(s) already recorded; the next one is "
              f"episode_{n:04d}")
        return dataset, dataset_root

    run_name = time.strftime("%Y%m%d_%H%M%S")

    dataset_root = (
        Path(DATASET_ROOT)
        / task_name
        / run_name
    )

    _check_fps_against_previous_runs(Path(DATASET_ROOT) / task_name,
                                     round(1.0 / control_dt))

    features = build_dataset_features(mode, active_cameras)

    # CHANGED (lerobot v0.6.0): stream frames to per-camera encoder threads while
    # recording instead of writing PNGs and encoding at episode end. This makes
    # save_episode() near-instant and removes the temp-image round-trip, so the
    # image_writer_* settings are no longer needed.
    #
    # encoder_queue_maxsize is the per-camera frame backlog. If encoding cannot
    # keep up the queue applies back-pressure and frames can be dropped, so this
    # is deliberately generous relative to the default of 30.
    ## VIDEO CODEC.  lerobot's default is libsvtav1 at CRF 30 with GOP 2 --
    ## a keyframe every second frame, chosen so training-time random access
    ## into an episode is cheap.  Measured on this machine (32 cores, 6
    ## cameras at 640x480, synthetic worst-case content):
    ##
    ##   libsvtav1 g=2  (default)   3.3x realtime, 37 MB per camera-minute
    ##   libsvtav1 g=30             8.6x realtime,  6 MB
    ##   h264_nvenc g=2             5.6x realtime, 11 MB
    ##
    ## So the default is NOT slower than recording -- if a save takes longer
    ## than the episode did, the encoders were starved of CPU during
    ## collection and the queue drains afterwards.  Check the per-episode
    ## "[save] ... saved N frames in Xs" line before changing anything here.
    ##
    ## GIAVA_VCODEC=auto picks a hardware encoder (h264_nvenc on this box, both
    ## GPUs have NVENC and six concurrent sessions were verified to work).
    ## That is a free 1.7x with SMALLER files and no change to decode cost.
    ## Raising GIAVA_VIDEO_GOP is the bigger win but it is NOT free: bigger
    ## GOPs make the dataloader decode more frames per random access.
    ## Unset, both leave lerobot's defaults alone, so datasets stay comparable.
    rgb_encoder = _rgb_encoder_from_env()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(dataset_root),
        fps=round(1.0 / control_dt),
        features=features,
        streaming_encoding=True,
        encoder_queue_maxsize=120,
        rgb_encoder=rgb_encoder,
    )

    save_dataset_metadata(
        dataset_root,
        task_name,
        mode,
        active_cameras,
        control_dt,
    )

    return dataset, dataset_root

def get_state_dim(mode):
    return len(build_state_names(ARM_MODES[mode]))

def get_action_dim(mode):
    return len(build_action_names(ARM_MODES[mode]))

def build_frame(
    mode,
    active_cameras,
    robot_states,
    robot_actions,
    ee_poses,
    timestamps,
    images,
):
    active_arms = ARM_MODES[mode]

    frame = {}

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    for camera in active_cameras:

        if camera == "oak_left" or camera == "oak_right":

            frame["observation.images.oak_left"] = torch.from_numpy(
                images["oak_left"]
            )

            frame["observation.images.oak_right"] = torch.from_numpy(
                images["oak_right"]
            )

            frame["observation.timestamps.oak_left"] = torch.tensor(
                [timestamps["oak_left"]],
                dtype=torch.float64,
            )

            frame["observation.timestamps.oak_right"] = torch.tensor(
                [timestamps["oak_right"]],
                dtype=torch.float64,
            )

            continue

        frame[f"observation.images.{camera}"] = torch.from_numpy(
            images[camera]
        )

        frame[f"observation.timestamps.{camera}"] = torch.tensor(
            [timestamps[camera]],
            dtype=torch.float64,
        )

    # ------------------------------------------------------------------
    # Observation state
    # ------------------------------------------------------------------

    ## observation.state / .velocity / .effort all share ONE layout -- per
    ## arm, joints then gripper -- so they line up index for index and a
    ## velocity can be read against the position it describes.  Built by a
    ## single helper rather than three copies of the same loop, because three
    ## copies is how the velocity column ends up indexed differently from the
    ## position column and nothing ever tells you.
    def _flatten(joint_key, gripper_key, default=0.0):
        out = []
        for arm in active_arms:
            n = ARM_CONFIG[arm]["num_joints"]
            vals = robot_states[arm].get(joint_key)
            if vals is None:
                out.extend([default] * n)
            else:
                out.extend(np.asarray(vals, dtype=np.float32).reshape(-1)[:n].tolist())
            if ARM_CONFIG[arm]["has_gripper"]:
                g = robot_states[arm].get(gripper_key)
                out.append(default if g is None else float(g))
        return out

    observation_state = _flatten("joints", "gripper")

    frame["observation.state"] = torch.tensor(
        observation_state,
        dtype=torch.float32,
    )

    ## Zeros stand in when the driver did not supply a channel (a joint_states
    ## message shorter than expected).  A zero row is honest and keeps the
    ## schema fixed for the whole run; dropping the frame would throw away a
    ## real observation over a missing auxiliary channel.
    for _kind in ("velocity", "effort"):
        frame[f"observation.{_kind}"] = torch.tensor(
            _flatten(_kind, f"gripper_{_kind}"),
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    action = []

    for arm in active_arms:

        action.extend(
            robot_actions[arm]["joints"]
        )

        if ARM_CONFIG[arm]["has_gripper"]:
            action.append(
                robot_actions[arm]["gripper"]
            )

    frame["action"] = torch.tensor(
        action,
        dtype=torch.float32,
    )

    assert len(observation_state) == get_state_dim(mode), (
        f"Observation dimension mismatch: "
        f"{len(observation_state)} != {get_state_dim(mode)}"
    )

    assert len(action) == get_action_dim(mode), (
        f"Action dimension mismatch: "
        f"{len(action)} != {get_action_dim(mode)}"
    )

    # ------------------------------------------------------------------
    # EE poses
    # ------------------------------------------------------------------

    for arm in active_arms:

        frame[f"observation.ee_pose.{arm}"] = torch.tensor(
            ee_poses[arm],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Robot timestamps
    # ------------------------------------------------------------------

    for arm in active_arms:

        frame[f"observation.timestamps.{arm}"] = torch.tensor(
            [timestamps[arm]],
            dtype=torch.float64,
        )

    return frame

def save_dataset_metadata(dataset_root, task_name, mode, active_cameras, control_dt):
    metadata = {
        "task": task_name,
        "mode": mode,
        "active_arms": ARM_MODES[mode],
        "joint_names": build_state_names(ARM_MODES[mode]),
        "active_cameras": active_cameras,
        "state_dim": get_state_dim(mode),
        "action_dim": get_action_dim(mode),
        "fps": round(1.0 / control_dt),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(dataset_root / "meta.json", "w") as f:
        json.dump(metadata, f, indent=2)


## Standalone entry point, so a dataset recorded by an earlier session (or by
## a session that ended badly) can be checked without re-recording anything:
##
##   python dataset.py dataset/lerobot/transfer_flower/20260903_214256
##   python dataset.py --meta-only dataset/lerobot/<task>/*/     # instant
##
## Exits non-zero if any dataset fails, so it drops into a shell loop or CI.
if __name__ == "__main__":
    import argparse
    import sys

    ## allow_abbrev=False on purpose.  With argparse's default prefix
    ## matching, '--verify' silently resolves to '--meta-only'-style flags and
    ## the run SKIPS the video check while still printing "VERIFY OK" -- a
    ## checker that can quietly not check is worse than no checker.
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Verify a finalized LeRobot v3 dataset against its own "
                    "metadata: episode counts, data parquet rows, and the "
                    "frames actually muxed into each video file.")
    ap.add_argument("root", nargs="+", help="dataset run directory")
    ap.add_argument("--meta-only", action="store_true",
                    help="skip counting frames in the videos (no ffprobe); "
                         "checks metadata consistency alone, instantly")
    args = ap.parse_args()

    failed = 0
    for r in args.root:
        print(f"\n=== {r}")
        ok, lines = verify_finalized_dataset(
            r, probe_videos=not args.meta_only)
        for line in lines:
            print(f"  {line}")
        failed += 0 if ok else 1
    if args.meta_only:
        print("\n(metadata only -- videos were NOT counted; "
              "re-run without --meta-only to check them)")
    sys.exit(1 if failed else 0)
