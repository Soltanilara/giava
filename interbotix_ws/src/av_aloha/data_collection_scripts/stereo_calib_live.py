"""Live stereo-calibration support for the teleop capture loop.

WHY THIS EXISTS
================
`oak_stereo_calibrate.py` is the authority on the fit and stays that way.
What it cannot do is tell the operator anything WHILE they are holding the
board inside the headset, and that is where calibration quality is actually
decided.  Three things belong at capture time:

  1. HANDEDNESS.  Which physical lens is called "left" is a wiring fact that
     nothing in the pipeline verifies (camera_manager labels CAM_B "left" by
     DepthAI convention).  It was wrong on this rig for months: the installed
     stereo.npz carried T[0] = +62.63 mm where a correctly ordered pair must
     be negative, i.e. the right lens's image was going to the LEFT eye and
     the operator was flying on inverted depth.
     `disparity_sign` settles it from ONE pair with the board in both eyes.

  2. COVERAGE.  What limits a calibration is pose DIVERSITY, not pair count:
     the corners of the frame constrain the distortion coefficients, and
     tilted views are the only thing that separates focal length from
     distance.  93 pairs shot from the same standoff, all frontal, all
     centred, fit worse than 25 spread properly.  `CoverageTracker` says
     which cell is empty while the board is still in your hand -- the DROID
     collection protocol's idea, applied to a checkerboard.

  3. A FIT YOU CAN SEE.  "board found" is not "board useful".  Running the
     real calibration on what has been captured so far, and installing its
     rectification into the live stream, turns a number into something the
     operator can look at.

WHAT IS DELIBERATELY NOT HERE
==============================
No new calibration mathematics.  `calibrate_now` shells out to
`oak_stereo_calibrate.py` -- same outlier rejection, same model choice, same
guards -- so there is exactly one implementation of the fit and no chance of
the live path and the offline path disagreeing about what a good calibration
is.  It runs as a SUBPROCESS rather than a thread: a full fit is 1-3 s of
CPU, and cv2 does not release the GIL for all of it, which at 50 Hz is a
hundred missed control ticks.
"""

from __future__ import annotations

import math
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent


## --------------------------------------------------------------------- ##
## 1. Handedness
## --------------------------------------------------------------------- ##

def _board_centre_u(bgr, board) -> Optional[float]:
    """Mean horizontal pixel coordinate of the detected board, or None."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    found, corners = cv2.findChessboardCorners(gray, board, flags)
    if not found:
        return None
    return float(np.asarray(corners).reshape(-1, 2)[:, 0].mean())


def disparity_sign(left_bgr, right_bgr, board) -> Optional[float]:
    """u_left - u_right for the board, in pixels.  None if not in both eyes.

    THE ENTIRE TEST, AND WHY IT IS SOUND WITH NO CALIBRATION AT ALL
    ===============================================================
    A left camera is displaced to the left, so anything both cameras see
    appears FURTHER RIGHT in the left image.  So for a correctly labelled
    pair, u_left > u_right for every feature -- positive disparity, always,
    for any scene at any depth in front of the cameras.  It needs no
    intrinsics, no extrinsics, no board geometry; it is pure sign.

    A NEGATIVE value means the image being called "left" was taken by the
    physically-right camera.  On this rig that measured -102 px, on 25 of 25
    pairs (2026-08-27).
    """
    ul = _board_centre_u(left_bgr, board)
    if ul is None:
        return None
    ur = _board_centre_u(right_bgr, board)
    if ur is None:
        return None
    return ul - ur


def handedness_warning(disp: float) -> str:
    """The message for a negative disparity.  Deliberately loud."""
    return (
        f"\n[HANDEDNESS] The eye labels are SWAPPED.\n"
        f"  Board disparity (u_left - u_right) = {disp:+.1f} px. It must be "
        f"POSITIVE:\n"
        f"  a left camera sees everything further RIGHT in its own image.\n"
        f"  Negative means the frame being called 'left' comes from the "
        f"physically-RIGHT lens,\n"
        f"  so the headset is showing you INVERTED DEPTH -- near reads as "
        f"far. Rectification\n"
        f"  does not fix this and no calibration metric can see it "
        f"(vertical-disparity quality\n"
        f"  is unchanged by a swap).\n"
        f"  Type  swap  to exchange the eye labels now, then recapture.")


## --------------------------------------------------------------------- ##
## 2. Coverage
## --------------------------------------------------------------------- ##

## Where the board must go.  Three bands each way over the LEFT image, which
## is what constrains the distortion model -- k1/k2/k3 are only observable
## where the distortion is large, i.e. at the edges and corners, and a
## capture that never leaves the middle third fits them from noise.
GRID = 3
## Fraction of pairs that must be genuinely oblique.  Tilt is what separates
## focal length from distance: on a frontal-only capture those two trade off
## almost exactly, which is how a fit reaches a low RMS with a field of view
## several degrees wrong.
TILT_TARGET = 0.35
TILT_OBLIQUE = 0.18       # foreshortening at roughly 30 deg
## Board apparent size, as a fraction of the frame's smaller dimension.
## Near AND far are both needed, for the same reason as tilt.
NEAR_FRAC = 0.45
FAR_FRAC = 0.22

_CELL_NAMES = (("top-left", "top-centre", "top-right"),
               ("mid-left", "CENTRE", "mid-right"),
               ("bottom-left", "bottom-centre", "bottom-right"))


def board_metrics(bgr, board) -> Optional[Dict]:
    """Where the board is, how big, and how oblique.  None if not found.

    FORESHORTENING WITHOUT INTRINSICS.  Tilt would normally come from
    solvePnP, which needs the camera matrix -- the thing being calibrated.
    Instead this uses the fact that a plane viewed head-on projects to a
    parallelogram, so its opposite sides are equal; perspective breaks that
    equality in proportion to tilt.  1 - (shorter/longer), maximised over
    both pairs of opposite sides, is 0 for a frontal view and grows with
    obliquity.  It is a proxy, not an angle -- which is all a coverage
    prompt needs.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    found, corners = cv2.findChessboardCorners(gray, board, flags)
    if not found:
        return None
    cols, rows = board
    pts = np.asarray(corners).reshape(rows, cols, 2)
    h, w = gray.shape[:2]

    tl, tr = pts[0, 0], pts[0, -1]
    bl, br = pts[-1, 0], pts[-1, -1]
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    leftE = np.linalg.norm(bl - tl)
    rightE = np.linalg.norm(br - tr)
    fore = max(1.0 - min(top, bottom) / max(top, bottom, 1e-6),
               1.0 - min(leftE, rightE) / max(leftE, rightE, 1e-6))

    centre = pts.reshape(-1, 2).mean(axis=0)
    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    area = abs(float(cv2.contourArea(quad)))
    size_frac = math.sqrt(area) / max(min(h, w), 1)

    ## Coverage is counted over the CORNERS, not the board centre.  What
    ## constrains k1/k2/k3 is having observations where the distortion is
    ## large, i.e. at particular PIXELS -- so a big board held centrally does
    ## cover the mid-band, and binning by its centre alone would wrongly call
    ## that "all in the middle".  This is the same thing the offline tool's
    ## coverage grid counts, so the two agree about what is missing.
    flat = pts.reshape(-1, 2)
    cx = np.clip((flat[:, 0] / w * GRID).astype(int), 0, GRID - 1)
    cy = np.clip((flat[:, 1] / h * GRID).astype(int), 0, GRID - 1)
    hit = np.zeros((GRID, GRID), dtype=bool)
    hit[cy, cx] = True

    gx = min(int(centre[0] / w * GRID), GRID - 1)
    gy = min(int(centre[1] / h * GRID), GRID - 1)
    return {"cell": (gy, gx), "cells_hit": hit, "foreshortening": float(fore),
            "size_frac": float(size_frac), "centre": (float(centre[0]),
                                                      float(centre[1]))}


class CoverageTracker:
    """What has been captured, and what to tell the operator to do next."""

    def __init__(self, grid: int = GRID):
        self.grid = grid
        self.cells = np.zeros((grid, grid), dtype=int)
        self.oblique = 0
        self.near = 0
        self.far = 0
        self.total = 0
        self._lock = threading.Lock()

    def add(self, m: Dict) -> None:
        with self._lock:
            self.total += 1
            self.cells += m["cells_hit"].astype(int)
            if m["foreshortening"] >= TILT_OBLIQUE:
                self.oblique += 1
            if m["size_frac"] >= NEAR_FRAC:
                self.near += 1
            elif m["size_frac"] <= FAR_FRAC:
                self.far += 1

    def _weakest_cell(self) -> Tuple[int, int]:
        with self._lock:
            k = int(np.argmin(self.cells))
        return divmod(k, self.grid)

    def instruction(self) -> str:
        """ONE thing to do next.  Not a list.

        A capture prompt competes with the operator flying two arms inside a
        headset, so it names the single weakest axis and stops.  Ranked by
        what costs the fit most when missing: an empty frame region (the
        distortion model has nothing to fit there), then tilt, then depth
        spread."""
        with self._lock:
            total, cells = self.total, self.cells.copy()
            oblique, near, far = self.oblique, self.near, self.far
        if total == 0:
            return ("hold the board in the CENTRE, filling about half the "
                    "view, and press 's'")
        gy, gx = divmod(int(np.argmin(cells)), self.grid)
        if int(cells[gy, gx]) < max(2, total // 12):
            return (f"move the board to the {_CELL_NAMES[gy][gx]} of the view "
                    f"({int(cells[gy, gx])} pairs there) -- the frame edges "
                    f"are where the distortion model is actually constrained")
        if oblique < TILT_TARGET * total:
            return (f"TILT the board ~30-40 deg (only {oblique}/{total} of "
                    f"your pairs are oblique) -- without tilt the fit cannot "
                    f"separate focal length from distance")
        if near < max(3, total // 8):
            return ("bring the board CLOSER, filling most of the view")
        if far < max(3, total // 8):
            return ("take the board FURTHER back, to about a quarter of the "
                    "view")
        return ("coverage looks good -- type  cal  to fit, or keep adding "
                "oblique corner poses")

    def report(self) -> str:
        with self._lock:
            cells, total = self.cells.copy(), self.total
            oblique, near, far = self.oblique, self.near, self.far
        lines = [f"[coverage] {total} pairs with the board found in the left "
                 f"eye; cells show how many of them put CORNERS there"]
        for gy in range(self.grid):
            lines.append("    " + "  ".join(f"{int(cells[gy, gx]):3d}"
                                            for gx in range(self.grid)))
        lines.append(f"    oblique {oblique}/{total} (want "
                     f">={int(TILT_TARGET * max(total, 1))})   "
                     f"near {near}   far {far}")
        lines.append(f"    NEXT: {self.instruction()}")
        return "\n".join(lines)


## --------------------------------------------------------------------- ##
## 3. Fit now, and install it live
## --------------------------------------------------------------------- ##

def calibrate_now(capture_dir: Path, board, square_m: float,
                  out_npz: Optional[Path] = None,
                  swap: bool = False, min_pairs: int = 12,
                  timeout_s: float = 300.0) -> Tuple[bool, Optional[Path], str]:
    """Run the real calibrator on what has been captured.  (ok, npz, output).

    A SUBPROCESS, not a thread: a full fit is 1-3 s of CPU and cv2 does not
    release the GIL throughout, which at 50 Hz is ~100 dropped control ticks
    with two arms live.  A separate process cannot stall the loop at all --
    the OS schedules it on another core and the worst case is that it is
    slow, not that the arms stutter.
    """
    out_npz = Path(out_npz or (capture_dir / "stereo_live.npz"))
    cmd = [sys.executable, str(_HERE / "oak_stereo_calibrate.py"),
           "--dir", str(capture_dir),
           "--board", f"{board[0]}x{board[1]}",
           "--square", str(square_m),
           "--out", str(out_npz),
           "--min-pairs", str(min_pairs),
           "--samples", "0"]
    if swap:
        cmd.append("--swap")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, None, f"calibration timed out after {timeout_s:.0f} s"
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0 or not out_npz.exists():
        return False, None, out
    return True, out_npz, out


def summarise_fit(text: str) -> str:
    """The handful of lines from the calibrator worth reading in a headset."""
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("Final fit", "left  rms", "right rms", "stereo rms",
                         "baseline", "inter-camera", "mean ", "  --> ",
                         "rectified field", "[HANDEDNESS")):
            keep.append("    " + s)
    return "\n".join(keep) if keep else "    (no summary lines parsed)"
