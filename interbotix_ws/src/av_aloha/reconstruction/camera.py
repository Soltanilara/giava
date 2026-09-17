"""One camera: intrinsics, distortion, and a pose in some reference frame.

STAGE TWO of the GIAVA chain.  `calibration/` establishes and measures
everything up to and including camera extrinsics; this package consumes
what it wrote and turns pixels into metres.  Nothing here writes a
calibration -- if a number is missing this raises rather than inventing one.


THE DISTORTION TRAP THIS FILE EXISTS TO AVOID
=============================================
There are two different, mutually inverse, conventions in play on this rig
and mixing them is silent -- the numbers stay plausible and are simply
wrong by a few pixels at the image edge.

    OpenCV  (plumb_bob / brown_conrady, what charuco_calibrate.py writes)
        the polynomial runs  RAY -> PIXEL.
        Distorting is closed form; undistorting must be iterated.

    librealsense (inverse_brown_conrady, what the D405 colour stream
        reports and what rs_intrinsics.py recorded)
        the polynomial runs  PIXEL -> RAY.
        Undistorting is closed form; distorting must be iterated.

Same five numbers, same symbols k1 k2 p1 p2 k3, opposite directions.
`compare_intrinsics.py` refuses to difference them term by term for exactly
this reason.

So this module never exposes "the coefficients".  It exposes two operations

    unproject(uv)      pixel -> normalised ray      (x/z, y/z)
    project(xyz)       camera-frame point -> pixel

and each model supplies whichever direction it has in closed form while
the other is obtained by damped fixed-point iteration to a stated
tolerance.  Every consumer of this file -- triangulation, epipolar curves,
reprojection error -- is then model-agnostic and correct for both.

The iteration is verified against the closed form in `selftest.py`, and
`unproject` / `project` are checked to be inverse to <1e-4 px over the
whole image for the real D405 coefficients.


CONVENTIONS (identical to calibration/common.py -- see its docstring)
====================================================================
    T_a_b       maps points from frame b into frame a; IS the pose of b in a.
    optical     OpenCV: +x image-right, +y image-down, +z along the view axis.
    units       metres, radians. Quaternions wxyz.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

## Reach calibration/ for common.py, so the two packages agree on
## save_json/provenance/SE(3) rather than growing a second copy.
_HERE = Path(__file__).resolve().parent
_CALIB = _HERE.parent / "calibration"
for _p in (str(_HERE), str(_CALIB), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


## ------------------------------------------------------------------ ##
## Distortion models
## ------------------------------------------------------------------ ##

## Model name -> which direction the polynomial runs.
##   "ray_to_pixel"  OpenCV convention: apply the polynomial when DISTORTING
##   "pixel_to_ray"  librealsense inverse convention: apply it when UNDISTORTING
_MODEL_DIRECTION = {
    "none": None,
    "distortion.none": None,
    "plumb_bob": "ray_to_pixel",
    "opencv_plumb_bob": "ray_to_pixel",
    "opencv_rational": "ray_to_pixel",
    "opencv_no_k3": "ray_to_pixel",
    "brown_conrady": "ray_to_pixel",
    "modified_brown_conrady": "ray_to_pixel",
    "distortion.modified_brown_conrady": "ray_to_pixel",
    "rational": "ray_to_pixel",
    "rational_polynomial": "ray_to_pixel",
    "no_k3": "ray_to_pixel",
    "inverse_brown_conrady": "pixel_to_ray",
    "distortion.inverse_brown_conrady": "pixel_to_ray",
}


def normalize_model(name: str) -> str:
    """Canonical model key, tolerating the 'distortion.' prefix rs uses.

    Canonicalising rather than merely validating: two spellings of the same
    model must compare equal, or a downstream `model == "..."` check
    quietly takes the wrong branch."""
    key = str(name).strip().lower()
    ## librealsense stringifies its enum as e.g. "distortion.inverse_brown_conrady".
    if key.startswith("distortion."):
        key = key[len("distortion."):]
    if key not in _MODEL_DIRECTION:
        raise ValueError(
            f"unknown distortion model '{name}'.\n"
            f"  Known: {sorted(_MODEL_DIRECTION)}\n"
            f"  A model this file does not know is a model whose direction "
            f"it cannot guess, and guessing is exactly the bug this module "
            f"exists to prevent.")
    return key


## Beyond this normalised radius the polynomial is meaningless -- r = 100
## is 89.4 degrees off axis, far outside any real lens -- and r^6 starts to
## overflow.  Points past it come back NaN rather than as a huge number
## that would look like a plausible pixel somewhere off-screen.
_R2_MAX = 1.0e4

## Acceptance tolerance for the inversion, in normalised units. 1e-9 is
## ~0.4 nanometres of pixel at this focal length -- far below any real
## error, so anything failing it genuinely did not converge.
_INVERT_ACCEPT = 1.0e-9


def _brown_conrady(xy: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """The Brown-Conrady polynomial, applied to normalised coordinates.

    This is one function used in BOTH directions -- which direction it
    MEANS is decided by the model, not by the maths.  Matches OpenCV's
    projectPoints and librealsense's rs2_deproject_pixel_to_point for
    inverse_brown_conrady, which are the same expression.

    Supports 5 coefficients (k1 k2 p1 p2 k3) and 8 (adding the rational
    denominator k4 k5 k6), which is what --model rational produces.

    Points where the model does not apply -- absurd radius, or a rational
    denominator crossing zero -- come back as NaN **per element**, not as
    an exception.  That distinction matters: this is called on whole
    epipolar curves, most of whose samples legitimately fall outside the
    destination image, and raising would discard the valid samples along
    with the invalid ones.  NaN is not silent: `in_image` rejects it and
    every consumer here propagates it visibly."""
    c = np.zeros(8)
    c[:len(coeffs)] = coeffs
    k1, k2, p1, p2, k3, k4, k5, k6 = c

    x, y = xy[..., 0], xy[..., 1]
    with np.errstate(over="ignore", invalid="ignore"):
        r2 = x * x + y * y
        bad = ~np.isfinite(r2) | (r2 > _R2_MAX)
        r2 = np.where(bad, 0.0, r2)
        r4 = r2 * r2
        r6 = r4 * r2

        radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6)
        denom = (1.0 + k4 * r2 + k5 * r4 + k6 * r6)
        ## The rational denominator can cross zero at large radius, which
        ## would mirror the point to the far side of the image.
        bad = bad | (np.abs(denom) < 1e-9)
        radial = radial / np.where(np.abs(denom) < 1e-9, 1.0, denom)

        dx = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = y * radial + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
    out = np.stack([np.where(bad, np.nan, dx), np.where(bad, np.nan, dy)],
                   axis=-1)
    return out


def _invert_brown_conrady(xy_target: np.ndarray, coeffs: np.ndarray,
                          iters: int = 30,
                          tol: float = 1e-10,
                          damping: float = 0.9) -> np.ndarray:
    """Find xy such that _brown_conrady(xy) == xy_target.

    Damped fixed-point iteration.  Undamped Newton-free iteration diverges
    near the corners of a wide lens; the damping factor keeps it stable at
    the cost of a few more cycles, and 30 cycles is far more than the ~6
    the D405 coefficients actually need (checked in selftest.py).

    Raises rather than returning a half-converged answer: a silently
    unconverged undistort is a millimetre-scale error downstream that
    nothing else would ever flag."""
    xy_target = np.atleast_2d(np.asarray(xy_target, dtype=float))
    xy = xy_target.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(iters):
            err = _brown_conrady(xy, coeffs) - xy_target
            finite = np.isfinite(err)
            if np.all(finite) and err.size and np.max(np.abs(err)) < tol:
                return xy
            ## The exact Jacobian is close to I for a lens like this, so a
            ## damped unit step converges quickly; the damping is what
            ## keeps it stable out at the corners where the radial term is
            ## largest. Rows that have gone non-finite are frozen rather
            ## than allowed to contaminate the rest of the batch.
            xy = xy - damping * np.where(finite, err, 0.0)

        ## Per-element verdict. A row that did not converge is NOT returned
        ## half-solved -- that would be a silent millimetre-scale error
        ## downstream, which is the whole reason this check exists.
        final = _brown_conrady(xy, coeffs) - xy_target
    ok = (np.isfinite(final).all(axis=-1)
          & (np.max(np.abs(final), axis=-1) < _INVERT_ACCEPT))
    return np.where(ok[..., None], xy, np.nan)


## ------------------------------------------------------------------ ##
## Camera
## ------------------------------------------------------------------ ##

@dataclass
class PinholeCamera:
    """Intrinsics + distortion + (optionally) a pose in a reference frame.

    `T_ref_cam` is the camera's pose in whatever frame the rig is anchored
    to -- the robot world frame `base` once the rig has been anchored, or
    the reference camera's own optical frame before that.  It is None for a
    camera whose extrinsics are not known, and every geometric method that
    needs it raises a named error rather than assuming identity."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "none"
    coeffs: Tuple[float, ...] = ()
    T_ref_cam: Optional[np.ndarray] = None
    ref_frame: str = "unknown"
    serial: Optional[str] = None
    source: str = "unspecified"
    provenance: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------- #
    def __post_init__(self) -> None:
        self.model = normalize_model(self.model)
        self.coeffs = tuple(float(c) for c in self.coeffs)
        if self.T_ref_cam is not None:
            self.T_ref_cam = np.asarray(self.T_ref_cam, dtype=float)
            if self.T_ref_cam.shape != (4, 4):
                raise ValueError(f"{self.name}: T_ref_cam must be 4x4")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"{self.name}: fx/fy must be positive")

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=float)

    @property
    def has_pose(self) -> bool:
        return self.T_ref_cam is not None

    @property
    def centre(self) -> np.ndarray:
        """Optical centre in the reference frame."""
        self._require_pose()
        return self.T_ref_cam[:3, 3].copy()

    def _require_pose(self) -> None:
        if self.T_ref_cam is None:
            raise ValueError(
                f"camera '{self.name}' has no extrinsic pose.\n"
                f"  Calibrate it (calibration/scene_extrinsics.py) and record "
                f"the result; this package does not invent extrinsics.")

    # ------------------------- intrinsics ------------------------- #
    def unproject(self, uv) -> np.ndarray:
        """Pixel(s) -> normalised ray(s) (x/z, y/z), distortion removed.

        Shape in (..., 2) -> shape out (..., 2).  This is the primitive:
        everything geometric in this package starts here."""
        uv = np.asarray(uv, dtype=float)
        single = uv.ndim == 1
        pts = np.atleast_2d(uv)
        xy = np.stack([(pts[:, 0] - self.cx) / self.fx,
                       (pts[:, 1] - self.cy) / self.fy], axis=-1)

        direction = _MODEL_DIRECTION[self.model]
        if direction is None or not np.any(np.asarray(self.coeffs)):
            out = xy
        elif direction == "pixel_to_ray":
            ## librealsense inverse_brown_conrady: closed form THIS way.
            out = _brown_conrady(xy, np.asarray(self.coeffs))
        else:
            ## OpenCV: the polynomial runs the other way, so invert it.
            out = _invert_brown_conrady(xy, np.asarray(self.coeffs))
        return out[0] if single else out

    def project(self, xyz) -> np.ndarray:
        """Camera-frame point(s) -> pixel(s), distortion applied.

        Points at or behind the optical centre have no pixel; they come
        back as NaN rather than as a plausible-looking wrong answer."""
        xyz = np.asarray(xyz, dtype=float)
        single = xyz.ndim == 1
        pts = np.atleast_2d(xyz)

        z = pts[:, 2]
        valid = z > 1e-9
        xy = np.full((len(pts), 2), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            xy[valid] = pts[valid, :2] / z[valid, None]

        out = np.full((len(pts), 2), np.nan)
        if np.any(valid):
            xyv = xy[valid]
            direction = _MODEL_DIRECTION[self.model]
            if direction is None or not np.any(np.asarray(self.coeffs)):
                d = xyv
            elif direction == "ray_to_pixel":
                d = _brown_conrady(xyv, np.asarray(self.coeffs))
            else:
                d = _invert_brown_conrady(xyv, np.asarray(self.coeffs))
            out[valid] = np.stack([d[:, 0] * self.fx + self.cx,
                                   d[:, 1] * self.fy + self.cy], axis=-1)
        return out[0] if single else out

    def in_image(self, uv, margin: float = 0.0) -> np.ndarray:
        """Which pixels fall inside the sensor (NaN counts as outside)."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        with np.errstate(invalid="ignore"):
            ok = ((uv[:, 0] >= -margin) & (uv[:, 0] <= self.width - 1 + margin)
                  & (uv[:, 1] >= -margin)
                  & (uv[:, 1] <= self.height - 1 + margin))
        return ok & np.isfinite(uv).all(axis=1)

    # -------------------------- geometry -------------------------- #
    def ray(self, uv) -> Tuple[np.ndarray, np.ndarray]:
        """A pixel as a ray in the REFERENCE frame: (origin, unit direction).

        The origin is the optical centre and is the same for every pixel;
        it is returned per call so callers never have to remember that."""
        self._require_pose()
        xy = np.atleast_2d(self.unproject(uv))
        d_cam = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
        d_ref = d_cam @ self.T_ref_cam[:3, :3].T
        d_ref = d_ref / np.linalg.norm(d_ref, axis=1, keepdims=True)
        o = np.broadcast_to(self.T_ref_cam[:3, 3], d_ref.shape)
        if np.asarray(uv).ndim == 1:
            return o[0].copy(), d_ref[0]
        return o.copy(), d_ref

    def point_at_depth(self, uv, depth_m: float) -> np.ndarray:
        """The reference-frame point on this pixel's ray at a given depth.

        `depth_m` is depth ALONG THE OPTICAL AXIS (the camera-frame z), not
        distance from the centre -- that is what a depth image stores and
        what RealSense's own deprojection means."""
        xy = np.atleast_2d(self.unproject(uv))
        p_cam = np.concatenate([xy * depth_m, np.full((len(xy), 1), depth_m)],
                               axis=1)
        if self.T_ref_cam is None:
            out = p_cam
        else:
            out = p_cam @ self.T_ref_cam[:3, :3].T + self.T_ref_cam[:3, 3]
        return out[0] if np.asarray(uv).ndim == 1 else out

    def world_to_cam(self, p_ref) -> np.ndarray:
        """Reference-frame point(s) -> this camera's optical frame."""
        self._require_pose()
        p = np.atleast_2d(np.asarray(p_ref, dtype=float))
        R = self.T_ref_cam[:3, :3]
        t = self.T_ref_cam[:3, 3]
        out = (p - t) @ R
        return out[0] if np.asarray(p_ref).ndim == 1 else out

    def project_ref(self, p_ref) -> np.ndarray:
        """Reference-frame point(s) -> pixel(s)."""
        return self.project(self.world_to_cam(p_ref))

    # --------------------------- report --------------------------- #
    def fov_deg(self) -> Tuple[float, float]:
        return (float(np.degrees(2 * np.arctan(self.width / (2 * self.fx)))),
                float(np.degrees(2 * np.arctan(self.height / (2 * self.fy)))))

    def scale_at(self, depth_m: float) -> float:
        """Metres per pixel at a given depth, on the optical axis.

        The single most useful sanity number: it converts a click error in
        pixels into a measurement error in millimetres."""
        return float(depth_m / ((self.fx + self.fy) / 2.0))

    def describe(self) -> str:
        fx_deg, fy_deg = self.fov_deg()
        lines = [
            f"{self.name}"
            + (f"  (serial {self.serial})" if self.serial else ""),
            f"    {self.width}x{self.height}  "
            f"fx {self.fx:.3f}  fy {self.fy:.3f}  "
            f"cx {self.cx:.3f}  cy {self.cy:.3f}",
            f"    fov {fx_deg:.2f} x {fy_deg:.2f} deg   model {self.model}",
            f"    intrinsics from: {self.source}",
        ]
        direction = _MODEL_DIRECTION[self.model]
        if direction is not None and np.any(np.asarray(self.coeffs)):
            lines.append(
                f"    coeffs run {direction.replace('_', ' ')}: "
                + ", ".join(f"{c:+.6f}" for c in self.coeffs))
        if self.T_ref_cam is None:
            lines.append("    pose: *** UNKNOWN -- not calibrated ***")
        else:
            t = self.T_ref_cam[:3, 3]
            lines.append(
                f"    pose in '{self.ref_frame}': "
                f"[{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}] m "
                f"({self.provenance})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "serial": self.serial,
            "width": self.width,
            "height": self.height,
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "model": self.model,
            "coeffs": list(self.coeffs),
            "distortion_direction": _MODEL_DIRECTION[self.model],
            "intrinsics_source": self.source,
            "ref_frame": self.ref_frame,
            "provenance": self.provenance,
            "T_ref_cam_row_major": (None if self.T_ref_cam is None
                                    else self.T_ref_cam.tolist()),
        }


## ------------------------------------------------------------------ ##
## Loading intrinsics from what calibration/ already wrote
## ------------------------------------------------------------------ ##

def from_intrinsics_dict(name: str, d: Dict[str, Any], *,
                         source: str = "dict",
                         serial: Optional[str] = None,
                         **kwargs) -> PinholeCamera:
    """Build a camera from an rs_intrinsics-shaped dict."""
    return PinholeCamera(
        name=name,
        width=int(d["width"]), height=int(d["height"]),
        fx=float(d["fx"]), fy=float(d["fy"]),
        cx=float(d.get("cx", d.get("ppx"))),
        cy=float(d.get("cy", d.get("ppy"))),
        model=d.get("model", "none"),
        coeffs=tuple(d.get("coeffs", ())),
        serial=serial, source=source, **kwargs)


def load_factory(camera: str, path: Optional[Path] = None,
                 stream: str = "color", **kwargs) -> PinholeCamera:
    """Newest `factory_intrinsics_*.json` for a camera (rs_intrinsics.py).

    These are the manufacturer's numbers read off the device.  They are
    real calibrations, not defaults -- but they are inverse_brown_conrady
    and were taken at the factory, not on this rig."""
    from common import DIR_CAMERAS, latest_matching, load_json

    if path is None:
        cand = sorted(Path(DIR_CAMERAS).glob(f"{camera}_*"))
        if not cand:
            raise FileNotFoundError(
                f"no intrinsics directory for '{camera}' under {DIR_CAMERAS}.\n"
                f"  Record them first:  python calibration/rs_intrinsics.py "
                f"--cameras {camera} --all")
        path = latest_matching(cand[-1], "factory_intrinsics_*.json")
        if path is None:
            raise FileNotFoundError(
                f"no factory_intrinsics_*.json in {cand[-1]}")

    data = load_json(Path(path))
    streams = data.get("streams", {})
    if stream not in streams:
        raise KeyError(
            f"{path} has no '{stream}' stream (has: {sorted(streams)})")
    return from_intrinsics_dict(
        camera, streams[stream]["intrinsics"],
        source=f"factory:{Path(path).name}",
        serial=data.get("serial"), **kwargs)


def load_charuco(camera: str, path: Optional[Path] = None,
                 **kwargs) -> PinholeCamera:
    """Newest OpenCV ChArUco calibration (charuco_calibrate.py).

    Preferred over factory when it exists: it was measured on THIS rig, at
    the resolution actually used, and its model is the OpenCV one that the
    rest of the CV world assumes."""
    from common import DIR_CAMERAS, latest_matching, load_json

    if path is None:
        cand = sorted(Path(DIR_CAMERAS).glob(f"{camera}_*"))
        if not cand:
            raise FileNotFoundError(f"no intrinsics directory for '{camera}'")
        path = latest_matching(cand[-1], "charuco_intrinsics_*.json")
        if path is None:
            raise FileNotFoundError(
                f"no charuco_intrinsics_*.json for '{camera}' in {cand[-1]}.\n"
                f"  Run charuco_capture.py then charuco_calibrate.py, or "
                f"fall back to load_factory().")

    data = load_json(Path(path))
    K = np.asarray(data["camera_matrix"], dtype=float)
    size = data["image_size"]
    return PinholeCamera(
        name=camera,
        width=int(size["width"]), height=int(size["height"]),
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        model=data.get("distortion_model", "opencv_plumb_bob"),
        coeffs=tuple(np.asarray(data["distortion_coefficients"],
                                dtype=float).ravel()),
        serial=data.get("serial"),
        source=f"charuco:{Path(path).name}",
        meta={"rms_px": data.get("rms_reprojection_error_px"),
              "n_images": data.get("n_images_used"),
              "board": data.get("board")},
        **kwargs)


def load_intrinsics(camera: str, prefer: str = "charuco",
                    **kwargs) -> PinholeCamera:
    """ChArUco if this rig has one, else the factory numbers.

    Says out loud which it used -- the choice changes every metric result
    downstream and must never be silent."""
    order = (["charuco", "factory"] if prefer == "charuco"
             else ["factory", "charuco"])
    errors = []
    for which in order:
        try:
            fn = load_charuco if which == "charuco" else load_factory
            return fn(camera, **kwargs)
        except (FileNotFoundError, KeyError) as e:
            errors.append(f"    {which}: {e}")
    raise FileNotFoundError(
        f"no intrinsics available for '{camera}':\n" + "\n".join(errors))
