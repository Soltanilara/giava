"""Turning pixels into metres: epipolar geometry and triangulation.

WHAT ONE CLICK BUYS YOU, AND WHAT IT DOES NOT
=============================================
A single pixel in a single camera is **a ray, not a point**.  Every
position along that ray produces the same pixel, so no amount of care with
intrinsics recovers the missing depth.  This is not a limitation of the
calibration; it is what a projection is.

There are exactly three honest ways out, and this module implements all
three so a measurement can be cross-checked rather than trusted:

  1. **A second view.**  The ray from `top_scene` projects into
     `low_scene` as the epipolar curve (`epipolar_curve`).  Every possible
     3D position of the clicked point lies on it, and each point of the
     curve is labelled with the depth that produced it.  Click the same
     feature there and the two rays intersect -- `triangulate`.  With two
     near-perpendicular cameras this is extremely well conditioned, which
     is the whole reason the rig is arranged that way.

  2. **A plane constraint.**  If the point is known to lie on a surface --
     the tabletop, a board -- intersecting the single ray with that plane
     gives the position outright (`ray_plane`).  This is what makes the
     top camera alone usable for "where on the table is that", and it is
     exact, not an approximation, as long as the point really is on the
     plane.

  3. **A known size.**  Two clicks on the ends of something whose length
     you know fixes the depth of that something (`depth_from_known_size`).
     Independent of the extrinsics entirely, which is what makes it a
     *check* on them rather than another consumer of them.

The word "epipolar line" is used loosely everywhere; with real lens
distortion it is a **curve**, and calling it a line is a several-pixel
error at the image edge on a 78-degree D405.  This module never
approximates it: the ray is sampled in 3D and every sample is projected
through the full distortion model of the destination camera.


HOW GOOD IS THE ANSWER
======================
Every result carries its own error bars, because a triangulated point
without them is not a measurement:

    ray_gap_mm          how close the two rays actually came to meeting.
                        Nonzero always. LARGE means you clicked two
                        different things, or the extrinsics are wrong.
    reprojection_px     where the answer lands back in each image
                        versus where you clicked.
    sigma_mm            what a plausible click error (default 1 px)
                        does to the answer, as a 3x3 covariance and its
                        principal axes.
    condition           geometric conditioning. Two cameras viewing along
                        nearly the same direction cannot separate depth,
                        and this says so numerically instead of returning
                        a confident wrong number.

`ray_gap_mm` and `sigma_mm` answer different questions and both matter:
the first is "are these two clicks consistent", the second is "how much
would being one pixel off cost me".


FRAMES
======
Every camera in a rig shares one reference frame, named in `ref_frame`.
Two are meaningful here:

    "base"          the robot world frame, once the rig has been anchored
                    to the robot (calibration/scene_extrinsics.py anchor).
                    Measurements come out directly comparable with FK.
    "<camera>"      before anchoring: the reference camera's own optical
                    frame. DISTANCES AND SIZES ARE ALREADY CORRECT HERE --
                    only their expression in robot coordinates is missing.
                    Measure first, anchor later.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_CALIB = _HERE.parent / "calibration"
for _p in (str(_HERE), str(_CALIB), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from camera import PinholeCamera  # noqa: E402


## ------------------------------------------------------------------ ##
## Results
## ------------------------------------------------------------------ ##

@dataclass
class Triangulation:
    """A 3D point plus every number needed to judge whether to believe it."""
    point: np.ndarray                       # (3,) in the rig reference frame
    cameras: List[str]
    pixels: Dict[str, np.ndarray]           # what was clicked
    reprojected: Dict[str, np.ndarray]      # where the answer lands
    reprojection_px: Dict[str, float]
    depth_m: Dict[str, float]               # along each camera's optical axis
    ray_gap_m: float
    condition: float
    covariance: Optional[np.ndarray] = None  # (3,3) m^2 for the assumed sigma
    sigma_px: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------- #
    @property
    def ray_gap_mm(self) -> float:
        return float(self.ray_gap_m * 1e3)

    @property
    def sigma_mm(self) -> Optional[np.ndarray]:
        """Per-axis 1-sigma position uncertainty, millimetres."""
        if self.covariance is None:
            return None
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None)) * 1e3

    @property
    def sigma_axes_mm(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Principal uncertainty axes: (lengths_mm, directions 3x3 columns).

        Worth looking at rather than the per-axis numbers: triangulation
        error is nearly always a cigar pointing along the worse-conditioned
        direction, and the axis-aligned sigmas hide that."""
        if self.covariance is None:
            return None
        evals, evecs = np.linalg.eigh(self.covariance)
        order = np.argsort(evals)[::-1]
        return (np.sqrt(np.clip(evals[order], 0.0, None)) * 1e3,
                evecs[:, order])

    @property
    def worst_reprojection_px(self) -> float:
        return max(self.reprojection_px.values()) if self.reprojection_px else 0.0

    def quality(self, gap_warn_mm: float = 5.0,
                reproj_warn_px: float = 2.0) -> Tuple[str, List[str]]:
        """A verdict and the reasons for it.

        Deliberately conservative. The failure this guards against is a
        confident millimetre readout produced from two clicks on different
        objects, which looks completely normal in the 3D view."""
        problems = []
        if self.ray_gap_mm > gap_warn_mm:
            problems.append(
                f"rays missed each other by {self.ray_gap_mm:.1f} mm "
                f"(> {gap_warn_mm:.1f}) -- most likely the two clicks are "
                f"not the same physical point")
        if self.worst_reprojection_px > reproj_warn_px:
            problems.append(
                f"reprojects {self.worst_reprojection_px:.2f} px from where "
                f"you clicked (> {reproj_warn_px:.1f})")
        if self.condition > 50.0:
            problems.append(
                f"poorly conditioned (condition {self.condition:.0f}) -- the "
                f"cameras view this point from too similar a direction")
        for cam, z in self.depth_m.items():
            if z <= 0:
                problems.append(f"point is BEHIND {cam}")
        return ("ok" if not problems else "suspect"), problems

    def describe(self, frame: str = "ref") -> str:
        p = self.point
        lines = [
            f"  position ({frame}): "
            f"[{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}] m"
            f"   = [{p[0]*1e3:+.1f}, {p[1]*1e3:+.1f}, {p[2]*1e3:+.1f}] mm",
            f"  ray gap: {self.ray_gap_mm:.2f} mm"
            f"    condition: {self.condition:.1f}",
        ]
        for cam in self.cameras:
            lines.append(
                f"  {cam:<11s} depth {self.depth_m[cam]*1e3:7.1f} mm   "
                f"reprojection {self.reprojection_px[cam]:.2f} px")
        s = self.sigma_axes_mm
        if s is not None:
            lengths, dirs = s
            lines.append(
                f"  1-sigma for a {self.sigma_px:.1f} px click error: "
                + " x ".join(f"{v:.2f}" for v in lengths) + " mm"
                + f"   (worst along [{dirs[0,0]:+.2f}, {dirs[1,0]:+.2f}, "
                  f"{dirs[2,0]:+.2f}])")
        verdict, problems = self.quality()
        lines.append(f"  verdict: {verdict.upper()}")
        for pr in problems:
            lines.append(f"    ! {pr}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        verdict, problems = self.quality()
        s = self.sigma_axes_mm
        return {
            "point_m": self.point.tolist(),
            "point_mm": (self.point * 1e3).tolist(),
            "cameras": list(self.cameras),
            "pixels": {k: np.asarray(v).tolist()
                       for k, v in self.pixels.items()},
            "reprojected_px": {k: np.asarray(v).tolist()
                               for k, v in self.reprojected.items()},
            "reprojection_error_px": dict(self.reprojection_px),
            "depth_along_optical_axis_m": dict(self.depth_m),
            "ray_gap_mm": self.ray_gap_mm,
            "condition": self.condition,
            "assumed_click_sigma_px": self.sigma_px,
            "sigma_mm_axis_aligned": (None if self.sigma_mm is None
                                      else self.sigma_mm.tolist()),
            "sigma_mm_principal": (None if s is None else s[0].tolist()),
            "sigma_principal_directions": (None if s is None
                                           else s[1].tolist()),
            "verdict": verdict,
            "problems": problems,
            **self.meta,
        }


## ------------------------------------------------------------------ ##
## The rig
## ------------------------------------------------------------------ ##

class StereoRig:
    """Two or more posed cameras sharing one reference frame."""

    def __init__(self, cameras: Sequence[PinholeCamera],
                 ref_frame: Optional[str] = None,
                 source: str = "constructed"):
        if len(cameras) < 1:
            raise ValueError("a rig needs at least one camera")
        self.cameras: Dict[str, PinholeCamera] = {c.name: c for c in cameras}
        if len(self.cameras) != len(cameras):
            raise ValueError("duplicate camera names in rig")

        frames = {c.ref_frame for c in cameras if c.has_pose}
        if len(frames) > 1:
            raise ValueError(
                f"cameras disagree about the reference frame: {sorted(frames)}. "
                f"A rig whose members are posed in different frames cannot "
                f"triangulate; re-express them first.")
        self.ref_frame = ref_frame or (frames.pop() if frames else "unknown")
        self.source = source

    # -------------------------------------------------------------- #
    def __getitem__(self, name: str) -> PinholeCamera:
        if name not in self.cameras:
            raise KeyError(
                f"no camera '{name}' in this rig (have: {sorted(self.cameras)})")
        return self.cameras[name]

    def __contains__(self, name: str) -> bool:
        return name in self.cameras

    @property
    def names(self) -> List[str]:
        return list(self.cameras)

    @property
    def posed(self) -> List[str]:
        return [n for n, c in self.cameras.items() if c.has_pose]

    def require_posed(self, *names: str) -> None:
        missing = [n for n in names if not self[n].has_pose]
        if missing:
            raise ValueError(
                f"camera(s) {missing} have no extrinsic pose, so nothing "
                f"geometric can be computed for them.\n"
                f"  Run: python calibration/scene_extrinsics.py solve ...")

    def baseline_m(self, a: str, b: str) -> float:
        self.require_posed(a, b)
        return float(np.linalg.norm(self[a].centre - self[b].centre))

    def viewing_angle_deg(self, a: str, b: str, point: np.ndarray) -> float:
        """Angle the two cameras subtend AT a point -- the conditioning.

        90 degrees is ideal (which is why the rig is built perpendicular);
        near 0 or 180 the depth direction is unobservable."""
        self.require_posed(a, b)
        va = np.asarray(point, dtype=float) - self[a].centre
        vb = np.asarray(point, dtype=float) - self[b].centre
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        cos = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    # ---------------------- epipolar geometry --------------------- #
    def epipolar_curve(self, src: str, uv, dst: str,
                       depth_range: Tuple[float, float] = (0.05, 2.0),
                       n: int = 240,
                       clip_to_image: bool = True) -> Dict[str, Any]:
        """Where a pixel in `src` can appear in `dst`, as a polyline.

        Returns the destination pixels, the 3D point each came from, and
        the source depth that produced it -- so the curve is not just a
        locus but a **depth scale you can read off the image**.

        `depth_range` is depth along `src`'s optical axis, in metres. It is
        a real constraint, not a rendering detail: it is the prior you are
        asserting about how far away the thing might be, and narrowing it
        shortens the curve and makes the correspondence unambiguous. The
        D405 sees usefully from ~0.05 m, and this rig's cameras are within
        ~2 m of everything, hence the default.

        Sampled in 3D and projected through the full distortion model --
        with a 78-degree lens the true locus is a curve and the textbook
        straight epipolar line is several pixels wrong near the border."""
        self.require_posed(src, dst)
        cam_s, cam_d = self[src], self[dst]

        lo, hi = float(depth_range[0]), float(depth_range[1])
        if not (0 < lo < hi):
            raise ValueError(f"depth_range must be 0 < lo < hi, got {depth_range}")
        ## Log spacing: near depths change the destination pixel far
        ## faster than far ones, so uniform spacing wastes most samples
        ## on the part of the curve that barely moves.
        depths = np.geomspace(lo, hi, int(n))

        pts3d = cam_s.point_at_depth(np.asarray(uv, dtype=float), 1.0)
        origin = cam_s.centre
        direction = pts3d - origin
        ## point_at_depth at unit depth gives origin + (ray scaled so that
        ## camera-frame z == 1), so scaling it by d gives depth d exactly.
        world = origin[None, :] + depths[:, None] * direction[None, :]

        uv_dst = cam_d.project_ref(world)
        inside = cam_d.in_image(uv_dst, margin=2.0)

        out = {
            "src": src, "dst": dst,
            "src_pixel": np.asarray(uv, dtype=float).tolist(),
            "depths_m": depths,
            "points_ref": world,
            "pixels": uv_dst,
            "inside": inside,
            "depth_range_m": [lo, hi],
            "epipole_dst": None,
        }
        ## The other camera's centre seen in this one: where the curve
        ## converges. NaN when it is behind the destination camera, which
        ## is normal for a perpendicular pair.
        e = cam_d.project_ref(cam_s.centre)
        if np.all(np.isfinite(e)):
            out["epipole_dst"] = e.tolist()

        if clip_to_image:
            out["pixels_visible"] = uv_dst[inside]
            out["depths_visible"] = depths[inside]
            out["points_visible"] = world[inside]
            out["fraction_visible"] = float(np.mean(inside))
        return out

    def closest_on_curve(self, curve: Dict[str, Any], uv,
                         refine: bool = True) -> Dict[str, Any]:
        """Snap a click in the destination image onto the epipolar curve.

        The user clicks near the curve, never exactly on it. Snapping first
        makes the correspondence consistent by construction, so the
        resulting `ray_gap` measures the calibration rather than the
        steadiness of a hand. The distance snapped is reported: a large
        snap means the click was not on the locus at all, which is the
        signal that the two clicks are different features.

        The curve is a POLYLINE SAMPLED FROM A CONTINUOUS LOCUS, and the
        distinction is not cosmetic.  Returning the nearest stored sample
        quantises the answer to the sample spacing -- ~5 mm of depth error
        at 240 samples over a metre, which would silently become the
        accuracy floor of every measurement made through this path.  So
        the nearest sample only brackets the answer, and the depth is then
        refined by golden-section search on the true continuous curve
        until it is exact to a millionth of a pixel."""
        uv = np.asarray(uv, dtype=float)
        px = np.asarray(curve["pixels"], dtype=float)
        depths = np.asarray(curve["depths_m"], dtype=float)
        ok = np.isfinite(px).all(axis=1)
        if not np.any(ok):
            raise ValueError("the epipolar curve does not appear in this image")

        idx_ok = np.flatnonzero(ok)
        d = np.linalg.norm(px[idx_ok] - uv[None, :], axis=1)
        i = int(idx_ok[int(np.argmin(d))])

        src, dst = curve["src"], curve["dst"]
        uv_src = np.asarray(curve["src_pixel"], dtype=float)
        cam_s, cam_d = self[src], self[dst]
        origin = cam_s.centre
        unit = cam_s.point_at_depth(uv_src, 1.0) - origin

        def miss(depth: float) -> float:
            q = cam_d.project_ref(origin + depth * unit)
            if not np.all(np.isfinite(q)):
                return np.inf
            return float(np.linalg.norm(q - uv))

        depth = float(depths[i])
        if refine and len(depths) > 2:
            lo = float(depths[max(i - 1, 0)])
            hi = float(depths[min(i + 1, len(depths) - 1)])
            depth = _golden_min(miss, lo, hi)

        point = origin + depth * unit
        pixel = cam_d.project_ref(point)
        return {
            "pixel": pixel,
            "snap_distance_px": miss(depth),
            "depth_m": depth,
            "point_ref": point,
            "index": i,
            "refined": bool(refine),
            "sample_spacing_m": float(abs(depths[min(i + 1, len(depths) - 1)]
                                          - depths[max(i - 1, 0)]) / 2.0),
        }

    # ------------------------ triangulation ----------------------- #
    def triangulate(self, observations: Dict[str, Any],
                    sigma_px: float = 1.0) -> Triangulation:
        """Least-squares 3D point from >=2 pixel observations.

        Minimises the sum of squared PERPENDICULAR distances from the point
        to each camera's ray -- the midpoint method, generalised to N
        cameras.  Chosen over the classic DLT because its residual has
        physical units (millimetres of miss) instead of an arbitrary
        algebraic scale, and because it stays well behaved when the rays
        are near-perpendicular, which is this rig's normal case."""
        names = list(observations)
        if len(names) < 2:
            raise ValueError(
                "triangulation needs at least two cameras.\n"
                "  One pixel is a ray. For a single view use ray_plane() "
                "with a surface the point is known to lie on, or "
                "depth_from_known_size().")
        self.require_posed(*names)

        origins, dirs = [], []
        for n in names:
            o, d = self[n].ray(np.asarray(observations[n], dtype=float))
            origins.append(o)
            dirs.append(d)
        point, cond = _least_squares_ray_intersection(origins, dirs)

        reproj, reproj_err, depth = {}, {}, {}
        for n in names:
            uv_click = np.asarray(observations[n], dtype=float)
            uv_back = self[n].project_ref(point)
            reproj[n] = uv_back
            reproj_err[n] = float(np.linalg.norm(uv_back - uv_click))
            depth[n] = float(self[n].world_to_cam(point)[2])

        gap = _ray_gap(origins, dirs, point)
        cov = self._covariance(names, observations, sigma_px)

        meta: Dict[str, Any] = {"ref_frame": self.ref_frame}
        if len(names) == 2:
            meta["baseline_m"] = self.baseline_m(*names)
            meta["viewing_angle_deg"] = self.viewing_angle_deg(
                names[0], names[1], point)
        return Triangulation(
            point=point, cameras=names,
            pixels={n: np.asarray(observations[n], dtype=float) for n in names},
            reprojected=reproj, reprojection_px=reproj_err, depth_m=depth,
            ray_gap_m=gap, condition=cond, covariance=cov,
            sigma_px=sigma_px, meta=meta)

    def _covariance(self, names, observations, sigma_px: float
                    ) -> Optional[np.ndarray]:
        """Propagate an assumed per-pixel click error into the 3D point.

        Numeric Jacobian, one column per (camera, pixel axis).  Numeric
        rather than analytic on purpose: it goes through the real
        distortion model and the real solve, so it cannot silently
        disagree with them the way a hand-derived formula would."""
        if sigma_px <= 0:
            return None
        base = {n: np.asarray(observations[n], dtype=float) for n in names}
        h = 0.5  # pixels; central difference

        def solve(obs):
            o, d = [], []
            for n in names:
                oo, dd = self[n].ray(obs[n])
                o.append(oo)
                d.append(dd)
            return _least_squares_ray_intersection(o, d)[0]

        cols = []
        for n in names:
            for axis in (0, 1):
                plus = {k: v.copy() for k, v in base.items()}
                minus = {k: v.copy() for k, v in base.items()}
                plus[n][axis] += h
                minus[n][axis] -= h
                cols.append((solve(plus) - solve(minus)) / (2.0 * h))
        J = np.stack(cols, axis=1)              # (3, 2*ncam), metres per pixel
        return J @ J.T * (sigma_px ** 2)

    # ------------------- single-view alternatives ----------------- #
    def ray_plane(self, camera: str, uv,
                  plane_point: Sequence[float] = (0.0, 0.0, 0.0),
                  plane_normal: Sequence[float] = (0.0, 0.0, 1.0)
                  ) -> Dict[str, Any]:
        """Intersect one camera's ray with a known plane.

        This is the exact answer for a point that really lies on the
        plane -- the tabletop for the top camera, a back wall for the low
        one -- and it needs only ONE view.  The catch is entirely physical:
        an object 20 mm thick sitting on the table is 20 mm off the plane,
        and the error that produces is `20 mm x tan(angle from normal)`,
        which this reports as `sensitivity_mm_per_mm` so it is never a
        surprise."""
        self.require_posed(camera)
        cam = self[camera]
        o, d = cam.ray(np.asarray(uv, dtype=float))
        p0 = np.asarray(plane_point, dtype=float)
        nrm = np.asarray(plane_normal, dtype=float)
        nrm = nrm / np.linalg.norm(nrm)

        denom = float(np.dot(d, nrm))
        angle = float(np.degrees(np.arccos(np.clip(abs(denom), 0, 1))))
        if abs(denom) < 1e-6:
            raise ValueError(
                f"the ray from {camera} is parallel to the plane "
                f"({angle:.2f} deg from grazing) -- no usable intersection")
        t = float(np.dot(p0 - o, nrm) / denom)
        if t <= 0:
            raise ValueError(
                f"the plane is BEHIND {camera} along this ray; "
                f"check the plane definition")
        point = o + t * d
        return {
            "camera": camera,
            "pixel": np.asarray(uv, dtype=float),
            "point_ref": point,
            "distance_along_ray_m": t,
            "depth_m": float(cam.world_to_cam(point)[2]),
            "incidence_deg": angle,
            ## How much a height error off the plane costs laterally.
            "sensitivity_mm_per_mm": float(
                np.linalg.norm(d - denom * nrm) / abs(denom)),
            "metres_per_pixel": cam.scale_at(float(cam.world_to_cam(point)[2])),
            "ref_frame": self.ref_frame,
            "assumption": (
                "the clicked feature lies exactly on the given plane; "
                "any height above it displaces the answer laterally by "
                "height x sensitivity_mm_per_mm"),
        }

    def depth_from_known_size(self, camera: str, uv_a, uv_b,
                              size_m: float) -> Dict[str, Any]:
        """Depth of an object of known length, from two clicks on its ends.

        Uses no extrinsics at all, which is the point: it is an INDEPENDENT
        check on a triangulated depth rather than another thing derived
        from the same calibration.  This is the check to run against a
        printed ruler, or against a link whose length you measured.

        The rigorous form: the two rays subtend an angle `theta` at the
        optical centre, and if the segment is perpendicular to their
        bisector then its midpoint sits at `(L/2) / tan(theta/2)`.  The
        familiar `f * L / pixels` is the small-angle version of that and
        is reported alongside so the two can be compared -- they diverge
        by ~1% at 40 degrees, which is well inside this lens.

        Both forms assume the segment is FRONTO-PARALLEL. A segment tilted
        by `t` away from the image plane reads short by `cos(t)`, so a 25
        degree tilt overstates the depth by 10%. Tilt is the dominant
        error here, not click precision."""
        cam = self[camera]
        if size_m <= 0:
            raise ValueError("size_m must be positive")
        a = np.asarray(uv_a, dtype=float)
        b = np.asarray(uv_b, dtype=float)

        ra = np.append(cam.unproject(a), 1.0)
        rb = np.append(cam.unproject(b), 1.0)
        ra /= np.linalg.norm(ra)
        rb /= np.linalg.norm(rb)
        cos = float(np.clip(np.dot(ra, rb), -1.0, 1.0))
        theta = float(np.arccos(cos))
        if theta < 1e-7:
            raise ValueError(
                "the two clicks are on the same ray -- no angle to work with")

        depth_exact = (size_m / 2.0) / np.tan(theta / 2.0)
        px = float(np.linalg.norm(b - a))
        f = (cam.fx + cam.fy) / 2.0
        depth_small_angle = f * size_m / px if px > 0 else float("nan")

        ## One pixel of click error, as a fraction of the measured span.
        rel = 1.0 / px if px > 0 else float("nan")
        return {
            "camera": camera,
            "pixels": [a.tolist(), b.tolist()],
            "known_size_m": size_m,
            "subtended_angle_deg": float(np.degrees(theta)),
            "pixel_separation": px,
            "depth_m": float(depth_exact),
            "depth_small_angle_m": float(depth_small_angle),
            "small_angle_disagreement_pct": float(
                100.0 * abs(depth_small_angle - depth_exact) / depth_exact),
            "sensitivity_pct_per_px": float(100.0 * rel),
            "assumption": (
                "the segment is perpendicular to the viewing direction; "
                "a tilt of t degrees away from the image plane makes the "
                "reported depth too LARGE by 1/cos(t)"),
        }

    # ---------------------------- report -------------------------- #
    def describe(self) -> str:
        lines = [f"rig reference frame: '{self.ref_frame}'   "
                 f"source: {self.source}"]
        for c in self.cameras.values():
            lines.append(c.describe())
        posed = self.posed
        for i in range(len(posed)):
            for j in range(i + 1, len(posed)):
                a, b = posed[i], posed[j]
                lines.append(
                    f"  baseline {a} <-> {b}: "
                    f"{self.baseline_m(a, b) * 1e3:.1f} mm")
        return "\n".join(lines)


## ------------------------------------------------------------------ ##
## Geometry primitives
## ------------------------------------------------------------------ ##

def _least_squares_ray_intersection(origins: Sequence[np.ndarray],
                                    dirs: Sequence[np.ndarray]
                                    ) -> Tuple[np.ndarray, float]:
    """The point minimising the summed squared distance to a set of rays.

    Each ray contributes the projector `I - d d^T`, which measures
    displacement PERPENDICULAR to it -- exactly the component the ray
    constrains.  The condition number of the accumulated matrix is the
    honest statement of whether the rays actually pin the point down: it
    blows up when they are parallel, which is when the answer is
    meaningless."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        d = np.asarray(d, dtype=float)
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ np.asarray(o, dtype=float)

    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > 1e12:
        raise ValueError(
            "the rays are parallel (or nearly so) -- they do not determine "
            "a point. Two cameras looking along the same direction cannot "
            "triangulate.")
    return np.linalg.solve(A, b), cond


def _golden_min(fn, lo: float, hi: float, tol: float = 1e-9,
                max_iter: int = 200) -> float:
    """Minimise a unimodal fn on [lo, hi] without a scipy dependency.

    The pixel-distance-along-the-epipolar-curve function is unimodal by
    construction (the curve is monotone in depth and does not self
    intersect), which is exactly the condition golden section needs."""
    if not (hi > lo):
        return lo
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(max_iter):
        if (b - a) < tol * max(1.0, abs(a) + abs(b)):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = fn(d)
    return 0.5 * (a + b)


def _ray_gap(origins: Sequence[np.ndarray], dirs: Sequence[np.ndarray],
             point: np.ndarray) -> float:
    """How far the rays came from meeting, in metres.

    For two rays this is the classic closest-approach distance between
    skew lines, computed exactly rather than from the solved point (which
    sits midway and would halve it).  For three or more it is the largest
    perpendicular distance from the solution to any ray, which is the
    worst-case reading and the one worth acting on."""
    if len(origins) == 2:
        o1, o2 = np.asarray(origins[0]), np.asarray(origins[1])
        d1 = np.asarray(dirs[0]) / np.linalg.norm(dirs[0])
        d2 = np.asarray(dirs[1]) / np.linalg.norm(dirs[1])
        n = np.cross(d1, d2)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            return float("inf")
        return float(abs(np.dot(o2 - o1, n / nn)))

    worst = 0.0
    for o, d in zip(origins, dirs):
        d = np.asarray(d) / np.linalg.norm(d)
        v = np.asarray(point) - np.asarray(o)
        worst = max(worst, float(np.linalg.norm(v - np.dot(v, d) * d)))
    return worst


def distance(a: Triangulation | np.ndarray,
             b: Triangulation | np.ndarray) -> float:
    """Metres between two triangulated points."""
    pa = a.point if isinstance(a, Triangulation) else np.asarray(a, dtype=float)
    pb = b.point if isinstance(b, Triangulation) else np.asarray(b, dtype=float)
    return float(np.linalg.norm(pa - pb))


def distance_with_error(a: Triangulation, b: Triangulation) -> Dict[str, Any]:
    """Distance between two triangulated points, with a propagated sigma.

    The two endpoints are treated as independent, which is right when the
    clicks are independent and slightly pessimistic when they share a
    systematic extrinsic error -- a common-mode calibration error largely
    cancels in a difference.  So this sigma bounds the random part; it
    says nothing about a wrong baseline, which biases every length by the
    same factor and can only be caught against a ruler."""
    d_vec = b.point - a.point
    d = float(np.linalg.norm(d_vec))
    out: Dict[str, Any] = {
        "distance_m": d, "distance_mm": d * 1e3,
        "delta_m": d_vec.tolist(),
    }
    if a.covariance is not None and b.covariance is not None and d > 1e-9:
        u = d_vec / d
        var = float(u @ (a.covariance + b.covariance) @ u)
        out["sigma_mm"] = float(np.sqrt(max(var, 0.0)) * 1e3)
        out["note"] = ("sigma covers click noise only; a scale error in the "
                       "extrinsics biases every length identically and is "
                       "invisible here -- check against a ruler")
    return out


def polyline_distance(px: np.ndarray, q) -> float:
    """Distance from a point to the polyline through `px`, ignoring NaN.

    The honest measure of "is this click on the curve": a sampled curve is
    drawn as segments, and point-to-nearest-VERTEX overstates the miss by
    up to half a segment length for no reason."""
    q = np.asarray(q, dtype=float)
    px = np.asarray(px, dtype=float)
    ok = np.isfinite(px).all(axis=1)
    ## Only segments whose BOTH ends are finite are real segments.
    seg = ok[:-1] & ok[1:]
    if not np.any(seg):
        finite = px[ok]
        if not len(finite):
            return float("inf")
        return float(np.min(np.linalg.norm(finite - q[None, :], axis=1)))
    a = px[:-1][seg]
    b = px[1:][seg]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    t = np.where(denom > 1e-12,
                 np.sum((q[None, :] - a) * ab, axis=1) / np.maximum(denom, 1e-12),
                 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(proj - q[None, :], axis=1)))
