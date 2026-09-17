"""Click-to-3D measurement panel for world_view.py.

Adds a **Measure** tab and, for every calibrated camera, a clickable image
plane sitting at that camera's real pose in the 3D scene.  Click a feature
in one camera, click the same feature in the other, and read its position
in the robot's world frame -- with the ray gap, the reprojection error and
the propagated click uncertainty next to it, because a coordinate without
those is not a measurement.

All geometry lives in `reconstruction/`; this file is the interface to it.


HOW A CLICK BECOMES A PIXEL, EXACTLY
====================================
Viser reports a click as a RAY in world coordinates, not as a pixel.  The
conversion could be fudged by assuming how viser lays out an image plane,
and that assumption would be wrong by a few pixels and never noticed.  It
is instead derived from the client's own geometry
(`CameraFrustumVariants.tsx`), which builds the frustum as

    y = tan(fov/2)          x = y * aspect          z = 1        (x scale)

with the image plane at local z, its texture rotated by pi about x so that
image +v runs along local -(-y) = OpenCV's +y down.  A plane textured with
a WxH image puts pixel centre u at fraction (u + 0.5)/W across it, so

    x_norm = (2 (u+0.5)/W - 1) * aspect * tan(fov/2)
    y_norm = (2 (v+0.5)/H - 1) * tan(fov/2)

Choosing the displayed camera to be an ideal pinhole with fx = fy = f
makes that invert to exactly

    u = x_norm * f + W/2 - 0.5          v = y_norm * f + H/2 - 0.5
    fov = 2 atan(H / 2f)                aspect = W / H

which is what this module passes to `add_camera_frustum`.  So a click ray,
rotated into the camera's own frame and divided by its z, IS the
undistorted normalised coordinate of the clicked pixel.  No approximation.


WHY THE DISPLAYED IMAGE IS UNDISTORTED
======================================
That derivation describes an ideal pinhole, and the D405 is not one: its
distortion moves a feature by several pixels near the border.  If the raw
image were pasted onto the frustum, the ray implied by a click and the
feature actually under the cursor would disagree by exactly that much --
worst precisely at the edges, where the interesting parts of the workspace
are.

So each camera's image is remapped to its ideal pinhole once at startup
(through `PinholeCamera.project`, which knows whether that camera's
coefficients run pixel->ray or ray->pixel).  What you see is then what the
geometry assumes.  Pixel coordinates RECORDED to disk are converted back
to the real distorted image, because that is the frame the raw capture is
in and the only one a later re-analysis could reproduce.
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
for _p in (str(HERE), str(SCRIPTS_DIR), str(HERE.parent / "reconstruction")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (DATA_ROOT, matrix_to_quat_wxyz, provenance,  # noqa: E402
                    save_json, timestamp)

DIR_MEASURE = DATA_ROOT / "measurements"

## Distinct, colour-blind-safe marks per camera role.
PICK_COLOURS = ((255, 90, 90), (90, 180, 255), (150, 255, 120), (255, 210, 90))


## ------------------------------------------------------------------ ##
## Ideal (undistorted) view of one camera
## ------------------------------------------------------------------ ##

@dataclass
class IdealView:
    """A camera rendered as the ideal pinhole its frustum actually is."""
    cam: Any                    # reconstruction.PinholeCamera
    f: float
    map_x: np.ndarray
    map_y: np.ndarray

    @property
    def width(self) -> int:
        return self.cam.width

    @property
    def height(self) -> int:
        return self.cam.height

    @property
    def cx(self) -> float:
        return self.width / 2.0 - 0.5

    @property
    def cy(self) -> float:
        return self.height / 2.0 - 0.5

    @property
    def fov(self) -> float:
        """Vertical field of view, radians -- what add_camera_frustum wants."""
        return 2.0 * float(np.arctan(self.height / (2.0 * self.f)))

    @property
    def aspect(self) -> float:
        return self.width / self.height

    # -------------------------------------------------------------- #
    def undistort(self, image: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def ideal_pixel(self, xy_norm) -> np.ndarray:
        xy = np.asarray(xy_norm, dtype=float)
        return np.array([xy[..., 0] * self.f + self.cx,
                         xy[..., 1] * self.f + self.cy]).T

    def norm_from_ideal(self, uv) -> np.ndarray:
        uv = np.asarray(uv, dtype=float)
        return np.stack([(uv[..., 0] - self.cx) / self.f,
                         (uv[..., 1] - self.cy) / self.f], axis=-1)

    def real_pixel(self, uv_ideal) -> np.ndarray:
        """Ideal-image pixel -> the pixel in the RAW camera frame.

        What gets written to disk: a coordinate in an image that exists."""
        xy = self.norm_from_ideal(uv_ideal)
        return self.cam.project(np.append(np.atleast_1d(xy).ravel()[:2], 1.0))

    def project_ref(self, p_ref) -> np.ndarray:
        """Reference-frame point -> ideal-image pixel."""
        p_cam = np.atleast_2d(self.cam.world_to_cam(p_ref))
        out = np.full((len(p_cam), 2), np.nan)
        good = p_cam[:, 2] > 1e-9
        if np.any(good):
            xy = p_cam[good, :2] / p_cam[good, 2:3]
            out[good] = np.stack([xy[:, 0] * self.f + self.cx,
                                  xy[:, 1] * self.f + self.cy], axis=-1)
        return out[0] if np.asarray(p_ref).ndim == 1 else out

    def pixel_from_click(self, ray_direction) -> Optional[np.ndarray]:
        """World-frame click ray -> ideal pixel, or None if behind/outside.

        The frustum was built with THIS view's fov and aspect, so the ray's
        camera-frame direction divided by its z is the clicked pixel's
        normalised coordinate exactly -- see the module docstring."""
        d = np.asarray(ray_direction, dtype=float)
        R = self.cam.T_ref_cam[:3, :3]
        d_cam = R.T @ d
        if d_cam[2] <= 1e-9:
            return None
        uv = np.array([d_cam[0] / d_cam[2] * self.f + self.cx,
                       d_cam[1] / d_cam[2] * self.f + self.cy])
        if not (-0.5 <= uv[0] <= self.width - 0.5
                and -0.5 <= uv[1] <= self.height - 0.5):
            return None
        return uv


def build_ideal_view(cam) -> IdealView:
    """Precompute the undistortion map for one camera.

    Uses the camera's own `project`, so it is correct for BOTH the
    librealsense inverse_brown_conrady factory numbers and an OpenCV
    ChArUco calibration -- the two run in opposite directions and a shared
    remap built from raw coefficients would be right for only one."""
    W, H = cam.width, cam.height
    f = (cam.fx + cam.fy) / 2.0
    cx, cy = W / 2.0 - 0.5, H / 2.0 - 0.5

    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64))
    xy = np.stack([(u.ravel() - cx) / f, (v.ravel() - cy) / f], axis=-1)
    pts = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
    src = cam.project(pts)
    return IdealView(
        cam=cam, f=f,
        map_x=np.nan_to_num(src[:, 0], nan=-1.0).reshape(H, W).astype(np.float32),
        map_y=np.nan_to_num(src[:, 1], nan=-1.0).reshape(H, W).astype(np.float32))


## ------------------------------------------------------------------ ##
## Overlay drawing
## ------------------------------------------------------------------ ##

def draw_overlay(img: np.ndarray, marks: Sequence[Dict[str, Any]],
                 curve: Optional[np.ndarray] = None,
                 curve_labels: Optional[Sequence[Tuple[np.ndarray, str]]] = None
                 ) -> np.ndarray:
    """Crosshairs and the epipolar curve, drawn into a copy of the image."""
    import cv2

    out = np.ascontiguousarray(img.copy())
    if curve is not None and len(curve):
        pts = np.asarray(curve, dtype=float)
        ok = np.isfinite(pts).all(axis=1)
        seg = ok[:-1] & ok[1:]
        a, b = pts[:-1][seg], pts[1:][seg]
        for p, q in zip(a, b):
            cv2.line(out, (int(round(p[0])), int(round(p[1]))),
                     (int(round(q[0])), int(round(q[1]))), (255, 220, 60), 1,
                     cv2.LINE_AA)
    for pt, text in (curve_labels or []):
        if not np.all(np.isfinite(pt)):
            continue
        p = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out, p, 2, (255, 220, 60), -1, cv2.LINE_AA)
        cv2.putText(out, text, (p[0] + 5, p[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 60), 1,
                    cv2.LINE_AA)
    for m in marks:
        uv = np.asarray(m["uv"], dtype=float)
        if not np.all(np.isfinite(uv)):
            continue
        x, y = int(round(uv[0])), int(round(uv[1]))
        c = tuple(int(v) for v in m.get("colour", (255, 90, 90)))
        cv2.line(out, (x - 11, y), (x - 3, y), c, 1, cv2.LINE_AA)
        cv2.line(out, (x + 3, y), (x + 11, y), c, 1, cv2.LINE_AA)
        cv2.line(out, (x, y - 11), (x, y - 3), c, 1, cv2.LINE_AA)
        cv2.line(out, (x, y + 3), (x, y + 11), c, 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 13, c, 1, cv2.LINE_AA)
        if m.get("label"):
            cv2.putText(out, m["label"], (x + 16, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out


## ------------------------------------------------------------------ ##
## The panel
## ------------------------------------------------------------------ ##

@dataclass
class Pick:
    camera: str
    uv_ideal: np.ndarray
    uv_real: np.ndarray
    when: float = field(default_factory=time.time)


class MeasurePanel:
    """Clickable camera planes + the Measure tab.

    Constructed once; `update()` is called from world_view's render loop to
    push fresh images (with overlays) into the frustums."""

    def __init__(self, server, tab, rig, workers: Dict[str, Any],
                 world_view=None, depth_enabled: bool = False,
                 frustum_scale: float = 0.35):
        self.server = server
        self.rig = rig
        self.workers = workers
        self.wv = world_view
        self.depth_enabled = depth_enabled
        self.frustum_scale = frustum_scale

        ## Only cameras that are BOTH posed and streaming can be measured
        ## with. The rest are reported, not silently dropped.
        self.usable = [n for n in rig.names
                       if n in workers and rig[n].has_pose]
        self.unusable = {
            n: ("not streaming" if n not in workers else "no calibrated pose")
            for n in rig.names if n not in self.usable}

        self.views: Dict[str, IdealView] = {
            n: build_ideal_view(rig[n]) for n in self.usable}
        self.colours = {n: PICK_COLOURS[i % len(PICK_COLOURS)]
                        for i, n in enumerate(self.usable)}

        self.picks: Dict[str, Pick] = {}
        self.curves: Dict[str, Dict[str, Any]] = {}
        self.tri = None
        self.saved: List[Dict[str, Any]] = []
        self.correspondences: List[Dict[str, Any]] = []
        self._nodes: Dict[str, Any] = {}
        self._marker = None
        self._ray_lines: Dict[str, Any] = {}

        self._build_scene()
        self._build_gui(tab)

    # ------------------------------------------------------------- #
    def _build_scene(self) -> None:
        for name in self.usable:
            iv = self.views[name]
            T = self.rig[name].T_ref_cam
            node = self.server.scene.add_camera_frustum(
                f"/measure/{name}", fov=iv.fov, aspect=iv.aspect,
                scale=self.frustum_scale, line_width=1.5,
                color=self.colours[name],
                image=np.zeros((iv.height, iv.width, 3), dtype=np.uint8),
                format="jpeg", jpeg_quality=75,
                position=tuple(T[:3, 3]),
                wxyz=tuple(matrix_to_quat_wxyz(T[:3, :3])))
            self._nodes[name] = node

            @node.on_click
            def _(event, _name=name) -> None:
                self._on_click(_name, event)

            self.server.scene.add_label(
                f"/measure/{name}/name", f"{name}  (click me)",
                position=(0.0, 0.0, -0.03))

    def _build_gui(self, tab) -> None:
        g = self.server.gui
        with tab:
            if not self.usable:
                g.add_markdown(self._nothing_usable_md())
                self.readout = g.add_markdown("")
                return

            self.readout = g.add_markdown(self._idle_md())

            with g.add_folder("Assumptions"):
                self.depth_lo = g.add_number(
                    "search from (m)", initial_value=0.15, min=0.02, max=5.0,
                    step=0.05)
                self.depth_hi = g.add_number(
                    "search to (m)", initial_value=2.0, min=0.05, max=10.0,
                    step=0.05)
                g.add_markdown(
                    "_How far away the thing might be. This is a real "
                    "constraint, not a display setting: it bounds the "
                    "epipolar curve and so bounds which matches are "
                    "considered._")
                self.sigma = g.add_number(
                    "click error (px)", initial_value=1.0, min=0.1, max=10.0,
                    step=0.1)
                self.snap = g.add_checkbox("snap 2nd click to the curve", True)
                g.add_markdown(
                    "_Snapping makes the correspondence consistent by "
                    "construction, so the reported ray gap measures the "
                    "CALIBRATION rather than the steadiness of your hand. "
                    "Turn it off to see the raw disagreement._")

            with g.add_folder("Table plane (single click)"):
                self.plane_on = g.add_checkbox("intersect with a plane", False)
                self.plane_z = g.add_number("plane height z (m)",
                                            initial_value=0.0, step=0.005)
                g.add_markdown(
                    "_One click is enough for a point KNOWN to lie on a "
                    "plane -- the tabletop. Exact if it really does; an "
                    "object of some height is displaced laterally, and the "
                    "readout says by how much per mm._")

            with g.add_folder("Points and distances"):
                self.add_btn = g.add_button("ADD THIS POINT")
                self.dist_btn = g.add_button("DISTANCE: last two points")
                self.clear_pts_btn = g.add_button("clear saved points")
                self.points_md = g.add_markdown("_no points saved_")

            with g.add_folder("Anchor to the robot"):
                arms = (list(getattr(self.wv, "arms", []) or ["right", "left"])
                        if self.wv else ["right", "left"])
                self.corr_arm = g.add_dropdown("arm", tuple(arms),
                                               initial_value=arms[0])
                self.corr_point = g.add_dropdown(
                    "robot feature", ("flange (gripper_base)", "TCP (grasp)"),
                    initial_value="flange (gripper_base)")
                self.corr_btn = g.add_button("RECORD CORRESPONDENCE")
                self.corr_save = g.add_button("SAVE correspondences")
                self.corr_md = g.add_markdown(self._corr_md())
                g.add_markdown(
                    "_Click the SAME physical feature in both cameras, then "
                    "record. Each one pairs 'where forward kinematics says "
                    "the arm is' with 'where the cameras see it'. Collect "
                    "8-15 spread over the workspace AND over several "
                    "heights, then run_ `scene_extrinsics.py anchor`.")

            self.clear_btn = g.add_button("CLEAR CLICKS")

        @self.clear_btn.on_click
        def _(_) -> None:
            self.clear()

        @self.add_btn.on_click
        def _(_) -> None:
            self._add_point()

        @self.dist_btn.on_click
        def _(_) -> None:
            self._measure_distance()

        @self.clear_pts_btn.on_click
        def _(_) -> None:
            self.saved.clear()
            self.points_md.content = "_no points saved_"

        @self.corr_btn.on_click
        def _(_) -> None:
            self._record_correspondence()

        @self.corr_save.on_click
        def _(_) -> None:
            self._save_correspondences()

    # ------------------------------------------------------------- #
    def _on_click(self, name: str, event) -> None:
        iv = self.views[name]
        uv = iv.pixel_from_click(event.ray_direction)
        if uv is None:
            return
        self.picks[name] = Pick(camera=name, uv_ideal=uv,
                                uv_real=np.asarray(iv.real_pixel(uv),
                                                   dtype=float).ravel())
        self._recompute()

    def clear(self) -> None:
        self.picks.clear()
        self.curves.clear()
        self.tri = None
        self._clear_scene_marks()
        self.readout.content = self._idle_md()

    def _clear_scene_marks(self) -> None:
        if self._marker is not None:
            self._marker.remove()
            self._marker = None
        for h in self._ray_lines.values():
            h.remove()
        self._ray_lines.clear()

    # ------------------------------------------------------------- #
    def _recompute(self) -> None:
        self._clear_scene_marks()
        self.curves.clear()
        self.tri = None

        names = list(self.picks)
        for n in names:
            self._draw_ray(n)

        if len(names) == 1:
            only = names[0]
            for other in self.usable:
                if other == only:
                    continue
                try:
                    self.curves[other] = self.rig.epipolar_curve(
                        only, self.picks[only].uv_real, other,
                        depth_range=(float(self.depth_lo.value),
                                     float(self.depth_hi.value)))
                except Exception:  # noqa: BLE001
                    pass
            self.readout.content = self._one_click_md(only)
            return

        obs = {n: self.picks[n].uv_real for n in names}
        if self.snap.value and len(names) == 2:
            first, second = sorted(names, key=lambda n: self.picks[n].when)
            try:
                curve = self.rig.epipolar_curve(
                    first, self.picks[first].uv_real, second,
                    depth_range=(float(self.depth_lo.value),
                                 float(self.depth_hi.value)))
                snapped = self.rig.closest_on_curve(
                    curve, self.picks[second].uv_real)
                obs[second] = np.asarray(snapped["pixel"], dtype=float)
                self._snap_px = snapped["snap_distance_px"]
                self.curves[second] = curve
            except Exception:  # noqa: BLE001
                self._snap_px = None
        else:
            self._snap_px = None

        try:
            self.tri = self.rig.triangulate(
                obs, sigma_px=float(self.sigma.value))
        except Exception as e:  # noqa: BLE001
            self.readout.content = f"**could not triangulate**\n\n`{e}`"
            return

        p = self.tri.point
        verdict, _ = self.tri.quality()
        self._marker = self.server.scene.add_icosphere(
            "/measure/point", radius=0.008,
            color=(90, 255, 140) if verdict == "ok" else (255, 140, 60),
            position=tuple(p))
        self.readout.content = self._result_md()

    def _draw_ray(self, name: str) -> None:
        try:
            o, d = self.rig[name].ray(self.picks[name].uv_real)
        except Exception:  # noqa: BLE001
            return
        far = float(self.depth_hi.value)
        self._ray_lines[name] = self.server.scene.add_spline_catmull_rom(
            f"/measure/ray_{name}",
            positions=np.stack([o + d * float(self.depth_lo.value),
                                o + d * far]),
            color=self.colours[name], line_width=1.5)

    # ------------------------------------------------------------- #
    def update(self) -> None:
        """Push the newest frames, undistorted, with overlays."""
        for name, node in self._nodes.items():
            rec = self.workers[name].snapshot()
            if rec is None or rec.image is None:
                continue
            iv = self.views[name]
            img = iv.undistort(rec.image)

            marks = []
            if name in self.picks:
                marks.append({"uv": self.picks[name].uv_ideal,
                              "colour": self.colours[name], "label": name[0].upper()})
            if self.tri is not None:
                back = iv.project_ref(self.tri.point)
                if np.all(np.isfinite(back)):
                    marks.append({"uv": back, "colour": (90, 255, 140),
                                  "label": "3D"})

            curve_px, labels = None, []
            c = self.curves.get(name)
            if c is not None:
                curve_px = iv.project_ref(np.asarray(c["points_ref"]))
                pts = np.asarray(c["points_ref"])
                depths = np.asarray(c["depths_m"])
                for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                    k = int(frac * (len(pts) - 1))
                    labels.append((iv.project_ref(pts[k]),
                                   f"{depths[k]*1e3:.0f}"))
            node.image = draw_overlay(img, marks, curve_px, labels)

    # ------------------------------------------------------------- #
    def _depth_check(self) -> Dict[str, Any]:
        """What the D405s' own depth says at the clicked pixels.

        A genuinely independent number: it comes from the stereo IR pair
        inside each camera and uses none of the extrinsics.  It is also the
        weaker measurement here -- the D405 is a SHORT-RANGE sensor, best
        from ~70 to ~500 mm, and scene cameras a metre away are outside
        that.  Treated as a cross-check that can disagree, never as truth."""
        out: Dict[str, Any] = {}
        for name, pick in self.picks.items():
            rec = self.workers[name].snapshot()
            if rec is None or rec.depth_m is None:
                continue
            d = rec.depth_at(pick.uv_real)
            entry: Dict[str, Any] = {"depth_m": d}
            if d is not None and self.tri is not None:
                entry["triangulated_depth_m"] = self.tri.depth_m.get(name)
                if entry["triangulated_depth_m"]:
                    entry["difference_mm"] = float(
                        (d - entry["triangulated_depth_m"]) * 1e3)
            out[name] = entry
        return out

    # ------------------------------------------------------------- #
    def _fk_point(self) -> Optional[Tuple[np.ndarray, str]]:
        """Where the robot says the selected feature is, in `base`."""
        if self.wv is None:
            return None
        arm = self.corr_arm.value
        try:
            q_urdf = self.wv.q_urdf()
            T = self.wv.frames.ee_pose(q_urdf, arm)
        except Exception:  # noqa: BLE001
            return None
        if self.corr_point.value.startswith("TCP"):
            import tcp as TCP
            T = TCP.tcp_pose(T, arm)
            return T[:3, 3].copy(), f"{arm} TCP (grasp point)"
        return T[:3, 3].copy(), f"{arm} flange ({self.wv.frames.ee_link(arm)})"

    def _record_correspondence(self) -> None:
        if self.tri is None:
            self.corr_md.content = ("**click the feature in BOTH cameras "
                                    "first** -- a correspondence needs a "
                                    "triangulated point.")
            return
        fk = self._fk_point()
        if fk is None:
            self.corr_md.content = (
                "**no live robot.** Correspondences pair forward kinematics "
                "with the cameras, so world_view must be running with "
                "`--from-robot`.")
            return
        p_base, label = fk
        verdict, problems = self.tri.quality()
        entry = {
            "label": label,
            "arm": self.corr_arm.value,
            "feature": ("tcp" if self.corr_point.value.startswith("TCP")
                        else "flange"),
            "p_base_fk": p_base.tolist(),
            "p_ref_triangulated": self.tri.point.tolist(),
            "pixels": {n: self.picks[n].uv_real.tolist() for n in self.picks},
            "ray_gap_mm": self.tri.ray_gap_mm,
            "reprojection_px": dict(self.tri.reprojection_px),
            "condition": self.tri.condition,
            "verdict": verdict,
            "problems": problems,
            "q_urdf": np.asarray(self.wv.q_urdf()).tolist(),
            "when": time.time(),
        }
        try:
            entry["q_driver"] = np.asarray(self.wv.read_q_driver()).tolist()
        except Exception:  # noqa: BLE001
            pass
        self.correspondences.append(entry)
        self.corr_md.content = self._corr_md()

    def _save_correspondences(self) -> None:
        if not self.correspondences:
            self.corr_md.content = "**nothing to save**"
            return
        DIR_MEASURE.mkdir(parents=True, exist_ok=True)
        doc = {
            "metadata": provenance(
                "world_view_correspondences",
                n=len(self.correspondences),
                cameras=self.usable,
                rig_source=self.rig.source),
            "ref_frame": self.rig.ref_frame,
            "stereo_file": self.rig.source,
            "convention": (
                "p_base_fk is the robot's own claim about where the feature "
                "is, from forward kinematics on the MEASURED joint angles. "
                "p_ref_triangulated is where the cameras see it, in the rig "
                "reference frame. Fitting one onto the other gives the rig's "
                "pose; the residual is the disagreement."),
            "correspondences": self.correspondences,
        }
        path = DIR_MEASURE / f"correspondences_{timestamp()}.json"
        save_json(doc, path)
        self.corr_md.content = (
            f"**saved {len(self.correspondences)}** -> `{path.name}`\n\n"
            f"```\npython calibration/scene_extrinsics.py anchor \\\n"
            f"    --points {path}\n```")

    # ------------------------------------------------------------- #
    def _add_point(self) -> None:
        if self.tri is None:
            self.points_md.content = "_click the feature in both cameras first_"
            return
        self.saved.append({"point": self.tri.point.copy(),
                           "tri": self.tri})
        self.points_md.content = self._points_md()

    def _measure_distance(self) -> None:
        if len(self.saved) < 2:
            self.points_md.content = ("_need two saved points -- ADD THIS "
                                      "POINT twice_")
            return
        from stereo import distance_with_error
        a, b = self.saved[-2]["tri"], self.saved[-1]["tri"]
        d = distance_with_error(a, b)
        sig = (f" +/- {d['sigma_mm']:.2f} mm" if "sigma_mm" in d else "")
        self.points_md.content = (
            self._points_md()
            + f"\n\n### {d['distance_mm']:.2f}{sig}  mm\n\n"
            + f"between the last two points, along "
              f"[{d['delta_m'][0]*1e3:+.1f}, {d['delta_m'][1]*1e3:+.1f}, "
              f"{d['delta_m'][2]*1e3:+.1f}] mm\n\n"
            + (f"_{d['note']}_" if "note" in d else ""))

    # ------------------------------------------------------------- #
    ## Readouts.  Every number that is an ASSUMPTION rather than a
    ## measurement says so where it is shown -- the failure mode this whole
    ## panel guards against is a confident millimetre readout produced from
    ## two clicks on different objects, which looks entirely normal in 3D.

    def _frame_note(self) -> str:
        if self.rig.ref_frame == "base":
            return ("Coordinates are in the robot world frame **`base`** "
                    "(+x operator's left, +y backward, +z up), so they are "
                    "directly comparable with forward kinematics.")
        return (f"Coordinates are in **`{self.rig.ref_frame}`**'s optical "
                f"frame -- the rig is not anchored to the robot yet. "
                f"**Lengths and distances are already correct**; only their "
                f"expression in robot coordinates is missing. Record "
                f"correspondences below and run `scene_extrinsics.py "
                f"anchor` to fix that.")

    def _nothing_usable_md(self) -> str:
        lines = ["### Nothing measurable yet", ""]
        for n, why in self.unusable.items():
            lines.append(f"- **{n}** — {why}")
        lines += [
            "",
            "A camera needs BOTH a live stream and a calibrated pose:",
            "",
            "```",
            "python calibration/world_view.py --from-robot \\",
            "    --cameras top_scene low_scene --measure",
            "python calibration/scene_extrinsics.py status",
            "```",
        ]
        return "\n".join(lines)

    def _idle_md(self) -> str:
        return (
            "### Click a feature in one camera, then the same feature in "
            "the other\n\n"
            "One click is a **ray**, not a point — the first click draws "
            "that ray in 3D and the yellow curve on the other image, "
            "labelled in **mm of depth**. Every position the feature could "
            "possibly be in lies on that curve. The second click picks "
            "which.\n\n"
            + self._frame_note()
            + f"\n\ncameras: {', '.join(self.usable)}"
            + ("" if not self.unusable else
               "\n\nunavailable: "
               + ", ".join(f"{n} ({w})" for n, w in self.unusable.items())))

    def _one_click_md(self, name: str) -> str:
        pick = self.picks[name]
        lines = [
            f"### one click — a ray, not yet a point",
            "",
            f"**{name}** at pixel "
            f"({pick.uv_real[0]:.1f}, {pick.uv_real[1]:.1f})",
            "",
            "The yellow curve on the other image is every place this "
            "feature could be, labelled in mm of depth. Click it there.",
        ]
        for other, c in self.curves.items():
            frac = c.get("fraction_visible")
            if frac is not None:
                lines.append("")
                lines.append(
                    f"- **{other}**: {frac*100:.0f}% of the "
                    f"{self.depth_lo.value:.2f}–{self.depth_hi.value:.2f} m "
                    f"search range falls inside the image"
                    + ("" if frac > 0.05 else
                       "  ← almost none of it; the depth range or the "
                       "extrinsics are wrong"))

        if self.plane_on.value:
            try:
                r = self.rig.ray_plane(
                    name, pick.uv_real,
                    (0.0, 0.0, float(self.plane_z.value)), (0, 0, 1))
                p = r["point_ref"]
                lines += [
                    "",
                    f"### on the z = {self.plane_z.value:.3f} m plane",
                    "",
                    f"**[{p[0]*1e3:+.1f}, {p[1]*1e3:+.1f}, {p[2]*1e3:+.1f}] mm**",
                    "",
                    f"- {r['metres_per_pixel']*1e3:.3f} mm per pixel there",
                    f"- incidence {r['incidence_deg']:.1f}° from the plane "
                    f"normal",
                    f"- **assumes the feature is exactly on the plane**; "
                    f"every 1 mm it sits above displaces this by "
                    f"{r['sensitivity_mm_per_mm']:.2f} mm sideways",
                ]
            except ValueError as e:
                lines += ["", f"_plane intersection: {e}_"]
        return "\n".join(lines)

    def _result_md(self) -> str:
        t = self.tri
        p = t.point
        verdict, problems = t.quality()
        s = t.sigma_axes_mm

        lines = [
            f"## [{p[0]*1e3:+.1f}, {p[1]*1e3:+.1f}, {p[2]*1e3:+.1f}] mm",
            "",
            self._frame_note(),
            "",
            "| | |",
            "|---|---|",
            f"| ray gap | **{t.ray_gap_mm:.2f} mm** — how close the two "
            f"rays came to meeting |",
        ]
        if s is not None:
            lines.append(
                f"| 1σ for a {t.sigma_px:.1f} px click | "
                + " × ".join(f"{v:.2f}" for v in s[0]) + " mm |")
        if "viewing_angle_deg" in t.meta:
            ang = t.meta["viewing_angle_deg"]
            lines.append(
                f"| viewing angle | {ang:.1f}° "
                + ("(near ideal)" if 60 < ang < 120 else
                   "(**poor** — the cameras see this from too similar a "
                   "direction)") + " |")
        if getattr(self, "_snap_px", None) is not None:
            lines.append(
                f"| snapped | {self._snap_px:.2f} px onto the epipolar "
                f"curve |")
        for n in t.cameras:
            lines.append(
                f"| {n} | depth {t.depth_m[n]*1e3:.0f} mm, reprojects "
                f"{t.reprojection_px[n]:.2f} px |")

        if self.depth_enabled:
            dc = self._depth_check()
            if dc:
                lines += ["", "**D405 depth cross-check** — from each "
                              "camera's own stereo pair, using none of the "
                              "extrinsics:", ""]
                for n, e in dc.items():
                    if e.get("depth_m") is None:
                        lines.append(f"- {n}: no depth return at that pixel")
                    else:
                        diff = e.get("difference_mm")
                        lines.append(
                            f"- {n}: {e['depth_m']*1e3:.0f} mm"
                            + (f" vs {e['triangulated_depth_m']*1e3:.0f} mm "
                               f"triangulated (**{diff:+.0f} mm**)"
                               if diff is not None else ""))
                lines.append(
                    "\n_The D405 is a short-range sensor (best ~70–500 mm). "
                    "At scene-camera distances it is the weaker of the two "
                    "numbers; a disagreement is not automatically the "
                    "triangulation's fault._")

        lines += ["", f"**{verdict.upper()}**"]
        for pr in problems:
            lines.append(f"- ⚠ {pr}")
        return "\n".join(lines)

    def _points_md(self) -> str:
        if not self.saved:
            return "_no points saved_"
        lines = ["| # | x | y | z | mm |", "|---|---|---|---|---|"]
        for i, s in enumerate(self.saved[-8:]):
            p = s["point"] * 1e3
            lines.append(f"| {len(self.saved)-len(self.saved[-8:])+i+1} | "
                         f"{p[0]:+.1f} | {p[1]:+.1f} | {p[2]:+.1f} | |")
        return "\n".join(lines)

    def _corr_md(self) -> str:
        n = len(self.correspondences)
        if n == 0:
            return ("_none recorded_ — an anchor wants **8–15**, spread "
                    "across the workspace and across several heights. "
                    "Points that are nearly coplanar leave the fit poorly "
                    "constrained out of that plane.")
        gaps = [c["ray_gap_mm"] for c in self.correspondences]
        span = np.ptp(np.array([c["p_base_fk"]
                                for c in self.correspondences]), axis=0) * 1e3
        return (f"**{n} recorded** — ray gaps "
                f"{min(gaps):.1f}–{max(gaps):.1f} mm\n\n"
                f"spread so far: {span[0]:.0f} × {span[1]:.0f} × "
                f"{span[2]:.0f} mm"
                + ("" if min(span) > 60 else
                   f"\n\n⚠ only {min(span):.0f} mm in the thinnest "
                   f"direction — add poses at different heights"))


## ------------------------------------------------------------------ ##
## Entry point used by world_view.py
## ------------------------------------------------------------------ ##

def attach(server, tabs, cameras: Sequence[str], workers: Dict[str, Any],
           world_view=None, depth_enabled: bool = False,
           frame: str = "auto", frustum_scale: float = 0.35
           ) -> Optional[MeasurePanel]:
    """Add the Measure tab. Returns None (with a note in the tab) if the
    rig cannot be assembled at all -- a missing calibration must not stop
    world_view from starting."""
    tab = tabs.add_tab("Measure")
    try:
        from rig import load_rig
        rig = load_rig(list(cameras), frame=frame)
    except Exception as e:  # noqa: BLE001
        with tab:
            server.gui.add_markdown(
                "### No measurement rig\n\n"
                f"```\n{e}\n```\n\n"
                "The 3D geometry needs the cameras' poses relative to each "
                "other. Nothing else in this viewer is affected.\n\n"
                "```\npython calibration/scene_extrinsics.py status\n```")
        return None
    return MeasurePanel(server, tab, rig, workers, world_view=world_view,
                        depth_enabled=depth_enabled,
                        frustum_scale=frustum_scale)
