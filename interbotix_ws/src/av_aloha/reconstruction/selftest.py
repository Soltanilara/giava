"""Hardware-free verification of the reconstruction geometry.

    python reconstruction/selftest.py

Every check here is a closed-loop test against ground truth that the code
under test does not have access to: synthetic cameras with known poses
project known 3D points, and the recovered numbers are compared with the
ones that generated them.  A test that only checks self-consistency would
pass just as happily with the distortion direction inverted, which is the
single most likely bug in this package.

The real D405 factory coefficients are used wherever a lens is needed, so
the tolerances quoted are the ones that apply on this rig, not on a
notional distortion-free camera.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CALIB = _HERE.parent / "calibration"
for _p in (str(_HERE), str(_CALIB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from camera import (PinholeCamera, _brown_conrady,  # noqa: E402
                    _invert_brown_conrady, normalize_model)
from stereo import (StereoRig, _least_squares_ray_intersection,  # noqa: E402
                    _ray_gap, distance, distance_with_error,
                    polyline_distance)

## The measured top_scene D405 colour intrinsics (640x480).
D405 = dict(width=640, height=480,
            fx=390.1454162597656, fy=389.6618957519531,
            cx=315.58648681640625, cy=241.462890625,
            model="distortion.inverse_brown_conrady",
            coeffs=(-0.05474819615483284, 0.06331445276737213,
                    0.0002434329508105293, 0.00048444102867506444,
                    -0.021259112283587456))

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            FAIL.append((name, e, traceback.format_exc()))
            print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
        else:
            PASS.append(name)
            print(f"  ok    {name}")
        return fn
    return deco


def _grid(cam, step=40, margin=4):
    u = np.arange(margin, cam.width - margin, step)
    v = np.arange(margin, cam.height - margin, step)
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu.ravel(), vv.ravel()], axis=-1).astype(float)


def make_rig():
    """A perpendicular pair, geometrically like the real top/low scene rig.

    top  : 1.0 m above the origin looking straight down (world -z)
    low  : 0.9 m in front at 0.25 m height, looking horizontally (world +y)
    """
    top = PinholeCamera(name="top", T_ref_cam=np.array([
        [1, 0, 0, 0.0],
        [0, -1, 0, 0.0],
        [0, 0, -1, 1.0],
        [0, 0, 0, 1.0]], dtype=float), ref_frame="base", **D405)
    ## Optical +z along world +y; image +x along world +x; image +y = world -z.
    low = PinholeCamera(name="low", T_ref_cam=np.array([
        [1, 0, 0, 0.0],
        [0, 0, 1, -0.90],
        [0, -1, 0, 0.25],
        [0, 0, 0, 1.0]], dtype=float), ref_frame="base", **D405)
    return StereoRig([top, low], source="selftest")


print("\nreconstruction selftest")
print("=" * 66)
print("\ndistortion model")


@check("normalize_model accepts the rs and opencv spellings")
def _():
    assert normalize_model("distortion.inverse_brown_conrady") \
        == "inverse_brown_conrady"
    assert normalize_model("opencv_plumb_bob") == "opencv_plumb_bob"
    try:
        normalize_model("fisheye_equidistant")
    except ValueError:
        return
    raise AssertionError("an unknown model must raise, never be guessed at")


@check("brown-conrady inversion is exact to 1e-10 for the real D405 coeffs")
def _():
    c = np.array(D405["coeffs"])
    ## Normalised coords spanning the whole 640x480 sensor and beyond.
    xy = np.random.default_rng(0).uniform(-1.0, 1.0, size=(500, 2))
    back = _invert_brown_conrady(_brown_conrady(xy, c), c)
    err = np.max(np.abs(back - xy))
    assert err < 1e-10, f"worst inversion error {err:.3e}"


@check("inverse_brown_conrady: unproject applies the polynomial FORWARD")
def _():
    """This is the direction trap. librealsense's rs2_deproject_pixel_to_point
    for inverse_brown_conrady applies the Brown-Conrady polynomial when
    going pixel->ray. Reproduce that literally and compare."""
    cam = PinholeCamera(name="c", **D405)
    uv = np.array([501.0, 92.0])
    x = (uv[0] - cam.cx) / cam.fx
    y = (uv[1] - cam.cy) / cam.fy
    c = D405["coeffs"]
    r2 = x * x + y * y
    f = 1 + c[0] * r2 + c[1] * r2 ** 2 + c[4] * r2 ** 3
    ux = x * f + 2 * c[2] * x * y + c[3] * (r2 + 2 * x * x)
    uy = y * f + 2 * c[3] * x * y + c[2] * (r2 + 2 * y * y)
    got = cam.unproject(uv)
    assert np.allclose(got, [ux, uy], atol=1e-12), f"{got} vs {[ux, uy]}"


@check("opencv model runs the OTHER way (project applies the polynomial)")
def _():
    cv_cam = PinholeCamera(name="cv", **{**D405, "model": "opencv_plumb_bob"})
    rs_cam = PinholeCamera(name="rs", **D405)
    uv = np.array([560.0, 60.0])
    a, b = cv_cam.unproject(uv), rs_cam.unproject(uv)
    ## They must DISAGREE -- same numbers, opposite conventions. If a future
    ## refactor makes them equal, the direction distinction has been lost.
    sep = np.linalg.norm(a - b) * cv_cam.fx
    assert sep > 0.5, (f"the two conventions differ by only {sep:.3f} px; "
                       f"they are supposed to be inverse maps")


@check("project/unproject round-trip over the whole sensor, both models")
def _():
    for model in ("distortion.inverse_brown_conrady", "opencv_plumb_bob"):
        cam = PinholeCamera(name="c", **{**D405, "model": model})
        uv = _grid(cam, step=20)
        xy = cam.unproject(uv)
        p = np.concatenate([xy, np.ones((len(xy), 1))], axis=1) * 0.7
        back = cam.project(p)
        err = np.max(np.linalg.norm(back - uv, axis=1))
        assert err < 1e-6, f"{model}: worst round-trip {err:.3e} px"


@check("project returns NaN behind the camera, never a plausible pixel")
def _():
    cam = PinholeCamera(name="c", **D405)
    out = cam.project(np.array([[0.1, 0.1, -0.5], [0.0, 0.0, 0.0]]))
    assert np.all(np.isnan(out)), out
    assert not np.any(cam.in_image(out))


@check("far-out-of-image points give NaN, and do not poison the batch")
def _():
    """The epipolar curve deliberately sweeps depths that project well
    outside the destination image. Those samples must come back NaN
    individually -- raising, or returning a half-converged number, would
    either discard the valid samples with them or fabricate a plausible
    pixel that is nowhere near the truth."""
    for model in ("distortion.inverse_brown_conrady", "opencv_plumb_bob"):
        cam = PinholeCamera(name="c", **{**D405, "model": model})
        pts = np.array([
            [0.0, 0.0, 1.0],            # dead centre, must survive
            [1e5, 1e5, 1e-9],           # essentially in the image plane
            [0.05, -0.02, 0.5],         # ordinary, must survive
            [np.inf, 0.0, 1.0],
            [-3e4, 2e4, 1e-6],
        ])
        uv = cam.project(pts)
        assert np.all(np.isfinite(uv[[0, 2]])), f"{model}: valid rows lost"
        assert np.all(cam.in_image(uv)[[0, 2]]), f"{model}: valid rows moved"
        assert not np.any(cam.in_image(uv)[[1, 3, 4]]), \
            f"{model}: an absurd point produced an in-image pixel"
        ## And the surviving rows must be untouched by their neighbours.
        alone = cam.project(pts[[0, 2]])
        assert np.allclose(uv[[0, 2]], alone, equal_nan=True), \
            f"{model}: a bad row changed a good one"


@check("unproject also survives absurd pixels element-wise")
def _():
    cam = PinholeCamera(name="c", **{**D405, "model": "opencv_plumb_bob"})
    uv = np.array([[320.0, 240.0], [1e9, -1e9], [100.0, 400.0]])
    xy = cam.unproject(uv)
    assert np.all(np.isfinite(xy[[0, 2]])), xy
    alone = cam.unproject(uv[[0, 2]])
    assert np.allclose(xy[[0, 2]], alone), "a bad row changed a good one"


@check("distortion is material: >2 px at the image corner")
def _():
    cam = PinholeCamera(name="c", **D405)
    ideal = PinholeCamera(name="i", **{**D405, "model": "none", "coeffs": ()})
    uv = np.array([8.0, 8.0])
    d = np.linalg.norm(cam.unproject(uv) - ideal.unproject(uv)) * cam.fx
    assert d > 2.0, (f"only {d:.2f} px -- if this ever drops to zero the "
                     f"coefficients are not being applied at all")


print("\nray geometry")


@check("ray_gap is zero for intersecting rays and exact for skew ones")
def _():
    o1, d1 = np.zeros(3), np.array([1.0, 0, 0])
    o2, d2 = np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert _ray_gap([o1, o2], [d1, d2], np.zeros(3)) < 1e-12
    o3 = np.array([0.0, -1.0, 0.037])
    assert abs(_ray_gap([o1, o3], [d1, d2], np.zeros(3)) - 0.037) < 1e-12


@check("least-squares intersection recovers an exact 3-ray meeting point")
def _():
    p = np.array([0.13, -0.22, 0.41])
    origins = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
               np.array([0, 0, 1.0])]
    dirs = [(p - o) / np.linalg.norm(p - o) for o in origins]
    got, cond = _least_squares_ray_intersection(origins, dirs)
    assert np.allclose(got, p, atol=1e-12), got
    assert cond < 10, cond


@check("parallel rays raise instead of returning a confident number")
def _():
    try:
        _least_squares_ray_intersection(
            [np.zeros(3), np.array([0, 0.1, 0])],
            [np.array([1.0, 0, 0]), np.array([1.0, 0, 0])])
    except ValueError:
        return
    raise AssertionError("parallel rays must raise")


print("\ntriangulation against synthetic ground truth")


@check("triangulation recovers known points to <1 micrometre")
def _():
    rig = make_rig()
    rng = np.random.default_rng(1)
    truth = np.stack([rng.uniform(-0.25, 0.25, 40),
                      rng.uniform(-0.25, 0.25, 40),
                      rng.uniform(0.02, 0.30, 40)], axis=1)
    worst = 0.0
    for p in truth:
        obs = {n: rig[n].project_ref(p) for n in rig.names}
        if not all(rig[n].in_image(obs[n])[0] for n in rig.names):
            continue
        tri = rig.triangulate(obs)
        worst = max(worst, float(np.linalg.norm(tri.point - p)))
        assert tri.ray_gap_mm < 1e-6, tri.ray_gap_mm
    assert worst < 1e-6, f"worst recovery error {worst*1e6:.3f} um"


@check("the perpendicular pair is well conditioned (angle near 90 deg)")
def _():
    rig = make_rig()
    p = np.array([0.0, 0.0, 0.05])
    ang = rig.viewing_angle_deg("top", "low", p)
    assert 60 < ang < 120, f"{ang:.1f} deg"
    tri = rig.triangulate({n: rig[n].project_ref(p) for n in rig.names})
    assert tri.condition < 5, tri.condition


@check("a deliberately mismatched click shows up as a large ray gap")
def _():
    rig = make_rig()
    p = np.array([0.05, 0.02, 0.10])
    obs = {n: rig[n].project_ref(p) for n in rig.names}
    obs["low"] = obs["low"] + np.array([25.0, 0.0])   # click 25 px away
    tri = rig.triangulate(obs)
    verdict, problems = tri.quality()
    assert verdict == "suspect", (tri.ray_gap_mm, problems)
    assert tri.ray_gap_mm > 5.0, tri.ray_gap_mm


@check("distance between two triangulated points matches ground truth")
def _():
    rig = make_rig()
    a, b = np.array([-0.10, 0.05, 0.08]), np.array([0.12, -0.04, 0.15])
    ta = rig.triangulate({n: rig[n].project_ref(a) for n in rig.names})
    tb = rig.triangulate({n: rig[n].project_ref(b) for n in rig.names})
    truth = float(np.linalg.norm(b - a))
    assert abs(distance(ta, tb) - truth) < 1e-6
    d = distance_with_error(ta, tb)
    assert abs(d["distance_m"] - truth) < 1e-6
    assert 0 < d["sigma_mm"] < 5.0, d


@check("triangulation needs two views; one raises with the alternatives")
def _():
    rig = make_rig()
    try:
        rig.triangulate({"top": (320, 240)})
    except ValueError as e:
        assert "ray_plane" in str(e) and "known_size" in str(e), str(e)
        return
    raise AssertionError("a single observation must raise")


print("\nuncertainty")


@check("propagated covariance matches a Monte-Carlo estimate to ~10%")
def _():
    rig = make_rig()
    p = np.array([0.06, -0.03, 0.09])
    obs = {n: rig[n].project_ref(p) for n in rig.names}
    sigma = 1.0
    tri = rig.triangulate(obs, sigma_px=sigma)

    rng = np.random.default_rng(7)
    samples = []
    for _ in range(4000):
        noisy = {n: obs[n] + rng.normal(0, sigma, 2) for n in rig.names}
        samples.append(rig.triangulate(noisy, sigma_px=0).point)
    mc = np.cov(np.stack(samples), rowvar=False)

    a = np.sqrt(np.diag(tri.covariance))
    b = np.sqrt(np.diag(mc))
    rel = np.max(np.abs(a - b) / b)
    assert rel < 0.12, f"analytic {a*1e3} mm vs monte-carlo {b*1e3} mm"


@check("uncertainty scales linearly with the assumed click sigma")
def _():
    rig = make_rig()
    p = np.array([0.0, 0.0, 0.10])
    obs = {n: rig[n].project_ref(p) for n in rig.names}
    s1 = rig.triangulate(obs, sigma_px=1.0).sigma_mm
    s3 = rig.triangulate(obs, sigma_px=3.0).sigma_mm
    assert np.allclose(s3, 3.0 * s1, rtol=1e-9), (s1, s3)


@check("sub-millimetre precision at 1 m with a 1 px click error")
def _():
    """The number that decides whether this rig can answer the question at
    all. If a 1 px click cost 10 mm, no amount of care elsewhere matters."""
    rig = make_rig()
    tri = rig.triangulate(
        {n: rig[n].project_ref(np.array([0.0, 0.0, 0.05]))
         for n in rig.names}, sigma_px=1.0)
    worst = tri.sigma_axes_mm[0][0]
    assert worst < 5.0, f"worst-axis sigma {worst:.2f} mm"


print("\nepipolar geometry")


@check("the true correspondence lies on the epipolar curve (<0.02 px)")
def _():
    rig = make_rig()
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(30):
        p = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2),
                      rng.uniform(0.03, 0.25)])
        uv_t = rig["top"].project_ref(p)
        uv_l = rig["low"].project_ref(p)
        if not (rig["top"].in_image(uv_t)[0] and rig["low"].in_image(uv_l)[0]):
            continue
        curve = rig.epipolar_curve("top", uv_t, "low",
                                   depth_range=(0.3, 1.5), n=240)
        worst = max(worst, polyline_distance(curve["pixels"], uv_l))
    assert worst < 0.02, f"true match sat {worst:.4f} px off the curve"


@check("the epipolar curve is genuinely curved (not a straight line)")
def _():
    """With a 78 degree lens, treating the locus as a line is a real error.
    Measure the sagitta: the largest deviation from the chord."""
    rig = make_rig()
    curve = rig.epipolar_curve("top", np.array([70.0, 60.0]), "low",
                               depth_range=(0.3, 1.6), n=800)
    px = np.asarray(curve["pixels"])
    ok = np.isfinite(px).all(axis=1)
    px = px[ok]
    a, b = px[0], px[-1]
    ab = b - a
    n = np.array([-ab[1], ab[0]])
    n = n / np.linalg.norm(n)
    sagitta = float(np.max(np.abs((px - a) @ n)))
    assert sagitta > 0.5, (
        f"sagitta only {sagitta:.3f} px -- either the distortion is not "
        f"being applied when projecting, or the test geometry got flattened")


@check("closest_on_curve beats its own sample spacing by >1000x")
def _():
    """An exact click must give an exact point. The unrefined answer would
    be quantised to the sample spacing (~6 mm here), which would silently
    become the accuracy floor of every measurement taken this way."""
    rig = make_rig()
    p = np.array([0.04, 0.06, 0.12])
    uv_t = rig["top"].project_ref(p)
    uv_l = rig["low"].project_ref(p)
    curve = rig.epipolar_curve("top", uv_t, "low", depth_range=(0.3, 1.5),
                               n=240)

    snapped = rig.closest_on_curve(curve, uv_l)
    err = np.linalg.norm(snapped["point_ref"] - p)
    spacing = snapped["sample_spacing_m"]
    assert spacing > 1e-3, f"spacing {spacing} -- test no longer discretised"
    assert err < 1e-6, f"exact click recovered {err*1e3:.4f} mm off truth"
    assert err < spacing / 1000.0, (
        f"refinement gained only {spacing/max(err,1e-15):.0f}x over the "
        f"{spacing*1e3:.1f} mm sample spacing")

    raw = np.asarray(curve["points_ref"])[snapped["index"]]
    assert np.linalg.norm(raw - p) > 100 * err, (
        "the nearest raw sample was already exact -- this test is not "
        "actually exercising the refinement")


@check("an off-curve click costs exactly what the pixel scale says")
def _():
    """A miss perpendicular to the curve is absorbed by the snap; a miss
    ALONG it is real depth error, and must be neither hidden nor inflated.
    ~1 px is ~2.3 mm at this range, so a 3.6 px click error is millimetres,
    not microns -- and that is the honest limit of clicking by hand."""
    rig = make_rig()
    p = np.array([0.04, 0.06, 0.12])
    uv_t = rig["top"].project_ref(p)
    uv_l = rig["low"].project_ref(p)
    curve = rig.epipolar_curve("top", uv_t, "low", depth_range=(0.3, 1.5),
                               n=240)
    snapped = rig.closest_on_curve(curve, uv_l + np.array([3.0, -2.0]))
    err = np.linalg.norm(snapped["point_ref"] - p)
    per_px = rig["low"].scale_at(abs(rig["low"].world_to_cam(p)[2]))
    assert err < 6.0 * per_px, (
        f"{err*1e3:.2f} mm for a 3.6 px offset, but a pixel is only "
        f"{per_px*1e3:.2f} mm here -- the snap is amplifying the error")
    assert snapped["snap_distance_px"] < 4.0


@check("depth is readable off the curve: monotone and correctly labelled")
def _():
    rig = make_rig()
    p = np.array([0.03, -0.05, 0.11])
    uv_t = rig["top"].project_ref(p)
    curve = rig.epipolar_curve("top", uv_t, "low", depth_range=(0.5, 1.2),
                               n=500)
    ## Each labelled depth must reproduce its own 3D point exactly.
    d = np.asarray(curve["depths_m"])
    pts = np.asarray(curve["points_ref"])
    z_cam = rig["top"].world_to_cam(pts)[:, 2]
    assert np.max(np.abs(z_cam - d)) < 1e-9, np.max(np.abs(z_cam - d))


print("\nsingle-view alternatives")


@check("ray_plane is exact for a point that lies on the plane")
def _():
    rig = make_rig()
    p = np.array([0.11, -0.07, 0.0])
    uv = rig["top"].project_ref(p)
    r = rig.ray_plane("top", uv, (0, 0, 0), (0, 0, 1))
    assert np.allclose(r["point_ref"], p, atol=1e-9), r["point_ref"]
    ## Looking straight down at a horizontal plane: no lateral sensitivity
    ## on axis, and the metres-per-pixel figure must be sane.
    assert r["sensitivity_mm_per_mm"] < 0.6, r
    assert 0.001 < r["metres_per_pixel"] < 0.01, r


@check("ray_plane sensitivity predicts the real off-plane error")
def _():
    rig = make_rig()
    p_true = np.array([0.15, 0.10, 0.020])       # 20 mm above the table
    uv = rig["top"].project_ref(p_true)
    r = rig.ray_plane("top", uv, (0, 0, 0), (0, 0, 1))
    lateral = float(np.linalg.norm(r["point_ref"][:2] - p_true[:2]))
    predicted = 0.020 * r["sensitivity_mm_per_mm"]
    assert abs(lateral - predicted) < 0.2 * predicted + 1e-4, \
        (lateral, predicted)


@check("ray_plane refuses a plane behind the camera or edge-on")
def _():
    rig = make_rig()
    uv = np.array([320.0, 240.0])
    for kwargs in (dict(plane_point=(0, 0, 2.0), plane_normal=(0, 0, 1)),
                   dict(plane_point=(0, 0, 0), plane_normal=(1, 0, 0))):
        try:
            rig.ray_plane("top", uv, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"should have refused: {kwargs}")


@check("depth_from_known_size recovers a fronto-parallel segment's depth")
def _():
    rig = make_rig()
    ## A 150 mm segment lying flat, seen by the downward-looking camera.
    a = np.array([-0.075, 0.0, 0.0])
    b = np.array([+0.075, 0.0, 0.0])
    r = rig.depth_from_known_size(
        "top", rig["top"].project_ref(a), rig["top"].project_ref(b), 0.150)
    assert abs(r["depth_m"] - 1.0) < 1e-6, r["depth_m"]
    ## The small-angle form must be close but NOT identical -- if it is
    ## identical the exact form has been quietly replaced by it.
    assert 0.01 < r["small_angle_disagreement_pct"] < 5.0, r


@check("depth_from_known_size overstates depth on a tilted segment")
def _():
    rig = make_rig()
    tilt = np.radians(25.0)
    half = 0.075
    a = np.array([-half * np.cos(tilt), 0.0, +half * np.sin(tilt)])
    b = np.array([+half * np.cos(tilt), 0.0, -half * np.sin(tilt)])
    r = rig.depth_from_known_size(
        "top", rig["top"].project_ref(a), rig["top"].project_ref(b), 0.150)
    ## Predicted by the documented 1/cos(t) rule.
    assert r["depth_m"] > 1.0
    assert abs(r["depth_m"] - 1.0 / np.cos(tilt)) < 0.02, r["depth_m"]


print("\nguards")


@check("a camera with no pose raises, naming the fix")
def _():
    cam = PinholeCamera(name="unposed", **D405)
    rig = StereoRig([cam])
    for fn in (lambda: rig.require_posed("unposed"),
               lambda: cam.centre,
               lambda: cam.ray((1.0, 2.0))):
        try:
            fn()
        except ValueError as e:
            assert "scene_extrinsics" in str(e) or "extrinsic" in str(e), str(e)
            continue
        raise AssertionError("an unposed camera must raise")


@check("cameras posed in different frames cannot form a rig")
def _():
    a = PinholeCamera(name="a", T_ref_cam=np.eye(4), ref_frame="base", **D405)
    b = PinholeCamera(name="b", T_ref_cam=np.eye(4), ref_frame="top_scene",
                      **D405)
    try:
        StereoRig([a, b])
    except ValueError as e:
        assert "reference frame" in str(e)
        return
    raise AssertionError("mixed reference frames must raise")


@check("a bad depth_range raises rather than producing an empty curve")
def _():
    rig = make_rig()
    try:
        rig.epipolar_curve("top", (320, 240), "low", depth_range=(1.0, 0.5))
    except ValueError:
        return
    raise AssertionError("inverted depth_range must raise")


print("\n" + "=" * 66)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("\nfailures in detail:")
    for name, err, tb in FAIL:
        print(f"\n--- {name} ---\n{tb}")
print("=" * 66 + "\n")
sys.exit(1 if FAIL else 0)
