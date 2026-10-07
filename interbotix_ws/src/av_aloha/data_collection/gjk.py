"""GJK: exact distance between convex bodies, from first principles.

WHY THIS BEATS EVERY FITTED PRIMITIVE
=====================================
Every model so far APPROXIMATES a link with something simpler and pays for
it in padding -- the gap between the model's surface and the real one:

    one capsule per link      28 mm median, 61 mm on the gripper bases
    8 capsules per link        9 mm median
    GJK on the convex pieces   0 mm  -- it IS the distance

GJK is not a fit.  Given two convex sets it returns their exact separation.
The only remaining approximation is the convex decomposition itself, and
that is handled separately by a measured leak term.


THE ONE IDEA
============
Define the Minkowski difference of two sets:

    A - B  =  { a - b  :  a in A, b in B }

Then for any a in A and b in B, the vector a-b lies in A-B, and |a-b| is
the distance between those two points.  So:

    dist(A, B)  =  min |a - b|  =  distance from the ORIGIN to (A - B)

Two bodies touch exactly when the origin lies inside A-B.  A
two-body problem has become a one-body problem: how far is a convex set
from a point?  That is the whole trick, and everything below is machinery
for answering it without ever building A-B (which for two 8-vertex hulls
would have up to 64 vertices, and far more in general).


HOW TO TOUCH A SET YOU NEVER BUILD: SUPPORT FUNCTIONS
=====================================================
The support function of a convex set returns its farthest point in a
direction d:

    s_A(d)  =  argmax_{x in A}  <x, d>

For a convex hull this is gloriously cheap -- the maximiser is always a
VERTEX, so it is one matrix-vector product and an argmax.  No faces, no
edges, no adjacency structure.

And support functions compose over the Minkowski difference:

    s_{A-B}(d)  =  s_A(d) - s_B(-d)

so we can probe A-B in any direction using only A's and B's vertices.
That is what makes GJK work on shapes it never explicitly constructs.


THE ITERATION
=============
Keep a simplex W (1 to 4 points) of points known to lie in A-B.  Its
convex hull is a subset of A-B, so the closest point of that hull to the
origin is an UPPER bound on the answer.

    v = closest point to the origin in conv(W)
    w = s_{A-B}(-v)            # probe further along the direction of v

Now the key inequality.  Because A-B is convex and w is its extreme point
in direction -v, EVERY point x of A-B satisfies <x, -v> <= <w, -v>, i.e.
the whole set lies on the far side of the plane through w with normal -v.
The distance from the origin to that plane is <v,w>/|v|, so

    <v, w> / |v|   <=   dist(0, A-B)   <=   |v|

A lower bound and an upper bound, both computable.  When they meet, we are
done -- and the gap between them is a genuine error bound, not a guess.
That is the termination test below.

Each iteration adds w to W, recomputes the closest point, and discards any
vertices of W that are not needed to express it.  W never exceeds 4 points
in 3D, because by Caratheodory's theorem the closest point of a convex
hull in R^3 is a combination of at most 4 vertices.


CONVERGENCE
===========
Monotone: |v| never increases.  In exact arithmetic GJK terminates finitely
on polytopes.  In floating point one uses a tolerance, which is what the
bound above provides honestly.  Typical convergence here is 3-6 iterations.


INTERSECTION
============
When the origin gets inside conv(W), the bodies overlap and the distance is
zero.  GJK alone does not give PENETRATION DEPTH (that needs EPA, which
expands a polytope outward).  For a safety gate that is fine and is stated
plainly: this returns 0.0 for any overlap, and the gate only ever asks
"is the distance at least the margin", to which 0 is already a decisive no.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

EPS = 1e-12


## ------------------------------------------------------------------ ##
## Closest point from the origin to a simplex
## ------------------------------------------------------------------ ##
##
## GJK needs, at every step, the point of conv(W) nearest the origin, plus
## WHICH vertices of W actually support it so the rest can be dropped.
## Written out per simplex dimension rather than as a general solver: the
## cases are few, each is a couple of dot products, and an explicit form is
## far easier to verify than a generic QP.

def _closest_point_segment(a, b):
    """Closest point to the origin on segment [a, b], and its support set."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < EPS:
        return a, (0,)
    t = float(-(a @ ab) / denom)
    if t <= 0.0:
        return a, (0,)
    if t >= 1.0:
        return b, (1,)
    return a + t * ab, (0, 1)


def _closest_point_triangle(a, b, c):
    """Closest point to the origin on triangle abc (Ericson, Voronoi regions).

    The triangle's plane is partitioned into seven regions -- three vertex,
    three edge, one interior -- and the barycentric tests below decide which
    one the origin projects into.  Checking regions rather than projecting
    and clamping is what keeps this exact on degenerate triangles."""
    ab, ac = b - a, c - a
    ap = -a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0 and d2 <= 0:
        return a, (0,)

    bp = -b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0 and d4 <= d3:
        return b, (1,)

    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        denom = d1 - d3
        t = d1 / denom if abs(denom) > EPS else 0.0
        return a + t * ab, (0, 1)

    cp = -c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0 and d5 <= d6:
        return c, (2,)

    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        denom = d2 - d6
        t = d2 / denom if abs(denom) > EPS else 0.0
        return a + t * ac, (0, 2)

    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        denom = (d4 - d3) + (d5 - d6)
        t = (d4 - d3) / denom if abs(denom) > EPS else 0.0
        return b + t * (c - b), (1, 2)

    denom = va + vb + vc
    if abs(denom) < EPS:
        ## Degenerate (collinear) triangle -- fall back to its longest edge.
        best, bset, bd = None, None, np.inf
        for i, j in ((0, 1), (0, 2), (1, 2)):
            pts = (a, b, c)
            p, sub = _closest_point_segment(pts[i], pts[j])
            if float(p @ p) < bd:
                bd, best = float(p @ p), p
                bset = tuple(sorted({(i, j)[k] for k in sub}))
        return best, bset
    v, w = vb / denom, vc / denom
    return a + ab * v + ac * w, (0, 1, 2)


def _closest_point_tetra(a, b, c, d):
    """Closest point to the origin on tetrahedron abcd.

    If the origin is inside, the distance is zero and the bodies overlap.
    Otherwise it lies outside at least one face, and the answer is the best
    of those faces' closest points."""
    pts = (a, b, c, d)
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    ## Inside test: same side as the opposite vertex for all four faces.
    inside = True
    for (i, j, k), opp in zip(faces, (3, 2, 1, 0)):
        n = np.cross(pts[j] - pts[i], pts[k] - pts[i])
        if float(n @ n) < EPS:
            continue
        s_o = float(n @ (pts[opp] - pts[i]))
        s_0 = float(n @ (-pts[i]))
        if s_o * s_0 < 0:
            inside = False
            break
    if inside:
        return np.zeros(3), (0, 1, 2, 3)

    best, bset, bd = None, None, np.inf
    for (i, j, k) in faces:
        p, sub = _closest_point_triangle(pts[i], pts[j], pts[k])
        dd = float(p @ p)
        if dd < bd:
            bd, best = dd, p
            bset = tuple(sorted({(i, j, k)[t] for t in sub}))
    return best, bset


def closest_point_on_simplex(W):
    """Closest point to the origin on conv(W), and the supporting subset."""
    n = len(W)
    if n == 1:
        return W[0], (0,)
    if n == 2:
        return _closest_point_segment(W[0], W[1])
    if n == 3:
        return _closest_point_triangle(W[0], W[1], W[2])
    return _closest_point_tetra(W[0], W[1], W[2], W[3])


## ------------------------------------------------------------------ ##
## GJK
## ------------------------------------------------------------------ ##

def gjk_distance(VA: np.ndarray, VB: np.ndarray,
                 max_iter: int = 64, tol: float = 1e-9
                 ) -> Tuple[float, int]:
    """Exact distance between the convex hulls of vertex sets VA and VB.

    Returns (distance, iterations).  Zero means the hulls overlap; GJK does
    not report penetration depth (that needs EPA), which is sufficient here
    because the gate only asks whether the distance reaches its margin.

    Neither hull is ever constructed: the sets are touched only through
    their support functions, which for a vertex cloud is `V @ d` and an
    argmax -- so passing raw vertices is not a shortcut, it is the intended
    interface.  Interior points are simply never selected."""
    VA = np.ascontiguousarray(VA, dtype=float)
    VB = np.ascontiguousarray(VB, dtype=float)

    def support(d):
        """s_{A-B}(d) = s_A(d) - s_B(-d) -- probing a set we never build."""
        return VA[int(np.argmax(VA @ d))] - VB[int(np.argmin(VB @ d))]

    v = VA[0] - VB[0]
    if float(v @ v) < EPS:
        v = np.array([1.0, 0.0, 0.0])
    W: list = []

    for it in range(1, max_iter + 1):
        w = support(-v)

        ## Termination on the DUALITY GAP, not on step size.
        ##   <v,w>/|v| <= dist <= |v|
        ## so |v|^2 - <v,w> is the width of the bracket, scaled by |v|.
        ## Testing this rather than "did v stop moving" is what makes the
        ## returned number provably converged instead of merely settled.
        vv = float(v @ v)
        if vv - float(v @ w) <= tol * max(vv, 1.0):
            return float(np.sqrt(vv)), it

        ## A repeated support point means no progress is possible.
        if any(float((w - x) @ (w - x)) < EPS for x in W):
            return float(np.sqrt(vv)), it

        W.append(w)
        v, keep = closest_point_on_simplex(W)
        W = [W[i] for i in keep]

        if float(v @ v) < EPS:
            return 0.0, it            # origin captured: the hulls overlap

    return float(np.linalg.norm(v)), max_iter


def gjk_distance_hulls(hull_a, hull_b, T_a=None, T_b=None) -> float:
    """Convenience wrapper for trimesh hulls with optional 4x4 poses."""
    VA = np.asarray(hull_a.vertices, dtype=float)
    VB = np.asarray(hull_b.vertices, dtype=float)
    if T_a is not None:
        VA = VA @ np.asarray(T_a)[:3, :3].T + np.asarray(T_a)[:3, 3]
    if T_b is not None:
        VB = VB @ np.asarray(T_b)[:3, :3].T + np.asarray(T_b)[:3, 3]
    return gjk_distance(VA, VB)[0]


## ------------------------------------------------------------------ ##
## The query a safety gate actually makes
## ------------------------------------------------------------------ ##
##
## `gjk_distance` computes the separation to full precision.  A gate never
## needs that -- it needs one bit: "is the separation at least `margin`?"
##
## The duality bracket derived in the module docstring,
##
##     <v,w>/|v|   <=   dist   <=   |v|
##
## answers that bit long before the bracket collapses:
##
##   * lower bound >= margin  -> PROVEN clear. Stop.
##   * upper bound <  margin  -> PROVEN too close. Stop.
##
## Only pairs straddling the margin need to iterate to convergence, and
## those are rare -- one or two out of sixty-four.  This is why a gate
## built on GJK is cheaper than one built on fitted primitives even though
## GJK is the more exact computation: exactness is only paid for where it
## changes the answer.

def gjk_separated(VA: np.ndarray, VB: np.ndarray, margin: float,
                  max_iter: int = 64, tol: float = 1e-9) -> Tuple[bool, float]:
    """Is dist(conv(VA), conv(VB)) >= margin?  Returns (verdict, bound).

    The second value is whichever bound decided it, so a caller can still
    report a number -- but it is a BOUND, not the distance, and is only
    tight when the verdict was close."""
    VA = np.ascontiguousarray(VA, dtype=float)
    VB = np.ascontiguousarray(VB, dtype=float)

    def support(d):
        return VA[int(np.argmax(VA @ d))] - VB[int(np.argmin(VB @ d))]

    v = VA[0] - VB[0]
    if float(v @ v) < EPS:
        v = np.array([1.0, 0.0, 0.0])
    W: list = []

    for _ in range(max_iter):
        w = support(-v)
        vv = float(v @ v)
        nv = np.sqrt(vv)

        ## Lower bound: the whole Minkowski difference lies beyond the
        ## supporting plane through w, so nothing can be nearer than this.
        if nv > EPS:
            lower = float(v @ w) / nv
            if lower >= margin:
                return True, lower          # proven clear, no more work
        ## Upper bound: |v| is achieved by an actual pair of points.
        if nv < margin:
            return False, nv                # proven too close

        if vv - float(v @ w) <= tol * max(vv, 1.0):
            return nv >= margin, nv
        if any(float((w - x) @ (w - x)) < EPS for x in W):
            return nv >= margin, nv

        W.append(w)
        v, keep = closest_point_on_simplex(W)
        W = [W[i] for i in keep]
        if float(v @ v) < EPS:
            return False, 0.0

    return float(np.linalg.norm(v)) >= margin, float(np.linalg.norm(v))


def bounding_spheres(pieces) -> Tuple[np.ndarray, np.ndarray]:
    """Centre and radius of each piece -- the broad-phase cull.

    A sphere test is three subtractions and a norm.  It cannot decide a
    close case, but it discards distant piece pairs before GJK is even
    entered, which is where most of the sixty-four go."""
    cs, rs = [], []
    for V in pieces:
        V = np.asarray(V, dtype=float)
        c = 0.5 * (V.max(axis=0) + V.min(axis=0))
        cs.append(c)
        rs.append(float(np.linalg.norm(V - c, axis=1).max()))
    return np.stack(cs), np.array(rs)


def pieces_separated(pieces_a, pieces_b, margin: float,
                     spheres_a=None, spheres_b=None) -> Tuple[bool, float]:
    """Are two decomposed links separated by at least `margin`?

    Broad phase (bounding spheres) then GJK, and the piece pairs are
    visited NEAREST FIRST: the whole query fails as soon as any single
    pair is too close, so the pair most likely to fail should be tested
    first.  Ordering costs one argsort and saves most of the work in the
    case that matters."""
    ca, ra = spheres_a if spheres_a is not None else bounding_spheres(pieces_a)
    cb, rb = spheres_b if spheres_b is not None else bounding_spheres(pieces_b)

    ## Sphere-gap matrix: a lower bound on every piece pair's distance.
    gap = (np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
           - ra[:, None] - rb[None, :])
    cand = np.argwhere(gap < margin)
    if len(cand) == 0:
        return True, float(gap.min())      # broad phase alone settled it

    order = np.argsort(gap[cand[:, 0], cand[:, 1]])
    worst = np.inf
    for idx in order:
        i, j = cand[idx]
        ok, b = gjk_separated(pieces_a[i], pieces_b[j], margin)
        worst = min(worst, b)
        if not ok:
            return False, worst
    return True, float(min(worst, gap.min() if gap.size else np.inf))
