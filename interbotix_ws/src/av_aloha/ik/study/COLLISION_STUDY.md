# Collision study — Part 1: implementations, threat model, pair analysis

Status: C1–C3 complete (2026-08-08). Benchmarking (C4+) pending user
inspection of the geometry in Viser (`view_collision.py`, port 8091).

## C1 — What PyRoki actually provides

Two geometry modes behind one `RobotCollision` class; everything else
(pair machinery, distances, costs) is shared.

### Capsule mode — `RobotCollision.from_urdf(urdf)`
- **One capsule per link**, fitted as the **minimum bounding cylinder** of the
  link's URDF *collision* mesh (`trimesh.bounds.minimum_cylinder`), used as a
  capsule: hemispherical caps are **added beyond the cylinder ends**. The fit
  is strictly conservative laterally *and* sticks out by one radius at each
  end — pathological for plate-like links (below).
- 31 links on giava; links with real geometry: 28.

### Sphere mode — `RobotCollision.from_sphere_decomposition(dict, urdf)`
- **PyRoki does not generate spheres** — the caller supplies
  `{link: {centers, radii}}`. Sources available to us: (a) curobo's
  voxel-based mesh→spheres fitter (in-repo:
  `curobo/_src/geom/sphere_fit/fit_spheres.py`), (b) decomposing the fitted
  capsules (`Capsule.decompose_to_spheres(n)`), (c) hand-tuning.
- Pairs are **geometry-level**: a link pair with k_i and k_j spheres
  contributes k_i·k_j pairs — pair count multiplies where capsules had 1.

### Shared machinery
- **Distances**: closed-form and cheap — capsule↔capsule = closest
  segment-segment points + radii; sphere↔sphere = center distance − radii.
  Smooth a.e. (parallel-segment edge case exists). The distance evaluation
  itself is *not* the expected cost driver.
- **Pairs**: all link pairs, upper triangle, minus URDF parent↔child
  adjacents, minus `user_ignore_pairs`. **435 active pairs** for giava.
  Self- vs inter-arm collision are *not* distinguished — one flat list.
- **Residual** (`self_collision_cost`): per pair,
  `weight · (−colldist_from_sdf(d, margin))` where `colldist_from_sdf` is a
  **smooth margin hinge** (arXiv 2310.17274): exactly 0 for d ≥ margin, a
  quadratic ramp inside the margin, linear in penetration. So the *shape* of
  the soft cost is already hinge/margin-based — the Phase-4 failure was not
  the penalty shape but the pair set and geometry.
- **Formulation**: self-collision exists **only as a soft LS cost** in
  PyRoki. The AL **constraint** form exists only for world collision;
  a self-collision constraint is a 3-line custom
  (`Cost.factory(kind="constraint_leq_zero")(self_collision_residual)`) — on
  the C7 menu, clearly labeled custom.
- **Gradients**: pure autodiff. jaxls `jac_mode="auto"` picks
  **jacfwd when residual_dim ≥ tangent_dim** — 435 > 23, so the collision
  Jacobian = 23 JVPs through FK + geometry transforms + all pair distances +
  hinge, per LM iteration. This is the structural-cost hypothesis (C9
  profiles it); Phase 4 measured 27× solve time at 17 vs 7.4 iterations
  (≈11× per-iteration cost) and an ~18 min XLA compile.

## C2 — What the three-arm setup must actually protect against

From the robot layout and the measured pair excursions (20,350 configs:
full frozen-suite baseline rollout + 3,000 uniform-random configs):

1. **Hand↔hand (inter-arm)** — the primary teleop risk. Gripper bases,
   fingers, wrists of opposite arms genuinely cross (suite min −133 mm on
   the designed crossing; real usable signal, huge dynamic range [−133, +2200]).
2. **Hands↔camera body/cover** — real and deployment-relevant: grasp
   reaches pass the camera at capsule-distance −40 mm.
3. **Within-arm folding** — base/shoulder↔forearm/wrist/gripper of the same
   arm during folds (`self_fold` bottomed at −157 mm on base↔lower_forearm,
   range [−201, +408] — genuine signal).
4. **Camera-arm self-collision** — mostly rigid neighborhoods (no signal);
   the varying camera↔own-forearm pairs are kept.
5. **NOT a target**: same-gripper finger↔finger — that contact is *grasping*
   (global min −35 mm occurs when the gripper closes). Penalizing it fights
   the gripper's job.

## C3 — Pair analysis and the margin-aware pruning rule

Distance statistics per pair over the 20,350-config corpus give three
disjoint classes (raw data: `results/pair_analysis.npz`):

| class | rule (margin m) | count @ m=3 cm | disposition |
|---|---|---|---|
| **structural** | max distance over corpus < m — the pair can *never* satisfy the margin | **12** | prune: zero discriminative signal, pure constant gradient bias (this is precisely what corrupted the Phase-4 run) |
| **functional** | same-gripper finger pairs | **2** | prune: intended contact |
| **inactive** | min distance over corpus > 5 cm ≥ m + buffer | **76** | prune for speed; correctness-neutral at m ≤ 3 cm |
| **kept** | everything else | **345** | the working set |

The 12 structural pairs, with corpus ranges (mm) — every one an intra-chain
near-rigid neighborhood, none inter-arm:

```
leftbase↔leftupper_arm        [−186, −91]   rightbase↔rightupper_arm  [−185, −45]
middle_base↔middle_upper_arm    [−179, −81]   camera_body↔camera_cover  [−85, −85] (rigid: constant)
leftwrist↔left_gripper_base    [−111, −81]   rightwrist↔right_gripper_base [−111, −81]
leftlower_forearm↔left_gripper_base  [−133, −38]   right… same           [−133, −38]
middle_pan↔camera_cover        [−45, −40]    middle_wrist↔camera_cover  [−40, −5]
middle_pan↔camera_body         [−9, −2]      middle_lower_forearm↔middle_pan [−54, +22]
```

**Root cause** (why "permanently inside any margin" happens): the
minimum-bounding-cylinder fit orients **plate-like links** with the cylinder
axis along the thin dimension, then the capsule caps add a full radius on
each flat side. Measured fits: the base plate (mesh 0.299×0.204×0.079 m)
becomes r=0.154 → effective thickness 0.39 m; `middle_camera_cover` (a 5 mm
plate) becomes a ~14.4 cm-diameter blob; `upper_forearm` (0.204×0.101×0.039)
gets r=0.108. Neighbouring links sit *inside* these inflated shells at every
configuration. These are **artifacts of the geometric approximation**, not
meaningful collisions — but note the same inflation also biases the *kept*
inter-arm pairs conservative by several cm (hands-vs-camera reading −40 mm
does not mean meshes touch). Part 2 will quantify capsule/sphere conservatism
against true mesh distances before interpreting any clearance numbers as
safety.

**Sampling caveat, stated**: "inactive" and "structural" classes rest on a
20k-config corpus (suite + uniform random). Structural pruning is robust
(those neighborhoods are kinematically rigid); inactive pruning carries a
2 cm buffer above the margin. The pruned list is generated by rule from
`pair_analysis.npz`, not hand-curated.

---

# Part 2 — Ground truth, corrected capsules, and the multi-sphere model

## Mesh ground truth (`results/mesh_ground_truth.csv`)

70 cases = 14 representative pairs × 5 configurations (home, self-fold apex,
arms-converge apex, grasp end, parked hands). Method: dense surface samples
(4k/mesh) + KD-tree for separation, convex-hull signed distance for
penetration (±3 mm accuracy; hull slightly over-estimates depth on
non-convex links). Findings that reframed the study:

- The self-fold apex — **a legitimate rest posture (user)** — is truly
  clear: +9…+89 mm, with one −5.8 mm light nesting (upper↔lower forearm),
  consistent with how the arms physically rest.
- The permanently-negative capsule pairs split into **artifacts** (e.g.
  base↔upper_arm truly +43 mm at home while capsules claim −129) and **one
  real permanent assembly contact** (wrist↔gripper_base, −4.8 mm at every
  configuration — correctly pruned as structural either way).
- Capsule conservatism on *kept* pairs is huge: home gripper↔gripper truth
  186 mm, pyroki capsule 31 mm — 155 mm of bimanual workspace falsely
  consumed.

## Corrected single capsule (`collision_models.tight_capsule_collision`)

User-diagnosed defect: `minimum_cylinder` minimizes *cylinder* volume, so
box/plate-like links (fingers 54×99×27 mm) get the axis along the *thin*
dimension, and the caps then add a radius on each flat side. The corrected
fit (axis = longest OBB dimension, exact vertex containment) halves most of
the error — but a single capsule still cannot represent boxes: it *inverts*
the home cross-finger case (truth +14 mm → −34 mm) because fingers face each
other thin-side-on and any capsule doubles a box's thin dimension. Single
capsules are structurally unable to meet the acceptance criteria.

## Multi-sphere model (`collision_models.sphere_collision`)

curobo's VOXEL algorithm (interior grid → inscribed radii), mirrored
dependency-free (warp/torch not importable in this env — flagged deviation),
with three engineering fixes required by this robot's meshes:
1. **Per-body convex hulls** — the link STLs are thin-shell housings
   (12–23% volume fill); inscribed spheres of the raw mesh live in the walls.
2. **Plate-aware fitting** — hulls thinner than 25 mm get a mid-plane 2-D
   grid at r = max(t/2, 12 mm): bounded out-of-plane protrusion (≤ 12 mm)
   instead of unbounded under-coverage.
3. **Spread-aware selection** + dropping negligible hardware bodies.

Result: **180 spheres** (`results/sphere_decomposition.json`, frozen &
reviewable), 14,491 raw sphere pairs before pruning. Sphere counts per link:
12 base plates, 20 gripper forks, 4 fingers (12 mm, along the finger), 3–8
arm links, 4 camera cover.

## The scorecard (Phases 2–4; `results/geometry_scorecard.csv`)

Accuracy vs mesh truth over the 70 cases, |error| in mm:

| scope | pyroki capsule | tight capsule | **spheres** |
|---|---|---|---|
| ALL mean / max | 97.9 / 254 | 73.9 / 159 | **11.1 / 31** |
| plates | 162 | 106 | **11.4** |
| gripper fork | 111 | 79 | **5.8** |
| fingers (inter-arm) | 36 | 40 | **9.9** |
| signed bias | −98 (conservative) | −74 | **+7.8 (optimistic)** |

Safety classification (threshold 0 / 25 mm near band):

| model | FP collisions | FN | near-FP | stolen workspace med/max mm |
|---|---|---|---|---|
| pyroki | 15 | 0 | 14 | 94 / 254 |
| tight | 16 | 0 | 13 | 78 / 159 |
| **spheres** | **0** | 7* | **0** | **0 / 27** |

*The sphere FNs are definitional, not misses: the inscribed fit is
optimistic, so true contacts read +1…+18 mm — **all 7 fall inside a 25 mm
margin band** ("near-detected 7/7"). The model must therefore be used with
`margin > 18.3 mm` (the worst optimistic reading on a true contact), and
"collision" must mean "distance < margin", never "distance < 0".

Acceptance criteria:

| criterion | pyroki | tight | **spheres** |
|---|---|---|---|
| C1 rest pose readable as OK | −158 mm, 6 severe false collisions | −120 mm, 4 | **+0.9 mm, 0 — PASS** |
| C2 bimanual stolen clearance (med/max) | 29/156 mm | 38/76 mm (+2 false collisions) | **0 / 18 mm — PASS** |
| C3 real contacts caught | 7/7 at d<0 (but with 15 FP) | 7/7 (16 FP) | **7/7 within 25 mm band, 0 FP — PASS with margin ≥ 20 mm** |

The two genuinely dynamic contact cases read +14.2 mm (designed gripper
crossing) and +18.3 mm (rest-pose nesting — which we *want* tolerated; at a
20–25 mm margin it sits at the gentle edge of the activation ramp, exactly
the desired "aware but not repelled" behaviour).

**Data-driven margin window: 20–25 mm** (> 18.3 mm worst optimistic error;
below the ~29 mm where capsule-era false repulsion began). Phase 9 sweeps
within this window on the solver.

---

# Part 3 — Complexity, pairs, and the solver (user-authorized full run)

## Phase 6 — sphere-count Pareto: 180 stands

Reduction operator: greedy merge of each link's two nearest spheres into
their minimal bounding sphere — merging (c₁,r₁),(c₂,r₂) with d = ‖c₂−c₁‖
gives r = (d+r₁+r₂)/2, c on the segment; the bound *contains* both parents,
so coverage never decreases and only conservatism can grow. Scored against
the 70 mesh-truth cases:

| variant | spheres | sphere pairs | mean\|e\| mm | FP | stolen max mm |
|---|---|---|---|---|---|
| full | 180 | 14,491 | 11.1 | 0 | 27 |
| reduced (forks 20→10, bases 12→8) | 144 | 9,359 | 17.9 | 1 | 63 |
| minimal (forks→6, ~half counts) | 104 | 4,893 | 32.3 | 6 | 121 |

Any meaningful reduction re-creates capsule-style conservatism exactly where
it hurts (the forks — C2 violation at 144). **Decision: 180 spheres.** The
complexity lever is pair pruning, not sphere count.

## Phase 7 — sphere-pair classification: the permanent class dissolves

Same 20,350-config corpus, sphere-pair granularity, margin 25 mm,
activation buffer 25 mm:

| class | rule | count |
|---|---|---|
| functional | same-gripper finger pairs (grasping) | 32 |
| permanent | corpus max < 25 mm (no signal, constant bias) | **12** |
| inactive | corpus min > 50 mm (never near the band) | 8,563 |
| **kept** | | **5,884** |

The capsule model had 12 *entire link pairs* permanently inside the margin;
the sphere model has 12 individual *sphere* pairs, scattered inside
otherwise-healthy link pairs, and **zero link pairs that are fully
permanent** — the geometry fix dissolved the artifact class rather than
hiding it. 92 of 355 link pairs prune away entirely (all their sphere pairs
inactive). Prune list generated by rule → `results/sphere_pair_pruning.npz`.

## Phase 8 setup — trajectories, variants, pilot

- Collision-stress subset (separate module, frozen suite untouched):
  `bimanual_parallel` (side-by-side downward grips at 16 cm centres, ±8 cm
  shared translation — correct behaviour is ZERO deviation) and
  `bimanual_handoff` (right hand closes to a 4 cm fork gap and holds —
  margin-edge behaviour). Both oracle-clean (0.0 mm reachability error).
- Solver variants: `collision_sphere` (soft margin-hinge cost, pruned 5,884
  pairs), `collision_sphere_al` (custom AL constraint = hard standoff at the
  margin; pyroki has no native self-collision constraint — 3-line wrap of
  the native residual, flagged), `collision_capsule` (corrected capsules +
  Part-1 link pruning — the head-to-head reference).
- **Pilot: the sphere model compiles in 2.5 s** where the 435-pair capsule
  model took ~18 min — the capsule segment-segment closest-point routine is
  the XLA compile bomb, not the pair count. Pilot solve: median 9.9 ms.

## Theory notes (the math behind the measurements)

### Why the capsule fit fails plates and fingers
`trimesh.bounds.minimum_cylinder` minimizes cylinder volume V = πr²h over
orientations. For a box a×b×t with a ≥ b ≫ t, the two candidate
orientations give
    axis ⊥ plate:  r ≈ √(a²+b²)/2,  h = t      → V ∝ (a²+b²)·t
    axis ∥ a:      r ≈ √(b²+t²)/2,  h ≈ a      → V ∝ (b²+t²)·a
For the finger (a,b,t = 99,54,27 mm) these volumes differ by ~3%, and the
⊥-orientation wins. But PyRoki *uses the cylinder as a capsule*: hemispherical
caps add a full radius at each end, so the ⊥ fit's effective thickness
becomes t + 2r — the 27 mm finger reads 129 mm thick. The corrected fit
constrains the axis to the longest OBB dimension and chooses (r, h) as the
smallest containment pair: r = max_i ‖p_i − axis‖, then
h = 2·max_i(|z_i| − √(r² − ρ_i²))₊ — every vertex satisfies the capsule
inequality |z| ≤ h/2 + √(r²−ρ²). Even optimally, one capsule inflates a box's
thin dimension to its mid dimension — the structural reason single primitives
fail the bimanual criterion.

### The sphere model's signed bias, and the margin bound
Interior-fitted spheres (center c inside the hull, radius = SDF(c)) are a
subset of the body: the model surface lies *inside* the true surface, so the
model distance overestimates the true distance:
    d_model = d_true + ε,   ε ≥ 0,   E[ε] ≈ 8 mm, max ε = 18.3 mm (measured
    on contact cases; ε comes from inter-sphere waists and plate protrusion
    trade-offs).
A margin-activated cost is therefore safe iff every true contact activates
the cost: d_true ≤ 0 ⇒ d_model ≤ m requires **m > max ε = 18.3 mm**. The
opposite failure (false repulsion) grows with m, bounded by the workspace
the band consumes; the capsule era showed repulsion damage from ~29 mm of
stolen clearance. Hence the window **m ∈ [20, 25] mm**, with the weight
setting how hard the boundary is held (measured in the Phase-9 sweep).

### The cost that enters the optimizer
Per active pair i, PyRoki's soft cost contributes the residual
    hᵢ(q) = −colldist_from_sdf(dᵢ(q), m)
          = 0                          for dᵢ ≥ m
          = (m − dᵢ)²/(2m)             for 0 < dᵢ < m   (smooth ramp)
          = m/2 − dᵢ                   for dᵢ ≤ 0       (linear in penetration)
and the LM objective gains Σᵢ w²hᵢ². C¹-continuity at both joints of the
piecewise definition is what keeps the boundary approach non-chattering.
The AL variant instead enforces hᵢ ≤ 0 (⇔ dᵢ ≥ m) with per-pair Lagrange
multipliers — a *hard standoff at the margin*, which is measurably the wrong
semantics for close bimanual work (the sweep quantifies it).

### Where the compute goes — the differentiation scaling law
jaxls's `jac_mode="auto"` chooses forward-mode when the residual dimension
exceeds the tangent dimension (23). The collision residual has P outputs
(5,884 pruned), so the Jacobian costs ~23 JVPs through
FK → 180 SE(3) transforms → P pair distances → hinge; the pair count enters
*once per JVP*, which is why the residual eval itself is cheap and the
Jacobian dominates. Two consequences, both measured in Phase 10:
1. Pair pruning cuts the Jacobian linearly (5,884/14,491 ≈ 0.41×).
2. **Aggregation**: replacing the P-vector residual with the scalar
   r = w·√(Σᵢ hᵢ² + ε) leaves the least-squares objective *identical*
   (r² = w²Σhᵢ² + w²ε) but flips auto-diff to reverse mode: ONE VJP instead
   of 23 JVPs. The price is a rank-1 Gauss-Newton approximation of the
   collision block (vs rank-P), which can cost LM iterations when many pairs
   are simultaneously active — an empirical trade measured at solve level.

### Classification as a safety problem
With truth t and model d, at activation threshold m:
  FP (d < m, t ≥ m): stolen workspace / false repulsion — costs capability.
  FN (d ≥ m, t < 0): undetected contact — costs hardware.
The two errors trade against the model's signed bias: conservative models
(capsules, bias −74…−98 mm) buy FN=0 at 15–16 FPs and 78–94 mm median
stolen clearance; the optimistic sphere model buys FP=0 and pushes FN risk
into the margin condition above. The sphere-plus-margin design is the only
one of the three that can satisfy all three acceptance criteria at once.

## Phase 9 — margin × weight sweep (results/sweeps/collision_margin_weight.csv)

Corrected collision subset (bimanual parallel/tight/handoff) + arms_converge,
self_fold, teleop_grasp; soft grid {15,20,25,30,40} mm × {10,30,100,300};
AL at {10,25} mm. Plot: results/plots/collision_sweep.png.

Findings:
1. **False repulsion is a non-issue in the practical range.** The clean
   bimanual tests hold 0.0 mm deviation for every margin ≤ 30 mm at
   w ≤ 100 (margin 20 stays 0.0 even at w = 300). The capsule-era fear of
   hand-jerking was pure geometry, not the cost formulation.
2. **Boundary-riding is graceful.** On bimanual_tight (commanded −5.4 mm)
   the solver deviates 1–4 mm to hold +12…+19 model clearance — the
   "deviate just enough" behaviour, scaling smoothly with weight.
3. **The weight threshold is sharp.** On the designed crossing, min model
   clearance vs weight crosses zero between w=30 and w=100 at every margin:
   w ≤ 30 cannot hold the boundary against pose weight 50; w = 100 holds
   +3…+8 mm; w = 300 holds +9…+18 mm at 45–66 ms during the conflict.
4. **Rest pose and the grasp path are untouched**: self_fold dev 0.3 mm
   (its 41.9 mm p95 is commanded infeasibility, same as baseline);
   teleop_grasp dev 0.4 mm with the camera now held at +19.7 model
   clearance (the baseline drove capsule-model −41 mm there).
5. **The AL constraint form is empirically rejected**, confirming the theory
   note: hard-standoff-at-margin fights boundary riding (m=25: 81 LM
   iterations, 80% non-convergence on bimanual_tight, 9 mm forced deviation)
   and *diverges* under infeasible commands (arms_converge: 67% nonconverged
   AND −28 mm clearance — worse than soft at 1/10th the cost).

**Operating point: margin 20 mm, weight 100** (window-interior, first
boundary-holding weight, zero false repulsion, all six trajectories at
4–6 ms median solve). w=300 documented as the "harder guarantee" option
(+6–10 mm more clearance on adversarial commands, ~55 ms during conflicts).

Honest caveat on adversarial commands: with the soft cost the held boundary
sits near model-zero, and the sphere model is optimistic by ε ≤ 18 mm
locally — so a deliberately interpenetrating command (arms_converge) is
*bounded* to ≈1–2 cm true penetration at w=100, not prevented outright;
w=300 or an added hard floor would tighten this at real cost. For
teleoperation (where the crossing command is operator error, not a task),
"resist and bound" is the appropriate semantics.

## Phase 8 — full-suite results (hash-verified vs every earlier variant)

All at margin 20 mm / weight 100, clean-tracking aggregates:

| variant | pos p95 mm | ori p95 ° | worst clearance | solve mean ms | worst-traj p95 ms | jumps | jerk | nonconv |
|---|---|---|---|---|---|---|---|---|
| baseline | 4.72 | 2.14 | −157 | 0.68 | 3.2 | 241 | 1905 | 0.002 |
| **+ sphere collision** | **4.72 (unchanged)** | **2.14** | **+3.1** | 8.03 | 67.0 | 248 | 1989 | 0.005 |
| + capsule collision (same margin/weight/pruning) | 13.05 | 4.52 | +3.1 | 43.1 | 121 | 735 | 9543 | **0.326** |
| smooth_center (no collision) | 0.60 | 0.24 | −157 | 0.56 | 3.7 | 138 | 378 | 0.001 |
| **smooth_center + sphere collision** | **0.60** | **0.24** | **+3.0** | 5.18 | 64.3 | 128 | 350 | 0.003 |
| smooth + sphere collision | 0.44 | 0.21 | +3.0 | 6.49 | 65.3 | 144 | 357 | 0.003 |
| center + sphere collision | 0.30 | 0.08 | +3.0 | 5.54 | 65.7 | 224 | 1783 | 0.004 |
| sphere collision + manipulability | (traps return, 47.5/38.4 mm; 257 s wall; max 246 ms) — manipulability rejected in every combination |

Reading:
- **Sphere collision costs zero tracking on every clean trajectory** — the
  full frozen suite reproduces baseline p95s to the display digit — while
  taking worst clearance from −157 mm to +3.1.
- The capsule reference at identical settings fails comprehensively: plain
  trans_x saturates the LM budget (100 iters, 97 ms), self_fold is blocked
  (80.6 mm), 32.6% non-convergence. Geometry alone separates a working
  system from a broken one.
- The deployment stack (smoothing 0.05 + centering 0.5 + sphere collision)
  composes without interference: 0.60 mm / 0.24° clean, all safety
  boundaries held, 5.2 ms mean (the pair *reduces* collision-variant
  iterations, so the full stack is cheaper than collision alone: 102 s vs
  145 s suite wall).
- A pleasant surprise: collision *incidentally prevents the yaw-grasp branch
  dead-end* (the dead-end path swings the hand into the camera's protection
  band, blocking entry to the bad branch): smooth+collision reaches ~1 mm on
  the whole grasp family even without centering.
- Known regression: `reach_limit` recovery 6.8 → 27–32 mm p95 (boundary
  stretch + extension self-proximity + smoothing lag compound). Boundary
  over-reach is operator error in deployment; acceptable, documented.

## The bimanual battery (results/collision_subset_eval.csv, plots/collision_battery.png)

max deviation from the baseline path / min model clearance:

| variant | parallel (clear) | tight (boundary-riding) | handoff (margin-edge) |
|---|---|---|---|
| baseline | 0.0 / +17.1 | 0.0 / **−5.4 (contact)** | 0.0 / +4.1 |
| smooth_center | 1.0 / +16.7 | 1.0 / −5.2 (contact) | 0.8 / +4.1 |
| sphere collision | **0.0 / +19.5** | **2.6 / +15.2** | **0.0 / +19.5** |
| smooth_center+collisionS | 1.0 / +18.7 | 2.7 / +15.2 | 0.8 / +18.7 |
| capsule collision | **59.6** / +19.5 | **76.2** / +19.5 | **62.9** / +19.5 |

The user's feared "hands jerking away from each other" is real and
quantified — for capsules (59–76 mm on every test). The sphere model's
deviation is 0.0 on clear tests and 2.6–2.7 mm on the boundary-riding test,
which is precisely the amount needed to hold +15 mm model clearance.

## Phase 10 — where the compute goes (quiet-CPU microbench)

| piece (ms/call, jitted) | spheres 5,884 | spheres 14,491 | capsules 421 |
|---|---|---|---|
| residual eval | 0.048 | 0.039 | 0.076 |
| Jacobian (jacfwd, what LM needs) | 0.569 | 0.649 | **1.437** |
| aggregated scalar grad | 0.360 | 0.568 | 0.544 |

- The residual is ~3× an FK call; **the Jacobian is ~92% of the collision
  cost per iteration** (12× the residual — XLA fuses below the naive 23×).
- The capsule Jacobian costs 2.5× the sphere Jacobian *despite 14× fewer
  pairs* — the segment-segment closest-point routine is expensive to
  differentiate (and is also the ~18-minute XLA compile bomb; the sphere
  model compiles in 2.5 s).
- Pair pruning buys principled signal, not much runtime (vectorized gather:
  5,884 vs 14,491 differ ~14% in Jacobian time).
- **Aggregation** (scalar r = w·√(Σh²+ε), identical LS objective): ~1.8×
  faster in normal regimes (2.3–2.5 vs 4.1–4.4 ms median) with behaviour
  identical to 0.1 mm; under adversarial many-active-pair conflict its
  rank-1 Gauss-Newton block costs iterations (70 vs 39). Documented option
  for latency-critical deployment; per-pair (pyroki-native) stays default.

## Final recommendation (evidence-based, per the study brief)

> We evaluated three collision representations against 70 mesh-ground-truth
> cases. The 180-sphere model reduced mean clearance error from 98 mm
> (pyroki capsules) / 74 mm (corrected capsules) to **11 mm**, eliminated
> all 15–16 false-positive collision classifications, and removed the
> pathological repulsion capsules produce (59–76 mm of bimanual hand-jerk →
> ≤ 2.7 mm, incurred only to hold a real boundary). With **180 spheres,
> 5,884 corpus-pruned pairs, a 20 mm margin and soft hinge weight 100**, it
> detects every mesh-true contact inside the activation band (optimistic
> bias ≤ 18.3 mm < margin), holds the designed arm-crossing at +3 mm model
> clearance, permits the self-fold rest posture, and adds **4–8 ms to the
> median solve** (worst-trajectory p95 ~65 ms, transient, during
> adversarial segments only). Capsule collision at identical settings
> degrades clean tracking 2.8×, blocks the rest pose, jerks the hands
> 59–76 mm, and runs 5× slower with 33% non-convergence. **Therefore: the
> sphere model, as configured above, integrated as
> `smooth_center_collisionS` (smoothing 0.05 + centering 0.5 + sphere
> collision 20 mm/100) — measured on the full frozen suite at 0.60 mm /
> 0.24° clean-trajectory p95 with every safety boundary held.**

Configuration answers to the brief's twelve questions:
1. **Geometry**: multi-sphere (180; capsules structurally cannot pass the
   acceptance criteria — not even optimally fitted ones).
2. **Pairs**: 5,884 sphere pairs (corpus rule, `sphere_pair_pruning.npz`).
3. **Pruned**: 32 functional (grasping), 12 permanent (signal-free),
   8,563 never-active — each class by stated rule, none hand-curated.
4. **Margin**: 20 mm (must exceed the 18.3 mm optimistic bias; 25 mm adds
   repulsion on the grasp path at high weight).
5. **Formulation**: soft margin-hinge residual (pyroki-native); the AL
   constraint is empirically rejected (fights boundary-riding, diverges on
   infeasible commands).
6. **Weight**: 100 (the measured threshold that holds the boundary; 300 =
   harder-guarantee option at ~55 ms during conflicts).
7. **Overhead**: +4–8 ms median (0.56 → 5.2 ms in the deployment stack);
   2.5 s compile.
8. **Tracking cost**: zero on clean trajectories (measured, full suite).
9. **Safety gained**: worst-case clearance −157 → +3 mm model (≈ bounded
   ≤ 1–2 cm true penetration under deliberately adversarial commands;
   contact-free in all realistic cases).
10. **Undesirable behaviour**: reach_limit recovery 6.8 → 32 mm p95 (the one
    regression); adversarial-command solves transiently hit ~65 ms.
11. **Complexity justified?** Yes — measured, and the added terms *reduce*
    solver iterations in combination.
12. All numbers reproducible: `run_study.py --variant <name>
    --clearance-model sphere` against suite hash `3815490046334af4`.
