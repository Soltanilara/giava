# Tube MPC: what it is, why it is here, how it is wired

`tube_mpc/` is a robust-MPC **reference filter** that sits between your IK
targets and the joint commands you send.  It tracks the operator when their
command is feasible and returns the closest feasible command when it is not —
with a guarantee that position, velocity and acceleration limits are *never*
violated, no matter what the operator does next.

The derivation lives in [`tube_mpc/MATH.md`](../tube_mpc/MATH.md) and the
package usage in [`tube_mpc/README.md`](../tube_mpc/README.md).  This document
is the connecting tissue: what the thing actually does, what decisions the
GIAVA integration made, and what happened when they were made wrong.

---

## 1. The problem it solves

The ordinary teleop pipeline is:

```
IK target  →  clamp position  →  clamp velocity  →  clamp acceleration  →  send
```

Clamping satisfies each limit **instantaneously** and has no braking-distance
reasoning.  Approaching `q_max` at `q̇_max` is "legal" right up to the moment
when no admissible deceleration can avoid the limit.  Del Prete (RA-L 2018)
states it formally: per-step control reaches states from which *every* future
command violates a bound.

Measured on the right arm from `giava.urdf`, against an operator "lunge"
0.4 rad past the joint limits:

| | position violations | worst overshoot | solve time |
|---|---|---|---|
| naive clamping | **115 steps** | 0.41 rad past the limit | — |
| tube MPC | **0** | — | ~1.3 ms mean (20 ms budget) |

Equal tracking error, equal smoothness.  The difference is entirely that one
of them plans ahead and the other does not.

---

## 2. How it works, in five steps

### Step 1 — model the joints as a double integrator

State `x = (q, q̇)`, input `u = q̈`, period Δt:

```
x[k+1] = A x[k] + B u[k] + w[k]

A = [ I   Δt·I ]      B = [ Δt²/2 · I ]
    [ 0     I  ]          [  Δt · I   ]
```

This is **exactly LTI** — no linearisation, no approximation.  That is the
central design choice and it is what makes everything else exact.

Why kinematic and not full dynamics: robust-MPC machinery is exact for linear
systems and polytopic sets.  Full manipulator dynamics forces *nonlinear*
robust MPC, which in the one published arm-scale demonstration (Nubert et al.,
RA-L 2020) could not run in real time and had to be approximated by a neural
network.  At the kinematic level the whole toolbox applies exactly and the QP
solves in ~1 ms.

The servos' own loops track the commanded trajectory; everything they get
wrong — tracking error, latency jitter, discretisation — is lumped into an
additive disturbance `w[k]` confined to a box `W`, **calibrated from data,
never guessed**.

### Step 2 — pick an ancillary feedback gain

Discrete LQR for `(A, B)` gives `K` with `A_K = A + BK` Schur stable.  If the
real trajectory deviates from a nominal plan, `e = x − z`, and the input is
`u = v + Ke`, the error obeys

```
e[k+1] = A_K e[k] + w[k]
```

`K` is the *tube controller*: it sets how fast execution error is squeezed
back toward the plan, and therefore how large the margins must be.

### Step 3 — compute constraint-tightening margins

Starting from `e₀ = 0` (the state is re-measured every tick), the error after
`i` steps lies in `E_i = ⊕ⱼ A_K^j W`.  A constraint row `cᵀx ≤ d` must be
tightened at prediction step `i` by the support function

```
m_i(c) = Σⱼ h_W((A_K^j)ᵀ c)      where  h_W(a) = Σⱼ |aⱼ| w̄ⱼ
```

Because every constraint here is a coordinate box, `sets.py` computes all rows
at once as `Σⱼ |A_K^j|ᵀ w̄` — no polytope library needed.

This is the **Chisci et al. (2001) constraint-restriction** form: pin the
nominal to the measured state (`z₀ = x[k]`), let margins grow along the
horizon, get feedback by replanning every tick.  (Mayne 2005 instead optimises
`z₀` inside a fixed tube; Chisci's is exact with boxes and simpler.)

The margins converge to the support of the **minimal robust positively
invariant** set, bounded via Raković–Kerrigan–Kouramas–Mayne (2005): find
`(s, α)` with `A_K^s W ⊆ α W`, then `m_∞ ≤ m_s / (1−α)`.
`sets.py::mrpi_alpha_s` finds the smallest `s` with α ≤ 0.05.

Construction **fails loudly** if the tightened sets are empty — that means `W`
is too large for the limits.  Recalibrate, or speed up `K`.  Do not force it.

### Step 4 — solve the QP every tick

```
min   Σᵢ ( ‖qᶻᵢ − rᵢ‖²_Q + ‖q̇ᶻᵢ‖²_Qv ) + Σᵢ ‖vᵢ‖²_R + ‖v₀ − uʰ‖²_Rh

s.t.  z₀ = x[k]
      z[i+1] = A zᵢ + B vᵢ
      zᵢ ∈ X ⊖ E_i          (box minus margin)
      vᵢ ∈ U ⊖ K E_i
      z_N ∈ X_f = {(q, 0) : q_min + m_∞ ≤ q ≤ q_max − m_∞}
```

Three things worth noticing:

- **`rᵢ` enters only the cost.**  Nothing the operator does can affect
  feasibility.  That is the property that makes this safe for teleoperation.
- **`X_f` is a safe-stop set**: zero velocity, strictly inside the tightened
  position box.  Honest teleop semantics — the plan always ends somewhere the
  robot can simply stay put.
- **`‖v₀ − uʰ‖²_Rh` is one knob spanning two philosophies.**  Crank `R_h` up
  and this is a Wabersich–Zeilinger safety filter (minimal intervention);
  crank the tracking term up and it is robust tracking MPC.  One formulation.

The applied command is `u = v₀*`, clamped to its known hard box so solver
tolerance can never leak past an actuator limit.

**Proposition (recursive feasibility).**  If the QP is feasible at `k`, then
for every `w ∈ W` it is feasible at `k+1`, and the closed loop satisfies
`x ∈ X`, `u ∈ U` for all time — *for arbitrary reference signals*.  Proof
sketch in `MATH.md` §4: shift the plan, append `K A_K^i w` corrections, show
the candidate satisfies the step-`i` margins by construction.

### Step 5 — degrade gracefully when the solve is late

If a solve misses its deadline, apply the **tube control law along the stored
plan**: `u = u_plan[i] + K(x − x_plan[i])` — exactly the candidate from the
proof, so a missed deadline degrades instead of violating.  An exhausted plan
brakes to zero velocity.

This is empirically necessary, not theoretical: applying the stored plan
*open-loop* let disturbances accumulate and produced violations.

---

## 3. Reference extrapolation

The MPC needs `r[1..N]` but the operator only reveals `r[k]`.  Since the
reference enters only the cost, the predictor affects **transparency, never
safety**.

Implemented: zero-order hold, and decayed-velocity extrapolation
`r[k+i] = r[k] + v̂ Δt Σⱼ γʲ` with γ ≈ 0.85–0.95 — a crude stand-in for the
bell-shaped speed profiles of human reaching (Flash & Hogan 1985).

Upgrade path: fit a quintic minimum-jerk segment online (Smith & Christensen,
IROS 2009) and use `W` to bound only the prediction *residual*.

`ref_jump_rad: 0.1` refuses to extrapolate across a target discontinuity —
counted as `ref_jumps`.

---

## 4. Calibrating W

From a log of commanded and measured joint positions: reconstruct commanded
acceleration, form the one-step residual on the measured stream,

```
w[k] = x_meas[k+1] − ( A x_meas[k] + B u_cmd[k] )
```

and bound it per joint by the 99.9th percentile × 1.25.

Two rules, both learned the hard way:

1. **Recalibrate per machine and per command mode.**  A velocity-commanded
   servo has different residuals than a position-commanded one.
2. **A huge W is a modelling bug, not a noisy robot.**  If the residual
   swallows systematic servo lag or a wrong Δt, the margins will *correctly*
   eat the whole workspace and construction will refuse.  Fix the model input.

---

## 5. The GIAVA integration

`data_collection_scripts/tube_mpc_hook.py` is the adapter seam.  It is shared
by `debug/tube_mpc_bench.py` (quantitative, one arm) and, under
`GIAVA_TUBE_MPC=1`, by `data_collection.py` — so both run the same filter and
the numbers transfer.

Three decisions the package deliberately leaves to the adapter:

### Position limits = URDF ∩ driver

It is the **driver** that refuses commands — and it refuses the *whole group
command* on any single joint.  Its limit arrays disagree with the URDF on some
joints: right `wrist_angle` is URDF +2.234 but the driver rejected +2.243.  A
filter certifying against the wrong array produces commands the driver
silently drops.  `TeleopConfig.driver_limit_margin` pulls in the intersection.

### State source: `command` (default), not `measured`

The arms are **position-servoed**: you publish a goal and the servo chases it
with its own PID.  Two ways to close the filter's loop:

| | `x[k]` is | consequence |
|---|---|---|
| **`command`** (default) | the filter's own last *planned* state | a virtual double integrator driven by the command stream. Exactly LTI with `w = 0`, so the guarantee applies to the **command trajectory**: nothing infeasible ever reaches the driver. Servo lag is outside the loop — measured separately, not fought. |
| `measured` | `(q, q̇)` off the encoders | the textbook tube. But on a position servo the goal is then only ever one plan step ahead of where the arm *is*, so the servo's P-loop sees a tiny error and crawls. `lookahead` (send `x_pred[lookahead]`) is the standard fix, and `W` must then cover servo lag. |

Start with `command`.  Switch only once the bench shows servo lag is small
enough that `calibrate_w` accepts `W`.

### Re-anchoring

The virtual state must be re-synced to the encoders whenever the command
stream and the arm can have diverged: teleop enable, a stall hold, a
driver-rejected command.  `reset(q_meas)` does that and drops the stored plan.
`step()` also re-anchors itself if the command has run more than
`reanchor_rad` (0.35) from the measured position — a jammed arm with the
command marching on is exactly the failure `servo_health.StallGate` exists
for, and the filter must not add to it.

---

## 6. The two config files, and the bug that produced them

`config.giava.yaml` and `config.giava.command.yaml` differ **only** in `w_q` /
`w_v`, and they differ because they describe two different loops.  Worth
reading as a case study in what "a huge W is a modelling bug" means in
practice.

On 2026-09-09, `calibrate_w` on the real right arm gave `w_v = 0.038–0.081`
rad/s.  That number was inflated by **280 ms of command→encoder lag** on
`wrist_angle` (14 ticks, measured by `tube_mpc_bench`).  Installed in a
*command-mode* loop it produced:

```
readiness check:  READY
worst terminal margin:  1.6384 rad  =  94.3% of the narrowest joint's range
```

The filter was reserving nearly the whole workspace against a disturbance
that, in that loop, **does not occur** — the servo's tracking error is outside
the command-mode plant by construction.  The same check with the command-mode
`W` reports 0.2969 rad, 18.5%.

So:

- `config.giava.yaml` — `W` from the *measured* stream. The servo's residual.
  The right `W` for `state_source=measured`.  **Keep it; do not delete it.**
- `config.giava.command.yaml` — `w_q = 0.002`, `w_v = 0.01`. Not zero, because
  the virtual plant is not perfectly followed either: the planned command is
  edited after the filter by the driver clamp, the stall hold and the waist
  travel limit, and the plan is re-anchored on enable and after a rejection.
  These are shipped defaults at solver-tolerance scale (compare `q_backoff`
  ≈ 2 mrad) — **a floor, not a measurement.**  Calibrating them honestly would
  mean logging the filter's *planned* command against what `data_collection`
  actually sent, which `calibrate_w` does not currently do.

That last paragraph is the most useful thing in the package.  It says exactly
what is not yet known and what it would take to know it.

---

## 7. Running it

```bash
python -m pytest tube_mpc/tests -q     # 11 tests, ~4 s
python -m tube_mpc.check               # 30-second readiness check — must print READY
python -m tube_mpc.demo_single_arm     # naive clamping vs tube MPC, with plot
python debug/tube_mpc_bench.py         # on-rig bench → debug/bench_logs/
GIAVA_TUBE_MPC=1 python data_collection.py    # in the live loop
```

Per-arm counters fold into `robustness.jsonl`:

```
ticks   fallbacks   reanchors   ref_jumps   solve_ms
intervention_rad     |q_cmd − q_target| per tick:
                     zero = transparent, large = it was braking for a limit
```

`tube_fallback_ticks` is the one to watch.  A rising fallback count means
solves are missing the deadline, and the filter is running on the stored plan.

### Files

| file | what |
|---|---|
| `model.py` | LTI double-integrator + `JointLimits` |
| `sets.py` | LQR gain, mRPI (s, α), tightening margins |
| `controller.py` | the QP (OSQP), safe-stop terminal set, tube fallback |
| `reference.py` | hold / decayed-velocity extrapolators |
| `config.py` | `FilterConfig`, YAML-backed |
| `adapter.py` | `TeleopAdapter` protocol, `FilterRunner`, `SimAdapter` |
| `urdf_limits.py` | parse position/velocity limits out of a URDF |
| `calibrate_w.py` | fit `W` from logged data |
| `check.py` | per-machine readiness check |
| `demo_single_arm.py` | the clamping-vs-MPC comparison |

The core depends only on `numpy`, `scipy`, `osqp`, `pyyaml`.  It never imports
lerobot, pyroki, or anything robot-specific — different IK weights, different
lerobot versions and velocity- vs position-commanded arms are all absorbed by
the config file and a ~20-line adapter.  That portability is why the bench and
the live loop provably run the same filter.

---

## 8. Where it goes next

From `MATH.md` §8, in the order that makes sense here:

1. **Collision constraints (Phase 2).**  Per-step half-space constraints
   `d(q̄ᵢ) + ∇d(q̄ᵢ)ᵀ(qᵢ − q̄ᵢ) ≥ d_safe + mᵢ`, with `d` from the sphere/GJK
   pipeline, linearised about the previous plan.  **Local, not global** — say
   so and measure it.  This is the interesting one: it would give the collision
   system a *predictive* layer, where today the IK cost is reactive and the
   gate is instantaneous.
2. **Task-space cost.**  Replace the joint-space reference with the
   Jacobian-linearised pose residual (one Gauss–Newton step per tick), so the
   filter tracks the *pose* target and the IK bias disappears.
3. **Three arms in one QP** with inter-arm distance constraints.
4. **Elastic tubes.**  Scale `w̄` online from an operator-activity estimate —
   margins shrink when the operator is calm (Raković's homothetic tube,
   box-simplified).

And the honest framing of where it sits relative to the rest of the stack:

```
IK collision cost   reactive, soft, differentiable    "resist and bound"
tube MPC            predictive, over a horizon        "never become infeasible"
capsule/GJK gate    instantaneous, hard               "stop here"
```

Three layers, three time horizons, three failure semantics.  Tube MPC is the
only one that reasons about the *future*, which is exactly the blind spot
clamping has.
