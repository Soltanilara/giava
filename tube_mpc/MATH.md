# The math behind `tube_mpc`

Everything the code does, derived. Notation follows the course modules:
constrained QPs (M2/M4), LQR (M3), invariant sets (M5), constrained
optimal control and MPC (M6/M7).

## 1. Plant model

State $x_k = (q_k, \dot q_k) \in \mathbb R^{2n}$, input $u_k = \ddot q_k \in \mathbb R^n$, period $\Delta t$:

$$
x_{k+1} = A x_k + B u_k + w_k,\qquad
A=\begin{bmatrix} I & \Delta t I\\ 0 & I\end{bmatrix},\quad
B=\begin{bmatrix} \tfrac{\Delta t^2}{2} I\\ \Delta t I\end{bmatrix}.
$$

This is **exactly LTI** — no linearization. The robot's own servo loops
track the commanded joint trajectory; everything they get wrong
(tracking error, latency jitter, discretization) is lumped into the
additive disturbance $w_k$, assumed to lie in a symmetric box

$$
\mathcal W = \{ w : |w_j| \le \bar w_j \}, \qquad \bar w \in \mathbb R^{2n}_{>0},
$$

**calibrated from data** (§7), never guessed. Constraints are boxes too:

$$
\mathcal X = \{x: q_{\min} \le q \le q_{\max},\ |\dot q| \le \dot q_{\max}\},\qquad
\mathcal U = \{u: |u| \le \ddot q_{\max}\}.
$$

Position and velocity limits come from the URDF; $\ddot q_{\max}$ must be
identified (URDFs don't carry it).

Why kinematic and not full dynamics: robust MPC machinery is exact for
linear systems and polytopic sets. Full manipulator dynamics would force
nonlinear robust MPC, which in the one published arm-scale demonstration
(Nubert et al., RA-L 2020) could not run in real time and had to be
approximated by a neural network. At the kinematic level, the entire
course toolbox applies *exactly* and the QP solves in ~1 ms.

## 2. Ancillary feedback and error dynamics (Module 3)

Choose the discrete LQR gain $K$ for $(A, B)$ with weights
$Q_{\mathrm{lqr}} = q_{\mathrm{lqr}} I$, $R_{\mathrm{lqr}} = r_{\mathrm{lqr}} I$
(solved via the DARE), so $A_K = A + BK$ is Schur stable. If the actual
trajectory deviates from a nominal one, $e = x - z$, and the input is
$u = v + Ke$, the error obeys

$$
e_{k+1} = A_K e_k + w_k .
$$

$K$ is the *tube controller*: it determines how fast execution error is
squeezed back toward the plan, and therefore how large the constraint
margins must be.

## 3. Support functions and tightening margins (Modules 4–5)

For the box $\mathcal W$, the support function in direction $a$ is
$h_{\mathcal W}(a) = \sum_j |a_j| \bar w_j$. Starting from $e_0 = 0$
(we re-measure the state every tick), the error after $i$ steps lies in

$$
\mathcal E_i = \bigoplus_{j=0}^{i-1} A_K^j \mathcal W
\quad\text{(Minkowski sum)} .
$$

A state-constraint row $c^\top x \le d$ must therefore be *tightened* at
prediction step $i$ by

$$
m_i(c) = \sum_{j=0}^{i-1} h_{\mathcal W}\!\big((A_K^j)^\top c\big),
$$

and an input row $g^\top u \le h$ by the same sum with $c$ replaced by
$K^\top g$ (because $u_i = v_i + K e_i$). Since all our constraints are
coordinate boxes, `sets.py` computes all rows at once as
$\sum_j |A_K^j|^\top \bar w$ — no polytope libraries needed. This is the
**Chisci et al. (2001) constraint-restriction** form of tube MPC: pin
the nominal to the measured state ($z_0 = x_k$), let margins grow along
the horizon, and get feedback by replanning every tick. (The Mayne 2005
variant instead optimizes $z_0$ inside a fixed tube; Chisci's is exact
with boxes and simpler, which is why it's implemented first.)

The margins converge as $i \to \infty$ to the support of the **minimal
robust positively invariant (mRPI) set**. We bound the limit with the
Raković–Kerrigan–Kouramas–Mayne (2005) construction: find $(s, \alpha)$
with $A_K^s \mathcal W \subseteq \alpha \mathcal W$ (for a box this is
the one-line check $\alpha(s) = \max_j ( |A_K^s|\bar w )_j / \bar w_j$),
then

$$
m_\infty(c) \le \frac{1}{1-\alpha}\, m_s(c).
$$

`sets.py::mrpi_alpha_s` finds the smallest $s$ with $\alpha \le 0.05$.
Construction fails loudly if the tightened sets are empty — that means
$\mathcal W$ is too large for the limits (recalibrate, or speed up $K$).

Two small honesty terms are added on top of the margins:
`q_backoff`/`v_backoff` (~2 mrad / 5 mrad/s) budget for the QP solver
returning an *approximately* feasible plan (OSQP at $\varepsilon = 10^{-4}$),
which the tube — built only for the physical disturbance — doesn't cover.

## 4. The receding-horizon QP (Modules 6–7)

Decision variables: nominal states $z_{0:N}$ and inputs $v_{0:N-1}$.

$$
\begin{aligned}
\min\ & \sum_{i=1}^{N} \Big( \|q^z_i - r_i\|_{Q}^2 + \|\dot q^z_i\|_{Q_v}^2 \Big)
      + \sum_{i=0}^{N-1} \|v_i\|_R^2
      + \|v_0 - u^{\mathrm h}\|^2_{R_h} \\
\text{s.t.}\ & z_0 = x_k \\
& z_{i+1} = A z_i + B v_i \\
& z_i \in \mathcal X \ominus \mathcal E_i \quad (i = 1..N-1)
  \qquad\text{(box minus margin } m_i\text{)}\\
& v_i \in \mathcal U \ominus K\mathcal E_i \\
& z_N \in \mathcal X_f := \{(q, 0): q_{\min} + m_\infty \le q \le q_{\max} - m_\infty\}.
\end{aligned}
$$

- $r_i$ is the extrapolated reference (§6); it enters **only the cost**,
  so nothing the operator does can affect feasibility.
- $\|v_0 - u^{\mathrm h}\|^2_{R_h}$ is the optional minimal-intervention
  term: crank $R_h$ up and this is a Wabersich–Zeilinger-style safety
  filter; crank the tracking term up and it's robust tracking MPC. One
  formulation, one knob.
- $\mathcal X_f$ is the **safe-stop set**: zero velocity, strictly inside
  the mRPI-tightened position box. Honest teleoperation semantics: the
  plan always ends in a state from which the robot can simply stay put.

The applied command is $u_k = v_0^\star$ (clamped to its known hard box
so solver tolerance can never leak past an actuator limit).

**Proposition 1 (recursive feasibility & constraint satisfaction).**
*If the QP is feasible at time $k$, then for every $w_k \in \mathcal W$
it is feasible at time $k+1$, and the closed-loop trajectory satisfies
$x \in \mathcal X$, $u \in \mathcal U$ for all time — for arbitrary
reference signals.*

*Proof sketch (the course argument, written for this exact QP).* Let
$(z^\star, v^\star)$ solve the QP at $k$. After applying $v_0^\star$,
the state is $x_{k+1} = z_1^\star + w_k$. Candidate at $k+1$: the
shifted plan with the ancillary correction,
$\tilde v_i = v_{i+1}^\star + K A_K^i w_k$ for $i = 0..N-2$, and
$\tilde v_{N-1} = K A_K^{N-1} w_k$ appended (hold still at the safe-stop
state plus tube feedback). The induced nominal states are
$\tilde z_i = z_{i+1}^\star + A_K^i w_k$. Because
$z_{i+1}^\star$ satisfied the step-$(i+1)$ margins and
$A_K^i w_k \in A_K^i \mathcal W$, each $\tilde z_i$ satisfies the
step-$i$ margins (the margin sequence is built from exactly these
sums); same for inputs via $K A_K^i \mathcal W$; and
$\tilde z_{N}$ stays in $\mathcal X_f$ because $\mathcal X_f$ is tightened
by the *limit* margin $m_\infty \ge m_i$ for all $i$ and the appended
velocity remains zero-mean within the tube. Feasibility of the candidate
gives feasibility of the optimum; constraint satisfaction of the true
state follows because the step-$i$ margins dominate the accumulated
error set $\mathcal E_i$. $\square$

Writing this proof out cleanly, with the exact index bookkeeping, is a
core deliverable of the course project (it is Module 7's terminal-set
argument instantiated on a real system).

**Fallback (implemented, matters in practice).** If a solve is late or
hits its time limit, the controller applies the *tube control law along
the stored plan*: $u = u^{\mathrm{plan}}_i + K(x - x^{\mathrm{plan}}_i)$
— exactly the candidate from the proof, so a missed deadline degrades
gracefully instead of violating constraints. An exhausted plan brakes to
zero velocity. (Empirically necessary: applying the stored plan
*open-loop* let disturbances accumulate and produced violations.)

## 5. What the naive pipeline lacks (why this beats IK + clamps)

The baseline in `demo_single_arm.py` is "IK target → clamp position →
clamp velocity → clamp acceleration", i.e. what most teleop stacks do.
Clamping satisfies each limit *instantaneously* but has no braking-
distance reasoning: approaching $q_{\max}$ at $\dot q_{\max}$ is
"legal" right up until no admissible deceleration can avoid the limit
(Del Prete, RA-L 2018: per-step control reaches states from which every
future command violates a bound). The demo shows precisely this: 115
violation steps for the clamp baseline, zero for the MPC, at equal
tracking error and smoothness.

## 6. Reference extrapolation

The MPC needs $r_{1:N}$ but the operator only reveals $r_k$. Since the
reference enters only the cost, the predictor affects *transparency*,
never safety. Implemented: zero-order hold, and decayed-velocity
extrapolation $r_{k+i} = r_k + \hat v \Delta t \sum_{j=1}^{i} \gamma^j$
with $\gamma \approx 0.85$ — a crude stand-in for the bell-shaped speed
profiles of human reaching (Flash & Hogan 1985). Upgrade path: fit a
quintic minimum-jerk segment online (Smith & Christensen, IROS 2009)
and use $\mathcal W$ to bound only the prediction *residual*.

## 7. Calibrating $\mathcal W$ (`calibrate_w.py`)

From a log of commanded and measured joint positions: reconstruct the
commanded acceleration, form the one-step model residual on the measured
stream,

$$
w_k = x^{\mathrm{meas}}_{k+1} - \big(A x^{\mathrm{meas}}_k + B u^{\mathrm{cmd}}_k\big),
$$

and bound it per joint by the 99.9th percentile × 1.25. Two rules:

1. **Recalibrate per machine and per command mode.** A velocity-commanded
   servo has different residuals than a position-commanded one.
2. **A huge $\mathcal W$ is a modeling bug, not a noisy robot.** If the
   residual swallows systematic servo lag or a wrong $\Delta t$, the
   margins will (correctly) eat the whole workspace and construction
   will refuse. Fix the model input, don't inflate the limits.

## 8. Roadmap

- **Phase 2 — collision constraints.** Add per-step half-space
  constraints $d(\bar q_i) + \nabla d(\bar q_i)^\top (q_i - \bar q_i)
  \ge d_{\mathrm{safe}} + m_i$, with $d$ from the GJK/sphere pipeline in
  `pyroki_ik_script.py`, linearized about the previous plan. Local, not
  global — state this honestly and measure it.
- **Phase 3 — three arms in one QP** (both manipulators + camera arm)
  with inter-arm distance constraints; optionally a cooperative
  relative-pose constraint for bimanual phases.
- **Task-space cost.** Replace the joint-space reference with the
  Jacobian-linearized pose residual (one Gauss–Newton step per tick —
  the real-time-iteration idea on a linear plant), so the filter tracks
  the *pose* target and the IK bias disappears.
- **Elastic tubes.** Scale $\bar w$ online from an operator-activity
  estimate; margins shrink when the operator is calm (Raković's
  homothetic-tube idea, box-simplified).

## 9. Connections to Tedrake's material (things to learn/implement)

- *Manipulation*, "Differential IK as a QP": the clamp-free version of
  the baseline, with the same one-step blind spot — implementing it as a
  third baseline is ~30 lines and makes the comparison sharper. Drake's
  `DifferentialInverseKinematicsIntegrator` even ships a `kStuck` status
  for exactly the failure Prop. 1 excludes.
- *Underactuated*, trajectory optimization: our QP **is** direct
  transcription (decision variables = states + inputs, dynamics as
  equality constraints), specialized to LTI.
- *Underactuated*, LQR: the ancillary gain $K$ and the DARE.
- Funnels / regions of attraction ↔ tubes: same geometric idea —
  a set-valued guarantee wrapped around a nominal motion.
- IRIS (Deits & Tedrake 2015) + GCS (Marcucci et al., Science Robotics
  2023): convex collision-free regions in configuration space. The
  principled upgrade for Phase 2: constrain $q_i$ to an IRIS polytope
  (tightened by the tube) instead of a one-step distance linearization,
  and let a region-sequence layer handle the global routing.
