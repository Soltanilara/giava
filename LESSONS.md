# Lessons — April 2026 to September 2026

Written 2026-09-12, from measurements in this repo rather than memory. Every
claim below names the file it came from so it can be re-checked.

This is not a postmortem of a failure. It is a record of what an eight-month
build actually taught, including the parts that were expensive.

---

## 1. The timebase bug, end to end

The single most consequential defect of the period, and worth understanding in
full because every part of it was individually reasonable.

**What happened.** The recorder wrote `fps: 30` into dataset metadata while the
teleop loop was actually capturing at roughly 0.7 Hz. The label was a constant,
not a measurement.

**How it propagated:**

1. `block_square/20260524_224911` records `fps: 30`, 21 episodes, 1614 frames.
2. 1614 / 21 = 77 frames per episode; 77 / 30 = **2.6 s**, so the dataset
   describes itself as holding ~2.6-second demonstrations.
3. The 271 report repeats that: *"Each demonstration trajectory lasts between
   1.5 and 3 seconds."*
4. Rollouts were configured to match — `rollout_seconds=3.0, control_hz=50.0`
   in every row of `act_code/6-2-rollout_notes.csv`.
5. The policy was therefore asked to execute, in 3 seconds, a motion that had
   been demonstrated over about two minutes.

**What it actually was.** `timing_logs/block_square/` (same 21 episodes) records
per-tick durations. They sum to **71–139 s per episode**, and the wall-clock
gaps between episode files are 157–433 s — the two agree. Median tick 579.78 ms,
i.e. **~1.7 Hz control, ~0.7 Hz capture.** The fps label was high by ~40x.

**Why nothing caught it.** The pipeline was self-consistent everywhere the same
code ran. Collection and replay both went through the slow teleop loop, so a
replay took the same two minutes and looked exactly like the demonstration —
which is the strongest possible false reassurance. It broke at the one boundary
where a different loop ran with a hardcoded rate: rollout, which had no IK in it
and was free to run as fast as it was told.

**The visible symptom, misread.** Episode videos played back in 1–3 seconds.
That looked like compression. It was 77 real frames in a container claiming
30 fps.

**What it explains.** The report's high-start/low-start asymmetry. ACT emits
absolute joint positions; commanded 40x too fast, the servo cannot track and
behaves as a low-pass filter, so the arm follows a smoothed, lagged version of
the correct trajectory. From the low start the remaining motion is short and the
approximation still grasps — reported as the strongest results. From the high
start the same lag leaves the gripper *"~7 cm short of the block."* One bug, two
symptoms.

### Carry forward

- **A rate written into metadata is a claim, not a measurement.** Measure the
  achieved rate and store *that*. `measure_latency.py` and the per-episode
  `robustness.jsonl` stats exist because of this.
- **Self-consistency is not validation.** It held across collect and replay and
  still hid a 40x error. The test that would have caught it is a cross-check
  against an independent clock — wall time, file mtimes, anything outside the
  loop's own bookkeeping.
- **Suspicious observations deserve five minutes.** The 1–3 second videos were
  the bug, surfacing early and plainly, explained away with a plausible story.
- **The results were not noise.** All six representation variants shared the
  bug, so the *relative* comparison survives. What does not survive is the
  absolute difficulty conclusion — the task was harder than it should have been
  for reasons unrelated to representation.

---

## 2. Control: three stages, and neither axis alone was the goal

Recovered from `ros_ik_trajectory_logs/` (git commit `1ad3c33`, restored
2026-09-12), `timing_logs/block_square/`, and current `meta/robustness.jsonl`.

| | May 19 — ROS IK | May 24 — pyroki per-arm | Sep — pyroki coupled |
|---|---|---|---|
| solve time | **5.1 ms** | 167.6 ms | ~14 ms |
| position error | 32 mm median, 470 mm max | — | **0.25 mm max** |
| orientation error | 9.2° median, 126° p95 | — | **0.02° max** |
| loop rate | — | 1.7 Hz | ~50 Hz |

Stage one was fast and wrong. Stage two was correct and unusable. Only stage
three is both. A speed number alone would have made stage one look like the
best of the three.

**Where the 580 ms tick went** — almost exactly equal thirds:

| | median | share |
|---|---|---|
| `ik_solve` | 167.6 ms | 29% |
| `ik_section` minus the solve | 242 ms | 42% |
| `log` | 168.3 ms | 29% |

The solve was *uniformly* slow (p25 = 126 ms, 90.7% over 100 ms, only 0.2%
under 20 ms) while its own minimum was 7.3 ms. That shape matters: a cold-JIT
problem looks the opposite — low median, fat tail. Uniform slowness with a fast
floor means the solve was doing too much work on every call, not paying a
one-time cost.

### Carry forward

- **Report accuracy and latency together, always.** Either alone ranks the
  wrong solver.
- **Profile the shape, not just the median.** Uniform-vs-spiky distinguishes
  "configured too expensive" from "paying a warm-up", and the fixes are
  unrelated.
- **Logging belongs out of the control loop.** 29% of every tick went to
  synchronous logging inside the loop.
- **Instrument in thirds before optimising.** Two of the three costs here were
  not the thing that got blamed.

---

## 3. Servo profile: velocity-based vs time-based

`Drive_Mode` bit 2 decides whether `Profile_Velocity` is a *duration* or a
*speed cap*. The interbotix layer writes `moving_time * 1000` into it either
way, so under a velocity profile `moving_time = 0.14` silently became a
3.36 rad/s cap instead of a 140 ms move — and the loop's step clamp, derived
from `moving_time`, was measuring something the servo was not doing.

The gap was 5.9x: the loop permitted 0.396 rad/tick against a servo that could
execute 0.067. Tracking error then grows without bound while the joint chases
its goal at stall current. Recorded on hardware as `middle_base` at 2095 mA
against a ~2300 mA overload latch, sitting at zero effort afterwards while the
command walked 48 degrees away — and the wedged bus stalled the 50 Hz loop for
up to 1.65 s, taking the arms, the headset and the camera stream down together.

### Carry forward

- **A register's units can depend on another register.** Read both, or the
  number is meaningless.
- **A clamp derived from a config value is only as true as the hardware's
  agreement with it.** Read the achieved capability off the device
  (`apply_profile_limits`) rather than trusting the config.
- **Mixed configurations within one group are a latent trap.** The middle arm
  ran five joints velocity-based and two time-based, and which one got reported
  depended on driver joint ordering.
- **A runaway is not a tuning preference.** The failure mode was not "jerky" —
  it was a latched overload and a downed bus.

---

## 4. Frames and single sources of truth

- **A correction that lives outside the model is invisible to everything that
  loads the model.** `middle_joint_offsets.json` was applied only by
  `study_ik.py` and `calibration/kinematics.py`, so RViz, the collision viewer
  and pyroki's collision model all drew a different robot than the one being
  commanded. Folding it into `giava.urdf` made there be one robot.
- **Do not trust a vendor description over the hardware.** Aligning the middle
  chain to `wx250s_7dof.urdf.xacro` made the geometry agree and sent the real
  arm to a completely wrong pose. The revert is recorded in
  `middle_joint_offsets.json`.
- **A joint parked on an encoder seam splits a dataset invisibly.** The camera
  waist sat 0.5° from the wrap, so boot readings landed on either branch;
  episodes 0–67 recorded near −3.03 rad and 68+ near +3.26 for the same physical
  pose. It surfaced as a normalisation problem — `observation.state[middle_base]`
  std 2.768 against 0.109 of real motion — and would have reached the network
  as "which half of the session was this" at 25x the amplitude of the signal.
- **Some fixes are mechanical.** `Homing_Offset` is inert in extended-position
  mode and capped at ±90° anyway, so moving the seam required re-clocking the
  motor by hand. The software knob existed and could not do the job.

---

## 5. Evaluation

- **Training loss did not predict rollout behaviour.** Six variants within
  0.006 of each other behaved very differently closed-loop — the report's own
  conclusion, and it held.
- **Rollout scores have day-scale variance large enough to invert a
  conclusion.** The same checkpoint scored 8/17 and 4/17 on identical scenes a
  day apart. Single rollouts per condition are coin flips; scene-paired blinded
  A/B is worth roughly triple the episode count.
- **Warm-up contaminates the first episode of every run.** Across 53 rollout
  runs the first episode reached a median 47.3 Hz against 49.1 Hz for later
  ones, and *every* rate-abort on record (11 of 11) was an episode 0.
- **Name artefacts by what they are.** Rollout outputs were keyed by timestamp
  alone, so finding a specific run meant opening images until one looked right.

---

## 6. What the period actually produced

Not a working policy. What it produced is the apparatus that makes the next
dataset trustworthy, and the failure modes were the price of it:

- a 36x faster control loop that is simultaneously ~128x more accurate in
  position and ~470x in orientation than where it started
- per-episode robustness telemetry, latency measurement, and config snapshots —
  all of which exist specifically because of section 1
- a calibrated geometry stack (`calibration/` → `reconstruction/`) with a
  33-check selftest against synthetic ground truth
- collision gating, tube-MPC reference filtering, and a frozen IK ablation study
- one robot description that sim and hardware agree on

That is a normal shape for infrastructure work. It does not look like progress
from the inside, and the record of what was tried — `training_runs_manifest.csv`,
128 runs — is worth more than the 248 GB of weights it describes.

---

## 7. The DAgger round, and the first result that worked (2026-09-12)

The first rollout in this repo's history where the policy did the task.

**The implementation.** DAgger-style correction lives in `data_collection.py`
(`_dagger_frames = {"policy": 0, "human": 0}`, `dagger_play`): the policy drives
the arm inside the normal recording loop, the operator takes over when it goes
wrong, and every frame is tagged with who was driving. Corrected actions are
recorded as demonstrations rather than discarded, so the failure states the
policy actually visits get labelled — which is the whole point of DAgger and
the thing a fixed demonstration set cannot give you.

**The checkpoint.** `transfer_flower_20260910_act_dagger_r1` — ACT, chunk 100,
three cameras (right_wrist, top_scene, low_scene), 7-dim action, 40k steps.
Trained on 107 episodes / 87,507 frames from `transfer_flower/20260903_214256`.

**The protocol.** 9 episodes, one per cell of a 3x3 placement grid, scored on
four stages (approach / grasp / transport / placement) with a free-text note.
Measured from the start snapshots, the nine placements form a clean grid:
x in {401, 418, 439} and y in {219, 236, 254}, about 19 x 17 px spacing.

### Result

| stage | |
|---|---|
| approach | **9/9** |
| grasp | **8/9** |
| transport | **8/9** |
| placement | **8/9** |

Loop health was clean throughout: 49.0 Hz mean against a 50 Hz target (min
47.9), 8,126 ticks, **zero camera stalls**, no driver limit clamps.

**The one failure is the one episode where the safety gates fired.** Episode 7,
cell (437, 220): 46 table-gate blocks, `min_alpha` 0.0 — the policy commanded
below the table plane, the gate refused, and the grasp never happened. Every
other episode fired zero gates. That is a usable online failure predictor: the
gate knows the policy has left the demonstrated envelope before the operator
can see it go wrong.

The tempting explanation was thin training coverage at that cell, and the data
does not support it. Counting training episodes whose start position lands
within 15 px of each rollout cell, the failed cell had 22 — and the *sparsest*
cell (12 episodes) succeeded. Coverage is not what separated them.

### The finding worth acting on

**Every one of the eight successful placements was rotated counter-clockwise.**
From the operator notes: "a bit counter clockwise", "a bit rotated", "maybe 45
degree counterclockwise", "slight counterclockwise", "~45 cc", "45 cc", "90 cc".
Eight for eight, ranging from slight to 90 degrees.

Translation error, by contrast, is small and *unbiased*: 0.5-1.0 cm, sometimes
left, sometimes high, sometimes both. The automatic measure agrees -- final
object-to-target distance 9-23 px, median about 12.

That asymmetry is the useful part. An unbiased 0.5 cm error is noise and needs
more or better data. A systematic one-directional rotation error across every
single success is a *correctable* defect, and it is exactly the shape of thing
a second DAgger round can fix: the policy has learned the task and has one
consistent bias in the final degree of freedom.

### Carry forward

- **Gate activations caught ONE failure mode, not failure in general.** The
  one DAgger episode that fired them did fail -- but the paired baseline run
  below produced SEVEN failures with zero gate activations. The gate detects
  "commanded below the table", which is one specific way to fail, not a
  general-purpose failure detector. Recorded here because the tidier claim was
  the first thing this data suggested, and it is wrong.
- **Separate biased error from unbiased error before collecting more data.**
  They have different fixes: bias is correctable by targeted demonstration,
  variance needs volume. Conflating them wastes a collection session.
- **The stage-wise score earns its keep.** "8/9" hides that approach was 9/9 --
  the policy always found the object, and failed only at contact. A single
  success/failure number would have lost the one thing that says what to fix.
- **This worked because the data was collected correctly.** Same task, same
  arm, same architecture as the April-June attempt. The differences were a
  measured 50 Hz instead of a claimed one, 107 episodes with real spatial
  coverage instead of 71 from a single spot, and corrections recorded where the
  policy actually fails. Section 1 is why the earlier attempt could not have
  worked; this is what it looks like when those are fixed.

---

## 8. The paired comparison: DAgger vs the 50-episode baseline (2026-09-12)

Run immediately after Section 7, on the **same nine placements**, using
`rollout_policy.py --replicate` to ghost each of the earlier run's start
scenes live until the operator matched it.

**The pairing held.** Measured from the snapshots, the nine placements were
reproduced to a **median of 1.0 px, max 1.7 px**, against a grid spacing of
19 x 17 px. That is a genuinely paired comparison, not two independent samples.

| | DAgger (94 eps, 40k steps) | base-50 (50 eps, 100k steps) |
|---|---|---|
| approach | 9/9 | 9/9 |
| grasp | **8/9** | **2/9** |
| transport | 8/9 | 2/9 |
| placement | 8/9 | 2/9 |

Per cell, with both policies on identical scenes:

```
DAgger wins 6    base wins 0    both 2    neither 1
exact one-sided binomial on the 6 discordant pairs:  p = 0.016
```

Every discordant cell went the same way. With nine episodes that is only
significant because the pairing is tight -- the same comparison run
unpaired would have been indistinguishable from the day-scale variance
recorded in section 5 (the same checkpoint scoring 8/17 and 4/17).

### What actually differed

**Approach was 9/9 for both.** Neither policy had trouble finding the object.
The entire difference is at contact, and the baseline's failure notes are one
defect stated seven ways: *"lowered on to the block's surface and nudged the
block away"*, *"approached a bit low"*, *"right finger contacted top right
corner"*, *"approached too far back"*. It descends onto the block instead of
around it. The automatic measure separates them cleanly too -- successful
episodes move the object 78-142 px, failed ones 3-44 px, which is the
signature of a nudge rather than a carry.

That is exactly the defect DAgger is shaped to fix: the human takes over at the
moment the grasp fails, so the corrected actions are labelled in precisely the
states where the policy is wrong, and nowhere else.

### Honest caveats

- **This is not a clean DAgger ablation.** The two checkpoints differ in three
  ways at once: DAgger corrections, 94 vs 50 episodes, and 40k vs 100k training
  steps. The result says "the current policy beats the earlier baseline on
  matched scenes", not "DAgger specifically caused it". Isolating that needs a
  94-episode non-DAgger control that does not exist.
- **n = 9.** The p-value is real but rests on six discordant pairs.
- The one cell both policies failed (bottom-right) is the same cell in both
  runs, which is weak evidence that something about that placement is harder --
  though section 7 showed training coverage there was not unusually thin.

### Carry forward

- **Scene replication is cheap and changes what nine episodes can support.**
  Ghosting the previous run's snapshot took seconds per episode and turned an
  underpowered comparison into a significant one. Use `--replicate` for every
  A-vs-B from now on.
- **Stage-wise scoring located the defect.** "2/9 vs 8/9" says the DAgger policy
  is better; "approach 9/9 for both, grasp 2/9 vs 8/9" says *what* got better,
  and that is what tells you where the next correction round should go.
- **Report the confounds with the result.** The number is strong enough that it
  does not need overselling, and the three-way confound is the first thing a
  reader should be told.

---

## 9. One night, six checkpoints, 52 scored episodes (2026-09-12)

Every run used the same nine placements, reproduced with `--replicate` to about
one pixel. Operator notes were recorded per episode.

| checkpoint | appr | grasp | transport | placement |
|---|---|---|---|---|
Scored **from the operator notes**, not the recorded stage flags -- six of the
52 flag sets disagree with what the note says happened.

| checkpoint | appr | grasp | transport | placement | flags said |
|---|---|---|---|---|---|
| DAgger, 94 ep, 40k | 9/9 | **8/9** | 8 | 8 | same |
| base, 50 ep, 100k, RGB | 9/9 | 3/9 | 2 | 2 | g2 |
| env-state, 10 ep, 50k | 8/9 | 3/9 | 3 | 3 | a9 g4 t4 p4 |
| RGB, 10 ep, 50k | **7/9** | 2/9 | 2 | 2 | a9 |
| env-state, 25 ep, 50k | 9/9 | 3/9 | 1 | 1 | same |
| env-state, 25 ep, rerun (5 cells) | 5/5 | 3/5 | 2 | 2 | g4 |

Two failure modes the flags erased entirely: a policy that never approached the
block at all (recorded as approach=1), and a grasp that succeeded before the
gripper opened early (recorded as approach-only).

### The variance floor, measured

The last two rows are **the same checkpoint on the same five placements, ten
minutes apart**:

```
cells 0-4, run 1:   . . # . .    1/5
cells 0-4, run 2:   . # # # #    4/5
```

A three-episode swing out of five. Every gap in the table except DAgger's is
one or two episodes, i.e. smaller than the noise. So the only ranking this
night supports is **DAgger vs everything else**; the representation and
sample-size comparisons are underpowered and must not be reported as results.

A tempting explanation for "25 did worse than 10" was contaminated
demonstrations. It was checked and it is false: counting gripper open-to-closed
transitions, **49 of 50 demos contain exactly one grasp**. The single
drop-and-regrasp (ep12) is in the 25-episode subset -- one episode in
twenty-five, which cannot move a policy by the amount in question.

To separate checkpoints one or two episodes apart needs roughly 25-30 rollouts
each, six or seven times this night's budget for that question alone.

### What the 51 notes agree on

Pattern-matched across every run:

| pattern | count |
|---|---|
| contact too low / lowered onto the block | 15 |
| nudged or pushed the block away | 10 |
| hesitation, repeated grasp attempts | 16 |
| rotation error, counter-clockwise | **12** |
| rotation error, clockwise | **2** |
| placed too high / too low | 10 / 5 |
| placed left / right | 8 / 2 |

**Approach was NOT uniform, and the stage flags hid it.** Re-read from the
operator notes: rgb-10 failed to approach the block at all in 2 of 9 episodes
("did not even try to approach block" -- the gripper appears to have started
closed), and env-10 in 1 of 9. The flags recorded approach=1 for all of them,
because the automatic suggestion defaults generously and was accepted.

Corrected per-run approach: DAgger 9/9, base-50 9/9, env-25 9/9, env-10 8/9,
rgb-10 **7/9**. The worst approach behaviour of the night belongs to the
smallest RGB-only model, which is the one place a sample-size effect shows up
clearly rather than inside the noise.

This was written here as "9/9 everywhere, every failure at contact" before the
notes were read. It was wrong, and it was wrong in the direction of the tidier
story -- twice in one session. The flags are a convenience; the notes are the
record.

**The placement bias is one-directional in all three degrees of freedom** --
counter-clockwise 6:1, high 2:1, left 4:1 -- and it appears in *every*
checkpoint regardless of training size or representation. A defect that
survives every variant is not a property of the learning; it is a property of
the data or the rig. The testable version: measure whether the DEMONSTRATIONS
place high, left and counter-clockwise relative to the target. If they do, the
policies are faithfully reproducing an operator habit and no amount of
retraining will remove it.

### Carry forward

- **Measure the variance floor before ranking anything.** Two runs of one
  checkpoint on the same scenes cost ten minutes and would have prevented two
  wrong conclusions tonight. Do it first, not last.
- **A defect present in every variant is upstream of all of them.** The
  contact-height failure and the directional placement bias are shared by
  checkpoints trained on 10, 25, 50 and 94 episodes, with and without
  object-centric features. Chasing them through representation changes cannot
  work.
- **Free text at a structured prompt gets parsed as structure.** Two episodes
  were mis-scored tonight because a note typed at the stages prompt contained
  the stage letters. Guard the prompt.
- **Stage-wise scoring paid for itself again.** "9/9 approach, 2-8/9 grasp"
  across six checkpoints is a far more useful sentence than any of the
  single-number success rates.
