# tube_mpc — a robust-MPC reference filter for teleoperation

A drop-in layer between your IK targets and your robot's joint commands.
It tracks the operator when their command is feasible and returns the
closest feasible command when it isn't — with a recursive-feasibility
guarantee that joint position/velocity/acceleration limits are **never**
violated, no matter what the operator does next. See `MATH.md` for the
full derivation and proof sketch.

The core depends only on `numpy`, `scipy`, `osqp`, `pyyaml`
(`matplotlib` for the demo). It never imports lerobot, pyroki, or
anything robot-specific — different IK weights, different lerobot
versions, and velocity- vs position-commanded arms are all absorbed by
the config file and a ~20-line adapter.

## Quickstart (this repo)

```bash
python -m pytest tube_mpc/tests -q     # 11 tests, ~4 s
python -m tube_mpc.check               # 30-second readiness check
python -m tube_mpc.demo_single_arm     # naive clamping vs tube MPC, with plot
```

Demo result on the right arm from `giava.urdf`, against an operator
"lunge" 0.4 rad past the joint limits: naive clamping violates position
limits on 115 steps (worst 0.41 rad past the limit); tube MPC has zero
violations, zero infeasible solves, ~1.3 ms mean solve time (20 ms
budget at 50 Hz).

## Transferring to another computer

1. **Copy the folder** (or `pip wheel ./tube_mpc` here and install the
   wheel there): from the parent directory, `pip install -e ./tube_mpc`
   — or just copy it next to your code and import it; it's flat Python.
2. **Copy `config.example.yaml` → `config.<machine>.yaml`** and set the
   machine-specific block: `urdf_path`, `joint_prefix`, `dt` (measure
   your real loop period, don't assume), `a_max`, and `command_mode`
   (`velocity` for your velocity-commanded setup).
3. **Calibrate W on that machine** — this is the one step you must not
   skip or copy from another machine:
   ```bash
   # record a few minutes of ordinary teleop, save an .npz with
   # q_cmd (T,n), q_meas (T,n), optionally v_meas, dt
   python -m tube_mpc.calibrate_w mylog.npz --config config.lab.yaml --write
   ```
   If calibration is rejected ("w_box too large"), the model is missing
   systematic behavior (wrong `dt`, heavy servo lag, wrong command
   mode) — fix that first; don't force it.
4. **Run the readiness check:** `python -m tube_mpc.check --config
   config.lab.yaml`. It must print `READY` before you touch the robot.
5. **Write the adapter** — implement the three methods of
   `TeleopAdapter` (`adapter.py`) against that machine's stack:

```python
from tube_mpc import FilterConfig, FilterRunner

class MyAdapter:                       # your lerobot/interbotix calls here
    def read_state(self):  ...         # -> (q, qdot), shape (n,) each
    def read_target(self): ...         # -> current IK target q_ref, (n,)
    def send(self, q, v, a, command):  # command is mode-selected (q/v/a)
        ...

cfg = FilterConfig.load("config.lab.yaml")
runner = FilterRunner(cfg, MyAdapter())
while teleoperating:
    runner.step()                      # one filtered control tick
```

`command_mode` decides what `send()` gets as `command`: planned
positions, planned velocities, or the acceleration input — all from the
same plan, so the guarantee is identical; pick what your driver accepts.

## What I need from you

- **Per machine:** the real control loop `dt`; the command interface
  (velocity units, saturation behavior, whether the driver filters
  internally); a short teleop log (`q_cmd`, `q_meas`, ideally `v_meas`)
  for W calibration; an `a_max` estimate (servo spec or logs).
- **Once:** which URDF is current on each machine (this repo's
  `giava.urdf` vs your updated one), and where in your teleop loop the
  IK target is produced, so the adapter reads the target *before* any
  legacy clamping/filtering (the filter replaces those, and stacking
  them hides its behavior).
- **For Phase 2 (collision constraints):** the collision pairs/geometry
  you want enforced — the sphere/GJK setup from `pyroki_ik_script.py`
  is the natural source.

## Files

| File | What it is |
|---|---|
| `model.py` | LTI double-integrator joint model + limits |
| `sets.py` | LQR gain, mRPI (s, α), constraint-tightening margins |
| `controller.py` | the tube-MPC QP (OSQP), safe-stop terminal set, tube fallback |
| `reference.py` | reference extrapolators (hold / decayed velocity) |
| `config.py` | `FilterConfig` — everything machine-specific, YAML-backed |
| `adapter.py` | `TeleopAdapter` protocol, `FilterRunner`, `SimAdapter` |
| `calibrate_w.py` | fit the disturbance box W from logged data |
| `check.py` | 30-second per-machine readiness check |
| `demo_single_arm.py` | naive clamping vs tube MPC on `giava.urdf` |
| `MATH.md` | derivation, guarantee, calibration method, roadmap |
