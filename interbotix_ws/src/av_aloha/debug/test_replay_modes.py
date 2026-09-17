"""Mode coverage for replay_episode: left/right/middle/bimanual/av(all).

    python debug/test_replay_modes.py        # no hardware, no data needed


Builds the action vector EXACTLY the way dataset.build_frame does (per arm:
joints then gripper if it has one), with a unique sentinel per joint, then
checks replay's layout routes every sentinel back to the arm it came from.
"""

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import sys, types
sys.path.insert(0, "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts")
sys.path.insert(0, "/opt/ros/noetic/lib/python3/dist-packages")
import numpy as np
import replay_episode as R
from arm_config import ARM_CONFIG
from data_col_config import ARM_MODES, ACTION_LAYOUTS, action_dim

fails = []
def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail and not cond else ""))
    if not cond: fails.append(label)

def build_action(arms):
    """Sentinels: arm index * 100 + joint index; gripper = arm index * 100 + 99."""
    vec, truth = [], {}
    for ai, arm in enumerate(arms):
        n = ARM_CONFIG[arm]["num_joints"]
        truth[f"{arm}_arm"] = [ai*100 + j for j in range(n)]
        vec += truth[f"{arm}_arm"]
        if ARM_CONFIG[arm]["has_gripper"]:
            truth[f"{arm}_gripper"] = ai*100 + 99
            vec.append(truth[f"{arm}_gripper"])
    return np.array(vec, dtype=np.float32), truth

def action_names(arms):
    out = []
    for a in arms:
        out += [f"{j}_cmd" for j in ARM_CONFIG[a]["joint_names"]]
        if ARM_CONFIG[a]["has_gripper"]: out.append(f"{a}_gripper_cmd")
    return out

class FakeDS:
    def __init__(self, arms):
        self.meta = types.SimpleNamespace(
            info={"features": {"action": {"names": action_names(arms)}}})

print("\n=== 1. every mode: layout routes each joint to the right arm ===")
for mode, arms in ARM_MODES.items():
    vec, truth = build_action(arms)
    check(f"{mode:9s} action dim {len(vec)} == action_dim()", len(vec) == action_dim(mode))
    parsed = R.parse_action(vec, mode)
    ok = True
    for key, expect in truth.items():
        got = parsed[key]
        got = list(np.asarray(got).reshape(-1).astype(int)) if isinstance(expect, list) else int(got)
        if got != expect: ok = False; print(f"      {key}: got {got} expected {expect}")
    check(f"{mode:9s} all joints/grippers routed correctly", ok)
    check(f"{mode:9s} no gripper key for middle",
          ("middle_gripper" not in parsed) if "middle" in arms else True)

print("\n=== 2. mode detection from action feature names (no meta.json) ===")
for mode, arms in ARM_MODES.items():
    det, how = R.detect_mode("/nonexistent", FakeDS(arms), action_dim(mode))
    same = det is not None and ARM_MODES[det] == arms
    check(f"{mode:9s} -> {det} via {how}", same, f"got {det}")

print("\n=== 3. left vs middle: both dim 7, names must disambiguate ===")
dl, _ = R.detect_mode("/nonexistent", FakeDS(["left"]), 7)
dm, _ = R.detect_mode("/nonexistent", FakeDS(["middle"]), 7)
check("left dataset detects as left", dl == "left", f"got {dl}")
check("middle dataset detects as middle", dm == "middle", f"got {dm}")
class NoNames:
    meta = types.SimpleNamespace(info={"features": {"action": {"names": None}}})
d, why = R.detect_mode("/nonexistent", NoNames(), 7)
check("dim 7 with no names is refused, not guessed", d is None, f"got {d}")
print(f"        reason: {why}")

print("\n=== 4. clock candidates across dataset generations ===")
cases = [
    ("current per-arm", {"timestamp": 0, "observation.timestamps.left": 1,
                         "observation.timestamps.right": 2,
                         "observation.timestamps.left_wrist": 9}, "bimanual",
     ["timestamp", "observation.timestamps.left", "observation.timestamps.right"]),
    ("legacy robot",    {"observation.timestamps.robot": 1, "timestamp": 2}, "right",
     ["timestamp", "observation.timestamps.robot"]),
    ("lerobot only",    {"timestamp": 2}, "right", ["timestamp"]),
    ("av per-arm",      {"timestamp": 0, "observation.timestamps.left": 1,
                         "observation.timestamps.middle": 3}, "av",
     ["timestamp", "observation.timestamps.left", "observation.timestamps.middle"]),
    ("middle only",     {"observation.timestamps.middle": 3}, "middle",
     ["observation.timestamps.middle"]),
    ("nothing",         {"observation.state": 0}, "right", []),
]
for label, sample, mode, expect in cases:
    got = R.candidate_clocks(sample, mode)
    check(f"{label:18s} -> {got}", got == expect, f"expected {expect}")

print("\n=== 5. camera timestamps are never mistaken for arm timestamps ===")
got = R.candidate_clocks({"observation.timestamps.left_wrist": 1, "timestamp": 2}, "bimanual")
check("left_wrist (camera) is not a clock candidate", got == ["timestamp"], f"got {got}")

print("\n=== 5b. a quantised clock is rejected, not used ===")
# absolute epoch values read back as float32: every step collapses to 0 elapsed
q = np.asarray([1.787353728e9 + i * 0.02 for i in range(50)], dtype=np.float32)
rows = [{"timestamp": float(q[i])} for i in range(50)]
ok, why = R.clock_quality(lambda i: rows[i], 0, 50, "timestamp", 50.0)
check("float32-quantised absolute clock rejected", not ok, f"ok={ok} why={why}")
print(f"        reason: {why}")
good = [{"timestamp": i * 0.02} for i in range(50)]
ok, why = R.clock_quality(lambda i: good[i], 0, 50, "timestamp", 50.0)
check("clean 50 Hz relative clock accepted", ok, why)

print("\n=== 6. cross-mode mismatch is detectable (the dangerous case) ===")
av_vec, _ = build_action(ARM_MODES["av"])
check("av vector (21) != bimanual dim (14)", len(av_vec) != action_dim("bimanual"))
check("av vector (21) != right dim (7)", len(av_vec) != action_dim("right"))
det, _ = R.detect_mode("/nonexistent", FakeDS(ARM_MODES["av"]), 21)
check("av dataset never detects as bimanual", ARM_MODES[det] != ARM_MODES["bimanual"])

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
