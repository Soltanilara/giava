"""Drive replay_episode.main() with the hardware layer stubbed.

Proves the loop itself -- not just mode detection -- sends each arm ITS OWN
recorded joints, never calls a gripper on the (gripperless) middle arm, and
holds the recorded timeline.  No robot, no ROS master, no driver required.

    python debug/make_synth.py /tmp/giava_replay_synth
    python debug/test_replay_loop.py /tmp/giava_replay_synth

make_synth.py writes camera-less `middle` and `av` datasets, because no
real recording of either mode exists on disk yet; the bimanual and legacy
single-arm cases below run against actual recorded data.
"""

import _giava_paths  # noqa: F401  (puts the shared giava trees on sys.path)

import sys, time, types
sys.path.insert(0, "/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts")
sys.path.insert(0, "/opt/ros/noetic/lib/python3/dist-packages")
import numpy as np
import replay_episode as R
from arm_config import ARM_CONFIG

S = sys.argv[1] if len(sys.argv) > 1 else '/tmp/giava_replay_synth'
fails = []
def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail and not cond else ""))
    if not cond: fails.append(label)

class FakeBot:
    def __init__(self, arm):
        n = ARM_CONFIG[arm]["num_joints"]
        self.dxl = types.SimpleNamespace(
            joint_states=types.SimpleNamespace(position=[0.0]*(n+2)))

def run(dataset_root, argv_extra=()):
    sent, grips, resets = {}, {}, []
    R.rospy = types.SimpleNamespace(
        init_node=lambda *a, **k: None,
        on_shutdown=lambda f: None,
        is_shutdown=lambda: False)
    R.create_and_configure_robot = lambda arm: FakeBot(arm)
    R.stop_robots = lambda robots: None
    R.reset_arm = lambda bot, arm: resets.append(arm)
    def _cmd(bot, q):
        for a, b in bots.items():
            if b is bot: sent.setdefault(a, []).append(np.asarray(q).copy())
    def _grip(bot, v):
        for a, b in bots.items():
            if b is bot: grips.setdefault(a, []).append(float(v))
    bots = {}
    _orig = R.create_and_configure_robot
    def _mk(arm):
        b = FakeBot(arm); bots[arm] = b; return b
    R.create_and_configure_robot = _mk
    R.replay_arm_command = _cmd
    R.command_gripper = _grip
    sys.argv = ["replay_episode.py", "--dataset-root", dataset_root,
                "--episode-idx", "0", "--max-steps", "20"] + list(argv_extra)
    t = time.monotonic()
    R.main()
    return sent, grips, resets, time.monotonic() - t

print("\n=== av (3 arms): every arm gets its own sentinel block ===")
sent, grips, resets, dt = run(f"{S}/synth_av/20260821_000000")
check("all three arms reset", sorted(resets) == ["left", "middle", "right"], str(resets))
check("all three arms commanded", sorted(sent) == ["left", "middle", "right"], str(sorted(sent)))
check("20 commands per arm", all(len(v) == 20 for v in sent.values()),
      str({k: len(v) for k, v in sent.items()}))
check("left  gets 6 joints starting at   0", np.allclose(sent["left"][0],  [0,1,2,3,4,5], atol=.01), str(sent["left"][0]))
check("right gets 6 joints starting at 100", np.allclose(sent["right"][0], [100,101,102,103,104,105], atol=.01), str(sent["right"][0]))
check("middle gets 7 joints starting at 200", np.allclose(sent["middle"][0], [200,201,202,203,204,205,206], atol=.01), str(sent["middle"][0]))
check("left+right grippers commanded", sorted(grips) == ["left", "right"], str(sorted(grips)))
check("middle gripper NEVER commanded", "middle" not in grips)
check(f"20 steps @50 Hz took {dt:.2f}s (~0.4 s expected, not 1.0 s)", 0.3 < dt < 0.75, f"{dt:.2f}s")

print("\n=== middle only: 7 joints, no gripper ===")
sent, grips, resets, dt = run(f"{S}/synth_middle/20260821_000000")
check("only the middle arm is built", sorted(sent) == ["middle"], str(sorted(sent)))
check("7 joints starting at 0", np.allclose(sent["middle"][0], [0,1,2,3,4,5,6], atol=.01), str(sent["middle"][0]))
check("no gripper call at all", grips == {}, str(grips))

print("\n=== bimanual (real recorded data) ===")
sent, grips, resets, dt = run("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/"
                              "dataset/lerobot/bimanual_data_collection/20260722_164356")
check("left + right only", sorted(sent) == ["left", "right"], str(sorted(sent)))
check("6 joints each", all(len(c[0]) == 6 for c in sent.values()))
check("both grippers commanded", sorted(grips) == ["left", "right"], str(sorted(grips)))

print("\n=== legacy single-arm (real recorded data, timestamps.robot) ===")
sent, grips, resets, dt = run("/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/"
                              "dataset/lerobot/block_square/20260528_131838")
check("right only", sorted(sent) == ["right"], str(sorted(sent)))
check("6 joints", len(sent["right"][0]) == 6)
check("gripper commanded", "right" in grips)

print("\n=== --fps override paces independently of the recording ===")
sent, _, _, dt = run(f"{S}/synth_av/20260821_000000", ["--fps", "100"])
check(f"20 steps @100 Hz took {dt:.2f}s (~0.2 s)", 0.1 < dt < 0.45, f"{dt:.2f}s")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
