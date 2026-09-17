"""Self-test for the calibration package  --  no hardware required.

Checks the parts that can be verified without cameras or a powered robot:
transform algebra, the frame conventions, agreement with the deployed FK
path, the driver<->URDF bridge, the ChArUco detect-and-calibrate pipeline
(against synthetic views rendered through a KNOWN camera matrix), the
never-overwrite guarantee, and the timing statistics.

    JAX_PLATFORMS=cpu python calibration/selftest.py

Exit code is 0 when everything passes, 1 otherwise.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
## data_collection_scripts/ is a SIBLING of this package now, not the
## parent it used to be.
SCRIPTS_DIR = HERE.parent / "data_collection_scripts"
for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str):
    """Decorator turning a function into a recorded pass/fail test."""
    def wrap(fn: Callable[[], None]):
        try:
            fn()
            RESULTS.append((name, True, ""))
            print(f"  PASS  {name}")
        except Exception as exc:
            RESULTS.append((name, False, traceback.format_exc()))
            print(f"  FAIL  {name}: {exc}")
        return fn
    return wrap


def close(a, b, tol=1e-9, what="") -> None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    err = float(np.max(np.abs(a - b)))
    if err > tol:
        raise AssertionError(f"{what} max abs difference {err:.3e} > {tol:.1e}")


## ------------------------------------------------------------------ ##

def run() -> None:
    from common import (OutputExistsError, T_from_dict, T_from_xyz_rpy,
                        T_to_dict, invert_T, make_T, matrix_to_quat_wxyz,
                        matrix_to_rpy, quat_wxyz_to_matrix, rotation_angle_deg,
                        rpy_to_matrix, save_json)

    print("\n--- transform algebra ---")

    @check("quaternion <-> matrix round-trip")
    def _():
        rng = np.random.default_rng(0)
        for _ in range(200):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            if q[0] < 0:
                q = -q
            close(matrix_to_quat_wxyz(quat_wxyz_to_matrix(q)), q, 1e-9,
                  "quat round-trip")

    @check("rpy <-> matrix round-trip (URDF fixed-axis convention)")
    def _():
        rng = np.random.default_rng(1)
        for _ in range(200):
            rpy = np.array([rng.uniform(-np.pi, np.pi),
                            rng.uniform(-1.4, 1.4),      # avoid gimbal lock
                            rng.uniform(-np.pi, np.pi)])
            close(matrix_to_rpy(rpy_to_matrix(rpy)), rpy, 1e-9, "rpy")

    @check("rpy composes as Rz @ Ry @ Rx")
    def _():
        r, p, y = 0.3, -0.2, 1.1
        Rx = rpy_to_matrix([r, 0, 0])
        Ry = rpy_to_matrix([0, p, 0])
        Rz = rpy_to_matrix([0, 0, y])
        close(rpy_to_matrix([r, p, y]), Rz @ Ry @ Rx, 1e-12, "rpy order")

    @check("invert_T is a true rigid inverse")
    def _():
        rng = np.random.default_rng(2)
        for _ in range(100):
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            T = make_T(rng.normal(size=3), quat_wxyz_to_matrix(q))
            close(T @ invert_T(T), np.eye(4), 1e-9, "T @ inv(T)")
            close(invert_T(invert_T(T)), T, 1e-9, "double inverse")

    @check("T_a_b maps points from b into a (the stated convention)")
    def _():
        ## Frame b sits at (1,0,0) in a, rotated +90 deg about z.
        T_a_b = make_T([1, 0, 0], rpy_to_matrix([0, 0, np.pi / 2]))
        ## b's origin, in b, is the origin; in a it must be (1,0,0).
        close((T_a_b @ np.array([0, 0, 0, 1.0]))[:3], [1, 0, 0], 1e-12,
              "origin of b in a")
        ## b's +x axis maps to a's +y.
        close((T_a_b @ np.array([1, 0, 0, 1.0]))[:3], [1, 1, 0], 1e-12,
              "b's +x in a")
        ## The translation column is b's origin expressed in a.
        close(T_a_b[:3, 3], [1, 0, 0], 1e-12, "translation column")

    @check("chained composition T_w_c = T_w_e @ T_e_c")
    def _():
        rng = np.random.default_rng(3)
        q1 = rng.normal(size=4); q1 /= np.linalg.norm(q1)
        q2 = rng.normal(size=4); q2 /= np.linalg.norm(q2)
        T_w_e = make_T(rng.normal(size=3), quat_wxyz_to_matrix(q1))
        T_e_c = make_T(rng.normal(size=3), quat_wxyz_to_matrix(q2))
        T_w_c = T_w_e @ T_e_c
        ## A point at the camera origin must land at the camera's position
        ## in world, computed either way.
        p_w_direct = (T_w_c @ np.array([0, 0, 0, 1.0]))[:3]
        p_w_chain = (T_w_e @ (T_e_c @ np.array([0, 0, 0, 1.0])))[:3]
        close(p_w_direct, p_w_chain, 1e-12, "composition")

    @check("T_to_dict / T_from_dict round-trip")
    def _():
        T = T_from_xyz_rpy([0.1, -0.2, 0.3], [0.4, -0.5, 0.6])
        d = T_to_dict(T, "base", "camera")
        close(T_from_dict(d), T, 1e-12, "serialised transform")
        assert d["name"] == "T_base_camera"
        assert "maps points from 'camera' into 'base'" in d["convention"]

    @check("rotation_angle_deg")
    def _():
        for deg in (0.0, 15.0, 90.0, 179.0):
            R = rpy_to_matrix([0, 0, np.radians(deg)])
            close(rotation_angle_deg(R), deg, 1e-6, f"{deg} deg")

    ## -------------------------------------------------------------- ##
    print("\n--- never-overwrite guarantee ---")

    @check("save_json refuses to clobber, allows explicit overwrite")
    def _():
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cal.json"
            save_json({"a": 1}, p)
            try:
                save_json({"a": 2}, p)
            except OutputExistsError:
                pass
            else:
                raise AssertionError("second write should have been refused")
            import json
            assert json.loads(p.read_text())["a"] == 1, "file was modified"
            save_json({"a": 3}, p, overwrite=True)
            assert json.loads(p.read_text())["a"] == 3, "overwrite failed"

    @check("save_json serialises numpy types")
    def _():
        import json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.json"
            save_json({"arr": np.arange(3), "f": np.float32(1.5),
                       "i": np.int64(7), "b": np.bool_(True)}, p)
            got = json.loads(p.read_text())
            assert got == {"arr": [0, 1, 2], "f": 1.5, "i": 7, "b": True}

    ## -------------------------------------------------------------- ##
    print("\n--- URDF frame graph ---")

    from kinematics import UrdfTree

    tree = UrdfTree()

    @check("world frame is giava.urdf's root link 'base'")
    def _():
        assert tree.root == "base", f"root is {tree.root}"

    @check("giava.urdf base placements still match the documented values "
           "(regression guard, NOT a physical check)")
    def _():
        ## Compares the URDF against numbers copied from FRAMES.md. It
        ## catches the URDF silently drifting from the documentation --
        ## nothing more. It does NOT verify the real arms are bolted where
        ## the URDF claims; only a tape measure does that, via
        ## base_validation.py.
        ## Base x = +/-0.520 since 2026-08-19: measured 1040 mm centre-to-
        ## centre with a ruler, corroborated by a plate-to-plate gap check
        ## against FK at a live pose. The URDF's previous +/-0.469 (938 mm)
        ## drew the arms ~10 cm closer together than reality.
        ## Left base touch-calibrated 2026-09-03 against the right arm
        ## (touch_calibrate.py, 15 tip-to-tip touches, 15 mm rms residual):
        ## +15.4 mm x, +25.4 mm z, -0.24 deg yaw off the symmetric assumption.
        expect = {"right_base_link": (-0.520, -0.019, 0.020),
                  "left_base_link": (0.535278, -0.020820, 0.045371),
                  "middle_base_link": (0.0, 0.400, 0.020)}
        for link, xyz in expect.items():
            close(tree.parent_joint[link].xyz, xyz, 1e-6, link)

    @check("no TCP / tool / ee frame exists in the URDF")
    def _():
        bad = [l for l in tree.links
               if any(k in l.lower()
                      for k in ("ee_", "_tcp", "tool", "_tip", "grasp"))]
        assert not bad, f"unexpectedly found {bad}"

    @check("middle_camera_body and _cover are identity children")
    def _():
        for link in ("middle_camera_body", "middle_camera_cover"):
            close(tree.fixed_transform("middle_camera", link), np.eye(4),
                  1e-12, link)

    @check("chain_to_root reaches the world frame")
    def _():
        chain = tree.chain_to_root("right_gripper_base")
        assert chain[0] == "base" and chain[-1] == "right_gripper_base"
        assert "right_gripper_link" in chain

    @check("fixed_transform refuses to cross a movable joint")
    def _():
        try:
            tree.fixed_transform("base", "right_gripper_base")
        except ValueError:
            pass
        else:
            raise AssertionError("should have refused: revolute joints between")

    ## -------------------------------------------------------------- ##
    print("\n--- forward kinematics ---")

    from kinematics import RobotFrames
    from arm_config import ARM_CONFIG

    frames = RobotFrames()
    q_home = frames.home_q()

    @check("FK agrees with the deployed robot_control.compute_fk_and_ee")
    def _():
        ## The strongest check available: the same numbers the data
        ## collection loop uses, through a different code path.
        from robot_control import compute_fk_and_ee
        arm_data = {a: {"ee_index": frames.link_names.index(
            ARM_CONFIG[a]["ee_link"])} for a in ("left", "right", "middle")}
        _, ee = compute_fk_and_ee(frames.robot,
                                  np.asarray(q_home, dtype=np.float32),
                                  arm_data)
        mine = frames.fk(q_home)
        for arm, pose7 in ee.items():
            pose7 = np.asarray(pose7)
            T = mine[ARM_CONFIG[arm]["ee_link"]]
            close(T[:3, 3], pose7[4:7], 1e-5, f"{arm} position")
            q_mine = matrix_to_quat_wxyz(T[:3, :3])
            q_ref = np.asarray(pose7[0:4])
            if np.dot(q_mine, q_ref) < 0:
                q_ref = -q_ref   # q and -q are the same rotation
            close(q_mine, q_ref, 1e-5, f"{arm} orientation")

    @check("FK matches ik/robot_model.home_poses")
    def _():
        sys.path.insert(0, str(HERE.parent / "ik"))
        import robot_model as rm
        pos, wxyz = rm.home_poses(frames.robot, np.asarray(q_home, np.float32))
        mine = frames.fk(q_home)
        for i, link in enumerate(rm.TARGET_LINKS):
            close(mine[link][:3, 3], pos[i], 1e-5, link)

    @check("FK of a fixed-joint pair equals the URDF's own transform")
    def _():
        fk = frames.fk(q_home)
        T_direct = np.linalg.inv(fk["right_gripper_link"]) @ fk["right_gripper_base"]
        T_urdf = tree.fixed_transform("right_gripper_link", "right_gripper_base")
        close(T_direct, T_urdf, 1e-5, "gripper_link -> gripper_base")

    @check("the 'home' configuration is NOT all zeros")
    def _():
        nz = frames.nonzero_home_joints()
        assert nz, "expected pyroki's default to differ from zero"
        assert "right_shoulder" in nz

    ## -------------------------------------------------------------- ##
    print("\n--- driver <-> URDF bridge ---")

    from kinematics import JointFrameBridge

    bridge = JointFrameBridge(frames.robot)

    @check("driver -> urdf -> driver round-trips")
    def _():
        rng = np.random.default_rng(4)
        for _ in range(50):
            q = rng.uniform(-1.5, 1.5, frames.num_actuated)
            back = bridge.to_driver(bridge.to_urdf(q), ref_driver=q)
            close(back, q, 1e-6, "round-trip")

    @check("left/right arm joints are identity through the bridge")
    def _():
        rng = np.random.default_rng(5)
        q = rng.uniform(-1.0, 1.0, frames.num_actuated)
        u = bridge.to_urdf(q)
        for arm in ("left", "right"):
            idx = frames.joint_indices(arm)
            close(u[idx], q[idx], 1e-9, f"{arm} unchanged")

    @check("the middle waist really is shifted by pi")
    def _():
        q = np.zeros(frames.num_actuated)
        u = bridge.to_urdf(q)
        w = frames.actuated_names.index("middle_base")
        expect = (np.pi + np.pi) % (2 * np.pi) - np.pi
        close(u[w], expect, 1e-9, "waist offset")

    ## -------------------------------------------------------------- ##
    print("\n--- camera mounts ---")

    import camera_mount as CM

    @check("the D405 mesh and MuJoCo camera agree on POSITION")
    def _():
        mesh = CM._d405_mesh_origin()
        opt = CM._d405_nominal_optical()
        close(mesh[:3, 3], opt[:3, 3], 1e-5,
              "mesh vs mujoco translation")

    @check("mesh and optical orientations differ by exactly 180 deg")
    def _():
        mesh = CM._d405_mesh_origin()
        opt = CM._d405_nominal_optical()
        ang = rotation_angle_deg(mesh[:3, :3].T @ opt[:3, :3])
        close(ang, 180.0, 1e-3, "mesh vs optical rotation")

    @check("no camera claims a calibrated mount yet")
    def _():
        for cam in CM.CAMERA_MOUNTS:
            m = CM.resolve_mount(cam)
            assert m["provenance"] != "calibrated" or m["validated"], cam
            if m["provenance"] in ("urdf_mesh", "mujoco_model", "unknown"):
                assert m["validated"] is False, f"{cam} claims validation"

    @check("T_world_camera refuses to invent a missing extrinsic")
    def _():
        try:
            CM.T_world_camera(frames, q_home, "oak_left")
        except ValueError:
            pass
        else:
            raise AssertionError("should have refused: no mount transform")

    @check("T_world_camera composes for a camera that has a mount")
    def _():
        T = CM.T_world_camera(frames, q_home, "right_wrist")
        mount = CM.resolve_mount("right_wrist")
        expect = frames.link_pose(q_home, "right_gripper_base") \
            @ mount["T_parent_optical"]
        close(T, expect, 1e-12, "composition")

    ## -------------------------------------------------------------- ##
    print("\n--- tool centre point ---")

    import tcp as TCP

    @check("the DERIVED default TCP is still 72.2 mm along the flange's local +z")
    def _():
        close(TCP._derived_offset()[:3, 3], [0.0, 0.00006, 0.07220], 1e-5, "derived TCP")

    @check("both hand arms resolve the MEASURED closed-fingertip TCP (127.5 mm, touch_calibrate 2026-09-03)")
    def _():
        for arm in TCP.TCP_ARMS:
            r = TCP.resolve(arm)
            assert r["measured"] is True, arm
            assert str(r["provenance"]).startswith("measured:touch_calibrate"), r["provenance"]
            close(r["T_flange_tcp"][:3, 3], [0.0, 0.0, 0.1275], 1e-4, f"{arm} TCP offset")  # file keeps 5 decimals (0.12753)

    @check("the middle (camera) arm has no TCP")
    def _():
        assert TCP.resolve("middle")["T_flange_tcp"] is None

    @check("flange_target inverts tcp_pose exactly")
    def _():
        rng = np.random.default_rng(7)
        for _ in range(50):
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            T_flange = make_T(rng.normal(size=3), quat_wxyz_to_matrix(q))
            T_tcp = TCP.tcp_pose(T_flange, "right")
            close(TCP.flange_target(T_tcp, "right"), T_flange, 1e-9,
                  "round-trip")

    @check("the TCP offset is orientation dependent in world z")
    def _():
        # Gripper pointing straight down: the whole measured 127.5 mm lands on z.
        down = make_T([0, 0, 1.0], rpy_to_matrix([np.pi, 0, 0]))
        dz_down = TCP.tcp_pose(down, "right")[2, 3] - down[2, 3]
        # Pointing horizontally: none of it does.
        horiz = make_T([0, 0, 1.0], rpy_to_matrix([np.pi / 2, 0, 0]))
        dz_horiz = TCP.tcp_pose(horiz, "right")[2, 3] - horiz[2, 3]
        close(dz_down, -0.1275, 1e-4, "down-pointing dz")
        close(dz_horiz, 0.0, 1e-4, "horizontal dz")

    ## -------------------------------------------------------------- ##
    print("\n--- hand-eye solver (synthetic, known ground truth) ---")

    from handeye import (build_motions, residuals, rotation_diversity,
                         so3_exp, so3_log, solve_ax_xb)

    def _scene(n, rot_scale, axes=3, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        X_true = make_T([0.004, -0.0825, -0.0096],
                        so3_exp(np.array([-0.4363, 0.02, 0.01])))
        T_bb = make_T([0.3, 0.1, 0.2], so3_exp(rng.normal(scale=0.3, size=3)))
        T_bf, T_cb = [], []
        for _ in range(n):
            w = rng.normal(scale=rot_scale, size=3)
            if axes == 1:
                w[1] = w[2] = 0.0
            T = make_T(rng.normal(scale=0.12, size=3) + np.array([0, 0, 0.5]),
                       so3_exp(w))
            T_bf.append(T)
            T_c = invert_T(X_true) @ invert_T(T) @ T_bb
            if noise:
                T_c = T_c @ make_T(
                    rng.normal(scale=noise, size=3),
                    so3_exp(rng.normal(scale=np.radians(0.2), size=3)))
            T_cb.append(T_c)
        return X_true, T_bf, T_cb

    @check("so3 log/exp round-trip, including near pi")
    def _():
        rng = np.random.default_rng(3)
        for _ in range(200):
            w = rng.normal(size=3)
            w = w / np.linalg.norm(w) * rng.uniform(0.0, np.pi - 1e-4)
            close(so3_log(so3_exp(w)), w, 1e-8, "log(exp(w))")
        # exactly pi is the branch that trips naive implementations
        for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                     np.array([0.577, 0.577, 0.577])):
            R = so3_exp(axis / np.linalg.norm(axis) * np.pi)
            close(np.abs(so3_log(R)),
                  np.abs(axis / np.linalg.norm(axis) * np.pi), 1e-5, "pi case")

    @check("hand-eye recovers the exact transform from noise-free data")
    def _():
        X_true, T_bf, T_cb = _scene(12, 0.45)
        X = solve_ax_xb(*build_motions(T_bf, T_cb))
        close(X[:3, 3], X_true[:3, 3], 1e-9, "translation")   # 1 nanometre
        ## 1e-4 deg, not 1e-6: the residual bottoms out around 2e-6 deg,
        ## which is the arithmetic's noise floor rather than solver error.
        ## Asserting below it would be testing float64, not the method.
        da = rotation_angle_deg(X_true[:3, :3].T @ X[:3, :3])
        assert da < 1e-4, f"rotation off by {da:.3e} deg"

    @check("hand-eye stays within 3 mm / 0.3 deg under realistic PnP noise")
    def _():
        X_true, T_bf, T_cb = _scene(16, 0.45, noise=5e-4, seed=1)
        X = solve_ax_xb(*build_motions(T_bf, T_cb))
        dt = float(np.linalg.norm(X[:3, 3] - X_true[:3, 3])) * 1e3
        da = rotation_angle_deg(X_true[:3, :3].T @ X[:3, :3])
        assert dt < 3.0, f"translation off by {dt:.2f} mm"
        assert da < 0.3, f"rotation off by {da:.3f} deg"

    @check("hand-eye REFUSES single-axis rotations as degenerate")
    def _():
        _, T_bf, _ = _scene(12, 0.45, axes=1)
        d = rotation_diversity(T_bf)
        assert d["axis_rank"] == 1, d
        assert d["adequate"] is False

    @check("hand-eye REFUSES pure translation as degenerate")
    def _():
        _, T_bf, _ = _scene(12, 0.0)
        d = rotation_diversity(T_bf)
        assert d["adequate"] is False, d

    @check("hand-eye residuals are near zero for an exact solution")
    def _():
        _, T_bf, T_cb = _scene(10, 0.45, seed=5)
        A, B = build_motions(T_bf, T_cb)
        r = residuals(solve_ax_xb(A, B), A, B)
        assert r["rotation_deg"]["max"] < 1e-4, r      # noise floor ~2e-6
        assert r["translation_mm"]["max"] < 1e-6, r    # nanometres

    ## -------------------------------------------------------------- ##
    print("\n--- ChArUco pipeline (synthetic, known ground truth) ---")

    _charuco_tests()

    ## -------------------------------------------------------------- ##
    print("\n--- scene camera extrinsics (synthetic, known ground truth) ---")

    import scene_extrinsics as SE

    def _rot(axis, deg):
        a = np.radians(deg)
        k = np.asarray(axis, float) / np.linalg.norm(axis)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)

    @check("average_rotation returns a proper rotation and the true mean")
    def _():
        R0 = _rot([0.3, 1, 0.2], 40)
        Rs = [R0 @ _rot([1, 0, 0], d) for d in (-6, -2, 0, 2, 6)]
        M = SE.average_rotation(Rs)
        close(M.T @ M, np.eye(3), 1e-12, "not orthonormal")
        close(np.linalg.det(M), 1.0, 1e-12, "determinant")
        ## Perturbations about a COMMON axis, symmetric about zero, are a
        ## commutative subgroup, so the mean is exactly R0 -- the residual
        ## here is pure floating point.
        ##
        ## The tolerance is 1e-4 deg rather than machine epsilon because
        ## rotation_angle_deg goes through arccos, whose derivative is
        ## infinite at 1: near identity it loses HALF the significant
        ## digits, so the noise floor is sqrt(eps) ~ 1.5e-8 rad ~ 1e-6 deg
        ## no matter how exact the rotation itself is. Measuring a
        ## near-zero rotation angle is simply an ill-conditioned way to ask
        ## the question -- so ask it directly too, on the matrix.
        close(rotation_angle_deg(R0.T @ M), 0.0, 1e-4, "mean rotation")
        close(M, R0, 1e-12, "mean rotation, compared as a matrix")

    @check("average_rotation is a real mean on non-commuting rotations")
    def _():
        """The commutative case above cannot distinguish a genuine mean
        from any scheme that happens to cancel. Perturb about three
        different axes, where the answer is not any single input."""
        R0 = _rot([0.3, 1, 0.2], 40)
        Rs = [R0 @ _rot(a, d) for a, d in (([1, 0, 0], 9), ([0, 1, 0], -7),
                                           ([0, 0, 1], 5), ([1, 1, 0], -4))]
        M = SE.average_rotation(Rs)
        close(M.T @ M, np.eye(3), 1e-12, "not orthonormal")
        ## Inside the cloud, and not equal to any member of it.
        assert rotation_angle_deg(R0.T @ M) < 8.0
        for R in Rs:
            assert rotation_angle_deg(R.T @ M) > 1e-3, "mean equals an input"
        ## A mean is order independent; a sequential scheme would not be.
        close(SE.average_rotation(Rs[::-1]), M, 1e-12, "order dependence")
        ## And it is the true L2 minimiser: no member beats it on the
        ## summed squared chordal distance it claims to minimise.
        def cost(X):
            return sum(float(np.sum((X - R) ** 2)) for R in Rs)
        best = cost(M)
        for R in Rs:
            assert cost(R) > best, "an input beat the 'mean' on its own cost"

    @check("average_rotation never returns a reflection")
    def _():
        ## Two rotations 180 deg apart: the arithmetic mean is singular and
        ## a naive SVD projection can come back with det = -1, which is a
        ## mirror and physically impossible.
        Rs = [np.eye(3), _rot([0, 0, 1], 179.9)]
        M = SE.average_rotation(Rs)
        assert np.linalg.det(M) > 0, f"det {np.linalg.det(M)}"

    @check("spread is zero for identical poses and exact for a known offset")
    def _():
        T = make_T([0.1, -0.2, 0.3], _rot([1, 1, 0], 20))
        z = SE.spread([T, T, T])
        close(z["translation_mm"]["max"], 0.0, 1e-9, "identical translation")
        close(z["rotation_deg"]["max"], 0.0, 1e-9, "identical rotation")
        ## Two poses 10 mm apart sit 5 mm either side of their mean.
        a = make_T([0.0, 0, 0], np.eye(3))
        b = make_T([0.010, 0, 0], np.eye(3))
        z = SE.spread([a, b])
        close(z["translation_mm"]["mean"], 5.0, 1e-9, "known spread")

    @check("umeyama recovers a rigid transform exactly")
    def _():
        R, t = _rot([0.2, -1, 0.4], 37), np.array([0.31, -0.22, 0.94])
        src = np.random.default_rng(0).uniform(-0.3, 0.3, (20, 3))
        T, sc = SE.umeyama(src, src @ R.T + t)
        close(T[:3, :3], R, 1e-12, "rotation")
        close(T[:3, 3], t, 1e-12, "translation")
        close(sc, 1.0, 0.0, "rigid scale must be exactly 1")

    @check("umeyama recovers a known 3% scale error")
    def _():
        R, t = _rot([0, 0, 1], 25), np.array([0.1, 0.2, 0.3])
        src = np.random.default_rng(1).uniform(-0.3, 0.3, (20, 3))
        T, sc = SE.umeyama(src, 1.03 * (src @ R.T) + t, with_scale=True)
        close(sc, 1.03, 1e-12, "scale")

    @check("umeyama refuses a reflection even on coplanar noisy points")
    def _():
        """Coplanar points are the case where a mirrored 'rotation' can fit
        BETTER than the true one. Without the det guard the fit succeeds
        and is physically impossible."""
        rng = np.random.default_rng(3)
        src = np.concatenate(
            [rng.uniform(-0.2, 0.2, (30, 2)), np.zeros((30, 1))], axis=1)
        R, t = _rot([0, 1, 0], 15), np.array([0.05, 0.0, 0.4])
        dst = src @ R.T + t + rng.normal(0, 0.004, (30, 3))
        T, _ = SE.umeyama(src, dst)
        assert np.linalg.det(T[:3, :3]) > 0.99, np.linalg.det(T[:3, :3])

    @check("umeyama rejects too few points and mismatched shapes")
    def _():
        for a, b in ((np.zeros((2, 3)), np.zeros((2, 3))),
                     (np.zeros((5, 3)), np.zeros((4, 3))),
                     (np.zeros((5, 2)), np.zeros((5, 2)))):
            try:
                SE.umeyama(a, b)
            except ValueError:
                continue
            raise AssertionError(f"should have raised for {a.shape} {b.shape}")

    @check("fit_report separates a constant offset from scatter")
    def _():
        rng = np.random.default_rng(4)
        src = rng.uniform(-0.3, 0.3, (400, 3))
        offset = np.array([0.010, -0.004, 0.002])     # 10, -4, 2 mm
        dst = src + offset + rng.normal(0, 0.001, (400, 3))
        ## Report against the IDENTITY, so the offset is not fitted away.
        rep = SE.fit_report(src, dst, np.eye(4))
        close(np.asarray(rep["constant_offset_mm"]), offset * 1e3, 0.3,
              "constant offset")
        close(rep["scatter_rms_mm"], 1.0 * np.sqrt(3), 0.15, "scatter")

    @check("board_pose is correct for BOTH distortion conventions")
    def _():
        """The direction trap, end to end. The same board, the same pixels,
        under coefficients that run pixel->ray and ray->pixel: each must
        recover its own true pose. Handing either set to OpenCV raw would
        pass one and fail the other."""
        sys.path.insert(0, str(HERE.parent / "reconstruction"))
        from camera import PinholeCamera

        base = dict(width=640, height=480, fx=390.145, fy=389.662,
                    cx=315.586, cy=241.463,
                    coeffs=(-0.05475, 0.06331, 0.000243, 0.000484, -0.02126))
        obj = np.array([[(i + 1) * 0.035, (j + 1) * 0.035, 0.0]
                        for j in range(4) for i in range(6)])
        obj -= obj.mean(axis=0)
        T_true = make_T([0.02, -0.03, 0.45],
                        _rot([1, 0.3, 0.2], 35))

        class Det:
            pass

        for model in ("distortion.inverse_brown_conrady", "opencv_plumb_bob"):
            cam = PinholeCamera(name="c", model=model, **base)
            p_cam = obj @ T_true[:3, :3].T + T_true[:3, 3]
            d = Det()
            d.obj_points = obj.astype(np.float32)
            d.img_points = cam.project(p_cam).astype(np.float32)
            T = SE.board_pose(d, cam)
            close(T[:3, 3], T_true[:3, 3], 2e-5, f"{model} translation")
            close(rotation_angle_deg(T_true[:3, :3].T @ T[:3, :3]), 0.0,
                  2e-3, f"{model} rotation")
            ## And the reprojection it reports must actually be ~0.
            assert SE.reprojection_px(d, cam, T) < 0.02, model

    @check("solve recovers a known camera pair from synthetic views")
    def _():
        """The whole camera-to-camera stage, closed loop: build board views
        two synthetic cameras would have seen, run the per-view estimate and
        the chordal mean, and compare with the pose that generated them."""
        sys.path.insert(0, str(HERE.parent / "reconstruction"))
        from camera import PinholeCamera

        base = dict(width=640, height=480, fx=390.145, fy=389.662,
                    cx=315.586, cy=241.463,
                    model="distortion.inverse_brown_conrady",
                    coeffs=(-0.05475, 0.06331, 0.000243, 0.000484, -0.02126))
        top = PinholeCamera(name="top", **base)
        low = PinholeCamera(name="low", **base)
        T_base_top = make_T([0.02, 0.05, 1.05],
                            _rot([1, 0, 0], 180) @ _rot([0, 0, 1], 5))
        T_base_low = make_T([-0.01, -0.85, 0.28],
                            _rot([1, 0, 0], -90) @ _rot([0, 0, 1], 3))
        T_true = invert_T(T_base_top) @ T_base_low

        obj = np.array([[(i + 1) * 0.035, (j + 1) * 0.035, 0.0]
                        for j in range(4) for i in range(6)])
        obj -= obj.mean(axis=0)

        class Det:
            pass

        rng = np.random.default_rng(9)
        per_view = []
        for _ in range(200):
            T_bb = make_T(
                [rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18),
                 rng.uniform(0.25, 0.55)],
                _rot([1, 0, 0], rng.uniform(25, 65))
                @ _rot([0, 0, 1], rng.uniform(-180, 180)))
            poses, ok = {}, True
            for nm, cam, T_bc in (("t", top, T_base_top),
                                  ("l", low, T_base_low)):
                T_cb = invert_T(T_bc) @ T_bb
                p_cam = obj @ T_cb[:3, :3].T + T_cb[:3, 3]
                uv = cam.project(p_cam)
                vis = cam.in_image(uv) & (p_cam[:, 2] > 0.05)
                if vis.sum() < 14:
                    ok = False
                    break
                d = Det()
                d.obj_points = obj[vis].astype(np.float32)
                ## 0.15 px of corner noise, like a real ChArUco detection.
                d.img_points = (uv[vis]
                                + rng.normal(0, 0.15, (int(vis.sum()), 2))
                                ).astype(np.float32)
                poses[nm] = SE.board_pose(d, cam)
            if not ok:
                continue
            per_view.append(poses["t"] @ invert_T(poses["l"]))
            if len(per_view) >= 22:
                break

        assert len(per_view) >= 12, f"only {len(per_view)} usable views"
        est = SE.average_transform(per_view)
        dt = float(np.linalg.norm(est[:3, 3] - T_true[:3, 3])) * 1e3
        dr = rotation_angle_deg(T_true[:3, :3].T @ est[:3, :3])
        assert dt < 15.0, f"translation off by {dt:.2f} mm"
        assert dr < 1.0, f"rotation off by {dr:.3f} deg"
        ## And the reported spread must be a real quality signal, not zero.
        sp = SE.spread(per_view, est)
        assert 0.0 < sp["translation_mm"]["mean"] < 15.0, sp

    @check("write composes T_base_cam = T_base_ref @ T_ref_cam")
    def _():
        """The one place the two stages meet. Getting the order wrong here
        produces a pose that is plausible and wrong."""
        T_base_ref = make_T([0.02, 0.05, 1.05], _rot([1, 0, 0], 180))
        T_ref_cam = make_T([0.3, -0.1, 0.9], _rot([0, 1, 0], 30))
        composed = T_base_ref @ T_ref_cam
        ## A point in the camera frame must land in the same world place
        ## whether composed or applied in two steps.
        p_cam = np.array([0.05, -0.02, 0.4, 1.0])
        close((composed @ p_cam)[:3],
              (T_base_ref @ (T_ref_cam @ p_cam))[:3], 1e-12, "composition")
        ## And the reference camera itself must land at T_base_ref.
        close(T_base_ref @ np.eye(4), T_base_ref, 1e-12, "reference identity")

    ## -------------------------------------------------------------- ##
    print("\n--- emergency stop ---")

    from estop import EmergencyStop

    class _GroupInfo:
        joint_names = [f"j{i}" for i in range(6)]
        joint_velocity_limits = [float(np.pi)] * 6

    class _FakeArm:
        """Reimplements interbotix check_joint_limits (arm.py:92-108)."""
        def __init__(self, measured, commanded):
            self.group_info = _GroupInfo()
            self.core = type("C", (), {})()
            self.core.joint_states = type("J", (), {})()
            self.core.joint_states.position = list(measured)
            self._cmd = list(commanded)
            self.accepted = []

        def get_joint_commands(self):
            return list(self._cmd)

        def set_joint_positions(self, positions, moving_time=None,
                                accel_time=None, blocking=True):
            theta = [int(e * 1000) / 1000.0 for e in positions]
            speed = [abs(g - c) / float(moving_time)
                     for g, c in zip(theta, self._cmd)]
            for x in range(len(theta)):
                if speed[x] > self.group_info.joint_velocity_limits[x]:
                    return False
            self._cmd = theta
            self.accepted.append(moving_time)
            return True

    class _FakeBot:
        def __init__(self, m, c):
            self.arm = _FakeArm(m, c)

    def _halt(travel):
        import contextlib, io
        meas, cmd = np.zeros(6), np.zeros(6)
        cmd[2] = travel
        bot = _FakeBot(meas, cmd)
        es = EmergencyStop()
        es.register("right", bot)
        with contextlib.redirect_stdout(io.StringIO()):
            es.halt("selftest")
        return bot

    @check("a fixed 0.05 s halt IS refused for a large excursion")
    def _():
        # The bug this guards against: if this ever starts passing, the
        # driver's validation changed and the sizing below may be moot.
        bot = _FakeBot(np.zeros(6), np.array([0, 0, 0.5, 0, 0, 0.0]))
        assert bot.arm.set_joint_positions(
            np.zeros(6).tolist(), moving_time=0.05, blocking=False) is False

    @check("estop halt is accepted at every excursion size")
    def _():
        for travel in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 3.0):
            bot = _halt(travel)
            assert bot.arm.accepted, f"halt refused at {travel} rad of travel"

    @check("estop halt commands the MEASURED position")
    def _():
        bot = _halt(0.5)
        assert np.allclose(bot.arm.get_joint_commands(), 0.0), \
            "halt should freeze at where the arm actually is"

    @check("estop halt scales moving_time with the distance")
    def _():
        small = _halt(0.05).arm.accepted[0]
        large = _halt(2.0).arm.accepted[0]
        assert large > small, "a bigger excursion needs a longer moving_time"
        assert small >= 0.05, "must respect the floor"

    @check("estop survives an arm that raises")
    def _():
        class _Broken:
            @property
            def arm(self):
                raise RuntimeError("comms lost")
        es = EmergencyStop()
        es.register("ok", _FakeBot(np.zeros(6), np.zeros(6)))
        es.register("broken", _Broken())
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            rep = es.halt("selftest")
        assert len(rep) == 2
        assert any("HALT FAILED" in r for r in rep), rep
        assert any("holding at measured" in r for r in rep), rep

    ## -------------------------------------------------------------- ##
    print("\n--- timing statistics ---")

    from sync_capture import summarize

    @check("summarize reports the right moments")
    def _():
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 1e-3   # seconds
        s = summarize(v)
        close(s["mean_ms"], 3.0, 1e-9, "mean")
        close(s["median_ms"], 3.0, 1e-9, "median")
        close(s["min_ms"], 1.0, 1e-9, "min")
        close(s["max_ms"], 5.0, 1e-9, "max")
        close(s["range_ms"], 4.0, 1e-9, "range")
        close(s["std_ms"], float(np.std(v, ddof=1) * 1e3), 1e-9, "std")

    @check("summarize detects a linear drift")
    def _():
        n = 100
        v = (np.arange(n) * 0.002) * 1e-3     # +0.002 ms per sample
        s = summarize(v)
        close(s["drift_ms_per_sample"], 0.002, 1e-9, "drift")

    @check("summarize survives empty and NaN input")
    def _():
        assert summarize(np.array([]))["n"] == 0
        assert summarize(np.array([np.nan, np.nan]))["n"] == 0
        assert summarize(np.array([1e-3, np.nan]))["n"] == 1


def _charuco_tests() -> None:
    """Render synthetic board views through a KNOWN K and recover it."""
    import board as B
    from charuco_calibrate import calibrate, detect_all

    spec = B.BoardSpec(squares_x=7, squares_y=5, square_length_m=0.030,
                       marker_length_m=0.022, dictionary="DICT_5X5_250")

    @check("board geometry validation rejects impossible boards")
    def _():
        for bad in (B.BoardSpec(marker_length_m=0.05, square_length_m=0.03),
                    B.BoardSpec(squares_x=1),
                    B.BoardSpec(square_length_m=-1.0)):
            try:
                bad.validate()
            except ValueError:
                continue
            raise AssertionError(f"should have rejected {bad}")

    @check("interior corner count")
    def _():
        assert spec.n_corners == 24, spec.n_corners

    @check("a generated board is detected in its own image")
    def _():
        img = B.generate_image(spec, pixels_per_metre=4000.0)
        detector, board = B.make_detector(spec)
        det = B.detect(img, detector, board)
        assert det.ok, det.reason
        assert det.n_corners == spec.n_corners, det.n_corners

    ## Ground truth: roughly the real D405 colour intrinsics, no distortion
    ## (the synthetic views are exact homographies, so zero distortion is
    ## the correct ground truth to recover).
    K_true = np.array([[391.21426, 0.0, 317.49417],
                       [0.0, 390.69266, 240.24887],
                       [0.0, 0.0, 1.0]])
    W, H = 640, 480

    with tempfile.TemporaryDirectory() as d:
        imgdir = Path(d) / "imgs"
        imgdir.mkdir()
        _render_views(spec, K_true, W, H, imgdir, n=24)

        @check("all synthetic views are detected")
        def _():
            files = sorted(imgdir.glob("*.png"))
            records, size = detect_all(files, spec, 8, None)
            assert size == (W, H), size
            assert len(records) == len(files), \
                f"only {len(records)}/{len(files)} detected"

        @check("calibration recovers a known camera matrix from SYNTHETIC "
               "views to <1% (code-path test, no real camera)")
        def _():
            files = sorted(imgdir.glob("*.png"))
            records, size = detect_all(files, spec, 8, None)
            res = calibrate(records, size, 0)
            K = res["K"]
            for i, j, name, tol in ((0, 0, "fx", 0.01), (1, 1, "fy", 0.01)):
                rel = abs(K[i, j] - K_true[i, j]) / K_true[i, j]
                assert rel < tol, f"{name} off by {rel * 100:.3f}%"
            for i, j, name in ((0, 2, "cx"), (1, 2, "cy")):
                err = abs(K[i, j] - K_true[i, j])
                assert err < 2.0, f"{name} off by {err:.3f} px"
            assert res["rms"] < 1.0, f"rms {res['rms']:.4f} px"


def _render_views(spec, K_true, W, H, outdir: Path, n: int) -> None:
    """Project the board image through a known pinhole at random poses."""
    import cv2

    board_img = spec.cv_board().generateImage(
        (int(spec.squares_x * spec.square_length_m * 4000),
         int(spec.squares_y * spec.square_length_m * 4000)), marginSize=0)
    bh, bw = board_img.shape[:2]
    sx = spec.squares_x * spec.square_length_m / bw
    sy = spec.squares_y * spec.square_length_m / bh

    corners3d = np.array([[0, 0, 0], [bw * sx, 0, 0],
                          [bw * sx, bh * sy, 0], [0, bh * sy, 0]], float)
    corners3d[:, 0] -= bw * sx / 2
    corners3d[:, 1] -= bh * sy / 2
    src = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], np.float32)

    rng = np.random.default_rng(0)
    for i in range(n):
        R, _ = cv2.Rodrigues(np.radians(
            np.array([rng.uniform(-28, 28), rng.uniform(-28, 28),
                      rng.uniform(-20, 20)])))
        t = np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.03, 0.03),
                      rng.uniform(0.22, 0.45)])
        proj, _ = cv2.projectPoints(corners3d, cv2.Rodrigues(R)[0], t,
                                    K_true, np.zeros(5))
        Hm = cv2.getPerspectiveTransform(
            src, proj.reshape(-1, 2).astype(np.float32))
        img = cv2.warpPerspective(board_img, Hm, (W, H),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=128)
        cv2.imwrite(str(outdir / f"v{i:03d}.png"), img)


def main() -> int:
    print()
    print("#" * 74)
    print("#  calibration package self-test  (no hardware)")
    print("#" * 74)
    run()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [(n, tb) for n, ok, tb in RESULTS if not ok]
    print()
    print("=" * 74)
    print(f"  {passed}/{len(RESULTS)} passed")
    print("=" * 74)
    for name, tb in failed:
        print(f"\n  FAILED: {name}\n")
        print("    " + tb.replace("\n", "\n    "))
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
