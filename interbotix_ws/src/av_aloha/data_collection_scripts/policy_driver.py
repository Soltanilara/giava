"""Run a trained policy as a command source inside data_collection.py.

This is the machinery for human-gated DAgger: the policy drives, and the
moment the operator grips a controller that arm reverts to teleop.  The
correction is recorded as part of an ordinary episode, so the whole existing
pipeline -- gates, dataset writer, verdicts -- is untouched.

WHY A SEPARATE MODULE.  data_collection.py's control loop already carries the
IK solve, three clamps, two collision gates and the recorder.  Policy loading,
observation assembly and the waist frame conversion have nothing to do with
any of that, and putting them inline would make the loop harder to read for a
feature that is off by default.  The loop asks this for a joint vector and
gets one, or None.

WHAT IT DOES NOT DO.  It does not command anything, does not clamp, does not
gate.  It returns the policy's proposed joint positions in DRIVER coordinates
and the loop puts them through exactly the same clamps, capsule gate and table
gate a teleoperated command goes through.  A policy command is not trusted
more than a human one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


class PolicyDriver:
    """A loaded checkpoint, ready to propose joint commands each tick."""

    def __init__(self, checkpoint, mode, arm_names, device="cuda",
                 dataset_root=None, task_string=None, n_action_steps=None):
        ## IMPORT ORDER IS LOAD-BEARING: lerobot.policies.factory MUST be
        ## imported before lerobot.configs or this build segfaults before
        ## either name is bound (see rollout_policy.load_policy).
        from lerobot.policies.factory import (get_policy_class,
                                              make_pre_post_processors)
        from lerobot.configs.policies import PreTrainedConfig
        import torch

        self.torch = torch
        self.device = device
        ckpt = Path(checkpoint)
        cfg = PreTrainedConfig.from_pretrained(str(ckpt))
        if n_action_steps is not None:
            ## Fewer executed actions per prediction = more closed-loop, at
            ## no training cost.  ACT ships n_action_steps == chunk_size,
            ## i.e. fully open-loop for the whole chunk.
            cfg.n_action_steps = int(n_action_steps)
        self.cfg = cfg
        ## PASS THE EDITED CONFIG IN.  from_pretrained(path) alone re-reads
        ## config.json from disk and the n_action_steps override above is
        ## silently lost -- which is exactly what happened on 2026-09-10: a
        ## session that printed "n_action_steps=20" ran the policy at 100.
        self.policy = get_policy_class(cfg.type).from_pretrained(str(ckpt), config=cfg)
        ## And report what the POLICY holds, not what we asked for.
        self.cfg = self.policy.config
        self.policy.to(device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(ckpt))

        self.image_keys = sorted(k for k in cfg.input_features
                                 if k.startswith("observation.images."))
        self.cameras = [k.removeprefix("observation.images.")
                        for k in self.image_keys]
        self.state_dim = int(np.prod(cfg.input_features["observation.state"].shape))
        self.action_dim = int(np.prod(cfg.output_features["action"].shape))
        self.env_key = ("observation.environment_state"
                        if "observation.environment_state" in cfg.input_features
                        else None)

        ## The waist frame the training data was canonicalized into.  A
        ## checkpoint trained on a canon dataset expects middle_base on that
        ## branch; the live servo boots on whichever branch it likes.  Absent
        ## key = never canonicalized = leave the value alone.
        self.waist_ref = None
        self.waist_idx = None
        ## THE RE-CLOCK GAP.  The waist motor was physically re-clocked on
        ## 2026-09-11, so the driver frame the arm reports today is offset from
        ## the one older datasets were recorded in.  A checkpoint speaks its
        ## TRAINING data's frame; the servo speaks the live one.  The gap is
        ##     live theta  -  the theta already baked into this dataset
        ## which is 0 for a dataset shifted by shift_waist_frame.py (its
        ## meta.json records the shift) and the full theta for one recorded
        ## before the rebuild.  Getting this wrong is not a small error: theta
        ## is about a half turn, so an uncompensated old checkpoint would drive
        ## the camera arm ~180 deg from everything it was trained on.
        self.waist_shift = 0.0
        waist_name = "middle_base"
        ds_reclock = 0.0
        if dataset_root is not None:
            try:
                meta = json.loads((Path(dataset_root) / "meta.json").read_text())
                self.waist_ref = meta.get("waist_reference")
                waist_name = meta.get("waist_joint", "middle_base")
                ds_reclock = float(meta.get("waist_reclock", 0.0))
            except Exception:
                pass
            self.waist_idx = _state_index(arm_names, waist_name)
            if self.waist_idx is None:
                self.waist_ref = None
            else:
                self.waist_shift = _live_waist_reclock() - ds_reclock

        self.task_string = task_string
        self.arm_names = list(arm_names)

    # ------------------------------------------------------------------
    def reset(self):
        """Drop the queued action chunk. Call at every episode boundary."""
        self.policy.reset()

    def describe(self):
        bits = [f"{self.cfg.type}", f"cameras={self.cameras}",
                f"state_dim={self.state_dim}",
                f"n_action_steps={getattr(self.cfg, 'n_action_steps', None)}"]
        if self.env_key:
            bits.append("env_state")
        if self.waist_ref is not None:
            bits.append(f"waist->{self.waist_ref:+.3f}")
        if abs(self.waist_shift) > 1e-9:
            bits.append(f"reclock{self.waist_shift:+.3f}")
        if self.task_string:
            bits.append(f"task={self.task_string!r}")
        return "  ".join(bits)

    # ------------------------------------------------------------------
    def act(self, state, frames, env_vec=None):
        """Proposed action for this tick, or None when a camera is missing.

        `state` and the return value are both in DRIVER coordinates and in the
        recorder's layout (per arm, joints then gripper).  Returning None must
        make the caller HOLD -- never command on a stale or partial
        observation.
        """
        torch = self.torch
        state = np.asarray(state, dtype=np.float32)
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"policy expects state_dim={self.state_dim}, got "
                f"{state.shape[0]} -- wrong --mode for this checkpoint?")

        ## THE QUEUE SHORTCUT.  A chunked policy only runs its network when
        ## the action queue is empty (ACT: modeling_act.select_action, "only
        ## calling select_actions when the queue is empty") -- on every other
        ## tick the observation is built, normalized, shipped to the GPU and
        ## then ignored.  Inside a 50 Hz control loop that waste is not free:
        ## measured at ~11 ms/tick, it pushed the loop mean from 16.7 ms to
        ## 22.6 ms and the arm felt sluggish.  So pop the queue directly when
        ## there is something in it and skip the observation entirely.
        ##
        ## Guarded on a private attribute, so any lerobot change just falls
        ## back to the honest path rather than breaking.
        q = getattr(self.policy, "_action_queue", None)
        if (q is not None and len(q) > 0
                and getattr(self.cfg, "temporal_ensemble_coeff", None) is None):
            with torch.inference_mode():
                queued = self.post(q.popleft())
            out = np.asarray(queued.detach().float().cpu().numpy()).reshape(-1)
            if self.waist_idx is not None:
                out[self.waist_idx] = self._action_to_driver(
                    out[self.waist_idx], self._waist_to_dataset(state))
            return out, None

        missing = [c for c in self.cameras if frames.get(c) is None]
        if missing:
            return None, f"no frame from {', '.join(missing)}"

        ## Into the policy's frame; the caller keeps the driver-frame copy.
        s_in = state
        if self.waist_idx is not None:
            s_in = state.copy()
            w = self._waist_to_dataset(state)
            ## Canonicalize only AFTER the re-clock: waist_reference is a
            ## branch in the DATASET's frame, and anchoring the live value to
            ## it before the shift would pick the wrong turn.
            s_in[self.waist_idx] = (
                _canon(w, self.waist_ref) if self.waist_ref is not None else w)

        obs = {"observation.state":
               torch.from_numpy(s_in).unsqueeze(0).to(self.device)}
        for key, cam in zip(self.image_keys, self.cameras):
            ## CONVERT ON THE GPU, NOT THE CPU.  Doing .float().div(255)
            ## before .to(device) builds an 11 MB float tensor per camera on
            ## the CPU and pushes all of it over PCIe; sending uint8 first is
            ## a quarter of the traffic and moves the conversion to hardware
            ## that does it for free.  Measured inside data_collection's 50 Hz
            ## loop: the CPU-side version cost ~19 ms/tick and pushed the loop
            ## mean to 22.6 ms against a 20 ms budget (54% overruns).
            obs[key] = (torch.from_numpy(np.ascontiguousarray(frames[cam]))
                        .to(self.device, non_blocking=True)
                        .permute(2, 0, 1).float().div_(255.0)
                        .unsqueeze(0))
        if self.env_key is not None:
            if env_vec is None:
                return None, f"{self.env_key} required but not supplied"
            obs[self.env_key] = (torch.from_numpy(np.asarray(env_vec, np.float32))
                                 .unsqueeze(0).to(self.device))
        if self.task_string is not None:
            obs["task"] = [self.task_string]

        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(obs)))
        action = np.asarray(action.detach().float().cpu().numpy()).reshape(-1)

        ## Back to the servo's branch: the policy speaks canonical, the motor
        ## speaks driver.  Anchored on the CURRENT measured value so the
        ## command lands on the branch the arm is actually on.
        if self.waist_idx is not None:
            action[self.waist_idx] = self._action_to_driver(
                action[self.waist_idx], self._waist_to_dataset(state))
        return action, None

    # ------------------------------------------------------------------
    def _waist_to_dataset(self, state):
        """Live driver waist -> the frame this checkpoint was trained in."""
        return float(state[self.waist_idx]) - self.waist_shift

    def _action_to_driver(self, value, state_ds):
        """Policy-frame waist action -> a driver command for the live arm.

        Anchored on the measured value (in the dataset's frame) so the command
        lands on the branch the arm is actually on, then shifted back."""
        if self.waist_ref is not None:
            value = _canon(value, state_ds)
        return float(value) + self.waist_shift


def _live_waist_reclock():
    """The live arm's configured waist re-clock [rad], or 0.0.

    Prefers robot_control so there is ONE definition, but policy_driver is
    imported in offline/eval contexts where the ROS import chain is not
    available, so the env var is honoured directly as a fallback."""
    raw = os.environ.get("GIAVA_MIDDLE_WAIST_RECLOCK")
    if raw is not None and str(raw).strip():
        try:
            return float(raw)
        except ValueError:
            pass
    try:
        from robot_control import MIDDLE_WAIST_RECLOCK_RAD
        return float(MIDDLE_WAIST_RECLOCK_RAD)
    except Exception:
        return 0.0


def _canon(value, reference):
    """`value` moved by whole turns onto `reference`'s branch (exact)."""
    two_pi = 2.0 * np.pi
    return float(value - two_pi * round((value - reference) / two_pi))


def _state_index(arm_names, joint_name):
    """Index of `joint_name` in the recorder's concatenated state vector."""
    from arm_config import ARM_CONFIG
    i = 0
    for arm in arm_names:
        for jn in ARM_CONFIG[arm]["joint_names"]:
            if jn == joint_name:
                return i
            i += 1
        if ARM_CONFIG[arm]["has_gripper"]:
            i += 1
    return None


def arm_slices(arm_names):
    """{arm: (slice for joints, gripper index or None)} in recorder layout."""
    from arm_config import ARM_CONFIG
    out, i = {}, 0
    for arm in arm_names:
        n = ARM_CONFIG[arm]["num_joints"]
        g = None
        if ARM_CONFIG[arm]["has_gripper"]:
            g = i + n
        out[arm] = (slice(i, i + n), g)
        i += n + (1 if ARM_CONFIG[arm]["has_gripper"] else 0)
    return out
