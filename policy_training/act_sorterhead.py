"""ACT with an auxiliary block->hole head for the shape sorter, dropped at inference.

THE EXPERIMENT
==============
`act_objhead` asked whether the flower policy improves when its backbone is
made to predict WHERE the object is, as a training signal only.  The shape
sorter asks a harder question.  Three pieces sit on the table and the task
string says which one to insert; a pixel-only ACT has no way to know which,
so pooled training is ill-posed (see SHAPE_SORTER.md).  The sorter head gives
the backbone that binding pressure without giving the policy the answer:

    predict (dx, dy, found) for the piece THIS episode is about,
    from the top_scene feature map, with a separate output per piece.

Which output is trained comes from `task_index`.  On a triangle episode only
the triangle output sees a gradient, and it has to fire on the triangle while
the cube and flower are also in frame -- so the shared features have to
encode identity, not just "a blob is here".  At inference the head is never
evaluated; `select_action` is inherited unchanged.

WHERE THE LABEL COMES FROM
==========================
`observation.environment_state` as built by build_envstate_dataset.py
--scene shape_sorter:

    [obj_cx, obj_cy, obj_found, obj_size, hole_cx, hole_cy,
     obj_to_hole_dx, obj_to_hole_dy]

Indices 6, 7 and 2.  The vector rides in the batch but is NOT in
`config.input_features`, so it is a target, never an input.  This makes the
three arms of the ablation share one dataset and one label:

    train_real.py   --env-state            the vector as an INPUT
    train_sorterhead.py --sorter-head-weight 1   the vector as a TARGET
    train_sorterhead.py --sorter-head-weight 0   same class, gradient off

The box is fixed (hole_cx/cy std 0.035 vs dx/dy std ~0.13 on the 2026-09-07
data), so dx,dy is the piece position up to a constant.  The offset framing
buys nothing geometrically; the conditioning is what carries the experiment.

READING THE DIAGNOSTIC
======================
lerobot sends the loss dict to wandb only, so the policy prints its own line
every 100 steps.  The number to watch is offset MAE against the target's own
spread: dx std 0.145, dy std 0.112 in normalized image units.  A head that
only predicts the dataset mean sits at MAE ~0.8 x std; anything well below
that is localizing.  Per-piece MAE is printed too -- if the three are equal
and large, the head found "a blob" and not the piece.

`found` is 62% of frames on this data.  The offset loss is masked to frames
where the detector saw the piece; the found logit is trained on all frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

## Import order matters: the factory module must be imported before the
## configs package, or lerobot's draccus choice registration races itself.
import lerobot.policies.factory as _factory  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

ENV_KEY = "observation.environment_state"
TASK_KEY = "task_index"
## Layout of the shape_sorter env vector; see build_envstate_dataset.py.
IDX_FOUND, IDX_DX, IDX_DY = 2, 6, 7
PIECE_NAMES = ("cube", "triangle", "flower")   # task_index order in tasks.parquet


@PreTrainedConfig.register_subclass("act_sorterhead")
@dataclass
class ACTSorterHeadConfig(ACTConfig):
    sorter_head_weight: float = 1.0
    sorter_head_camera: str = "observation.images.top_scene"
    sorter_head_n_pieces: int = 3
    ## Width of the 1x1 reducer applied to the backbone map before the
    ## per-piece linear read-out.  15x20 cells x 32 ch = 9600 inputs per head.
    sorter_head_channels: int = 32
    ## Relative weight of the found-BCE against the offset Huber.
    sorter_head_found_weight: float = 1.0

    @property
    def type(self) -> str:
        return "act_sorterhead"


class ACTSorterHeadPolicy(ACTPolicy):
    config_class = ACTSorterHeadConfig
    name = "act_sorterhead"

    def __init__(self, config: ACTSorterHeadConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config: ACTSorterHeadConfig = config

        if config.sorter_head_camera not in config.input_features:
            raise ValueError(
                f"sorter_head_camera {config.sorter_head_camera!r} is not a "
                f"policy input; cameras are {sorted(config.image_features)}")
        if ENV_KEY in config.input_features:
            raise ValueError(
                f"{ENV_KEY} is in input_features. The head's label must not "
                f"also be a policy input -- drop --env-state.")
        self._cam_index = list(config.image_features).index(
            config.sorter_head_camera)

        feat_ch = self.model.encoder_img_feat_input_proj.in_channels
        self.sorter_reduce = nn.Conv2d(feat_ch, config.sorter_head_channels, 1)
        ## The read-out must exist BEFORE lerobot builds the optimizer from
        ## policy.parameters(), or it silently never trains.  resnet18 has
        ## stride 32, so the map is (H//32, W//32) of the configured camera;
        ## the first forward asserts this held.
        _, h, w = config.input_features[config.sorter_head_camera].shape
        self._map_hw = (h // 32, w // 32)
        n_in = config.sorter_head_channels * self._map_hw[0] * self._map_hw[1]
        self.sorter_heads = nn.ModuleList(
            nn.Linear(n_in, 3) for _ in range(config.sorter_head_n_pieces))
        self._checked_shape = False

        self._captured: list[Tensor] = []
        self._capturing = False
        self._step = 0
        self._report_every = 100
        self.model.backbone.register_forward_hook(self._grab_feature_map)

    def _grab_feature_map(self, _module, _inputs, output):
        if self._capturing:
            self._captured.append(output["feature_map"])

    def _check_shape(self, reduced: Tensor):
        got = tuple(reduced.shape[-2:])
        if got != self._map_hw:
            raise RuntimeError(
                f"feature map is {got}, but the read-out was sized for "
                f"{self._map_hw} from the configured camera resolution")
        print(f"[sorter_head] read-out: {reduced.shape[1]}x{got[0]}x{got[1]} "
              f"-> 3 outputs x {self.config.sorter_head_n_pieces} pieces, "
              f"weight {self.config.sorter_head_weight}", flush=True)
        self._checked_shape = True

    def _labels(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        if ENV_KEY not in batch:
            raise KeyError(
                f"{ENV_KEY} is not in the batch. Train on the "
                f"shape_sorter_envstate tree built by build_envstate_dataset.py.")
        if TASK_KEY not in batch:
            raise KeyError(f"{TASK_KEY} is not in the batch; cannot condition "
                           f"the head on the target piece.")
        env = batch[ENV_KEY]
        if env.ndim == 3:            # (B, T, 8) if a delta window was requested
            env = env[:, -1]
        piece = batch[TASK_KEY].long().reshape(-1)
        target = env[:, [IDX_DX, IDX_DY]].float()
        found = (env[:, IDX_FOUND] > 0.5)
        return piece, target, found

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        self._captured = []
        self._capturing = True
        try:
            loss, loss_dict = super().forward(batch)
        finally:
            self._capturing = False

        if len(self._captured) <= self._cam_index:
            raise RuntimeError(
                f"backbone produced {len(self._captured)} feature maps, need "
                f"index {self._cam_index} for {self.config.sorter_head_camera}")
        feat = self._captured[self._cam_index]
        self._captured = []

        reduced = F.relu(self.sorter_reduce(feat))
        if not self._checked_shape:
            self._check_shape(reduced)
        flat = reduced.flatten(1)
        ## One read-out per piece, then gather the target piece's row.
        all_out = torch.stack([h(flat) for h in self.sorter_heads], dim=1)  # (B, P, 3)
        piece, target, found = self._labels(batch)
        piece = piece.clamp_(0, self.config.sorter_head_n_pieces - 1)
        out = all_out[torch.arange(all_out.shape[0], device=all_out.device), piece]

        found_loss = F.binary_cross_entropy_with_logits(
            out[:, 2], found.to(out.dtype))
        if found.any():
            off_loss = F.smooth_l1_loss(out[found, :2], target[found], beta=0.05)
        else:
            off_loss = out.new_zeros(())
        head_loss = off_loss + self.config.sorter_head_found_weight * found_loss

        with torch.no_grad():
            mae = (out[:, :2] - target).abs()
            loss_dict["sorter_head_loss"] = head_loss.item()
            loss_dict["sorter_off_mae"] = float(mae[found].mean()) if found.any() else float("nan")
            loss_dict["sorter_found_acc"] = float(((out[:, 2] > 0) == found).float().mean())
            loss_dict["sorter_found_rate"] = float(found.float().mean())
        if self.config.sorter_head_weight:
            loss = loss + self.config.sorter_head_weight * head_loss

        self._step += 1
        if self._step % self._report_every == 0:
            self._report(out, target, found, piece, off_loss, found_loss)
        return loss, loss_dict

    @torch.no_grad()
    def _report(self, out, target, found, piece, off_loss, found_loss):
        parts = []
        for p, name in enumerate(PIECE_NAMES[: self.config.sorter_head_n_pieces]):
            m = found & (piece == p)
            if m.any():
                e = (out[m, :2] - target[m]).abs().mean(0)
                parts.append(f"{name} {e[0]:.3f}/{e[1]:.3f}")
            else:
                parts.append(f"{name} --")
        acc = float(((out[:, 2] > 0) == found).float().mean())
        ## Head weight norm: drifts when the aux gradient is on, stays at its
        ## init value to the last digit when weight is 0.  That is how you
        ## can SEE the control arm is really a control.
        wnorm = float(self.sorter_heads[0].weight.norm())
        print(f"[sorter_head] step~{self._step} off {off_loss.item():.4f} "
              f"found-bce {found_loss.item():.3f} acc {acc:.2f} "
              f"seen {100 * float(found.float().mean()):.0f}% | MAE dx/dy "
              f"{' · '.join(parts)} (std .145/.112) | w0-norm {wnorm:.5f} "
              f"(weight {self.config.sorter_head_weight})", flush=True)


## lerobot's factory dispatches on a hardcoded if/elif chain, so a registered
## config alone is not enough to make the policy loadable.
_ORIGINAL_GET_POLICY_CLASS = _factory.get_policy_class


def _get_policy_class(name: str):
    if name == "act_sorterhead":
        return ACTSorterHeadPolicy
    return _ORIGINAL_GET_POLICY_CLASS(name)


_factory.get_policy_class = _get_policy_class
