"""ACT with an auxiliary object-detection head that is dropped at inference.

THE EXPERIMENT
==============
Every object-centric variant that beat the RGB baseline on flower transfer
(env-state vector 73.9%, obj_mask as a fourth camera 62.5%, against 39.5%)
feeds the object in as a POLICY INPUT.  That means the blue-blob detector has
to run at rollout, and every detector failure is a policy failure.

This asks the other question: give the encoder the same information as a
TRAINING SIGNAL only, and take it away at test time.  The head predicts where
the object is from the shared visual backbone's feature map; the gradient
flows back into the backbone; `select_action` never touches the head.  A
checkpoint trained this way has byte-identical inference behaviour to the
baseline apart from the backbone weights themselves -- same inputs, same
forward path, same cost.

WHAT IT PREDICTS
================
A 1x1 conv on the backbone's layer4 feature map (15x20 for 480x640 input)
gives one logit per cell: "is the object in this cell".  The label is the
`observation.images.obj_mask` stream that `build_mask_dataset.py` already
renders -- the same detector rule `rollout_policy.py` uses -- max-pooled down
to the feature grid.  Max, not mean: the flower covers a few percent of the
frame, and averaging would wash it below the noise floor.

The mask rides in the dataset but NOT in `config.input_features`, so it never
becomes a policy input.  If the key is missing from the batch this raises
rather than silently training a plain ACT -- a silent no-op would look exactly
like "the auxiliary loss didn't help".

THE BASELINE ARM
================
`--obj-head-weight 0` gives the same class, same architecture, same data path
and the same dataloader order, with the auxiliary gradient switched off.  That
is the honest control for this ablation: the ONLY difference between the two
runs is whether the head's loss reaches the backbone.

WHY IT IS A SEPARATE POLICY TYPE
================================
Registering `act_objhead` rather than quietly wrapping `act` means the
checkpoint says what it is.  At rollout, importing this module registers the
type and the checkpoint loads normally; `select_action` is inherited
unchanged, so the head is present in the state dict and never evaluated.
lerobot's `get_policy_class` is a hardcoded if/elif chain, so it is patched
here rather than in the vendored tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

## Import order matters: the factory module must be imported before the
## configs package, or lerobot's draccus choice registration races itself.
## See the 2026-09-xx import-order segfault note.
import lerobot.policies.factory as _factory  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

OBJ_MASK_KEY = "observation.images.obj_mask"


@PreTrainedConfig.register_subclass("act_objhead")
@dataclass
class ACTObjHeadConfig(ACTConfig):
    """ACT plus the auxiliary head's three knobs.

    `obj_head_camera` must name a key in `input_features`; the head attaches
    to that camera's feature map only.  The backbone is shared across cameras
    in ACT, so supervising one camera still shapes the weights the others use
    -- and top_scene is the only camera whose detector rule subtracts the
    static printed targets, so it is the one whose label is trustworthy.
    """

    obj_head_weight: float = 1.0
    obj_head_camera: str = "observation.images.top_scene"
    ## MEASURED on transfer_flower_merged_mask: the flower is 0.06% of the
    ## top_scene frame, and 0.6% of cells on the 15x20 feature grid -- one to
    ## four cells out of 2400.  Unweighted BCE is therefore minimised by
    ## predicting "no object" everywhere, which would train a head that
    ## teaches the backbone nothing while its loss curve looks excellent.
    ## 150 ~= n_negative / n_positive, so the two classes contribute equally.
    obj_head_pos_weight: float = 150.0
    ## Grow the target by this many cells before the loss.  0 keeps the label
    ## exactly where the detector put it; 1 turns each positive cell into its
    ## 3x3 neighbourhood, which is a softer "roughly here" signal and a
    ## reasonable thing to ablate if the crisp target proves too sparse.
    obj_head_dilate: int = 0

    @property
    def type(self) -> str:
        return "act_objhead"


class ACTObjHeadPolicy(ACTPolicy):
    config_class = ACTObjHeadConfig
    name = "act_objhead"

    def __init__(self, config: ACTObjHeadConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config: ACTObjHeadConfig = config

        if config.obj_head_camera not in config.input_features:
            raise ValueError(
                f"obj_head_camera {config.obj_head_camera!r} is not a policy "
                f"input; cameras are {sorted(config.input_features)}")
        ## Position in the loop ACT.forward runs over batch[OBS_IMAGES],
        ## which it builds as [batch[k] for k in config.image_features].
        self._cam_index = list(config.image_features).index(
            config.obj_head_camera)

        ## in_channels of the projection ACT already applies to the feature
        ## map: the backbone's output width, without hardcoding resnet18's 512.
        feat_ch = self.model.encoder_img_feat_input_proj.in_channels
        self.obj_head = nn.Conv2d(feat_ch, 1, kernel_size=1)

        ## The backbone is ONE module called once per camera, so the hook
        ## fires n_cameras times per forward, in image_features order.
        self._captured: list[Tensor] = []
        self._capturing = False
        self._step = 0
        self._report_every = 100
        self.model.backbone.register_forward_hook(self._grab_feature_map)

    def _grab_feature_map(self, _module, _inputs, output):
        if self._capturing:
            self._captured.append(output["feature_map"])

    def _mask_target(self, batch: dict[str, Tensor], like: Tensor) -> Tensor:
        if OBJ_MASK_KEY not in batch:
            raise KeyError(
                f"{OBJ_MASK_KEY} is not in the batch. Train on a dataset built "
                f"by build_mask_dataset.py, and do not pass obj_mask to "
                f"--cameras (it is the label, not an input).")
        mask = batch[OBJ_MASK_KEY]
        ## Rendered as a 3-channel greyscale video in [0, 1]; one channel is
        ## the whole signal.
        if mask.ndim == 4 and mask.shape[1] > 1:
            mask = mask[:, :1]
        ## Max, not mean: a cell is positive if ANY of its pixels is object.
        target = F.adaptive_max_pool2d(mask.float(), like.shape[-2:])
        target = (target > 0.5).to(dtype=like.dtype)
        if self.config.obj_head_dilate:
            k = 2 * self.config.obj_head_dilate + 1
            target = F.max_pool2d(target, k, stride=1, padding=k // 2)
        return target

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
                f"index {self._cam_index} for {self.config.obj_head_camera}")

        feat = self._captured[self._cam_index]
        self._captured = []

        logits = self.obj_head(feat)
        target = self._mask_target(batch, logits)
        pos_weight = torch.as_tensor(self.config.obj_head_pos_weight,
                                     device=logits.device, dtype=logits.dtype)
        obj_loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight)

        loss_dict["obj_head_loss"] = obj_loss.item()
        ## Reported even at weight 0, so the baseline run logs what the head
        ## WOULD have cost and the two curves are directly comparable.
        recall, pred_rate = _recall(logits, target)
        loss_dict["obj_head_recall"] = recall
        loss_dict["obj_head_pred_rate"] = pred_rate
        loss_dict["obj_head_label_rate"] = float(target.mean())
        if self.config.obj_head_weight:
            loss = loss + self.config.obj_head_weight * obj_loss

        ## lerobot routes the loss dict to WANDB ONLY -- the console line
        ## carries just loss/grad-norm.  Without this, a head that collapsed
        ## to "no object anywhere" would be invisible in a nohup log, and its
        ## BCE would look healthy while it taught the backbone nothing.
        ## Recall is the number to watch: it is what accuracy hides.
        self._step += 1
        if self._step % self._report_every == 0:
            print(f"[obj_head] step~{self._step} bce {obj_loss.item():.4f} "
                  f"recall {recall:.3f} predicted {100 * pred_rate:.2f}% "
                  f"of cells vs label {100 * float(target.mean()):.2f}% "
                  f"(weight {self.config.obj_head_weight})", flush=True)
        return loss, loss_dict


@torch.no_grad()
def _recall(logits: Tensor, target: Tensor) -> tuple[float, float]:
    """(recall, predicted-positive rate).

    Neither number means anything alone, which is why both are reported.
    Accuracy is useless here -- all-negative scores 99.4%.  Recall alone is
    just as useless in the other direction: pos_weight 150 pushes the head
    toward predicting positive, and "everything is object" also scores 1.000.
    A head that is genuinely localizing has recall near 1 AND a predicted
    rate near the label's own ~0.6%.  A rate drifting toward 1.0 means the
    weighting is too aggressive, not that the head is working.
    """
    pos = target > 0.5
    pred = logits > 0
    rate = float(pred.float().mean())
    n = int(pos.sum())
    if n == 0:
        return float("nan"), rate
    return float((pred & pos).sum()) / n, rate


## lerobot's factory dispatches on a hardcoded if/elif chain, so a registered
## config alone is not enough to make the policy loadable.
_ORIGINAL_GET_POLICY_CLASS = _factory.get_policy_class


def _get_policy_class(name: str):
    if name == "act_objhead":
        return ACTObjHeadPolicy
    return _ORIGINAL_GET_POLICY_CLASS(name)


_factory.get_policy_class = _get_policy_class
