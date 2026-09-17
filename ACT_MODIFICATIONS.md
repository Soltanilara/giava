# ACT modifications — what is stock LeRobot and what is ours

The goal is that `lerobot/` stays **pristine upstream code** (currently v0.6.0) so you can
always read the industry-standard implementation, and everything project-specific is
recorded here.

Our historical changes live in the old lerobot commit `903a61be` ("Pushing lerobot
changes", 2026-07-16), against the pre-v3 `lerobot/common/...` layout. To view any of
them yourself:

```bash
cd lerobot
git show 903a61be -- <path>
```

---

## 1. `configuration_act.py` — hyperparameter experiments (no functional change)

We added **commented-out** alternatives next to the defaults:

```python
# dim_model: int = 256      <- ours, commented
# n_heads: int = 4          <- ours, commented
dim_model: int = 512        <- upstream default, active
n_heads: int = 8            <- upstream default, active
# n_encoder_layers = 2      <- ours, commented
# latent_dim: int = 16      <- ours, commented
```

**Nothing here was ever active.** The values that ran are stock LeRobot defaults, so
there is nothing to port to v0.6.0. These were a record of a smaller-model experiment.

## 2. `lerobot_dataset.py` — debug prints + a library-compat fix

Two unrelated things got mixed together:

- **Debug prints** dumping FPS, timestamp/episode-index shapes, and `episode_data_index`
  boundaries. This was diagnosing a `check_timestamps_sync` failure — episode boundaries
  not lining up with timestamp jumps.
- **A real compat fix**: wrapping `torch.stack(...)` arguments in `list(...)`, e.g.
  `torch.stack(list(self.hf_dataset["timestamp"]))`. Newer versions of the `datasets`
  library return a lazy column object rather than a list, which `torch.stack` rejects.

**Neither needs porting.** The debug prints were temporary, and upstream v0.6.0 already
handles the `datasets` column type correctly.

## 3. `modeling_act copy.py` — the YOLO-feature ACT variant (**this is the real work**)

This is the one substantive modification, and the only thing worth re-validating. It is
a *second copy* of `modeling_act.py`, not an edit of it, so the stock model was never
disturbed.

### What it does

Stock ACT feeds the transformer encoder one set of visual tokens per camera, from a
ResNet backbone. This variant feeds it **two** sets per camera: the stock ResNet tokens
**plus** tokens from a frozen YOLO detector, so the policy gets object-detection-flavored
features alongside generic visual ones.

Concretely:

- Loads a frozen YOLO checkpoint (`best_10epoch.pt`), sets `requires_grad = False` on all
  its parameters and puts it in `eval()` mode — it is a fixed feature extractor, not trained.
- `extract_yolo_features()` registers a forward hook on `yolo.model.model[22]` (a neck
  layer) to grab an intermediate feature map, runs the image through YOLO under
  `torch.no_grad()`, then removes the hook.
- A new `encoder_yolo_feat_input_proj` (`nn.Conv2d(256 -> dim_model, kernel_size=1)`)
  projects those 256-channel features into the transformer's width.
- In the encoder, for each camera image the YOLO features get a positional embedding
  (reusing `encoder_cam_feat_pos_embed`), are projected, rearranged to
  `(sequence, batch, dim)`, and appended to `encoder_in_tokens` / `encoder_in_pos_embed`
  after the stock camera tokens.

Net effect: the encoder input sequence roughly **doubles in length** for the visual part.

### Things to check when you validate it

These are observations from reading the code, not confirmed bugs — worth resolving before
you trust results:

1. **Dead pooling.** `self.yolo_pool = nn.AdaptiveAvgPool2d((4,4))` is re-assigned and
   applied to `yolo_features` *after* the camera loop, but the pooled result is never
   used — the tokens were already appended unpooled inside the loop. So the intended
   4x4 downsampling of YOLO features likely never took effect, and the sequence is much
   longer than intended.
2. **Duplicate projection layer.** `encoder_yolo_feat_input_proj` is created in `__init__`
   and then created *again* inside the encoder path, which would discard the trained
   weights on every call in that code path.
3. **Hardcoded checkpoint path.** The YOLO path points at the old
   `lerobot/lerobot/common/policies/act/` tree, which no longer exists in v0.6.0.
4. **Leftover debug print** guarded by `_printed_yolo_shape`.
5. **Normalization**: no `observation.images` normalization is applied before YOLO sees
   the image, so YOLO receives ACT-normalized input rather than what it was trained on.

### Porting it to v0.6.0

Not yet done. The recommended approach is **not** to copy a modified `modeling_act.py`
into `lerobot/` again, but to subclass the stock ACT model from a file in *our* repo.
That keeps `lerobot/` untouched and makes the diff between stock and ours self-evident.

---

## Convention going forward

- **Never edit files under `lerobot/`.** Treat it as a read-only vendored library.
- Project-specific model code lives in our repo and subclasses or wraps the stock classes.
- Mark every project-specific line with a `# CHANGED:` or `# GIAVA:` comment explaining
  *why*, the way the data-collection scripts already do.
- Record any new deviation in this file.
