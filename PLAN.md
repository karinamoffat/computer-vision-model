# Improvement Plan

Remediation plan for every gap identified in the code review of this repo
(`modelSS.py`, `modelSS_train.py`, `README.md`).

Tiers are ordered first by effort, then by payoff within each tier.
Items marked **(STAR)** are the highest-leverage work in the plan.

---

## Assumptions

- **Retraining is possible.** T1-1 and T2-1 invalidate the current
  `modelSS_weights.pth` and both plots; they require a fresh run. Without GPU
  access, do all code + hygiene work first, label existing numbers as stale,
  and gate retraining to the end.
- **Work happens on a branch** (`fix/segmentation-overhaul`), not `main`.
- **Scope is this repo.** Renaming the repo (T3-6) is a GitHub UI action.

---

## Tier 0 - Prep (must happen first, ~15 min)

| #    | Task                                                                                               | Verify                                                                         |
| ---- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| T0-1 | Create branch; add `requirements.txt` with pinned `torch` / `torchvision` / `matplotlib` / `numpy` | `pip install -r requirements.txt` succeeds; `torch.cuda.is_available()` prints |
| T0-2 | Add `torch.manual_seed` / `np.random.seed` / `random.seed` behind a `--seed` flag                  | Two 1-epoch runs on the same seed give identical loss to 4 decimals            |
| T0-3 | Add smoke test `tests/test_model.py`: one forward pass, assert output shape `(2, 21, 256, 256)`    | `pytest` passes                                                                |

T0-2 and T0-3 come first because they are the verification harness for
everything after them.

---

## Tier 1 - Quick wins: low effort, highest payoff

### T1-1. Wire the BatchNorms into `forward()`; fix the duplicated `bn2`

`modelSS.py:17,34-48` - all five BN layers are constructed and never called,
and line 17 clobbers the 128-channel `bn2`. Rename the 256-channel layer to
`bn3` and apply all five as `relu(bn(conv(x)))`. (~10 lines)

- **Verify:** smoke test passes; a 3-epoch run beats current code's epoch-3 loss.
- **Why:** likely the actual cause of the plateau. Converts the README's
  weakest sentence into a debugging story.

### T1-2. Exclude void (255) pixels from mIoU

`modelSS_train.py:38-55` - the loss uses `ignore_index=255` but the metric does
not, so void pixels inflate the union. Mask both `pred` and `target` with
`target != 255`. (~3 lines)

- **Verify:** on one batch, mIoU is strictly higher than the unmasked value.

### T1-3. Make mask resizing explicit: `InterpolationMode.NEAREST`

`modelSS_train.py:23-27` - currently correct only because Pillow silently
forces NEAREST on `P`-mode images. (~2 lines)

- **Verify:** `np.unique(target)` after transform is a subset of `{0..20, 255}`.

### T1-4. Add `.gitignore` + `LICENSE`; remove the 12 MB `.pth` from the tree

`git rm --cached` the weights, ignore `*.pth` and `data/`, re-attach weights to
a GitHub Release.

- **Verify:** `git ls-files` shows no `.pth`; clone size drops below 100 KB.

### T1-5. Fix the figure leak and the missing legend

`modelSS_train.py:112-132` - `plt.figure()` every epoch with no `plt.close()`;
loss plot sets `label=` but never calls `plt.legend()`. Drop the redundant
`plt.clf()`. (~4 lines)

- **Verify:** after a 5-epoch run, `len(plt.get_fignums()) <= 1`.

### T1-6. Clean up the argparse block

`modelSS_train.py:136,153-169` - `parse_args()` called twice, parsing happens
after the dataset download, `global learning_rate` is never assigned, and
`lr` / `weight_decay` are not exposed. (~15 lines)

- **Verify:** `--help` lists all six params; `-e 1 --lr 1e-3` changes behavior.

### T1-7. Delete the commented-out dead code blocks

`modelSS_train.py:104-105,172-176,183-189`.

- **Verify:** no triple-quoted dead blocks remain.

---

## Tier 2 - High payoff, moderate effort (a focused day)

### T2-1. Add a real validation split (STAR - most important item in this plan)

Every current metric - loss, mIoU, both plots, the README headline numbers - is
measured on training data. Load `image_set='val'`, add an `evaluate()` function
under `model.eval()` + `torch.no_grad()`, plot train and val curves together,
and checkpoint on **best val mIoU** instead of overwriting every epoch
(`modelSS_train.py:108-109`).

- **Verify:** both curves appear in the plots; saved checkpoint's epoch matches
  the argmax of val mIoU; README reports a val number.

### T2-2. Replace per-image mIoU with a vectorized confusion matrix

The current metric averages per-image `nanmean` over classes present in that
image - not comparable to any published VOC number - and runs 64 numpy
round-trips per batch on CPU (`modelSS_train.py:80-83`). Use a single
`torch.bincount`-based 21x21 accumulator on-GPU, IoU computed at epoch end.
Closes the non-standard-metric gap and the performance gap together, and
subsumes T1-2.

- **Verify:** per-class IoU table prints (background high, small classes near
  zero); epoch wall-clock drops substantially.

### T2-3. Write `predict.py` and put a qualitative grid atop the README (STAR)

This is a vision project whose only images are two line charts. Load the
checkpoint, run N val images, save an image / ground-truth / prediction grid
using the VOC palette.

- **Verify:** `results/qualitative.png` is committed and rendered in the README.
- **Why:** best 60 minutes available in this plan.

### T2-4. Rewrite the README around honest results

Val mIoU (not train), the qualitative grid, a baseline comparison line
(torchvision FCN-ResNet50 is approximately 0.66 on VOC2012 val vs. yours), a
per-class IoU table, and a short "what I learned / what's next".

- **Verify:** no training-only metric is presented as _the_ result anywhere.

### T2-5. Make the transform picklable; enable `num_workers` + `pin_memory`

`modelSS_train.py:151,179` - the `transforms=lambda ...` is exactly why workers
cannot be enabled under Windows spawn. Replace with a module-level callable class.

- **Verify:** `--num-workers 4` runs without a pickling error; epoch time drops.

### T2-6. Add CI

GitHub Action running `ruff check` + `pytest` on push.

- **Verify:** green check on the repo front page.

---

## Tier 3 - Low effort, moderate payoff (polish)

| #    | Task                                                                                                                                                            | Verify                                                        |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| T3-1 | Docstrings + type hints on all public functions                                                                                                                 | `ruff` clean with docstring rules on                          |
| T3-2 | Swap `print` for `logging` with a `--log-level` flag                                                                                                            | `--log-level DEBUG` changes verbosity                         |
| T3-3 | `scheduler.step()` should consume mean epoch loss, not the sum (`modelSS_train.py:96`); switch to **val** loss once T2-1 lands                                  | LR reductions log at sane epochs                              |
| T3-4 | Hyperparameter sanity: Adam `weight_decay=1e-3` is aggressive for segmentation (try 1e-4); `batch_size=64` at 256x256 OOMs most consumer GPUs - default to 8-16 | Documented default trains on a 12 GB card                     |
| T3-5 | Add augmentation (random horizontal flip, random scale/crop) applied identically to image and mask                                                              | Val mIoU improves or is unchanged; flipped pairs stay aligned |
| T3-6 | Rename repo `computer-vision-model` to `voc-semantic-segmentation` (GitHub settings)                                                                            | Old URL redirects; README links resolve                       |
| T3-7 | Optional: mixed precision (`torch.amp`) + epoch wall-clock in README                                                                                            | Loss curve matches FP32 within noise; step time drops         |

---

## Tier 4 - High effort, high payoff (the differentiator)

### T4-1. Architecture ablation study (STAR)

Three variants behind a single `--arch` flag, one table:

| Variant                                          | Val mIoU |
| ------------------------------------------------ | -------- |
| From-scratch encoder-decoder (current, BN fixed) | 0.0610   |
| plus U-Net skip connections                      | 0.0649   |
| plus pretrained ResNet-18 encoder                | 0.4315   |

**DONE** (2026-08-25, Colab T4, `--seed 0 -e 30 -b 16 --augment --amp`).
Pretraining is worth 7x; the skips gap (+0.0039) is within single-seed noise and
is reported as uncalled, not as a win. Both from-scratch variants plateaued near
background-only prediction (~0.033 floor).

- **Verify:** all three train from the same script, same val split, same seed;
  table lands in the README.
- **Why:** demonstrates experimental discipline, which architecture code alone
  does not. Also converts the README's "Future Work" list from things not done
  into things measured.

---

## Gap to tier coverage

| Original gap                                         | Tier                    |
| ---------------------------------------------------- | ----------------------- |
| Dead BatchNorms / duplicate `bn2`                    | T1-1                    |
| mIoU counts void pixels                              | T1-2 (subsumed by T2-2) |
| Non-standard mIoU                                    | T2-2                    |
| Implicit mask interpolation                          | T1-3                    |
| **No validation set**                                | T2-1                    |
| No qualitative results                               | T2-3                    |
| No baseline comparison                               | T2-4                    |
| No requirements / gitignore / license / tests / seed | T0-1, T0-3, T1-4        |
| 12 MB `.pth` in git                                  | T1-4                    |
| No CI                                                | T2-6                    |
| Generic repo name                                    | T3-6                    |
| Slow per-sample mIoU loop                            | T2-2                    |
| No dataloader workers / unpicklable lambda           | T2-5                    |
| Globals + duplicated argparse + unexposed LR         | T1-6                    |
| Figure leak + missing legend                         | T1-5                    |
| Commented-out dead code                              | T1-7                    |
| No docstrings / type hints / logging                 | T3-1, T3-2              |
| Scheduler on summed loss                             | T3-3                    |
| Hyperparameters, no augmentation, no AMP             | T3-4, T3-5, T3-7        |
| Pretrained encoder + skip connections                | T4-1                    |

---

## Execution order

**T0 to all of T1 (one commit each) to retrain baseline to T2-1, T2-2, T2-5 to
retrain to T2-3, T2-4 to T2-6 to T3 to T4.**

The two retrain points are the gates: everything before the first is code-only
and safe; everything after depends on real numbers existing.

If GPU time is the constraint: T1 + T2-3 + T2-4 + T2-6 + T3-1 + T3-2 require no
training run at all, and alone fix the README, the hygiene, and the visuals.
