# Semantic Segmentation on Pascal VOC 2012

A from-scratch convolutional encoder–decoder for pixel-wise semantic segmentation,
trained on Pascal VOC 2012 (21 classes). Built to understand segmentation
end-to-end rather than to beat a benchmark — the interesting part is what the
baseline gets wrong and why.

---

## Model

Three architectures share one training script, selected with `--arch`:

| `--arch` | Encoder | Decoder | Params |
| -------- | ------- | ------- | ------ |
| `baseline` | 3 × Conv stride 2, from scratch | 3 × ConvTranspose, no skips | 3.1M |
| `unet` | same as baseline | same widths, plus skip connections | 3.2M |
| `resnet18` | ImageNet-pretrained ResNet-18 | 5 × ConvTranspose, with skips | 13.2M |

All take `(N, 3, 256, 256)` and return `(N, 21, 256, 256)` logits.

`baseline` and `unet` differ by 0.1M parameters, so a gap between them would be
attributable to the skip connections rather than to capacity. In practice the
measured gap was too small to call — see Results. `resnet18` is a much larger
model and is not a controlled comparison; it is there to show how far a
pretrained encoder moves the number.

---

## Results

Trained on the VOC 2012 `train` split (1464 images), evaluated on `val` (1449),
256×256, 30 epochs, Adam `lr=1e-4`, batch 16, flip + scale/crop augmentation,
mixed precision, `--seed 0`. mIoU is a 21-class confusion matrix accumulated
over the full val split, void (255) pixels excluded.

### Architecture ablation

Same script, same val split, same seed — only `--arch` changes:

| Variant | Val mIoU |
| ------- | -------- |
| `baseline` — from-scratch encoder-decoder | 0.0610 |
| `unet` — plus skip connections | 0.0649 |
| `resnet18` — plus pretrained encoder | **0.4315** |
| _reference:_ torchvision FCN-ResNet50 | ≈ 0.66 |

Three things this table says, in order of how much they matter:

**Pretraining is worth 7× here.** `resnet18` scores 0.4315 against the
from-scratch baseline's 0.0610. With 1464 training images for 21 classes — about
70 examples per class — the encoder cannot learn general visual features from
the data available, so importing them is not an optimisation, it is the whole
task.

**The skip connections did not measurably help.** `unet` beats `baseline` by
0.0039. That is a single-seed difference on a model that is barely learning, and
it should not be read as evidence that skips work; it is within the run-to-run
variation you would expect from changing nothing but the seed. Confirming or
refuting it needs several seeds per variant, which has not been run.

**Both from-scratch variants converge to roughly background-only prediction.**
VOC is mostly background, and a model that predicts background everywhere scores
about 0.70 on that one class and 0 on the other twenty — a 21-class mean near
0.033. At 0.0610 and 0.0649 these two are only just above that floor. Their loss
curves had flattened by epoch 30, so this is a plateau rather than an
interrupted run: more epochs at this learning rate would not have rescued them.

### Not comparable to the number this README used to report

An earlier version of this README reported mIoU ≈ 0.30. That figure is not a
better result than the 0.0610 above; it is a different measurement. It was taken
on the **training** split, before the BatchNorm bug below was fixed, and with a
per-image metric that dropped absent classes via `nanmean` — so it averaged over
whichever handful of classes appeared in a batch rather than over all 21, and
every class the model never learned was excluded instead of scoring zero. The
number here counts those failures. It was removed rather than updated because no
honest arithmetic converts one into the other.

Reproduce with:

```bash
for a in baseline unet resnet18; do
  python modelSS_train.py --arch $a --seed 0 -e 30 -b 16 \
    --augment --amp --num-workers 2 -w weights_$a.pth -p $a.png
done
```

`python predict.py --arch resnet18 -w weights_resnet18.pth` writes
`results/qualitative.png` — an image / ground-truth / prediction grid using the
standard VOC palette. `--log-level DEBUG` adds a per-class IoU table each epoch.

---

## What was wrong with the first version

Three bugs worth recording, because they explain the original plateau:

1. **The BatchNorm layers were never called.** All five were constructed in
   `__init__` but `forward()` applied only `conv → relu`. A sixth layer
   (`self.bn2`) was also defined twice, silently discarding the 128-channel one.
2. **The metric counted void pixels.** The loss correctly used
   `ignore_index=255`, but mIoU did not, so VOC's unlabeled boundary pixels
   inflated every union and dragged the score down.
3. **There was no validation split.** Every reported metric was training-set
   performance, which says nothing about generalization.

The metric is now a vectorized 21×21 confusion matrix accumulated on-device,
with IoU computed at epoch end — standard, and much faster than the previous
per-image NumPy loop.

---

## Usage

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

Train (downloads VOC 2012 on first run, ~2 GB):

```bash
python modelSS_train.py -e 30 -b 16 --lr 1e-4 --num-workers 4
```

| Flag | Meaning | Default |
| ---- | ------- | ------- |
| `--arch` | `baseline`, `unet` or `resnet18` | `baseline` |
| `--no-pretrained` | train `resnet18` from scratch | off |
| `-e` | epochs | 30 |
| `-b` | batch size | 16 |
| `-w` | checkpoint path | `modelSS_weights.pth` |
| `-p` | plot filename suffix | `plot.png` |
| `--lr` | learning rate | 1e-4 |
| `--weight-decay` | Adam weight decay | 1e-4 |
| `--num-workers` | dataloader workers | 0 |
| `--augment` | random flip + scale/crop on the train split | off |
| `--amp` | mixed precision (CUDA only) | off |
| `--seed` | random seed | 0 |
| `--log-level` | logging verbosity | `INFO` |

The checkpoint is saved on **best validation mIoU**, not every epoch.
`--log-level DEBUG` adds a per-class IoU table each epoch.

Qualitative results:

```bash
python predict.py -w modelSS_weights.pth -n 6
```

Tests:

```bash
pytest
```

---

## Notes on defaults

Two defaults were changed from the original run. Batch size dropped from 64 to
16, because 64 at 256×256 OOMs most consumer GPUs, and Adam `weight_decay` from
1e-3 to 1e-4, which is the more usual choice for segmentation — 1e-3 was likely
over-regularising a model this small. Pass `-b 64 --weight-decay 1e-3` to
reproduce the original configuration.

Augmentation is off by default and applies to the training split only; the
validation split is never augmented, so val numbers stay comparable across runs.

---

## Next

* Multiple seeds per variant, so the `baseline` vs `unet` gap can be called
  either way instead of left as noise
* Train on the SBD-augmented split (~10k images) — the 1464-image `train` split
  is the binding constraint on the from-scratch variants, not the architecture
* Longer schedules and a learning-rate sweep per variant

---

## License

MIT — see [LICENSE](LICENSE).
