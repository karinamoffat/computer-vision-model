# Semantic Segmentation on Pascal VOC 2012

A from-scratch convolutional encoder–decoder for pixel-wise semantic segmentation,
trained on Pascal VOC 2012 (21 classes). Built to understand segmentation
end-to-end rather than to beat a benchmark — the interesting part is what the
baseline gets wrong and why.

---

## Model

| Stage | Layers |
| ----- | ------ |
| Encoder | 3 × (Conv 3×3 stride 2 → BatchNorm → ReLU), 3 → 64 → 128 → 256 |
| Bottleneck | Conv 3×3 → BatchNorm → ReLU, 256 → 512 |
| Decoder | 3 × (ConvTranspose 3×3 stride 2 → BatchNorm → ReLU), 512 → 256 → 128 → 21 |

Input `(N, 3, 256, 256)` → output `(N, 21, 256, 256)` logits. No skip
connections and no pretrained weights — deliberately, so later variants have
something to improve on.

---

## Results

> **Status: pending retrain.** The numbers previously reported here
> (train loss ≈ 1.25, mIoU ≈ 0.30) have been removed rather than updated. They
> were invalid in three separate ways: measured on the *training* split, produced
> before the BatchNorm bug below was fixed, and computed with a per-image metric
> that is not comparable to published VOC numbers. Re-running is required before
> any number belongs in this table.

| Metric | Value |
| ------ | ----- |
| Val mIoU | _pending_ |
| Val loss | _pending_ |
| Baseline: torchvision FCN-ResNet50 | ≈ 0.66 val mIoU |

Once trained, `python predict.py` writes `results/qualitative.png` — an
image / ground-truth / prediction grid using the standard VOC palette — and the
training run prints a per-class IoU table each epoch.

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

* U-Net skip connections and a pretrained ResNet-18 encoder, as an ablation
  against the current from-scratch baseline

---

## License

MIT — see [LICENSE](LICENSE).
