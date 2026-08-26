import numpy as np
import torch
from PIL import Image

from modelSS_train import VOID_LABEL, SegmentationTransform


def make_pair(w=256, h=256):
    """Left half is red / class 1, right half is blue / class 2."""
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, : w // 2] = (255, 0, 0)
    rgb[:, w // 2:] = (0, 0, 255)

    m = np.zeros((h, w), dtype=np.uint8)
    m[:, : w // 2] = 1
    m[:, w // 2:] = 2

    return Image.fromarray(rgb, 'RGB'), Image.fromarray(m, mode='P')


def test_no_augment_is_deterministic():
    t = SegmentationTransform(augment=False)
    image, target = make_pair()

    a_img, a_tgt = t(image, target)
    b_img, b_tgt = t(image, target)

    assert torch.equal(a_img, b_img)
    assert torch.equal(a_tgt, b_tgt)


def test_augment_preserves_shape_and_labels():
    t = SegmentationTransform(augment=True)
    image, target = make_pair()

    torch.manual_seed(0)
    for _ in range(10):
        img, tgt = t(image, target)
        assert img.shape == (3, 256, 256)
        assert tgt.shape == (256, 256)
        # nearest-neighbour resampling must never invent a label
        assert set(torch.unique(tgt).tolist()) <= {0, 1, 2, VOID_LABEL}


def test_flip_keeps_image_and_mask_aligned():
    """With scale fixed at 1.0 only flip/crop apply, so alignment is exact."""
    t = SegmentationTransform(augment=True, scale_range=(1.0, 1.0))
    image, target = make_pair()

    torch.manual_seed(0)
    for _ in range(20):
        img, tgt = t(image, target)

        # undo ImageNet normalisation to recover the red/blue channels
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb = img * std + mean

        red_px = rgb[0] > rgb[2]   # red channel dominates
        blue_px = rgb[2] > rgb[0]  # blue channel dominates

        # wherever the image is red the mask must say class 1, and blue -> 2
        assert torch.all(tgt[red_px] == 1)
        assert torch.all(tgt[blue_px] == 2)


def test_downscale_pads_mask_with_void():
    """A <1.0 scale must pad the mask with the ignore label, not class 0."""
    t = SegmentationTransform(augment=True, scale_range=(0.5, 0.5))
    image, target = make_pair()

    torch.manual_seed(0)
    saw_void = False
    for _ in range(10):
        _, tgt = t(image, target)
        if (tgt == VOID_LABEL).any():
            saw_void = True
            break

    assert saw_void, 'downscaling should leave void-labelled padding'


def test_val_transform_never_augments():
    t = SegmentationTransform(augment=False)
    image, target = make_pair()

    _, tgt = t(image, target)

    # untouched left/right split: no flip, no crop, no void padding
    assert tgt[0, 0].item() == 1
    assert tgt[0, -1].item() == 2
    assert not (tgt == VOID_LABEL).any()
