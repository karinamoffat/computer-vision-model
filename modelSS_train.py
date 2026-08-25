"""Train the modelSS encoder-decoder on Pascal VOC 2012 semantic segmentation."""

import argparse
import logging
import random
import time
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode, RandomCrop, Resize
from torchvision.transforms import functional as F

from modelSS import modelSS

log = logging.getLogger(__name__)

# Default parameters
save_file = 'modelSS_weights.pth'
n_epochs = 30
batch_size = 16
learning_rate = 1e-4
adam_decay = 1e-4
plot_file = 'plot.png'
num_workers = 0
seed = 0
log_level = 'INFO'

NUM_CLASSES = 21
VOID_LABEL = 255

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#module-level callable so DataLoader workers can pickle it under Windows spawn
class SegmentationTransform:
    """Resize an (image, mask) pair to a fixed size, optionally augmenting first.

    Augmentation is train-only: random horizontal flip plus a random rescale
    followed by a crop back to ``size``. Both operations are applied to the
    image and the mask identically, the mask always with nearest-neighbour
    interpolation so class ids are never blended. Padding introduced by a
    downscale fills the mask with the void label, which the loss ignores.
    """

    def __init__(
        self,
        size: tuple[int, int] = (256, 256),
        augment: bool = False,
        scale_range: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        """Store the target size and build the two resize ops."""
        self.size = size
        self.augment = augment
        self.scale_range = scale_range
        self.resize = Resize(size)
        self.resize_mask = Resize(size, interpolation=InterpolationMode.NEAREST)

    def _augment(self, image, target):
        """Apply flip + random-scale-crop to image and mask in lockstep."""
        #torch RNG (not `random`) so DataLoader workers get distinct streams
        if torch.rand(1).item() < 0.5:
            image = F.hflip(image)
            target = F.hflip(target)

        out_h, out_w = self.size
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        scaled = [int(round(out_h * scale)), int(round(out_w * scale))]
        image = F.resize(image, scaled)
        target = F.resize(target, scaled, interpolation=InterpolationMode.NEAREST)

        #pad when the rescale left us smaller than the crop; mask pads with void
        pad_h = max(0, out_h - scaled[0])
        pad_w = max(0, out_w - scaled[1])
        if pad_h or pad_w:
            image = F.pad(image, [0, 0, pad_w, pad_h], fill=0)
            target = F.pad(target, [0, 0, pad_w, pad_h], fill=VOID_LABEL)

        i, j, h, w = RandomCrop.get_params(image, output_size=self.size)
        return F.crop(image, i, j, h, w), F.crop(target, i, j, h, w)

    def __call__(self, image, target) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform a PIL (image, mask) pair into normalised tensors."""
        if self.augment:
            image, target = self._augment(image, target)
        else:
            image = self.resize(image)
            target = self.resize_mask(target)

        #convert image to tensor and normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        #convert target to tensor
        target = torch.as_tensor(np.array(target), dtype=torch.int64)

        return image, target


def update_confusion_matrix(
    conf_mat: torch.Tensor,
    outputs: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Accumulate predictions into a ``num_classes`` square matrix, void excluded."""
    pred = torch.argmax(outputs, dim=1)
    valid = target != VOID_LABEL
    idx = target[valid] * num_classes + pred[valid]
    conf_mat += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf_mat


def iou_from_confusion_matrix(conf_mat: torch.Tensor) -> torch.Tensor:
    """Per-class IoU = TP / (TP + FP + FN). Classes absent from pred and target are nan."""
    conf_mat = conf_mat.double()
    intersection = torch.diag(conf_mat)
    union = conf_mat.sum(dim=0) + conf_mat.sum(dim=1) - intersection
    return torch.where(union > 0, intersection / union, torch.full_like(union, float('nan')))


def mean_iou(conf_mat: torch.Tensor) -> float:
    """Mean IoU over the classes actually present, ignoring nan entries."""
    return torch.nanmean(iou_from_confusion_matrix(conf_mat)).item()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    use_amp: bool = False,
) -> tuple[float, float, torch.Tensor]:
    """Run one no-grad pass over ``loader``.

    Returns:
        Mean loss, mean IoU, and the per-class IoU tensor.
    """
    model.eval()
    total_loss = 0.0
    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            conf_mat = update_confusion_matrix(conf_mat, outputs.float(), labels, num_classes)

    return total_loss / len(loader), mean_iou(conf_mat), iou_from_confusion_matrix(conf_mat)


def log_per_class_iou(ious: torch.Tensor) -> None:
    """Log a per-class IoU table at DEBUG level."""
    if not log.isEnabledFor(logging.DEBUG):
        return
    log.debug('  per-class IoU:')
    for name, iou in zip(VOC_CLASSES, ious.tolist()):
        shown = 'n/a' if iou != iou else f'{iou:.4f}'
        log.debug('    %-14s %s', name, shown)


def save_plots(
    plot_file: str,
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_mious: Sequence[float],
    val_mious: Sequence[float],
) -> None:
    """Write the train/val loss and mIoU curves next to each other."""
    #loss plot
    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(f"loss_{plot_file}")
    plt.close(fig)

    #mIoU plot
    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_mious, label='Train mIoU', color='green')
    plt.plot(val_mious, label='Val mIoU', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.title('Mean IoU')
    plt.legend()
    plt.savefig(f"mIoU_{plot_file}")
    plt.close(fig)


def train(
    n_epochs: int,
    optimizer: optim.Optimizer,
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scheduler: object | None,
    device: torch.device,
    save_file: str | None = None,
    plot_file: str | None = None,
    num_classes: int = NUM_CLASSES,
    use_amp: bool = False,
) -> list[float]:
    """Train for ``n_epochs``, validating each epoch.

    The checkpoint is written only when validation mIoU improves, and the
    scheduler steps on mean validation loss.

    Returns:
        Validation mIoU per epoch.
    """
    log.info('Starting training...')

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_mious: list[float] = []
    val_mious: list[float] = []
    best_val_miou = float('-inf')
    best_epoch = None
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.perf_counter()

        model.train()
        epoch_loss = 0.0
        conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            #forward pass and loss, under autocast when AMP is on
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            #bring gradients to 0
            optimizer.zero_grad()
            #back propagation, scaled so fp16 gradients do not underflow
            scaler.scale(loss).backward()
            #update weights
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            #accumulate train confusion matrix (no grad needed for the metric)
            with torch.no_grad():
                conf_mat = update_confusion_matrix(conf_mat, outputs.float(), labels, num_classes)

        train_loss = epoch_loss / len(train_loader)
        train_miou = mean_iou(conf_mat)

        #validation pass
        val_loss, val_miou, val_class_ious = evaluate(
            model, val_loader, loss_fn, device, num_classes, use_amp)

        #scheduler steps on mean val loss, not the summed train loss
        if scheduler:
            scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mious.append(train_miou)
        val_mious.append(val_miou)

        elapsed = time.perf_counter() - epoch_start
        log.info(
            'Epoch %d/%d (%.1fs) train loss: %.4f, train mIoU: %.4f, val loss: %.4f, val mIoU: %.4f',
            epoch, n_epochs, elapsed, train_loss, train_miou, val_loss, val_miou,
        )
        log_per_class_iou(val_class_ious)

        #checkpoint on best val mIoU rather than overwriting every epoch
        if save_file and val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            torch.save(model.state_dict(), save_file)
            log.info('  new best val mIoU, saved checkpoint to %s', save_file)

        # Update and save plots after each epoch
        if plot_file:
            log.debug('Saving plots to loss_%s and mIoU_%s', plot_file, plot_file)
            save_plots(plot_file, train_losses, val_losses, train_mious, val_mious)

    if best_epoch is not None:
        log.info('Best val mIoU %.4f at epoch %d (checkpoint: %s)',
                 best_val_miou, best_epoch, save_file)

    return val_mious


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the training CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-w', metavar='weights', type=str, help='Path to save weights file (.pth)', default=save_file)
    parser.add_argument('-e', metavar='epochs', type=int, help='Number of epochs', default=n_epochs)
    parser.add_argument('-b', metavar='batch_size', type=int, help='Batch size', default=batch_size)
    parser.add_argument('-p', metavar='plot', type=str, help='Path to save loss plot (.png)', default=plot_file)
    parser.add_argument('--lr', metavar='learning_rate', type=float, help='Learning rate', default=learning_rate)
    parser.add_argument('--weight-decay', metavar='weight_decay', type=float, help='Adam weight decay', default=adam_decay)
    parser.add_argument('--num-workers', metavar='num_workers', type=int, help='DataLoader worker processes', default=num_workers)
    parser.add_argument('--augment', action='store_true', help='Enable train-time flip and scale/crop augmentation')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision (CUDA only)')
    parser.add_argument('--seed', type=int, help='Random seed', default=seed)
    parser.add_argument('--log-level', type=str, default=log_level,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging verbosity')
    return parser


def main() -> None:
    """Parse arguments, build the datasets and model, and run training."""
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info('Using device: %s', device)

    use_amp = args.amp
    if use_amp and device.type != 'cuda':
        log.warning('--amp requested but no CUDA device is available; continuing in fp32')
        use_amp = False

    #model load
    model = modelSS(num_classes=NUM_CLASSES)
    model.to(device)

    #load datasets - train split augmented, val split never is
    train_dataset = VOCSegmentation(
        root='./data', year='2012', image_set='train', download=True,
        transforms=SegmentationTransform(augment=args.augment))
    val_dataset = VOCSegmentation(
        root='./data', year='2012', image_set='val', download=True,
        transforms=SegmentationTransform(augment=False))
    log.info('train images: %d, val images: %d', len(train_dataset), len(val_dataset))

    #define data loaders
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.b, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)

    #define optimizer, loss, and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.CrossEntropyLoss(ignore_index=VOID_LABEL)

    # Train the model
    train(
        n_epochs=args.e,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        save_file=args.w,
        plot_file=args.p,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    main()
