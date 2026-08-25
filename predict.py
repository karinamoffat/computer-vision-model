"""Render qualitative segmentation results from a trained modelSS checkpoint."""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode, Resize
from torchvision.transforms import functional as F

from modelSS import modelSS
from modelSS_train import NUM_CLASSES, set_seed

log = logging.getLogger(__name__)

# Default parameters
weights_file = 'modelSS_weights.pth'
num_images = 6
output_file = 'results/qualitative.png'
data_root = './data'
seed = 0
log_level = 'INFO'


def voc_cmap(N: int = 256) -> np.ndarray:
    """Build the standard VOC colour palette; entry 255 is the void colour."""
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << (7 - j))
            g = g | (bitget(c, 1) << (7 - j))
            b = b | (bitget(c, 2) << (7 - j))
            c = c >> 3
        cmap[i] = np.array([r, g, b])

    return cmap


def colorize(mask, cmap: np.ndarray) -> np.ndarray:
    """Map a 2D array of class ids (0-20, plus 255 for void) to RGB."""
    return cmap[np.asarray(mask, dtype=np.uint8)]


def preprocess(image, target) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Resize and normalise a PIL pair, also returning an un-normalised RGB copy.

    The display copy is why this cannot be handed to ``VOCSegmentation``'s own
    ``transforms`` hook, which unpacks exactly two values.
    """
    resize = Resize((256, 256))
    resize_mask = Resize((256, 256), interpolation=InterpolationMode.NEAREST)

    #resize both image and target, same as training
    image = resize(image)
    target = resize_mask(target)

    #keep an un-normalized copy of the resized RGB image for display
    display = np.array(image.convert('RGB'))

    #convert image to tensor and normalize
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    #convert target to tensor
    target = torch.as_tensor(np.array(target), dtype=torch.int64)

    return image, target, display


def predict(
    model: torch.nn.Module,
    dataset,
    n: int,
    device: torch.device,
    cmap: np.ndarray,
    output_file: str,
) -> str:
    """Save an image / ground-truth / prediction grid for the first ``n`` samples.

    Returns:
        The path the grid was written to.
    """
    n = min(n, len(dataset))
    cols = ['Image', 'Ground Truth', 'Prediction']

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    #keep indexing uniform when only one row is requested
    axes = np.atleast_2d(axes)

    model.eval()
    with torch.no_grad():
        for row in range(n):
            #dataset is loaded raw (PIL) so preprocess can also hand back a display copy;
            #VOCSegmentation's own transforms hook only unpacks two values
            image, target, display = preprocess(*dataset[row])

            #add batch dim and run the model
            output = model(image.unsqueeze(0).to(device))
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            panels = [display, colorize(target.numpy(), cmap), colorize(pred, cmap)]
            for col in range(3):
                ax = axes[row][col]
                ax.imshow(panels[col])
                ax.axis('off')
                if row == 0:
                    ax.set_title(cols[col])

    plt.tight_layout()

    #make sure the output directory exists
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_file)
    plt.close(fig)
    log.info('Saved qualitative results to %s', output_file)
    return output_file


def main() -> None:
    """Parse arguments, load a checkpoint, and render the qualitative grid."""
    argParser = argparse.ArgumentParser(description=__doc__)
    argParser.add_argument('-w', '--weights', type=str, help='Path to weights file (.pth)', default=weights_file)
    argParser.add_argument('-n', '--num-images', type=int, help='Number of validation images to show', default=num_images)
    argParser.add_argument('-o', '--output', type=str, help='Path to save the results grid (.png)', default=output_file)
    argParser.add_argument('--data-root', type=str, help='Root directory of the VOC dataset', default=data_root)
    argParser.add_argument('--seed', type=int, help='Random seed', default=seed)
    argParser.add_argument('--log-level', type=str, default=log_level,
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging verbosity')
    args = argParser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info('Using device: %s', device)

    #model load
    model = modelSS(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    #load the validation split raw; preprocessing happens in predict()
    val_dataset = VOCSegmentation(
        root=args.data_root, year='2012', image_set='val', download=True)

    predict(model, val_dataset, args.num_images, device, voc_cmap(), args.output)


if __name__ == "__main__":
    main()
