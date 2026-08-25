import torch
import numpy as np
import argparse
import random
import os
import matplotlib.pyplot as plt
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import functional as F
from torchvision.transforms import Resize, InterpolationMode
from modelSS import modelSS

# Default parameters
weights_file = 'modelSS_weights.pth'
num_images = 6
output_file = 'results/qualitative.png'
data_root = './data'
seed = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def voc_cmap(N=256):
    #standard VOC palette via bit-shifting, entry 255 is the void colour
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


def colorize(mask, cmap):
    #mask is a 2D array of class ids (0-20) plus 255 for void
    return cmap[np.asarray(mask, dtype=np.uint8)]


def preprocess(image, target):
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


def predict(model, dataset, n, device, cmap, output_file):
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
    print(f'Saved qualitative results to {output_file}')


def main():
    print('running main ...')

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-w', '--weights', type=str, help='Path to weights file (.pth)', default=weights_file)
    argParser.add_argument('-n', '--num-images', type=int, help='Number of validation images to show', default=num_images)
    argParser.add_argument('-o', '--output', type=str, help='Path to save the results grid (.png)', default=output_file)
    argParser.add_argument('--data-root', type=str, help='Root directory of the VOC dataset', default=data_root)
    argParser.add_argument('--seed', type=int, help='Random seed', default=seed)
    args = argParser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #model load
    model = modelSS(num_classes=21)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    #load the validation split raw; preprocessing happens in predict()
    val_dataset = VOCSegmentation(
        root=args.data_root, year='2012', image_set='val', download=True)

    predict(model, val_dataset, args.num_images, device, voc_cmap(), args.output)


if __name__ == "__main__":
    main()
