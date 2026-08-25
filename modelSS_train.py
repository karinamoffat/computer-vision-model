import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import functional as F
from modelSS import modelSS
import numpy as np
import argparse
import datetime
import random
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, InterpolationMode

# Default parameters
save_file = 'modelSS_weights.pth'
n_epochs = 30
batch_size = 64
learning_rate = 1e-4
adam_decay = 1e-3
plot_file = 'plot.png'
num_workers = 0
seed = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#module-level callable so DataLoader workers can pickle it under Windows spawn
class SegmentationTransform:
    def __init__(self, size=(256, 256)):
        self.resize = Resize(size)
        self.resize_mask = Resize(size, interpolation=InterpolationMode.NEAREST)

    def __call__(self, image, target):
        #resize both image and target
        image = self.resize(image)
        target = self.resize_mask(target)

        #convert image to tensor and normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        #convert target to tensor
        target = torch.as_tensor(np.array(target), dtype=torch.int64)

        return image, target

def update_confusion_matrix(conf_mat, outputs, target, num_classes):
    #accumulate a num_classes x num_classes matrix on-device, void pixels excluded
    pred = torch.argmax(outputs, dim=1)
    valid = target != 255
    idx = target[valid] * num_classes + pred[valid]
    conf_mat += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf_mat

def iou_from_confusion_matrix(conf_mat):
    #IoU per class = TP / (TP + FP + FN); classes absent from both pred and target are nan
    conf_mat = conf_mat.double()
    intersection = torch.diag(conf_mat)
    union = conf_mat.sum(dim=0) + conf_mat.sum(dim=1) - intersection
    ious = torch.where(union > 0, intersection / union, torch.full_like(union, float('nan')))
    return ious

def mean_iou(conf_mat):
    ious = iou_from_confusion_matrix(conf_mat)
    return torch.nanmean(ious).item()

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

def evaluate(model, loader, loss_fn, device, num_classes=21):
    #run one pass over loader without gradients; returns (mean loss, mIoU, per-class IoU)
    model.eval()
    total_loss = 0.0
    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item()
            conf_mat = update_confusion_matrix(conf_mat, outputs, labels, num_classes)

    return total_loss / len(loader), mean_iou(conf_mat), iou_from_confusion_matrix(conf_mat)

def print_per_class_iou(ious):
    print('  per-class IoU:')
    for name, iou in zip(VOC_CLASSES, ious.tolist()):
        shown = 'n/a' if iou != iou else f'{iou:.4f}'
        print(f'    {name:<14s} {shown}')

def save_plots(plot_file, train_losses, val_losses, train_mious, val_mious):
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

def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device, save_file=None, plot_file=None, num_classes=21):
    print('Starting training...')

    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []
    best_val_miou = float('-inf')
    best_epoch = None

    for epoch in range(1, n_epochs + 1):
        print(f'Epoch [{epoch}/{n_epochs}]')

        model.train()
        epoch_loss = 0.0
        conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            #forward pass
            outputs = model(images)
            #calc loss
            loss = loss_fn(outputs, labels)

            #bring gradients to 0
            optimizer.zero_grad()
            #back propagation
            loss.backward()
            #update weights
            optimizer.step()

            epoch_loss += loss.item()

            #accumulate train confusion matrix (no grad needed for the metric)
            with torch.no_grad():
                conf_mat = update_confusion_matrix(conf_mat, outputs, labels, num_classes)

        train_loss = epoch_loss / len(train_loader)
        train_miou = mean_iou(conf_mat)

        #validation pass
        val_loss, val_miou, val_class_ious = evaluate(model, val_loader, loss_fn, device, num_classes)

        #scheduler steps on mean val loss, not the summed train loss
        if scheduler:
            scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mious.append(train_miou)
        val_mious.append(val_miou)

        #for tracking while running - time, epoch, and both train and val metrics
        print(f'{datetime.datetime.now()} Epoch {epoch}, '
              f'train loss: {train_loss:.4f}, train mIoU: {train_miou:.4f}, '
              f'val loss: {val_loss:.4f}, val mIoU: {val_miou:.4f}')
        print_per_class_iou(val_class_ious)

        #checkpoint on best val mIoU rather than overwriting every epoch
        if save_file and val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            torch.save(model.state_dict(), save_file)
            print(f'  new best val mIoU, saved checkpoint to {save_file}')

        # Update and save plots after each epoch
        if plot_file:
            print(f'Saving plots to loss_{plot_file} and mIoU_{plot_file}')
            save_plots(plot_file, train_losses, val_losses, train_mious, val_mious)

    if best_epoch is not None:
        print(f'Best val mIoU {best_val_miou:.4f} at epoch {best_epoch} (checkpoint: {save_file})')


def main():
    global save_file, n_epochs, batch_size, learning_rate, adam_decay, plot_file, num_workers, seed

    print('running main ...')

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-w', metavar='weights', type=str, help='Path to save weights file (.pth)', default=save_file)
    argParser.add_argument('-e', metavar='epochs', type=int, help='Number of epochs', default=n_epochs)
    argParser.add_argument('-b', metavar='batch_size', type=int, help='Batch size', default=batch_size)
    argParser.add_argument('-p', metavar='plot', type=str, help='Path to save loss plot (.png)', default=plot_file)
    argParser.add_argument('--lr', metavar='learning_rate', type=float, help='Learning rate', default=learning_rate)
    argParser.add_argument('--weight-decay', metavar='weight_decay', type=float, help='Adam weight decay', default=adam_decay)
    argParser.add_argument('--num-workers', metavar='num_workers', type=int, help='DataLoader worker processes', default=num_workers)
    argParser.add_argument('--seed', type=int, help='Random seed', default=seed)
    args = argParser.parse_args()

    #every flag has a default, so these are always populated
    save_file = args.w
    n_epochs = args.e
    batch_size = args.b
    plot_file = args.p
    learning_rate = args.lr
    adam_decay = args.weight_decay
    num_workers = args.num_workers
    seed = args.seed

    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #model load
    model = modelSS(num_classes=21)
    model.to(device)


    #load datasets - train and val splits
    data_transform = SegmentationTransform()
    train_dataset = VOCSegmentation(
        root='./data', year='2012', image_set='train', download=True,
        transforms=data_transform)
    val_dataset = VOCSegmentation(
        root='./data', year='2012', image_set='val', download=True,
        transforms=data_transform)
    print(f'train images: {len(train_dataset)}, val images: {len(val_dataset)}')

    #define data loaders
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    #define optimizer, loss, and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=adam_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    # Train the model
    train(
        n_epochs=n_epochs,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        save_file=save_file,
        plot_file=plot_file
    )


if __name__ == "__main__":
    main()
