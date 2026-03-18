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
import matplotlib.pyplot as plt
from torchvision.transforms import Resize

# Default parameters
save_file = 'modelSS_weights.pth'
n_epochs = 30
batch_size = 64
learning_rate = 1e-4
adam_decay = 1e-3
plot_file = 'plot.png'

def transform(image, target):
    resize = Resize((256, 256))
    
    #eesize both image and target
    image = resize(image)
    target = resize(target)
    
    #convert image to tensor and normalize
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    #convert target to tensor
    target = torch.as_tensor(np.array(target), dtype=torch.int64)
    
    return image, target

def compute_mIoU(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    target = target.squeeze(0).cpu().numpy()

    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()

        if union == 0:
            ious.append(float('nan'))

        else: 
            ious.append(intersection/union)

    return np.nanmean(ious)

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, save_file=None, plot_file=None):
    print('Starting training...')
    model.train()

    train_losses = []
    miou_scores = []

    for epoch in range(1, n_epochs + 1):
        print(f'Epoch [{epoch}/{n_epochs}]')

        epoch_loss = 0.0
        epoch_miou = 0.0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            #forward pass
            outputs = model(images)
            #calc loss
            loss = loss_fn(outputs, labels)

            #compute mIoU
            batch_miou = 0.0
            for i in range(images.size(0)):
                batch_miou += compute_mIoU(outputs[i].unsqueeze(0), labels[i].unsqueeze(0), num_classes=21)
            miou = batch_miou / images.size(0)
            epoch_miou += miou

            #bring gradients to 0 
            optimizer.zero_grad()
            #back propagation
            loss.backward()
            #update weights
            optimizer.step()

            epoch_loss += loss.item()

        if scheduler:
            scheduler.step(epoch_loss)

        train_losses.append(epoch_loss / len(train_loader))
        miou_scores.append(epoch_miou / len(train_loader))

        #for tracking while running - time, epoch and training loss outputted
        print(f'{datetime.datetime.now()} Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}, mIoU: {epoch_miou / len(train_loader):.4f}')

        '''print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, epoch_loss/len(train_loader)))'''
        
        #save weights after each epoch
        if save_file:
            torch.save(model.state_dict(), save_file)

        # Update and save plot after each epoch
        if plot_file:
            #loss plot
            plt.figure(figsize=(10, 5))
            plt.clf()
            plt.plot(train_losses, label='Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Training Metrics')
            print(f'Saving loss plot to {plot_file}')
            plt.savefig(f"loss_{plot_file}")

            #mIoU plot
            plt.figure(figsize=(10, 5))
            plt.clf()
            plt.plot(miou_scores, label='Mean mIoU', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Mean mIoU')
            plt.title('Training Mean mIoU')
            plt.legend()
            print(f'Saving mIoU plot to {plot_file}')
            plt.savefig(f"mIoU_{plot_file}")


def main():
    global save_file, n_epochs, batch_size, learning_rate, plot_file

    print('running main ...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #model load
    model = modelSS(num_classes=21)
    model.to(device)


    #load dataset
    train_dataset = VOCSegmentation(
    root='./data', year='2012', image_set='train', download=True, 
    transforms=lambda img, tgt: transform(img, tgt))

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-w', metavar='weights', type=str, help='Path to save weights file (.pth)', default=save_file)
    argParser.add_argument('-e', metavar='epochs', type=int, help='Number of epochs', default=n_epochs)
    argParser.add_argument('-b', metavar='batch_size', type=int, help='Batch size', default=batch_size)
    argParser.add_argument('-p', metavar='plot', type=str, help='Path to save loss plot (.png)', default=plot_file)
    args = argParser.parse_args()

    args = argParser.parse_args()

    if args.w != None:
        save_file = args.w
    if args.e != None:
        n_epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.p != None:
        plot_file = args.p

    #print vars
    '''print(f'\t\tn epochs = {n_epochs}')
    print(f'\t\tbatch size = {batch_size}')
    print(f'\t\tlearning rate = {learning_rate}')
    print(f'\t\tsave file = {save_file}')
    print(f'\t\tplot file = {plot_file}')'''

    #define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #define optimizer, loss, and learning rate scheduler

    '''optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-2,
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)'''

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
        scheduler=scheduler,
        device=device,
        save_file=save_file,
        plot_file=plot_file
    )


if __name__ == "__main__":
    main()
