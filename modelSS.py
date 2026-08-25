"""A small from-scratch encoder-decoder for semantic segmentation."""

import torch
import torch.nn as nn  # For defining neural network layers


class modelSS(nn.Module):
    """Convolutional encoder-decoder producing per-pixel class logits.

    Three stride-2 convolutions downsample by 8x, a bottleneck convolution
    widens to 512 channels, and three transpose convolutions upsample back to
    the input resolution. Every convolution except the final one is followed by
    batch normalisation and ReLU. There are no skip connections.

    Args:
        num_classes: Number of output channels, one logit per class.
    """

    def __init__(self, num_classes: int = 21) -> None:
        """Build the encoder, bottleneck and decoder layers."""
        super().__init__()

        #conv layers - encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        #bottleneck
        self.bottleneck = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn_bottleneck = nn.BatchNorm2d(512)

        #deconv - decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_deconv1 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_deconv2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a ``(N, 3, H, W)`` image batch to ``(N, num_classes, H, W)`` logits."""
        #encoder
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        #bottleneck
        x = self.relu(self.bn_bottleneck(self.bottleneck(x)))

        #decoder
        x = self.relu(self.bn_deconv1(self.deconv1(x)))
        x = self.relu(self.bn_deconv2(self.deconv2(x)))
        x = self.deconv3(x)  # Output segmentation map

        return x
