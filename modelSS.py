"""Segmentation architectures: a from-scratch encoder-decoder and two variants.

Three architectures share one interface so they can be compared under identical
training conditions:

``baseline``
    The original encoder-decoder. No skip connections, trained from scratch.
``unet``
    The same encoder and decoder, with skip connections concatenating encoder
    features into the matching decoder stage.
``resnet18``
    A pretrained ImageNet ResNet-18 encoder with a U-Net style decoder.

Use :func:`build_model` to construct one by name.
"""

import torch
import torch.nn as nn  # For defining neural network layers
from torchvision.models import ResNet18_Weights, resnet18

ARCHITECTURES = ('baseline', 'unet', 'resnet18')


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


class modelSSUNet(nn.Module):
    """The baseline encoder-decoder with U-Net skip connections.

    Identical layer widths to :class:`modelSS`, so a difference in validation
    mIoU is attributable to the skips rather than to capacity. Each decoder
    stage concatenates the encoder feature map at the matching resolution,
    which widens the following transpose convolution's input.

    Args:
        num_classes: Number of output channels, one logit per class.
    """

    def __init__(self, num_classes: int = 21) -> None:
        """Build the encoder, bottleneck and skip-aware decoder."""
        super().__init__()

        #encoder - same widths as the baseline
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        #bottleneck
        self.bottleneck = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn_bottleneck = nn.BatchNorm2d(512)

        #decoder - in_channels grow where an encoder map is concatenated
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_deconv1 = nn.BatchNorm2d(256)

        #256 from deconv1 + 128 from conv2
        self.deconv2 = nn.ConvTranspose2d(256 + 128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_deconv2 = nn.BatchNorm2d(128)

        #128 from deconv2 + 64 from conv1
        self.deconv3 = nn.ConvTranspose2d(128 + 64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a ``(N, 3, H, W)`` image batch to ``(N, num_classes, H, W)`` logits."""
        #encoder, keeping the maps that feed the skips
        s1 = self.relu(self.bn1(self.conv1(x)))    # H/2,  64ch
        s2 = self.relu(self.bn2(self.conv2(s1)))   # H/4, 128ch
        x = self.relu(self.bn3(self.conv3(s2)))    # H/8, 256ch

        #bottleneck
        x = self.relu(self.bn_bottleneck(self.bottleneck(x)))

        #decoder, concatenating the matching encoder map at each step
        x = self.relu(self.bn_deconv1(self.deconv1(x)))       # H/4, 256ch
        x = torch.cat([x, s2], dim=1)                         # H/4, 384ch
        x = self.relu(self.bn_deconv2(self.deconv2(x)))       # H/2, 128ch
        x = torch.cat([x, s1], dim=1)                         # H/2, 192ch
        x = self.deconv3(x)                                   # H,   num_classes

        return x


class modelSSResNet18(nn.Module):
    """A pretrained ResNet-18 encoder with a U-Net style decoder.

    The encoder downsamples by 32x, so the decoder has five upsampling stages
    rather than three. Inputs are expected to carry the usual ImageNet
    normalisation, which the training transform already applies.

    Args:
        num_classes: Number of output channels, one logit per class.
        pretrained: Load ImageNet weights. Set False for offline use and tests.
    """

    def __init__(self, num_classes: int = 21, pretrained: bool = True) -> None:
        """Build the ResNet-18 encoder stages and the decoder."""
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        #encoder stages, split so their outputs can feed skips
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # H/2,   64ch
        self.pool = backbone.maxpool                                            # H/4
        self.layer1 = backbone.layer1                                           # H/4,   64ch
        self.layer2 = backbone.layer2                                           # H/8,  128ch
        self.layer3 = backbone.layer3                                           # H/16, 256ch
        self.layer4 = backbone.layer4                                           # H/32, 512ch

        self.up1 = self._up(512, 256)
        self.up2 = self._up(256 + 256, 128)
        self.up3 = self._up(128 + 128, 64)
        self.up4 = self._up(64 + 64, 64)
        self.up5 = nn.ConvTranspose2d(64 + 64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

    @staticmethod
    def _up(in_channels: int, out_channels: int) -> nn.Sequential:
        """One upsampling stage: transpose conv, batch norm, ReLU."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a ``(N, 3, H, W)`` image batch to ``(N, num_classes, H, W)`` logits."""
        s0 = self.stem(x)              # H/2,   64ch
        s1 = self.layer1(self.pool(s0))  # H/4,  64ch
        s2 = self.layer2(s1)           # H/8,  128ch
        s3 = self.layer3(s2)           # H/16, 256ch
        x = self.layer4(s3)            # H/32, 512ch

        x = self.up1(x)                          # H/16, 256ch
        x = self.up2(torch.cat([x, s3], dim=1))  # H/8,  128ch
        x = self.up3(torch.cat([x, s2], dim=1))  # H/4,   64ch
        x = self.up4(torch.cat([x, s1], dim=1))  # H/2,   64ch
        x = self.up5(torch.cat([x, s0], dim=1))  # H,    num_classes

        return x


def build_model(arch: str = 'baseline', num_classes: int = 21, pretrained: bool = True) -> nn.Module:
    """Construct one of the architectures in :data:`ARCHITECTURES` by name.

    Args:
        arch: One of ``baseline``, ``unet`` or ``resnet18``.
        num_classes: Number of output channels.
        pretrained: Only meaningful for ``resnet18``; ignored otherwise.

    Returns:
        The constructed model.

    Raises:
        ValueError: If ``arch`` is not a known architecture.
    """
    if arch == 'baseline':
        return modelSS(num_classes=num_classes)
    if arch == 'unet':
        return modelSSUNet(num_classes=num_classes)
    if arch == 'resnet18':
        return modelSSResNet18(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f'unknown arch {arch!r}, expected one of {ARCHITECTURES}')
