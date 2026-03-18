import torch  # For tensor operations
import torch.nn as nn  # For defining neural network layers
import torch.nn.functional as F  # For using activation functions, etc.

class modelSS(nn.Module):
    def __init__(self, num_classes=21):
        super(modelSS, self).__init__()

        #conv layers - encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

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

    def forward(self, x):
        #encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        #bottleneck
        x = self.relu(self.bottleneck(x))
        
        #decoder
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)  # Output segmentation map
        
        return x
