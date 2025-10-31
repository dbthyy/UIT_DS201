import torch
import torch.nn as nn

class BaseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class InceptionBlocks(nn.Module):
    def __init__(self, in_channels, channel1, channel2, channel3, channel4):
        super().__init__()

        self.b1 = nn.Sequential(
            BaseConv2d(in_channels, channel1, kernel_size=1),
        )

        self.b2 = nn.Sequential(
            BaseConv2d(in_channels, channel2[0], kernel_size=1),
            BaseConv2d(channel2[0], channel2[1], kernel_size=3, padding=1),
        )

        self.b3 = nn.Sequential(
            BaseConv2d(in_channels, channel3[0], kernel_size=1),
            BaseConv2d(channel3[0], channel3[1], kernel_size=5, padding=2),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BaseConv2d(in_channels, channel4, kernel_size=1),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)

class GoogleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.main_path = nn.Sequential(
            self.branch1(), 
            self.branch2(), 
            self.branch3(),
            self.branch4(), 
            self.branch5(),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.main_path(x)
        x = self.classifier(x)
        return x

    def branch1(self):
        return nn.Sequential(
            # Conv 7x7/2 --> 112x112x64
            BaseConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            # Maxpool 3x3/2 --> 56x56x64
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        )
    
    def branch2(self):
        return nn.Sequential(
            # Conv 1x1
            BaseConv2d(in_channels=64, out_channels=64, kernel_size=1),
            # Conv 3x3/1 --> 56x56x192
            BaseConv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            # Maxpool 3x3/2 --> 28x28x192
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        )
    
    def branch3(self):
        return nn.Sequential(
            # Inception 3a --> 28x28x256
            InceptionBlocks(192, 64, (96, 128), (16, 32), 32),
            # Inception 3b --> 28x28x480
            InceptionBlocks(256, 128, (128, 192), (32, 96), 64),
            # Maxpool 3x3/2 --> 14x14x480
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
            )
    
    def branch4(self):
        return nn.Sequential(
            # Inception 4a --> 14x14x512
            InceptionBlocks(480, 192, (96, 208), (16, 48), 64),
            # Inception 4b --> 14x14x512
            InceptionBlocks(512, 160, (112, 224), (24, 64), 64),
            # Inception 4c --> 14x14x512
            InceptionBlocks(512, 128, (128, 256), (24, 64), 64),
            # Inception 4d --> 14x14x528
            InceptionBlocks(512, 112, (144, 288), (32, 64), 64),
            # Inception 4e --> 14x14x832
            InceptionBlocks(528, 256, (160, 320), (32, 128), 128),
            # Maxpool 3x3/2 --> 7x7x832
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
            )
    
    def branch5(self):
        return nn.Sequential(
            # Inception 5a --> 7x7x832
            InceptionBlocks(832, 256, (160, 320), (32, 128), 128),
            # Inception 5b --> 7x7x1024
            InceptionBlocks(832, 384, (192, 384), (48, 128), 128),
            # Avgpool - 7x7/1 --> 1x1x1024
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(start_dim=1)
        )