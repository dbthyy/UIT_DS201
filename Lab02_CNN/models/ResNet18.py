import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.bn_1 = nn.BatchNorm2d(out_channels)

        self.conv3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.conv1x1 = nn.Identity()
        if in_channels != out_channels:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
                )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv3x3_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv3x3_2(out)
        out = self.bn_2(out)

        identity = self.conv1x1(x)
        out += identity
        out = self.relu(out)

        return(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=21, in_channels=3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)
        self.layer4 = self._make_layer(256, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)     # 224x224 → 112x112
        x = self.maxpool(x)  # 112x112 → 55x55

        # conv2_18
        x = self.layer1(x)
        x = self.maxpool(x)  # 55x55 → 27x27

        # conv3_18
        x = self.layer2(x)
        x = self.maxpool(x)  # 27x27 → 13x13

        # conv4_18
        x = self.layer3(x)
        x = self.maxpool(x)  # 13x13 → 6x6

        # conv5_18
        x = self.layer4(x)

        # Classifier
        x = self.avgpool(x)  # 6x6 → 1x1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, in_channels, out_channels):
        # APPLY FOR TWO BLOCKS - RESNET-18 ONLY
        return nn.Sequential(
            ResnetBlock(in_channels, out_channels, strides=1),
            ResnetBlock(out_channels, out_channels, strides=1)
        )