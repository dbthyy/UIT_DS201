import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()

        # Convolution with 5x5 kernel + 2 padding: 28x28x6
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=2
        )
        # Pool with 2x2 average kernel + 2 stride: 14x14x6
        self.pool1 = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=0
        )

        # Convolution with 5x5 kernel: 10x10x16
        self.conv2 = nn.Conv2d(
            in_channels=6, 
            out_channels=16, 
            kernel_size=5, 
            padding=0
        ) 
        # Pool with 2x2 average kernel + 2 stride: 5x5x16
        self.pool2 = nn.AvgPool2d(
            kernel_size=2, 
            stride=2,
            padding=0
            )
        
        # Dense: 120 fully connected neurons
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            padding=0
        )
        # Dense: 84 fully connected neurons
        self.fc = nn.Linear(
            in_features=120,
            out_features=84
        )
        # Output layer: 10 fully connected neurons
        self.output = nn.Linear(
            in_features=84,
            out_features=num_classes
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (batch_size, 28, 28) -> (batch_size, 1, 28, 28)
        # images = images.unsqueeze(1)

        # Layer 1: Conv -> Sigmoid -> Pool
        features = F.sigmoid(self.conv1(images))      # (batch_size, 6, 28, 28)
        features = self.pool1(features)               # (batch_size, 6, 14, 14)

        # Layer 2: Conv -> Sigmoid -> Pool
        features = F.sigmoid(self.conv2(features))    # (batch_size, 16, 10, 10)
        features = self.pool2(features)               # (batch_size, 16, 5, 5)

        # Layer 3: Conv -> Sigmoid -> Flatten
        features = F.sigmoid(self.conv3(features))    # (batch_size, 120, 1, 1)
        features = features.view(-1, 120)            # (batch_size, 120)

        # Layer 4: Dense -> Sigmoid -> Output
        features = F.sigmoid(self.fc(features))       # (batch_size, 84)
        outputs = self.output(features)               # (batch_size, 10)

        return outputs