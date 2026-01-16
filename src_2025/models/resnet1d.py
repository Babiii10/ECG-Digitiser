#!/usr/bin/env python3
"""
ResNet-1D architecture for ECG classification - PhysioNet Challenge 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """1D Residual Block for ECG signals"""

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, downsample=None):
        super(ResBlock1D, self).__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """ResNet-1D for ECG classification"""

    def __init__(self,
                 num_leads=12,
                 num_classes=1,
                 initial_filters=64,
                 num_blocks=[2, 2, 2, 2],
                 kernel_size=7,
                 dropout=0.3):
        """
        Initialize ResNet-1D

        Args:
            num_leads: Number of ECG leads (input channels)
            num_classes: Number of output classes
            initial_filters: Number of filters in first layer
            num_blocks: Number of residual blocks in each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super(ResNet1D, self).__init__()

        self.in_channels = initial_filters

        # Initial convolution
        self.conv1 = nn.Conv1d(num_leads, initial_filters, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(initial_filters, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(initial_filters * 2, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(initial_filters * 4, num_blocks[2], kernel_size, stride=2)
        self.layer4 = self._make_layer(initial_filters * 8, num_blocks[3], kernel_size, stride=2)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(initial_filters * 8, num_classes)

    def _make_layer(self, out_channels, num_blocks, kernel_size, stride):
        """Create a layer with multiple residual blocks"""
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(ResBlock1D(self.in_channels, out_channels, kernel_size, stride, downsample))

        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResBlock1D(out_channels, out_channels, kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, num_leads, seq_length)

        Returns:
            Output tensor (batch_size, num_classes)
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classifier
        x = self.dropout(x)
        x = self.fc(x)

        return x


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""

    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()

        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        batch, channels, _ = x.size()

        # Squeeze: Global average pooling
        squeeze = F.adaptive_avg_pool1d(x, 1).view(batch, channels)

        # Excitation: FC layers with ReLU and Sigmoid
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(batch, channels, 1)

        # Scale
        return x * excitation.expand_as(x)


class SEResBlock1D(nn.Module):
    """Residual block with Squeeze-and-Excitation"""

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, downsample=None, se_reduction=16):
        super(SEResBlock1D, self).__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock1D(out_channels, se_reduction)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE block
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEResNet1D(nn.Module):
    """SE-ResNet-1D with squeeze-and-excitation blocks"""

    def __init__(self,
                 num_leads=12,
                 num_classes=1,
                 initial_filters=64,
                 num_blocks=[2, 2, 2, 2],
                 kernel_size=7,
                 dropout=0.3,
                 se_reduction=16):
        super(SEResNet1D, self).__init__()

        self.in_channels = initial_filters

        # Initial convolution
        self.conv1 = nn.Conv1d(num_leads, initial_filters, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers with SE blocks
        self.layer1 = self._make_layer(initial_filters, num_blocks[0], kernel_size, 1, se_reduction)
        self.layer2 = self._make_layer(initial_filters * 2, num_blocks[1], kernel_size, 2, se_reduction)
        self.layer3 = self._make_layer(initial_filters * 4, num_blocks[2], kernel_size, 2, se_reduction)
        self.layer4 = self._make_layer(initial_filters * 8, num_blocks[3], kernel_size, 2, se_reduction)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(initial_filters * 8, num_classes)

    def _make_layer(self, out_channels, num_blocks, kernel_size, stride, se_reduction):
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(SEResBlock1D(self.in_channels, out_channels, kernel_size, stride, downsample, se_reduction))

        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(SEResBlock1D(out_channels, out_channels, kernel_size, se_reduction=se_reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classifier
        x = self.dropout(x)
        x = self.fc(x)

        return x


def create_resnet1d_small():
    """Create small ResNet-1D for quick experimentation"""
    return ResNet1D(
        num_leads=12,
        num_classes=1,
        initial_filters=32,
        num_blocks=[1, 1, 1, 1],
        kernel_size=7,
        dropout=0.3
    )


def create_resnet1d_medium():
    """Create medium ResNet-1D (recommended)"""
    return ResNet1D(
        num_leads=12,
        num_classes=1,
        initial_filters=64,
        num_blocks=[2, 2, 2, 2],
        kernel_size=7,
        dropout=0.3
    )


def create_resnet1d_large():
    """Create large ResNet-1D for high performance"""
    return ResNet1D(
        num_leads=12,
        num_classes=1,
        initial_filters=64,
        num_blocks=[3, 4, 6, 3],
        kernel_size=7,
        dropout=0.4
    )


def create_seresnet1d():
    """Create SE-ResNet-1D with attention"""
    return SEResNet1D(
        num_leads=12,
        num_classes=1,
        initial_filters=64,
        num_blocks=[2, 2, 2, 2],
        kernel_size=7,
        dropout=0.3,
        se_reduction=16
    )


if __name__ == "__main__":
    # Test models
    print("Testing ResNet-1D models...")

    batch_size = 4
    num_leads = 12
    seq_length = 5000

    # Create dummy input
    x = torch.randn(batch_size, num_leads, seq_length)

    # Test small model
    model_small = create_resnet1d_small()
    output = model_small(x)
    print(f"Small ResNet-1D output shape: {output.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in model_small.parameters())
    print(f"Small ResNet-1D parameters: {num_params:,}")

    # Test medium model
    model_medium = create_resnet1d_medium()
    output = model_medium(x)
    print(f"Medium ResNet-1D output shape: {output.shape}")

    num_params = sum(p.numel() for p in model_medium.parameters())
    print(f"Medium ResNet-1D parameters: {num_params:,}")

    # Test SE-ResNet
    model_se = create_seresnet1d()
    output = model_se(x)
    print(f"SE-ResNet-1D output shape: {output.shape}")

    num_params = sum(p.numel() for p in model_se.parameters())
    print(f"SE-ResNet-1D parameters: {num_params:,}")

    print("✓ All models working correctly!")
