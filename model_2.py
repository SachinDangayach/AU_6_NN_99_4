"""
MNIST Digit Classification Model

This module contains the CNN architecture for MNIST digit classification.
The model achieves high accuracy with a compact design of 10,066 parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    MNIST Digit Classification CNN

    Architecture:
    - Input: 28x28x1 (grayscale MNIST images)
    - Output: 10 classes (digits 0-9)
    - Total Parameters: 10,066

    Network Flow:
    28x28x1 → 26x26x8 → 24x24x16 → 22x22x16 → 11x11x16 → 11x11x8 → 9x9x16 → 7x7x32 → 1x1x32 → 1x1x10
    """

    def __init__(self):
        super(Net, self).__init__()
        
        # INPUT        - OUTPUT       - RECEPTIVE FIELD
        self.conv1 = nn.Conv2d(1, 8, 3)                   # 28*28*1      - 26*26*8      - 3*3
        self.bn1   = nn.BatchNorm2d(8)
        self.dp1   = nn.Dropout(p=0.10)

        self.conv2 = nn.Conv2d(8, 16, 3)                   # 26*26*8      - 24*24*16     - 5*5
        self.bn2   = nn.BatchNorm2d(16)
        self.dp2   = nn.Dropout(p=0.10)

        self.conv3 = nn.Conv2d(16, 32, 3)                  # 24*24*16     - 22*22*16     - 7*7
        self.bn3   = nn.BatchNorm2d(32)
        self.dp3   = nn.Dropout(p=0.10)

        self.pool1 = nn.MaxPool2d(2, 2)                    # 22*22*16     - 11*11*16     - 14*14
        self.bn4   = nn.BatchNorm2d(32)
        self.dp4   = nn.Dropout(p=0.10)

        self.dj1   = nn.Conv2d(32, 8, 1)                   # 11*11*16     - 11*11*8      - 14*14
        self.bn5   = nn.BatchNorm2d(8)
        self.dp5   = nn.Dropout(p=0.10)

        self.conv4 = nn.Conv2d(8, 16, 3)                   # 11*11*8      - 9*9*16       - 16*16
        self.bn6   = nn.BatchNorm2d(16)
        self.dp6   = nn.Dropout(p=0.10)

        self.conv5 = nn.Conv2d(16, 32, 3)                  # 9*9*16       - 7*7*32       - 18*18
        self.bn7   = nn.BatchNorm2d(32)
        self.dp7   = nn.Dropout(p=0.10)

        self.gap1  = nn.AvgPool2d(7, 7)                    # 7*7*32       - 1*1*32
        self.conv6 = nn.Conv2d(32, 10, 1)                  # 1*1*32       - 1*1*10       - 22*22

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output tensor of shape (batch_size, 10) with log probabilities
        """

        # ========================================
        # FIRST CONVOLUTION BLOCK
        # ========================================
        # conv1: 28x28x1 -> 26x26x8
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dp1(x)

        # conv2: 26x26x8 -> 24x24x16
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.dp2(x)

        # conv3: 24x24x16 -> 22x22x16
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.dp3(x)

        # ========================================
        # POOLING LAYER
        # ========================================
        # pool1: 22x22x16 -> 11x11x16
        x = self.pool1(x)
        x = self.bn4(x)
        x = self.dp4(x)

        # ========================================
        # TRANSITION LAYER (1x1 CONVOLUTION)
        # ========================================
        # dj1: 11x11x16 -> 11x11x8
        x = F.relu(self.dj1(x))
        x = self.bn5(x)
        x = self.dp5(x)

        # ========================================
        # SECOND CONVOLUTION BLOCK
        # ========================================
        # conv4: 11x11x8 -> 9x9x16
        x = F.relu(self.conv4(x))
        x = self.bn6(x)
        x = self.dp6(x)

        # conv5: 9x9x16 -> 7x7x32
        x = F.relu(self.conv5(x))
        x = self.bn7(x)
        x = self.dp7(x)

        # ========================================
        # GLOBAL AVERAGE POOLING & CLASSIFICATION
        # ========================================
        # gap1: 7x7x32 -> 1x1x32
        x = self.gap1(x)

        # conv6: 1x1x32 -> 1x1x10
        x = self.conv6(x)

        # Flatten and apply log softmax
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)


def get_model_summary():
    """
    Get a summary of the model architecture
    
    Returns:
        str: Model summary information
    """
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
Model Architecture Summary:
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
- Input Shape: (batch_size, 1, 28, 28)
- Output Shape: (batch_size, 10)
- Architecture: CNN with BatchNorm, Dropout, and Global Average Pooling
"""
    return summary


if __name__ == "__main__":
    # Test the model
    model = Net()
    print("Model created successfully!")
    print(get_model_summary())
    
    # Test forward pass
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"Test forward pass: Input shape {x.shape} -> Output shape {output.shape}")
