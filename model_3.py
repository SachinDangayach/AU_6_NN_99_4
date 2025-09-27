"""
MNIST Digit Classification CNN with Dropout and Global Average Pooling

This module implements a compact Convolutional Neural Network (CNN) for MNIST digit 
classification with Dropout regularization and Global Average Pooling (GAP). The 
architecture is designed to achieve high accuracy while maintaining computational 
efficiency through strategic use of:

- Convolutional layers for feature extraction
- Batch Normalization for training stability and faster convergence
- Dropout layers for regularization and preventing overfitting
- Global Average Pooling for parameter reduction and better generalization
- Max pooling for spatial dimension reduction
- 1x1 convolutions for channel reduction and computational efficiency
- ReLU activations for non-linearity
- Log-softmax for final classification

Architecture Overview:
    Input: 28x28x1 grayscale images
    Output: 10 classes (digits 0-9)
    Total Parameters: ~8K (efficient design with GAP)
    Final Receptive Field: 28x28 (covers entire input)

Key Design Features:
    - Dropout regularization: Prevents overfitting with 0.1 dropout rate
    - Batch Normalization: Improves training stability and convergence speed
    - Global Average Pooling: Reduces parameters and improves generalization
    - No bias terms in convolutions (bias=False) for cleaner gradients
    - Progressive channel increase: 1→8→16→32 channels
    - Strategic downsampling with max pooling
    - Channel reduction with 1x1 convolutions

Architecture Flow:
    28x28x1 → 26x26x8 → 26x26x16 → 13x13x16 → 13x13x8 → 11x11x8 
    → 9x9x16 → 7x7x32 → 1x1x32 → 1x1x10

Usage:
    model = Net()
    x = torch.randn(batch_size, 1, 28, 28)
    log_probs = model(x)
    predictions = torch.exp(log_probs)

Author: Neural Network Implementation
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Dropout rate for regularization
dropout_value = 0.1


class Net(nn.Module):
    """
    MNIST Digit Classification CNN with Dropout and Global Average Pooling
    
    A compact convolutional neural network designed for MNIST digit classification
    with Dropout regularization and Global Average Pooling (GAP). This architecture
    combines the benefits of Batch Normalization, Dropout, and GAP for improved
    generalization and parameter efficiency.
    
    Architecture Flow:
        28x28x1 → 26x26x8 → 26x26x16 → 13x13x16 → 13x13x8 → 11x11x8 
        → 9x9x16 → 7x7x32 → 1x1x32 → 1x1x10
    
    Receptive Field Progression:
        RF: 3 → 5 → 5 → 6 → 12 → 16 → 20 → 24 → 28
    
    Key Components:
        - Input Block: Initial feature extraction (1→8 channels) + BatchNorm + Dropout
        - Conv Block 1: Progressive feature learning (8→16 channels) + BatchNorm + Dropout
        - Transition Block: Channel reduction (16→8) + spatial downsampling
        - Conv Block 2: Deep feature extraction (8→16→32 channels) + BatchNorm + Dropout
        - Output Block: Global Average Pooling + classification (32→10 channels)
    
    Design Rationale:
        - Dropout (0.1): Prevents overfitting by randomly zeroing 10% of activations
        - Batch Normalization: Normalizes inputs to each layer, improving training stability
        - Global Average Pooling: Reduces parameters by replacing FC layers with spatial averaging
        - No bias terms: BatchNorm includes learnable bias, making conv bias redundant
        - ReLU activations: Provides non-linearity and gradient stability
        - Max pooling: Reduces spatial dimensions while preserving important features
        - 1x1 convolutions: Efficient channel reduction without spatial information loss
        - Progressive channels: 1→8→16→32 for balanced complexity and efficiency
    
    Attributes:
        convblock1-7 (nn.Sequential): Convolutional blocks with BatchNorm + ReLU + Dropout
        pool1 (nn.MaxPool2d): Spatial downsampling layer
        gap (nn.AvgPool2d): Global Average Pooling layer
    
    Example:
        >>> model = Net()
        >>> x = torch.randn(1, 1, 28, 28)
        >>> log_probs = model(x)
        >>> print(f"Output shape: {log_probs.shape}")  # torch.Size([1, 10])
    """
    def __init__(self) -> None:
        """
        Initialize the MNIST CNN architecture with Dropout and Global Average Pooling.
        
        Creates all convolutional blocks, pooling layers, and regularization components
        for improved generalization. Uses BatchNorm for stability, Dropout for
        regularization, and GAP for parameter efficiency.
        """
        super(Net, self).__init__()
        
        # Input Block: Initial feature extraction with regularization
        # Design: 3x3 kernel captures local patterns, BatchNorm stabilizes, Dropout regularizes
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        )  # 28x28x1 → 26x26x8, RF=3

        # Conv Block 1: Progressive feature learning with regularization
        # Strategy: Increase channels with padding=1 to maintain spatial dimensions
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )  # 26x26x8 → 26x26x16, RF=5

        # Transition Block: Channel reduction + spatial downsampling
        # Max pooling reduces spatial dimensions by half
        self.pool1 = nn.MaxPool2d(2, 2)  # 26x26x16 → 13x13x16, RF=6
        # 1x1 convolution reduces channels without spatial information loss
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        )  # 13x13x16 → 13x13x8, RF=6

        # Conv Block 2: Deep feature extraction with regularization
        # Pattern: Rebuild channel depth for richer feature representation
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        )  # 13x13x8 → 11x11x8, RF=12
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )  # 11x11x8 → 9x9x16, RF=16
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        )  # 9x9x16 → 7x7x32, RF=20

        # Output Block: Global Average Pooling + classification
        # GAP reduces 7x7x32 to 1x1x32, eliminating need for FC layers
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        )  # 7x7x32 → 1x1x32, RF=28
        
        # Final 1x1 convolution maps 32 features to 10 classes
        # No BatchNorm/ReLU/Dropout before final classification layer
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),  # Commented out for final layer
            # nn.ReLU(),           # Commented out for final layer
            # nn.Dropout(dropout_value)  # Commented out for final layer
        )  # 1x1x32 → 1x1x10, RF=28

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN architecture with Dropout and Global Average Pooling.
        
        Processes input images through all convolutional blocks with BatchNorm, Dropout,
        and Global Average Pooling. The architecture uses regularization techniques
        for improved generalization and parameter efficiency.
        
        Args:
            x (torch.Tensor): Input batch of grayscale images
                Shape: (batch_size, 1, 28, 28)
                Expected dtype: torch.float32
                Expected range: [0, 1] for normalized images or [-1, 1] for standardized
        
        Returns:
            torch.Tensor: Log probabilities for each digit class
                Shape: (batch_size, 10)
                Dtype: torch.float32
                Range: (-∞, 0] (log probabilities)
        
        Forward Pass Flow:
            1. Feature Extraction: Extract low-level features with BatchNorm + Dropout
            2. Progressive Learning: Build complex feature representations with regularization
            3. Spatial Reduction: Reduce spatial dimensions while preserving features
            4. Channel Management: Balance computational efficiency and representation power
            5. Global Average Pooling: Replace FC layers with spatial averaging
            6. Classification: Map features to class probabilities
        
        Regularization Benefits:
            - Dropout: Prevents overfitting by randomly zeroing 10% of activations
            - Batch Normalization: Normalizes inputs to each layer, improving training stability
            - Global Average Pooling: Reduces parameters and improves generalization
            - No bias terms: BatchNorm includes learnable bias, making conv bias redundant
        
        Note:
            - Uses log_softmax for numerical stability in training
            - Final output can be converted to probabilities with torch.exp()
            - Dropout is active during training, disabled during inference
        
        Example:
            >>> model = Net()
            >>> x = torch.randn(2, 1, 28, 28)  # Batch of 2 images
            >>> log_probs = model(x)
            >>> probs = torch.exp(log_probs)    # Convert to probabilities
            >>> predictions = torch.argmax(probs, dim=1)  # Get predicted classes
        """
        # Stage 1: Initial feature extraction with regularization (low-level patterns)
        x = self.convblock1(x)  # 28x28x1 → 26x26x8, RF=3
        x = self.convblock2(x)  # 26x26x8 → 26x26x16, RF=5
        
        # Stage 2: Channel reduction and spatial downsampling
        x = self.pool1(x)       # 26x26x16 → 13x13x16, RF=6 (max pooling)
        x = self.convblock3(x)  # 13x13x16 → 13x13x8, RF=6 (channel reduction)
        
        # Stage 3: Deep feature learning with regularization (high-level patterns)
        x = self.convblock4(x)  # 13x13x8 → 11x11x8, RF=12
        x = self.convblock5(x)  # 11x11x8 → 9x9x16, RF=16
        x = self.convblock6(x)  # 9x9x16 → 7x7x32, RF=20
        
        # Stage 4: Global Average Pooling + classification
        x = self.gap(x)         # 7x7x32 → 1x1x32, RF=28 (global average pooling)
        x = self.convblock7(x)  # 1x1x32 → 1x1x10, RF=28 (feature-to-class mapping)
        
        # Flatten and apply log-softmax for stable training
        x = x.view(-1, 10)  # Reshape to (batch_size, 10)
        return F.log_softmax(x, dim=-1)  # Apply log-softmax along class dimension

def get_model_summary():
    """
    Get model architecture summary
    
    Returns:
        str: Model summary with parameter counts
    """
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
Model Summary:
- Parameters: {total_params:,} (trainable: {trainable_params:,})
- Input: (batch_size, 1, 28, 28)
- Output: (batch_size, 10)
- Architecture: CNN with ReLU, MaxPool2d, 1x1 convs
- Final RF: 28x28 (covers full input)
"""
    return summary
