"""
MNIST Digit Classification CNN with Batch Normalization

This module implements a compact Convolutional Neural Network (CNN) for MNIST digit 
classification with Batch Normalization. The architecture is designed to achieve 
high accuracy while maintaining computational efficiency through strategic use of:

- Convolutional layers for feature extraction
- Batch Normalization for training stability and faster convergence
- Max pooling for spatial dimension reduction
- 1x1 convolutions for channel reduction and computational efficiency
- ReLU activations for non-linearity
- Log-softmax for final classification

Architecture Overview:
    Input: 28x28x1 grayscale images
    Output: 10 classes (digits 0-9)
    Total Parameters: ~6K (compact design with BatchNorm)
    Final Receptive Field: 28x28 (covers entire input)

Key Design Features:
    - Batch Normalization: Improves training stability and convergence speed
    - No bias terms in convolutions (bias=False) for cleaner gradients
    - Compact channel progression: 1→10→20 channels
    - Strategic downsampling with max pooling
    - Channel reduction with 1x1 convolutions
    - Global average pooling equivalent with 7x7 convolution

Architecture Flow:
    28x28x1 → 26x26x10 → 24x24x10 → 22x22x20 → 11x11x20 → 11x11x10 
    → 9x9x10 → 7x7x20 → 7x7x10 → 1x1x10

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


class Net(nn.Module):
    """
    MNIST Digit Classification CNN with Batch Normalization
    
    A compact convolutional neural network designed for MNIST digit classification
    with Batch Normalization layers. This architecture uses a smaller channel count
    (10, 20) compared to model_1.py but includes BatchNorm for improved training
    stability and faster convergence.
    
    Architecture Flow:
        28x28x1 → 26x26x10 → 24x24x10 → 22x22x20 → 11x11x20 → 11x11x10 
        → 9x9x10 → 7x7x20 → 7x7x10 → 1x1x10
    
    Receptive Field Progression:
        RF: 3 → 5 → 7 → 8 → 8 → 12 → 16 → 16 → 28
    
    Key Components:
        - Input Block: Initial feature extraction (1→10 channels) + BatchNorm
        - Conv Block 1: Progressive feature learning (10→10→20 channels) + BatchNorm
        - Transition Block: Spatial downsampling + channel reduction + BatchNorm
        - Conv Block 2: Deep feature extraction (10→10→20 channels) + BatchNorm
        - Output Block: Classification preparation (20→10 channels) + BatchNorm
    
    Design Rationale:
        - Batch Normalization: Normalizes inputs to each layer, improving training stability
        - No bias terms: BatchNorm includes learnable bias, making conv bias redundant
        - ReLU activations: Provides non-linearity and gradient stability
        - Max pooling: Reduces spatial dimensions while preserving important features
        - 1x1 convolutions: Efficient channel reduction without spatial information loss
        - 7x7 final convolution: Acts as global average pooling for classification
        - Compact design: Fewer parameters (~6K) for faster training and inference
    
    Attributes:
        convblock1-8 (nn.Sequential): Convolutional blocks with BatchNorm + ReLU
        pool1 (nn.MaxPool2d): Spatial downsampling layer
    
    Example:
        >>> model = Net()
        >>> x = torch.randn(1, 1, 28, 28)
        >>> log_probs = model(x)
        >>> print(f"Output shape: {log_probs.shape}")  # torch.Size([1, 10])
    """
    def __init__(self) -> None:
        """
        Initialize the MNIST CNN architecture with Batch Normalization.
        
        Creates all convolutional blocks and pooling layers with BatchNorm for
        improved training stability. Uses bias=False since BatchNorm includes
        learnable bias parameters, making convolutional bias redundant.
        """
        super(Net, self).__init__()
        
        # Input Block: Initial feature extraction with BatchNorm
        # Design: 3x3 kernel captures local patterns, BatchNorm stabilizes training
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )  # 28x28x1 → 26x26x10, RF=3

        # Conv Block 1: Progressive feature learning with BatchNorm
        # Strategy: Maintain channels initially, then increase for complexity
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )  # 26x26x10 → 24x24x10, RF=5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        )  # 24x24x10 → 22x22x20, RF=7

        # Transition Block: Spatial downsampling + computational efficiency
        # Max pooling preserves important features while reducing computation
        self.pool1 = nn.MaxPool2d(2, 2)  # 22x22x20 → 11x11x20, RF=8
        # 1x1 convolution reduces channels without spatial information loss
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )  # 11x11x20 → 11x11x10, RF=8

        # Conv Block 2: Deep feature extraction with BatchNorm
        # Pattern: Rebuild channel depth for richer feature representation
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )  # 11x11x10 → 9x9x10, RF=12
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        )  # 9x9x10 → 7x7x20, RF=16

        # Output Block: Classification preparation with BatchNorm
        # 1x1 convolution maps 20 features to 10 classes efficiently
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )  # 7x7x20 → 7x7x10, RF=16
        # 7x7 convolution acts as global average pooling for final classification
        # No ReLU before final layer to allow negative logits
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False)
        )  # 7x7x10 → 1x1x10, RF=28

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN architecture with Batch Normalization.
        
        Processes input images through all convolutional blocks with BatchNorm,
        applying feature extraction, spatial downsampling, and classification
        preparation. BatchNorm layers normalize activations for stable training.
        
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
            1. Feature Extraction: Extract low-level features with BatchNorm stabilization
            2. Progressive Learning: Build complex feature representations with normalization
            3. Spatial Reduction: Reduce spatial dimensions while preserving features
            4. Channel Management: Balance computational efficiency and representation power
            5. Classification: Map features to class probabilities
        
        Batch Normalization Benefits:
            - Normalizes inputs to each layer (mean=0, std=1)
            - Reduces internal covariate shift
            - Allows higher learning rates
            - Provides regularization effect
            - Improves gradient flow during backpropagation
        
        Note:
            - Uses log_softmax for numerical stability in training
            - Final output can be converted to probabilities with torch.exp()
            - BatchNorm layers are trainable and adapt during training
        
        Example:
            >>> model = Net()
            >>> x = torch.randn(2, 1, 28, 28)  # Batch of 2 images
            >>> log_probs = model(x)
            >>> probs = torch.exp(log_probs)    # Convert to probabilities
            >>> predictions = torch.argmax(probs, dim=1)  # Get predicted classes
        """
        # Stage 1: Initial feature extraction with BatchNorm (low-level patterns)
        x = self.convblock1(x)  # 28x28x1 → 26x26x10, RF=3
        x = self.convblock2(x)  # 26x26x10 → 24x24x10, RF=5
        x = self.convblock3(x)  # 24x24x10 → 22x22x20, RF=7
        
        # Stage 2: Spatial reduction and computational efficiency
        x = self.pool1(x)       # 22x22x20 → 11x11x20, RF=8 (max pooling)
        x = self.convblock4(x)  # 11x11x20 → 11x11x10, RF=8 (channel reduction)
        
        # Stage 3: Deep feature learning with BatchNorm (high-level patterns)
        x = self.convblock5(x)  # 11x11x10 → 9x9x10, RF=12
        x = self.convblock6(x)  # 9x9x10 → 7x7x20, RF=16
        
        # Stage 4: Classification preparation with BatchNorm
        x = self.convblock7(x)  # 7x7x20 → 7x7x10, RF=16 (feature-to-class mapping)
        x = self.convblock8(x)  # 7x7x10 → 1x1x10, RF=28 (global pooling equivalent)
        
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
