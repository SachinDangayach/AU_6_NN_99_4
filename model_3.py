"""
MNIST Digit Classification CNN

This module implements a compact Convolutional Neural Network (CNN) for MNIST digit 
classification. The architecture is designed to achieve high accuracy while maintaining 
computational efficiency through strategic use of:

- Convolutional layers for feature extraction
- Max pooling for spatial dimension reduction
- 1x1 convolutions for channel reduction and computational efficiency
- ReLU activations for non-linearity
- Log-softmax for final classification

Architecture Overview:
    Input: 28x28x1 grayscale images
    Output: 10 classes (digits 0-9)
    Total Parameters: ~10K (varies based on implementation)
    Final Receptive Field: 28x28 (covers entire input)

Key Design Features:
    - No bias terms in convolutions (bias=False) for cleaner gradients
    - Progressive channel increase: 1→32→64→128
    - Strategic downsampling with max pooling
    - Channel reduction with 1x1 convolutions
    - Global average pooling equivalent with 7x7 convolution

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
    MNIST Digit Classification CNN
    
    A compact convolutional neural network designed for MNIST digit classification.
    The architecture follows a progressive feature extraction approach with strategic
    downsampling and channel management.
    
    Architecture Flow:
        28x28x1 → 26x26x32 → 24x24x64 → 22x22x128 → 11x11x128 → 11x11x32 
        → 9x9x64 → 7x7x128 → 7x7x10 → 1x1x10
    
    Receptive Field Progression:
        RF: 3 → 5 → 7 → 8 → 8 → 12 → 16 → 16 → 28
    
    Key Components:
        - Input Block: Initial feature extraction (1→32 channels)
        - Conv Block 1: Progressive feature learning (32→64→128 channels)
        - Transition Block: Spatial downsampling + channel reduction
        - Conv Block 2: Deep feature extraction (32→64→128 channels)
        - Output Block: Classification preparation (128→10 channels)
    
    Design Rationale:
        - No bias terms: Reduces parameters and improves gradient flow
        - ReLU activations: Provides non-linearity and gradient stability
        - Max pooling: Reduces spatial dimensions while preserving important features
        - 1x1 convolutions: Efficient channel reduction without spatial information loss
        - 7x7 final convolution: Acts as global average pooling for classification
    
    Attributes:
        convblock1-8 (nn.Sequential): Convolutional blocks with ReLU activations
        pool1 (nn.MaxPool2d): Spatial downsampling layer
    
    Example:
        >>> model = Net()
        >>> x = torch.randn(1, 1, 28, 28)
        >>> log_probs = model(x)
        >>> print(f"Output shape: {log_probs.shape}")  # torch.Size([1, 10])
    """
    def __init__(self) -> None:
        """
        Initialize the MNIST CNN architecture.
        
        Creates all convolutional blocks and pooling layers with optimized parameters
        for MNIST digit classification. Uses bias=False for cleaner gradients and
        consistent ReLU activations for non-linearity.
        """
        super(Net, self).__init__()
        
        # Input Block: Initial feature extraction from grayscale to 32 feature maps
        # Design: 3x3 kernel captures local patterns, no bias for cleaner gradients
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # 28x28x1 → 26x26x32, RF=3

        # Conv Block 1: Progressive feature learning with increasing complexity
        # Strategy: Double channels while reducing spatial dimensions
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # 26x26x32 → 24x24x64, RF=5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # 24x24x64 → 22x22x128, RF=7

        # Transition Block: Spatial downsampling + computational efficiency
        # Max pooling preserves important features while reducing computation
        self.pool1 = nn.MaxPool2d(2, 2)  # 22x22x128 → 11x11x128, RF=8
        # 1x1 convolution reduces channels without spatial information loss
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )  # 11x11x128 → 11x11x32, RF=8

        # Conv Block 2: Deep feature extraction with increased complexity
        # Pattern: Rebuild channel depth for richer feature representation
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # 11x11x32 → 9x9x64, RF=12
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )  # 9x9x64 → 7x7x128, RF=16

        # Output Block: Classification preparation
        # 1x1 convolution maps 128 features to 10 classes efficiently
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )  # 7x7x128 → 7x7x10, RF=16
        # 7x7 convolution acts as global average pooling for final classification
        # No ReLU before final layer to allow negative logits
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False)
        )  # 7x7x10 → 1x1x10, RF=28

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN architecture.
        
        Processes input images through all convolutional blocks, applying feature
        extraction, spatial downsampling, and classification preparation.
        
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
            1. Feature Extraction: Extract low-level features (edges, corners)
            2. Progressive Learning: Build complex feature representations
            3. Spatial Reduction: Reduce spatial dimensions while preserving features
            4. Channel Management: Balance computational efficiency and representation power
            5. Classification: Map features to class probabilities
        
        Note:
            - Uses log_softmax for numerical stability in training
            - Final output can be converted to probabilities with torch.exp()
            - No dropout or batch normalization for simplicity
        
        Example:
            >>> model = Net()
            >>> x = torch.randn(2, 1, 28, 28)  # Batch of 2 images
            >>> log_probs = model(x)
            >>> probs = torch.exp(log_probs)    # Convert to probabilities
            >>> predictions = torch.argmax(probs, dim=1)  # Get predicted classes
        """
        # Stage 1: Initial feature extraction (low-level patterns)
        x = self.convblock1(x)  # 28x28x1 → 26x26x32, RF=3
        x = self.convblock2(x)  # 26x26x32 → 24x24x64, RF=5
        x = self.convblock3(x)  # 24x24x64 → 22x22x128, RF=7
        
        # Stage 2: Spatial reduction and computational efficiency
        x = self.pool1(x)       # 22x22x128 → 11x11x128, RF=8 (max pooling)
        x = self.convblock4(x)  # 11x11x128 → 11x11x32, RF=8 (channel reduction)
        
        # Stage 3: Deep feature learning (high-level patterns)
        x = self.convblock5(x)  # 11x11x32 → 9x9x64, RF=12
        x = self.convblock6(x)  # 9x9x64 → 7x7x128, RF=16
        
        # Stage 4: Classification preparation
        x = self.convblock7(x)  # 7x7x128 → 7x7x10, RF=16 (feature-to-class mapping)
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
