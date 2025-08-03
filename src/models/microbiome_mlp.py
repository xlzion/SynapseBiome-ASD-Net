"""
Sparse Multi-Layer Perceptron for microbiome data processing.

This module implements a sparse neural network architecture specifically designed
for high-dimensional microbiome data with L1 regularization for feature selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class SparseMLP(nn.Module):
    """
    Sparse Multi-Layer Perceptron for microbiome feature processing.
    
    This model uses L1 regularization to encourage sparsity in the learned
    representations, helping identify the most relevant microbial features.
    """
    
    def __init__(self, input_dim: int, num_classes: int = 2,
                 hidden_dims: List[int] = [2048, 1024, 512],
                 dropout: float = 0.5, sparsity_lambda: float = 0.01,
                 activation: str = 'gelu'):
        """
        Initialize the SparseMLP model.
        
        Args:
            input_dim: Input dimension (number of microbiome features)
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            sparsity_lambda: L1 regularization coefficient
            activation: Activation function ('gelu', 'relu', 'tanh')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.sparsity_lambda = sparsity_lambda
        
        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build feature extractor layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Classifier
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the SparseMLP.
        
        Args:
            x: Input microbiome features [batch_size, input_dim]
            return_features: Whether to return intermediate features
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
            features: Intermediate features (if return_features=True)
        """
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        
        return logits
    
    def l1_regularization(self) -> torch.Tensor:
        """
        Compute L1 regularization term for sparsity.
        
        Returns:
            l1_reg: L1 regularization term
        """
        l1_reg = torch.tensor(0., device=next(self.parameters()).device)
        
        for param in self.feature_extractor.parameters():
            l1_reg += torch.norm(param, p=1)
        
        return self.sparsity_lambda * l1_reg
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance scores using gradient-based attribution.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            importance: Feature importance scores [batch_size, input_dim]
        """
        x.requires_grad_(True)
        
        # Forward pass
        logits = self.forward(x)
        
        # Compute gradients with respect to input
        logits.sum().backward()
        
        # Feature importance as absolute gradient values
        importance = torch.abs(x.grad)
        
        return importance
    
    def get_sparse_features(self, x: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """
        Get sparse feature representations by applying thresholding.
        
        Args:
            x: Input features [batch_size, input_dim]
            threshold: Threshold for feature selection
            
        Returns:
            sparse_features: Thresholded feature representations
        """
        features = self.feature_extractor(x)
        
        # Apply thresholding for sparsity
        sparse_features = torch.where(
            torch.abs(features) > threshold,
            features,
            torch.zeros_like(features)
        )
        
        return sparse_features
    
    def get_layer_representations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get intermediate layer representations.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            representations: List of layer representations
        """
        representations = []
        h = x
        
        for i, layer in enumerate(self.feature_extractor):
            h = layer(h)
            
            # Store representations after activation layers
            if isinstance(layer, (nn.GELU, nn.ReLU, nn.Tanh)):
                representations.append(h)
        
        return representations


class MicrobiomeDataLoader:
    """
    Data loader for microbiome data with preprocessing and augmentation.
    """
    
    def __init__(self, data_path: str, batch_size: int = 32, 
                 shuffle: bool = True, num_workers: int = 4):
        """
        Initialize the microbiome data loader.
        
        Args:
            data_path: Path to microbiome data file
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Load and preprocess data
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess microbiome data."""
        import pandas as pd
        import numpy as np
        
        # Load data
        data = pd.read_csv(self.data_path)
        
        # Extract features and labels
        self.features = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(data.iloc[:, -1].values, dtype=torch.long)
        
        # Normalize features
        self.features = (self.features - self.features.mean(dim=0)) / (self.features.std(dim=0) + 1e-8)
        
        # Create dataset
        self.dataset = torch.utils.data.TensorDataset(self.features, self.labels)
    
    def get_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Get PyTorch DataLoader.
        
        Returns:
            dataloader: PyTorch DataLoader
        """
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_train_val_loaders(self, train_ratio: float = 0.8, 
                             val_ratio: float = 0.1) -> Tuple[torch.utils.data.DataLoader, 
                                                             torch.utils.data.DataLoader]:
        """
        Split data into train and validation sets.
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            
        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        total_size = len(self.dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader 