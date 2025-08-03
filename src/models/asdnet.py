"""
SynapseBiome ASD-Net: Main model architecture.

This module provides the main SynapseBiomeASDNet class that integrates all components
for multimodal ASD diagnosis using fMRI and microbiome data.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import torch.nn.functional as F

from .fmri_gnn import fMRI3DGNN
from .microbiome_mlp import SparseMLP
from .contrastive import ContrastiveModel, LabelAwareContrastiveLoss


class SynapseBiomeASDNet(nn.Module):
    """
    SynapseBiome ASD-Net: Multimodal deep learning framework for ASD diagnosis.
    
    This is the main model that integrates fMRI and microbiome processing through
    contrastive learning for robust ASD diagnosis and biomarker discovery.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SynapseBiome ASD-Net.
        
        Args:
            config: Configuration dictionary containing all model parameters
        """
        super().__init__()
        
        # Configuration
        self.config = config
        self.fmri_input_dim = config.get('fmri_input_dim', 40000)
        self.microbiome_input_dim = config.get('microbiome_input_dim', 2503)
        self.contrastive_dim = config.get('contrastive_feat_dim', 256)
        self.num_classes = config.get('num_classes', 2)
        
        # Initialize base models
        self._initialize_base_models()
        
        # Initialize contrastive model
        self._initialize_contrastive_model()
        
        # Loss functions
        self._initialize_loss_functions()
        
        # Training state
        self.training_mode = 'contrastive'  # 'pretrain', 'contrastive', 'finetune'
    
    def _initialize_base_models(self):
        """Initialize fMRI and microbiome base models."""
        # fMRI GNN model
        fmri_config = {
            'fmri_input_dim': self.fmri_input_dim,
            'num_nodes': self.config.get('num_nodes', 200),
            'feature_dim': self.config.get('feature_dim', 16),
            'gnn_hidden_dims': self.config.get('gnn_hidden_dims', [128, 256, 512]),
            'num_heads': self.config.get('num_heads', [8, 4, 1]),
            'dropout': self.config.get('dropout', 0.3),
            'num_classes': self.num_classes
        }
        self.fmri_gnn = fMRI3DGNN(fmri_config)
        
        # Microbiome MLP model
        mlp_config = {
            'input_dim': self.microbiome_input_dim,
            'num_classes': self.num_classes,
            'hidden_dims': self.config.get('mlp_hidden_dims', [2048, 1024, 512]),
            'dropout': self.config.get('dropout', 0.5),
            'sparsity_lambda': self.config.get('sparsity_lambda', 0.01),
            'activation': self.config.get('activation', 'gelu')
        }
        self.microbiome_mlp = SparseMLP(**mlp_config)
    
    def _initialize_contrastive_model(self):
        """Initialize contrastive learning model."""
        contrastive_config = {
            'contrastive_feat_dim': self.contrastive_dim,
            'contrastive_dropout': self.config.get('contrastive_dropout', 0.3),
            'num_classes': self.num_classes,
            'freeze_base_models': self.config.get('freeze_base_models', True),
            'mlp_hidden_dims': self.config.get('mlp_hidden_dims', [2048, 1024, 512]),
            'gnn_output_dim': self.config.get('gnn_hidden_dims', [128, 256, 512])[-1]
        }
        
        self.contrastive_model = ContrastiveModel(
            mlp_model=self.microbiome_mlp,
            gnn_model=self.fmri_gnn,
            config=contrastive_config
        )
    
    def _initialize_loss_functions(self):
        """Initialize loss functions."""
        # Contrastive loss
        self.contrastive_loss = LabelAwareContrastiveLoss(
            temperature=self.config.get('contrastive_temperature', 0.07),
            hard_neg_ratio=self.config.get('hard_neg_ratio', 0.2),
            margin=self.config.get('triplet_margin', 0.5)
        )
        
        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Sparsity loss
        self.sparsity_lambda = self.config.get('sparsity_lambda', 0.01)
    
    def forward(self, microbe_input: torch.Tensor, fmri_input: torch.Tensor,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass through SynapseBiome ASD-Net.
        
        Args:
            microbe_input: Microbiome features [batch_size, microbiome_dim]
            fmri_input: fMRI features [batch_size, fmri_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
            attention_weights: GNN attention weights (if return_attention=True)
        """
        if self.training_mode == 'contrastive':
            return self._contrastive_forward(microbe_input, fmri_input, return_attention)
        elif self.training_mode == 'pretrain':
            return self._pretrain_forward(microbe_input, fmri_input)
        else:
            return self._finetune_forward(microbe_input, fmri_input)
    
    def _contrastive_forward(self, microbe_input: torch.Tensor, fmri_input: torch.Tensor,
                           return_attention: bool = False) -> Union[torch.Tensor, Tuple]:
        """Forward pass in contrastive learning mode."""
        if return_attention:
            h_microbe, h_fmri, logits, attention_weights = self.contrastive_model(
                microbe_input, fmri_input, return_gnn_attention=True
            )
            return logits, attention_weights
        else:
            h_microbe, h_fmri, logits = self.contrastive_model(
                microbe_input, fmri_input
            )
            return logits
    
    def _pretrain_forward(self, microbe_input: torch.Tensor, 
                         fmri_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass in pretraining mode."""
        # Separate forward passes for base models
        mlp_logits = self.microbiome_mlp(microbe_input)
        gnn_logits = self.fmri_gnn(fmri_input)
        
        return mlp_logits, gnn_logits
    
    def _finetune_forward(self, microbe_input: torch.Tensor, 
                         fmri_input: torch.Tensor) -> torch.Tensor:
        """Forward pass in finetuning mode."""
        return self.contrastive_model(microbe_input, fmri_input)[-1]  # Only return logits
    
    def compute_loss(self, microbe_input: torch.Tensor, fmri_input: torch.Tensor,
                    labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            microbe_input: Microbiome features
            fmri_input: fMRI features
            labels: Ground truth labels
            
        Returns:
            loss_dict: Dictionary containing all loss components
        """
        if self.training_mode == 'contrastive':
            return self._compute_contrastive_loss(microbe_input, fmri_input, labels)
        elif self.training_mode == 'pretrain':
            return self._compute_pretrain_loss(microbe_input, fmri_input, labels)
        else:
            return self._compute_finetune_loss(microbe_input, fmri_input, labels)
    
    def _compute_contrastive_loss(self, microbe_input: torch.Tensor, 
                                fmri_input: torch.Tensor, 
                                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute contrastive learning loss."""
        # Get contrastive features
        h_microbe, h_fmri, logits = self.contrastive_model(microbe_input, fmri_input)
        
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(h_microbe, h_fmri, labels)
        
        # Classification loss
        classification_loss = self.classification_loss(logits, labels)
        
        # Sparsity loss
        sparsity_loss = self.microbiome_mlp.l1_regularization()
        
        # Total loss
        total_loss = (
            self.config.get('contrastive_weight', 1.0) * contrastive_loss +
            self.config.get('classification_weight', 1.0) * classification_loss +
            self.sparsity_lambda * sparsity_loss
        )
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'classification_loss': classification_loss,
            'sparsity_loss': sparsity_loss
        }
    
    def _compute_pretrain_loss(self, microbe_input: torch.Tensor, 
                             fmri_input: torch.Tensor, 
                             labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute pretraining loss for base models."""
        mlp_logits, gnn_logits = self._pretrain_forward(microbe_input, fmri_input)
        
        # Individual classification losses
        mlp_loss = self.classification_loss(mlp_logits, labels)
        gnn_loss = self.classification_loss(gnn_logits, labels)
        
        # Sparsity loss
        sparsity_loss = self.microbiome_mlp.l1_regularization()
        
        # Total loss
        total_loss = mlp_loss + gnn_loss + self.sparsity_lambda * sparsity_loss
        
        return {
            'total_loss': total_loss,
            'mlp_loss': mlp_loss,
            'gnn_loss': gnn_loss,
            'sparsity_loss': sparsity_loss
        }
    
    def _compute_finetune_loss(self, microbe_input: torch.Tensor, 
                             fmri_input: torch.Tensor, 
                             labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute finetuning loss."""
        logits = self._finetune_forward(microbe_input, fmri_input)
        
        # Classification loss
        classification_loss = self.classification_loss(logits, labels)
        
        return {
            'total_loss': classification_loss,
            'classification_loss': classification_loss
        }
    
    def set_training_mode(self, mode: str):
        """
        Set training mode.
        
        Args:
            mode: Training mode ('pretrain', 'contrastive', 'finetune')
        """
        assert mode in ['pretrain', 'contrastive', 'finetune'], f"Invalid mode: {mode}"
        self.training_mode = mode
        
        # Set model states
        if mode == 'pretrain':
            self.contrastive_model.eval()
            self.microbiome_mlp.train()
            self.fmri_gnn.train()
        elif mode == 'contrastive':
            self.contrastive_model.train()
            if self.config.get('freeze_base_models', True):
                self.microbiome_mlp.eval()
                self.fmri_gnn.eval()
        else:  # finetune
            self.contrastive_model.train()
            self.microbiome_mlp.train()
            self.fmri_gnn.train()
    
    def get_feature_importance(self, microbe_input: torch.Tensor, 
                             fmri_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get feature importance scores for interpretability.
        
        Args:
            microbe_input: Microbiome features
            fmri_input: fMRI features
            
        Returns:
            importance_dict: Dictionary containing importance scores
        """
        # Microbiome feature importance
        microbe_importance = self.microbiome_mlp.get_feature_importance(microbe_input)
        
        # fMRI feature importance (using gradients)
        fmri_input.requires_grad_(True)
        logits = self.forward(microbe_input, fmri_input)
        logits.sum().backward()
        fmri_importance = torch.abs(fmri_input.grad)
        
        return {
            'microbiome_importance': microbe_importance,
            'fmri_importance': fmri_importance
        }
    
    def get_attention_weights(self) -> List[List[torch.Tensor]]:
        """Get GNN attention weights from the last forward pass."""
        return self.fmri_gnn.get_attention_weights()
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_mode': self.training_mode
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_mode = checkpoint.get('training_mode', 'contrastive')
    
    def get_model_summary(self) -> Dict:
        """Get model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'fmri_input_dim': self.fmri_input_dim,
            'microbiome_input_dim': self.microbiome_input_dim,
            'contrastive_dim': self.contrastive_dim,
            'num_classes': self.num_classes,
            'training_mode': self.training_mode
        } 