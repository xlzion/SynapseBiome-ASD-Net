"""
Contrastive learning framework for multimodal ASD diagnosis.

This module implements the contrastive learning approach that learns modality-invariant
representations by projecting fMRI and microbiome features into a shared latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class ContrastiveModel(nn.Module):
    """
    Contrastive learning model for multimodal ASD diagnosis.
    
    This model combines fMRI and microbiome features through contrastive learning,
    learning modality-invariant representations in a shared latent space.
    """
    
    def __init__(self, mlp_model: nn.Module, gnn_model: nn.Module, config: Dict):
        """
        Initialize the contrastive model.
        
        Args:
            mlp_model: Pre-trained SparseMLP for microbiome processing
            gnn_model: Pre-trained fMRI3DGNN for fMRI processing
            config: Configuration dictionary
        """
        super().__init__()
        
        self.mlp = mlp_model
        self.gnn = gnn_model
        
        # Configuration
        self.contrastive_dim = config.get('contrastive_feat_dim', 256)
        self.dropout = config.get('contrastive_dropout', 0.3)
        self.num_classes = config.get('num_classes', 2)
        self.freeze_base_models = config.get('freeze_base_models', True)
        
        # Get intermediate feature dimensions
        mlp_hidden_dims = config.get('mlp_hidden_dims', [2048, 1024, 512])
        mlp_intermediate_dim = mlp_hidden_dims[-1] if mlp_hidden_dims else 512
        gnn_intermediate_dim = config.get('gnn_output_dim', 512)
        
        # Freeze base models if specified
        if self.freeze_base_models:
            for param in self.mlp.parameters():
                param.requires_grad_(False)
            for param in self.gnn.parameters():
                param.requires_grad_(False)
        
        # Projection heads for contrastive learning
        self.mlp_proj = nn.Sequential(
            nn.LayerNorm(mlp_intermediate_dim),
            nn.Linear(mlp_intermediate_dim, self.contrastive_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        self.gnn_proj = nn.Sequential(
            nn.LayerNorm(gnn_intermediate_dim),
            nn.Linear(gnn_intermediate_dim, self.contrastive_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.contrastive_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize projection and classifier weights."""
        for module in [self.mlp_proj, self.gnn_proj, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, microbe_input: torch.Tensor, fmri_input: torch.Tensor,
                return_gnn_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the contrastive model.
        
        Args:
            microbe_input: Microbiome features [batch_size, microbiome_dim]
            fmri_input: fMRI features [batch_size, fmri_dim]
            return_gnn_attention: Whether to return GNN attention weights
            
        Returns:
            h_microbe: Projected microbiome features [batch_size, contrastive_dim]
            h_fmri: Projected fMRI features [batch_size, contrastive_dim]
            logits: Classification logits [batch_size, num_classes]
        """
        # Extract features from base models
        if return_gnn_attention:
            gnn_output, attention_weights = self.gnn(fmri_input, return_attention=True)
        else:
            gnn_output = self.gnn(fmri_input)
        
        # Get intermediate features
        mlp_features = self.mlp.feature_extractor(microbe_input)
        gnn_features = self.gnn.get_graph_representations(fmri_input)
        
        # Project to contrastive space
        h_microbe = F.normalize(self.mlp_proj(mlp_features), dim=1)
        h_fmri = F.normalize(self.gnn_proj(gnn_features), dim=1)
        
        # Combine features for classification
        combined = torch.cat([h_microbe, h_fmri], dim=1)
        logits = self.classifier(combined)
        
        if return_gnn_attention:
            return h_microbe, h_fmri, logits, attention_weights
        
        return h_microbe, h_fmri, logits
    
    def get_contrastive_features(self, microbe_input: torch.Tensor, 
                               fmri_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get contrastive features without classification.
        
        Args:
            microbe_input: Microbiome features
            fmri_input: fMRI features
            
        Returns:
            h_microbe: Projected microbiome features
            h_fmri: Projected fMRI features
        """
        mlp_features = self.mlp.feature_extractor(microbe_input)
        gnn_features = self.gnn.get_graph_representations(fmri_input)
        
        h_microbe = F.normalize(self.mlp_proj(mlp_features), dim=1)
        h_fmri = F.normalize(self.gnn_proj(gnn_features), dim=1)
        
        return h_microbe, h_fmri
    
    def compute_similarity_matrix(self, h_microbe: torch.Tensor, 
                                h_fmri: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix between microbiome and fMRI features.
        
        Args:
            h_microbe: Projected microbiome features [batch_size, contrastive_dim]
            h_fmri: Projected fMRI features [batch_size, contrastive_dim]
            
        Returns:
            similarity: Similarity matrix [batch_size, batch_size]
        """
        return torch.mm(h_microbe, h_fmri.t())


class LabelAwareContrastiveLoss(nn.Module):
    """
    Label-aware contrastive loss for multimodal learning.
    
    This loss function encourages same-class samples from different modalities
    to be closer in the shared latent space, while pushing different-class
    samples apart.
    """
    
    def __init__(self, temperature: float = 0.07, hard_neg_ratio: float = 0.2,
                 margin: float = 0.5):
        """
        Initialize the label-aware contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax
            hard_neg_ratio: Ratio of hard negative samples to use
            margin: Margin for triplet loss component
        """
        super().__init__()
        self.temperature = temperature
        self.hard_neg_ratio = hard_neg_ratio
        self.margin = margin
    
    def forward(self, h_microbe: torch.Tensor, h_fmri: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute label-aware contrastive loss.
        
        Args:
            h_microbe: Projected microbiome features [batch_size, contrastive_dim]
            h_fmri: Projected fMRI features [batch_size, contrastive_dim]
            labels: Ground truth labels [batch_size]
            
        Returns:
            loss: Contrastive loss value
        """
        batch_size = h_microbe.size(0)
        device = h_microbe.device
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(h_microbe, h_fmri.t()) / self.temperature
        
        # Create label matrix
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # [batch_size, batch_size]
        
        # Positive pairs (same class, different modalities)
        positive_mask = label_matrix.float()
        
        # Negative pairs (different classes)
        negative_mask = (~label_matrix).float()
        
        # InfoNCE loss for positive pairs
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mask out diagonal (same sample)
        mask = torch.eye(batch_size, device=device).bool()
        positive_mask = positive_mask.masked_fill(mask, 0)
        
        # Compute positive loss
        positive_loss = -(log_prob * positive_mask).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
        positive_loss = positive_loss.mean()
        
        # Hard negative mining
        if self.hard_neg_ratio > 0:
            # Find hardest negative samples
            negative_similarities = similarity_matrix * negative_mask
            num_hard_neg = max(1, int(batch_size * self.hard_neg_ratio))
            
            # Get top-k hardest negatives
            hard_neg_sim, _ = torch.topk(negative_similarities, k=num_hard_neg, dim=1)
            
            # Hard negative loss
            hard_neg_loss = torch.log(1 + torch.exp(hard_neg_sim)).mean()
        else:
            hard_neg_loss = torch.tensor(0.0, device=device)
        
        # Triplet loss component
        triplet_loss = self._compute_triplet_loss(h_microbe, h_fmri, labels)
        
        # Total loss
        total_loss = positive_loss + hard_neg_loss + triplet_loss
        
        return total_loss
    
    def _compute_triplet_loss(self, h_microbe: torch.Tensor, h_fmri: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss component.
        
        Args:
            h_microbe: Projected microbiome features
            h_fmri: Projected fMRI features
            labels: Ground truth labels
            
        Returns:
            triplet_loss: Triplet loss value
        """
        batch_size = h_microbe.size(0)
        device = h_microbe.device
        
        # Combine features
        h_combined = torch.cat([h_microbe, h_fmri], dim=0)
        labels_combined = torch.cat([labels, labels], dim=0)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(h_combined, h_combined, p=2)
        
        # Find positive and negative pairs
        label_matrix = labels_combined.unsqueeze(1) == labels_combined.unsqueeze(0)
        
        # Mask diagonal
        mask = torch.eye(2 * batch_size, device=device).bool()
        positive_mask = label_matrix.masked_fill(mask, False)
        negative_mask = (~label_matrix).masked_fill(mask, False)
        
        # Sample triplets
        triplet_loss = torch.tensor(0.0, device=device)
        num_triplets = 0
        
        for i in range(2 * batch_size):
            # Find positive pairs
            positive_indices = torch.where(positive_mask[i])[0]
            negative_indices = torch.where(negative_mask[i])[0]
            
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                # Sample positive and negative
                pos_idx = positive_indices[torch.randint(0, len(positive_indices), (1,))]
                neg_idx = negative_indices[torch.randint(0, len(negative_indices), (1,))]
                
                # Compute triplet loss
                pos_dist = dist_matrix[i, pos_idx]
                neg_dist = dist_matrix[i, neg_idx]
                
                triplet_loss += F.relu(pos_dist - neg_dist + self.margin)
                num_triplets += 1
        
        if num_triplets > 0:
            triplet_loss = triplet_loss / num_triplets
        
        return triplet_loss


class AdversarialContrastiveLoss(nn.Module):
    """
    Adversarial contrastive loss for improved robustness.
    
    This loss adds an adversarial component to make the model more robust
    to perturbations in the input data.
    """
    
    def __init__(self, base_loss: LabelAwareContrastiveLoss, 
                 epsilon: float = 0.01, alpha: float = 0.1):
        """
        Initialize adversarial contrastive loss.
        
        Args:
            base_loss: Base contrastive loss function
            epsilon: Perturbation magnitude
            alpha: Adversarial loss weight
        """
        super().__init__()
        self.base_loss = base_loss
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, h_microbe: torch.Tensor, h_fmri: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute adversarial contrastive loss.
        
        Args:
            h_microbe: Projected microbiome features
            h_fmri: Projected fMRI features
            labels: Ground truth labels
            
        Returns:
            loss: Combined loss value
        """
        # Base contrastive loss
        base_loss = self.base_loss(h_microbe, h_fmri, labels)
        
        # Adversarial perturbation
        h_microbe_adv = h_microbe + self.epsilon * torch.randn_like(h_microbe)
        h_fmri_adv = h_fmri + self.epsilon * torch.randn_like(h_fmri)
        
        # Adversarial loss
        adv_loss = self.base_loss(h_microbe_adv, h_fmri_adv, labels)
        
        # Combined loss
        total_loss = base_loss + self.alpha * adv_loss
        
        return total_loss 