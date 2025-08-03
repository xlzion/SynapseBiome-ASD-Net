"""
Dynamic fMRI Graph Neural Network for brain connectivity analysis.

This module implements a novel dynamic graph construction approach for fMRI data,
where each subject's brain connectivity is represented as a unique graph structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tg_nn
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Optional, Tuple


class DynamicGraphBuilder(nn.Module):
    """
    Dynamic graph builder that constructs subject-specific brain connectivity graphs.
    
    This module transforms flattened fMRI connectivity vectors into adjacency matrices
    and constructs graph structures with enhanced node features.
    """
    
    def __init__(self, input_dim: int = 40000, num_nodes: int = 200, 
                 feature_dim: int = 16, threshold: float = 0.5):
        """
        Initialize the dynamic graph builder.
        
        Args:
            input_dim: Dimension of flattened fMRI connectivity vector
            num_nodes: Number of brain regions (nodes) in the graph
            feature_dim: Dimension of enhanced node features
            threshold: Threshold for edge pruning
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.threshold = threshold
        
        # Graph construction network
        self.graph_builder = nn.Sequential(
            nn.Linear(input_dim, num_nodes * num_nodes),
            nn.Sigmoid()
        )
        
        # Node feature enhancement network
        self.feature_enhancer = nn.Sequential(
            nn.Linear(2, 8),  # mean, std of adjacency matrix
            nn.GELU(),
            nn.Linear(8, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def build_graph(self, fc_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph from functional connectivity matrix.
        
        Args:
            fc_matrix: Functional connectivity matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            edge_index: Graph edge indices [2, num_edges]
            node_features: Enhanced node features [num_nodes, feature_dim]
        """
        batch_size = fc_matrix.shape[0]
        edge_indices = []
        node_features_list = []
        
        for i in range(batch_size):
            adj_matrix = fc_matrix[i]
            
            # Prune weak connections
            adj_matrix = torch.where(
                adj_matrix > self.threshold, 
                adj_matrix, 
                torch.zeros_like(adj_matrix)
            )
            
            # Create edge indices
            edge_index = torch.nonzero(adj_matrix, as_tuple=False).t()
            
            # Compute node features from adjacency matrix statistics
            node_stats = torch.stack([
                adj_matrix.mean(dim=1),  # mean connectivity
                adj_matrix.std(dim=1)    # std connectivity
            ], dim=1)
            
            # Enhance node features
            node_features = self.feature_enhancer(node_stats)
            
            edge_indices.append(edge_index)
            node_features_list.append(node_features)
        
        return edge_indices, node_features_list
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass for graph construction.
        
        Args:
            x: Flattened fMRI connectivity vectors [batch_size, input_dim]
            
        Returns:
            edge_indices: List of edge indices for each sample
            node_features: List of node features for each sample
        """
        # Reshape to connectivity matrices
        batch_size = x.shape[0]
        fc_matrices = self.graph_builder(x).view(batch_size, self.num_nodes, self.num_nodes)
        
        # Build graphs
        edge_indices, node_features = self.build_graph(fc_matrices)
        
        return edge_indices, node_features


class fMRI3DGNN(nn.Module):
    """
    Dynamic fMRI Graph Neural Network for brain connectivity analysis.
    
    This model constructs dynamic graphs from fMRI data and processes them using
    Graph Attention Networks to learn contextual representations of brain regions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the fMRI3DGNN model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        
        # Configuration
        self.input_dim = config.get('fmri_input_dim', 40000)
        self.num_nodes = config.get('num_nodes', 200)
        self.feature_dim = config.get('feature_dim', 16)
        self.hidden_dims = config.get('gnn_hidden_dims', [128, 256, 512])
        self.num_heads = config.get('num_heads', [8, 4, 1])
        self.dropout = config.get('dropout', 0.3)
        self.num_classes = config.get('num_classes', 2)
        
        # Dynamic graph builder
        self.graph_builder = DynamicGraphBuilder(
            input_dim=self.input_dim,
            num_nodes=self.num_nodes,
            feature_dim=self.feature_dim
        )
        
        # Graph Attention layers
        self.convs = nn.ModuleList()
        prev_dim = self.feature_dim
        
        for i, (hidden_dim, num_head) in enumerate(zip(self.hidden_dims, self.num_heads)):
            conv = tg_nn.GATv2Conv(
                in_channels=prev_dim,
                out_channels=hidden_dim,
                heads=num_head,
                dropout=self.dropout,
                add_self_loops=(i > 0)  # No self-loops for first layer
            )
            self.convs.append(conv)
            prev_dim = hidden_dim * num_head
        
        # Global pooling
        self.global_pool = tg_nn.global_mean_pool
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes)
        )
        
        # Store attention weights for interpretability
        self.attention_weights = []
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass through the fMRI3DGNN.
        
        Args:
            x: Flattened fMRI connectivity vectors [batch_size, input_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Build dynamic graphs
        edge_indices, node_features = self.graph_builder(x)
        
        # Process each sample in the batch
        batch_outputs = []
        attention_weights = []
        
        for i in range(batch_size):
            # Create PyTorch Geometric Data object
            data = Data(
                x=node_features[i],
                edge_index=edge_indices[i]
            )
            
            # Process through GAT layers
            h = data.x
            sample_attention = []
            
            for conv in self.convs:
                h, attention = conv(h, data.edge_index, return_attention_weights=True)
                sample_attention.append(attention)
                h = F.gelu(h)
            
            # Global pooling
            h = self.global_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))
            
            batch_outputs.append(h)
            attention_weights.append(sample_attention)
        
        # Stack batch outputs
        batch_output = torch.stack(batch_outputs, dim=0)
        
        # Classification
        logits = self.classifier(batch_output)
        
        if return_attention:
            self.attention_weights = attention_weights
            return logits, attention_weights
        
        return logits
    
    def get_attention_weights(self) -> List[List[torch.Tensor]]:
        """Get attention weights from the last forward pass."""
        return self.attention_weights
    
    def get_graph_representations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get graph-level representations without classification.
        
        Args:
            x: Flattened fMRI connectivity vectors [batch_size, input_dim]
            
        Returns:
            representations: Graph-level representations [batch_size, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # Build dynamic graphs
        edge_indices, node_features = self.graph_builder(x)
        
        # Process each sample
        batch_outputs = []
        
        for i in range(batch_size):
            data = Data(
                x=node_features[i],
                edge_index=edge_indices[i]
            )
            
            # Process through GAT layers
            h = data.x
            for conv in self.convs:
                h = conv(h, data.edge_index)
                h = F.gelu(h)
            
            # Global pooling
            h = self.global_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))
            batch_outputs.append(h)
        
        return torch.stack(batch_outputs, dim=0) 