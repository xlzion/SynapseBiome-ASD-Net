#对应contrastive_model1.pth

import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import sys
sys.path.append("/home/yangzongxian/xlz/ASD_GCN/main/pre_train")
from MLP import MicrobeDataLoader, SparseMLP
#from GNN import fMRI3DGNN
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import logging
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import h5py
import math
import torch_geometric.nn as tg_nn
from torch_geometric.data import Data, Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
import pandas as pd


# 设置设备
#torch.cuda.set_device(3) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


class LeaveOneSiteOutLoader:
    def __init__(self, file_path, graph_type, batch_size=32):
        self.file_path = file_path
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.sites = self._get_sites()
        
    def _get_sites(self):
        with h5py.File(self.file_path, "r") as f:
            # Dynamically get all sites from the HDF5 file
            sites = []
            for key in f["/experiments"].keys():
                if key.startswith(f"{self.graph_type}_leavesiteout-"):
                    site_name = key.split('-')[1]
                    sites.append(site_name)
            if not sites:
                raise ValueError("No sites found in the HDF5 file for the given graph type.")
            return sites
    
    def get_site_data(self, test_site):
        graph_dataset = []
        labels = []

        with h5py.File(self.file_path, "r") as f:
            # Get the experiment group for the specific site
            exp_group_path = f"/experiments/{self.graph_type}_leavesiteout-{test_site}"
            if exp_group_path not in f:
                raise ValueError(f"Could not find data for site {test_site} in the HDF5 file.")
            exp_group = f[exp_group_path]

            # --- START OF CORRECTION ---
            # Your HDF5 file has folds (groups '0', '1', ...). We'll select fold '0' by default.
            # The original code was missing this step.
            fold_to_use = '0'
            if fold_to_use not in exp_group:
                raise ValueError(f"Could not find fold '{fold_to_use}' for site {test_site}")
            
            fold_group = exp_group[fold_to_use]

            # Get pre-split dataset IDs from the selected fold
            test_subjects = [s.decode('utf-8') if isinstance(s, bytes) else s for s in fold_group["test"][:]]
            train_subjects = [s.decode('utf-8') if isinstance(s, bytes) else s for s in fold_group["train"][:]]
            val_subjects = [s.decode('utf-8') if isinstance(s, bytes) else s for s in fold_group["valid"][:]]
            # --- END OF CORRECTION ---
            
            print(f"\nData distribution for site {test_site} (using fold {fold_to_use}):")
            print(f"Test set: {len(test_subjects)} samples")
            print(f"Training set: {len(train_subjects)} samples")
            print(f"Validation set: {len(val_subjects)} samples")
            
            # Process all subject data
            patients_group = f["/patients"]
            
            # Iterate through all subjects in the patients group
            for subject_id in patients_group.keys():
                subject_group = patients_group[subject_id]
                if self.graph_type in subject_group:
                    # Extract data
                    triu_vector = subject_group[self.graph_type][:]
                    matrix = reconstruct_fc(triu_vector)
                    edge_index = self._get_brain_connectivity_edges(matrix)
                    label = torch.tensor(subject_group.attrs["y"], dtype=torch.long)
                    flat_vector = matrix.flatten()
                    
                    # Create graph data
                    graph_data = Data(
                        x=torch.FloatTensor(flat_vector),
                        edge_index=edge_index,
                        y=label
                    )
                    
                    # Assign to the corresponding dataset based on ID
                    if subject_id in test_subjects:
                        graph_dataset.append(("test", graph_data))
                        labels.append(("test", label))
                    elif subject_id in train_subjects:
                        graph_dataset.append(("train", graph_data))
                        labels.append(("train", label))
                    elif subject_id in val_subjects:
                        graph_dataset.append(("valid", graph_data))
                        labels.append(("valid", label))
            
            # Print the number of actually loaded samples
            test_count = len([d for d in graph_dataset if d[0] == "test"])
            train_count = len([d for d in graph_dataset if d[0] == "train"])
            valid_count = len([d for d in graph_dataset if d[0] == "valid"])
            
            print("\nNumber of samples actually loaded:")
            print(f"Test set: {test_count}/{len(test_subjects)} ({test_count/len(test_subjects):.2%})")
            print(f"Training set: {train_count}/{len(train_subjects)} ({train_count/len(train_subjects):.2%})")
            print(f"Validation set: {valid_count}/{len(val_subjects)} ({valid_count/len(val_subjects):.2%})")
        
        # Organize data
        data_splits = {
            "train": ([d[1] for d in graph_dataset if d[0] == "train"], 
                    [l[1] for l in labels if l[0] == "train"]),
            "valid": ([d[1] for d in graph_dataset if d[0] == "valid"], 
                    [l[1] for l in labels if l[0] == "valid"]),
            "test": ([d[1] for d in graph_dataset if d[0] == "test"], 
                    [l[1] for l in labels if l[0] == "test"])
        }
    
        return data_splits
    
    def _get_brain_connectivity_edges(self, matrix, threshold=0.3):
        rows, cols = np.triu_indices_from(matrix, k=1)
        mask = matrix[rows, cols] > threshold
        edge_index = np.array([rows[mask], cols[mask]])
        edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
        return torch.tensor(edge_index, dtype=torch.long)
    
    def _create_dataloader(self, data, labels):
        if len(data) == 0:  # Return None if the dataset is empty
            return None
        return DataLoader(
            data, 
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: Batch.from_data_list(batch)
        )
    
    def get_dataloaders(self, test_site):
        data_splits = self.get_site_data(test_site)
        return {
            "train": self._create_dataloader(*data_splits["train"]),
            "valid": self._create_dataloader(*data_splits["valid"]),
            "test": self._create_dataloader(*data_splits["test"]),
        }

class fMRI3DGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.graph_builder = nn.Sequential(
            nn.Linear(40000, 200*200),  # Convert 40000-dim input to a 200x200 matrix
            nn.Sigmoid()
        )
        # Modified convolutional layer definitions
        self.convs = nn.ModuleList([
            # First GAT layer: input feature dimension needs to match the enhanced node features
            tg_nn.GATv2Conv(
                in_channels=16,  # Changed to the dimension after feature enhancement
                out_channels=128,
                heads=8,
                dropout=config['dropout'],
                add_self_loops=False
            ),
            # Second GAT layer
            tg_nn.GATv2Conv(
                in_channels=128*8,  # Output dimension of multi-head attention
                out_channels=256,
                heads=4,
                dropout=config['dropout']
            ),
            # Third GAT layer
            tg_nn.GATv2Conv(
                in_channels=256*4,
                out_channels=512,
                heads=1,
                dropout=config['dropout']
            )
        ])
        
        # Added feature enhancement layer
        self.feature_enhancer = nn.Sequential(
            nn.Linear(2, 8),  # Expand node feature dimension
            nn.GELU(),
            nn.Linear(8, 16),
            nn.LayerNorm(16)
        )

        # Update classifier input dimension
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, config['num_classes'])
        )
    def build_graph(self, fc_matrix):
        """Corrected graph construction method"""
        batch_size = fc_matrix.size(0)
        
        # Generate adjacency matrix via graph_builder
        adj = self.graph_builder(fc_matrix).view(batch_size, 200, 200).float()
        adj = (adj + adj.transpose(1,2)) / 2  # Ensure symmetry

        # Generate enhanced node features (dimension validation)
        node_features = []
        edge_indices = []
        for b in range(batch_size):
            # Basic statistical features
            
            means = adj[b].mean(dim=1, keepdim=True)  # (200,1)
            stds = adj[b].std(dim=1, keepdim=True)    # (200,1)
            base_feat = torch.cat([means, stds], dim=1)  # (200,2)
            
            # Feature enhancement (output dimension 16)
            enhanced_feat = self.feature_enhancer(base_feat)  # (200,16)
            
            # Dynamic edge generation (with threshold limit)
            assert adj[b].dtype == torch.float32, f"Incorrect data type for adjacency matrix: {adj[b].dtype}"
            threshold = torch.quantile(adj[b].flatten(), 0.75)
            mask = (adj[b] > threshold).float()
            row, col = mask.nonzero(as_tuple=False).t()
            edge_index = torch.stack([row, col], dim=0)
            
            node_features.append(enhanced_feat)
            edge_indices.append(edge_index)

        return Batch.from_data_list([
            Data(x=feat, edge_index=edge) 
            for feat, edge in zip(node_features, edge_indices)
        ])
    

    def _generate_adaptive_edges(self, node_feat):
        """Dynamically generate edge connections"""
        # Spatial constraints
        spatial_dist = torch.cdist(self.spatial_emb.weight, self.spatial_emb.weight)
        
        # Feature similarity
        feat_sim = torch.mm(node_feat, node_feat.t())
        
        # Combined edge weights
        combined = (feat_sim * (1 / (spatial_dist + 1e-6)))
        
        # Generate adjacency matrix
        adj = (combined > self.threshold).float()
        
        # Ensure minimum number of connections
        topk = torch.topk(combined, self.k_neighbors, dim=1)
        adj[topk.indices] = 1.0
        
        return adj
   
    def forward(self, raw_fc):
        # Input dimension validation
        assert raw_fc.dim() == 2, f"Input should be a 2D tensor, but current dimension is: {raw_fc.dim()}"
        assert raw_fc.size(1) == 40000, f"Incorrect input feature dimension, expected 40000, but got {raw_fc.size(1)}"
        
        # Build dynamic graph
        batch_graph = self.build_graph(raw_fc)
        
        # Graph convolution processing
        x = batch_graph.x.to(raw_fc.device)
        edge_index = batch_graph.edge_index.to(raw_fc.device)
        
        for i, conv in enumerate(self.convs):
            # Dimension compatibility check
            assert x.size(1) == conv.in_channels, \
                f"Input dimension mismatch for layer {i+1}! Expected {conv.in_channels}, but got {x.size(1)}"
            
            x = conv(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # Global pooling
        x = tg_nn.global_mean_pool(x, batch_graph.batch)
        return x
    
    def classify(self, x):
        return self.classifier(x)
    
    def _adjust_model_parameters(self, new_dim):
        """Dynamically adjust model parameters"""
        # Adjust spatial embedding dimension
        old_emb = self.spatial_emb
        self.spatial_emb = nn.Embedding(new_dim, 3)
        with torch.no_grad():
            min_dim = min(old_emb.weight.size(0), new_dim)
            self.spatial_emb.weight[:min_dim] = old_emb.weight[:min_dim]
        
        # Adjust input dimension of graph convolutional layers
        if self.convs[0].in_channels != new_dim:
            first_conv = self.convs[0]
            new_conv = tg_nn.GATv2Conv(
                new_dim, 
                first_conv.out_channels,
                heads=first_conv.heads,
                dropout=first_conv.dropout
            )
            self.convs[0] = new_conv

def load_pretrained_mlp():
    
    microbe_loader = MicrobeDataLoader(
        csv_path="/home/yangzongxian/xlz/ASD_GCN/main/data2/microbe_data.csv",
        biom_path="/home/yangzongxian/xlz/ASD_GCN/main/data2/feature-table.biom"
    )
    feature_dim = microbe_loader.features_tensor.size(1)
    num_classes = len(torch.unique(microbe_loader.labels_tensor))
    
    
    model = SparseMLP(
        input_dim= 2503,
        num_classes= 2,
        hidden_dims=[2048, 1024, 512],
        dropout=0.6,
        sparsity_lambda=0.05
    )
    model.load_state_dict(torch.load("/home/yangzongxian/xlz/ASD_GCN/main/down/sparse_mlp.pth"))
    return model

def load_pretrained_gnn():
    GNN_CONFIG = {
        "gnn_layers": 3,
        "hidden_channels": 128,
        "num_classes": 2,
        "dropout": 0.4
    }
    model = fMRI3DGNN(GNN_CONFIG)
    checkpoint = torch.load("/home/yangzongxian/xlz/ASD_GCN/main/down/fmri_gnn.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)




class ContrastiveModel(nn.Module):
    def __init__(self, mlp_model, gnn_model, feat_dim=128):
        super().__init__()
        # Freeze pre-trained model parameters
        self.mlp = mlp_model
        self.gnn = gnn_model
        for param in mlp_model.parameters():
            param.requires_grad_(False)
        for param in gnn_model.parameters():
            param.requires_grad_(False)
        
        mlp_feat_dim = self.mlp.classifier.in_features  # Match MLP final layer input
        gnn_feat_dim = self.gnn.classifier[0].in_features  # Match GNN final layer input
        
        # Microbiome feature projection
        self.mlp_proj = nn.Sequential(
            nn.LayerNorm(mlp_feat_dim),
            nn.Linear(mlp_feat_dim, feat_dim),
            nn.GELU()
        )
        
        print(f"GNN feature dim: {gnn_feat_dim}")
        # fMRI feature projection
        self.gnn_proj = nn.Sequential(
            nn.LayerNorm(gnn_feat_dim),
            nn.Linear(gnn_feat_dim, feat_dim),
            nn.GELU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim*2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, microbe_input, fmri_input):
        # Modify GNN input processing
       
        #print(f"Microbe input shape: {microbe_input.shape}") #(64,2503)
        #print(f"fMRI input shape: {fmri_input.shape}")#(64,40000)
        mlp_feat = self.mlp.feature_extractor(microbe_input)
        gnn_feat = self.gnn(fmri_input)
            
        
        # Projection and classification remain unchanged
        h_microbe = F.normalize(self.mlp_proj(mlp_feat), dim=1)
        h_fmri = F.normalize(self.gnn_proj(gnn_feat), dim=1)
        combined = torch.cat([h_microbe, h_fmri], dim=1)
        return h_microbe, h_fmri, self.classifier(combined)
    

class LabelAwareContrastiveLoss(nn.Module):
    def __init__(self, temp=0.07, hard_neg_ratio=0.2):
        super().__init__()
        self.temp = temp
        self.hard_neg_ratio = hard_neg_ratio
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, h_microbe, h_fmri, labels):
        # Calculate inter-modality similarity matrix
        logits = torch.mm(h_microbe, h_fmri.T) / self.temp
        
        # Create label mask matrix
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        
       # Hard negative mining
        neg_mask = ~label_mask
        neg_logits = logits * neg_mask.float()
        
        # Select the hardest negative samples
        k = int(self.hard_neg_ratio * neg_mask.sum(dim=1).float().mean())
        hard_neg_indices = neg_logits.topk(k, dim=1).indices
        
        # Construct an enhanced target matrix
        targets = label_mask.float()
        targets.scatter_(1, hard_neg_indices, 0.5)  # Partial weight for hard negative samples
        
        # Symmetric loss calculation
        loss = -torch.mean(
            F.log_softmax(logits, dim=1) * targets +
            F.log_softmax(logits.T, dim=1) * targets.T
        )
        
        return loss
    
class PairedDataset(Dataset):
    def __init__(self, microbe_data, fmri_data, train=True):
        """
        microbe_data: (features, labels)
        fmri_data: (features, labels)
        """
        self.microbe_features, self.microbe_labels = microbe_data
        self.fmri_features, self.fmri_labels = fmri_data
        self.train = train
        
        # Build bimodal label index
        self.label_to_indices = {
            label: {
                'microbe': np.where(self.microbe_labels == label)[0],
                'fmri': np.where(self.fmri_labels == label)[0]
            }
            for label in np.unique(self.microbe_labels)
        }
        
        if not train:
            # Create fixed pairs for the test set
            self.fmri_pairs = [None] * len(self.microbe_features)
            for label in np.unique(self.microbe_labels):
                microbe_indices = self.label_to_indices[label]['microbe']
                fmri_indices = self.label_to_indices[label]['fmri']
                
                # Ensure enough fMRI samples
                if len(fmri_indices) < len(microbe_indices):
                    # If fMRI samples are insufficient, cycle through existing samples
                    fmri_indices = np.tile(fmri_indices, int(np.ceil(len(microbe_indices) / len(fmri_indices))))
                    fmri_indices = fmri_indices[:len(microbe_indices)]
                
                for m_idx, f_idx in zip(microbe_indices, fmri_indices):
                    self.fmri_pairs[m_idx] = f_idx

    def __len__(self):
        return len(self.microbe_features)

    def __getitem__(self, idx):
        microbe_feat = self.microbe_features[idx]
        label = self.microbe_labels[idx]
        
        if self.train:
            # Add random noise to the training set
            microbe_feat += np.random.normal(0, 0.1, size=microbe_feat.shape)
            # Randomly select an fMRI sample
            fmri_idx = np.random.choice(self.label_to_indices[label]['fmri'])
            fmri_feat = self.fmri_features[fmri_idx]
        else:
            # Do not add noise to the test set, use fixed pairs
            fmri_idx = self.fmri_pairs[idx]
            fmri_feat = self.fmri_features[fmri_idx]
        
        return {
            'microbe': torch.FloatTensor(microbe_feat),
            'fmri': torch.FloatTensor(fmri_feat),
            'label': torch.LongTensor([label])
        }
        

    
    
def train_contrastive_adversarial(model, train_loader, optimizer, criterion, device, epsilon=0.1):
    model.train()
    total_loss = 0.0
    accuracies = []
    with tqdm(train_loader, desc="Training", unit="batch") as bar:
        for batch in bar:
            microbe = batch['microbe'].to(device)
            fmri = batch['fmri'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            # Generate adversarial microbiome inputs
            microbe.requires_grad = True
            h_m, h_f, logits = model(microbe, fmri)
            loss = criterion(h_m, h_f, labels) + 0.5 * F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, microbe)[0]
            delta = epsilon * torch.sign(grad)
            microbe_adv = microbe + delta
            microbe_adv = microbe_adv.detach()
            
            # Train with adversarial inputs
            optimizer.zero_grad()
            h_m_adv, h_f, logits_adv = model(microbe_adv, fmri)
            loss_adv = criterion(h_m_adv, h_f, labels) + 0.5 * F.cross_entropy(logits_adv, labels)
            loss_adv.backward()
            optimizer.step()
            
            # Calculate metrics
            preds = torch.argmax(logits_adv, dim=1)
            acc = (preds == labels).float().mean()
            total_loss += loss_adv.item()
            accuracies.append(acc.item())
            bar.set_postfix({
                'loss': f"{total_loss/(bar.n+1):.4f}",
                'acc': f"{np.mean(accuracies):.2%}"
            })
    return total_loss / len(train_loader), np.mean(accuracies)


 
def validate(model, criterion, valid_loader, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_pred = []
    with torch.no_grad():
        for batch in valid_loader:
            microbe = batch['microbe'].to(device)
            fmri = batch['fmri'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            h_m, h_f, logits = model(microbe, fmri)
            loss = criterion(h_m, h_f, labels) + 0.5 * F.cross_entropy(logits, labels)
            total_loss += loss.item()
            
            pred = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_pred)
    avg_loss = total_loss / len(valid_loader)
    return avg_loss, accuracy

    
def test_adversarial(model, test_loader, device, criterion, epsilon=0.1):
    model.eval()
    all_labels = []
    all_pred_original = []
    all_pred_perturbed = []
    
    for batch in test_loader:
        microbe = batch['microbe'].to(device)
        fmri = batch['fmri'].to(device)
        labels = batch['label'].squeeze().to(device)
        
        # Original prediction
        with torch.no_grad():
            _, _, logits_original = model(microbe, fmri)
            pred_original = torch.argmax(logits_original, dim=1)
        
        # Generate perturbed microbiome inputs
        microbe = microbe.clone().detach().requires_grad_(True)
        h_m, h_f, _ = model(microbe, fmri)
        loss = criterion(h_m, h_f, labels) + 0.5 * F.cross_entropy(logits_original, labels)
        grad = torch.autograd.grad(loss, microbe)[0]
        delta = epsilon * torch.sign(grad)
        microbe_perturbed = microbe + delta
        
        # Perturbed prediction
        with torch.no_grad():
            _, _, logits_perturbed = model(microbe_perturbed, fmri)
            pred_perturbed = torch.argmax(logits_perturbed, dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_pred_original.extend(pred_original.cpu().numpy())
        all_pred_perturbed.extend(pred_perturbed.cpu().numpy())
    
    acc_original = accuracy_score(all_labels, all_pred_original)
    acc_perturbed = accuracy_score(all_labels, all_pred_perturbed)
    return acc_original, acc_perturbed
    
def reconstruct_fc(vector):
    """Reconstruct a symmetric matrix from an upper triangular vector"""
    # Create an empty matrix
    matrix = np.zeros((200, 200))
    # Extract upper triangular indices (excluding the diagonal)
    triu_indices = np.triu_indices(200, k=1)
    # Fill the upper triangle
    matrix[triu_indices] = vector
    # Symmetrically copy to the lower triangle
    matrix = matrix + matrix.T - np.diag(matrix.diagonal())    
    return matrix



def extract_graph_features(loader):
    """Extract global features from a PyG data loader"""
    all_features = []
    all_labels = []
    #gnn_model.to(device)
    for batch in loader:
        batch = batch.to(device)
        raw_features = batch.x.view(batch.num_graphs, -1).cpu().numpy()  # 40000-dim
        all_features.append(raw_features)
        all_labels.append(batch.y.cpu().numpy())
    
    return np.concatenate(all_features), np.concatenate(all_labels)

def extract_microbe_features(loader):
    all_features = []
    all_labels = []
    for batch in loader:
        if not batch:  # Check for empty batches
            continue
        features, labels = batch[0].numpy(), batch[1].numpy()
        if features.shape[0] == 0:  # Check for empty data
            continue
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

def expand_to_full_matrix(vector):
    """Convert a 19900-dim upper triangular vector to a 40000-dim flattened full matrix"""
    assert len(vector) == 19900, f"Input should be 19900-dimensional, but is {len(vector)}"
    matrix = np.zeros((200, 200))
    triu_indices = np.triu_indices(200, k=1)
    matrix[triu_indices] = vector
    matrix = matrix + matrix.T  # Symmetric copy
    np.fill_diagonal(matrix, 1.0)  # Ensure diagonal is 1.0
    return matrix.flatten()  # Return 40000-dim vector

def normalize_features(features, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
    return (features - mean) / std, mean, std

# Special handling for fMRI data (consistent with GNN pre-processing)
def process_fmri_features(features):
    """Process into 40000-dimensional input required by GNN"""
    # Add noise
    noise = np.random.normal(scale=0.1, size=features.shape)
    noisy_features = np.clip(features + noise, -1, 1)
    
    # Convert to 40000 dimensions
    expanded_features = np.array([expand_to_full_matrix(vec) for vec in noisy_features])
    assert expanded_features.shape[1] == 40000, "Dimension of processed features should be 40000"
    return expanded_features  # Return (n_samples, 40000)

def get_accuracy_from_report(report_str):
    """Safely extract accuracy from a classification report"""
    for line in report_str.split('\n'):
        if 'accuracy' in line:
            return float(line.split()[-1])
    raise ValueError("Could not find accuracy information in the report")

def main():
    # Load pre-trained models
    mlp_model = load_pretrained_mlp()
    gnn_model = load_pretrained_gnn()

    file_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/abide.hdf5"
    graph_type = "cc200"
    csv_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/microbe_data.csv"
    biom_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/feature-table.biom"

    # Store results for all sites
    all_site_results = []
    
    # Create the loader once to get all sites
    fmri_loader = LeaveOneSiteOutLoader(file_path=file_path, graph_type=graph_type, batch_size=32)
    all_sites = fmri_loader.sites
    print(f"Found {len(all_sites)} sites to process: {all_sites}")

    # Train and validate on each site
    for site_name in all_sites:
        print(f"\nTraining on site: {site_name}")
        
        # Create a microbe data loader
        microbe_loader = MicrobeDataLoader(csv_path=csv_path, biom_path=biom_path, batch_size=32)
        
        # Get data loaders for the current site
        fmri_loaders = fmri_loader.get_dataloaders(site_name)
        fmri_train_loader = fmri_loaders["train"]
        fmri_val_loader = fmri_loaders["valid"]
        fmri_test_loader = fmri_loaders["test"]
        
        microbe_train_loader = microbe_loader.get_loaders()[0]
        microbe_val_loader = microbe_loader.get_loaders()[1]
        microbe_test_loader = microbe_loader.get_loaders()[2]

        # Extract features
        fmri_train_features, fmri_train_labels = extract_graph_features(fmri_train_loader)
        fmri_val_features, fmri_val_labels = extract_graph_features(fmri_val_loader)
        fmri_test_features, fmri_test_labels = extract_graph_features(fmri_test_loader)

        microbe_train_features, microbe_train_labels = extract_microbe_features(microbe_train_loader)
        microbe_val_features, microbe_val_labels = extract_microbe_features(microbe_val_loader)
        microbe_test_features, microbe_test_labels = extract_microbe_features(microbe_test_loader)

        # Normalize features
        microbe_train_features, microbe_mean, microbe_std = normalize_features(microbe_train_features)
        microbe_val_features = normalize_features(microbe_val_features, microbe_mean, microbe_std)[0]
        microbe_test_features = normalize_features(microbe_test_features, microbe_mean, microbe_std)[0]

        # Create contrastive model
        contrast_model = ContrastiveModel(mlp_model, gnn_model).to(device)
        contrast_model.mlp.eval()
        contrast_model.gnn.eval()

        # Prepare data
        microbe_data = (microbe_train_features, microbe_train_labels)
        fmri_data = (fmri_train_features, fmri_train_labels)
        paired_dataset = PairedDataset(microbe_data, fmri_data)
        train_loader = DataLoader(paired_dataset, batch_size=64, shuffle=True)

        # Create validation dataset
        valid_paired_dataset = PairedDataset(
            (microbe_val_features, microbe_val_labels),
            (fmri_val_features, fmri_val_labels)
        )
        valid_loader = DataLoader(valid_paired_dataset, batch_size=64, shuffle=False)

        # Training configuration
        optimizer = torch.optim.AdamW([
            {'params': contrast_model.mlp_proj.parameters()},
            {'params': contrast_model.gnn_proj.parameters()},
            {'params': contrast_model.classifier.parameters()}
        ], lr=1e-4, weight_decay=1e-5)
        criterion = LabelAwareContrastiveLoss(temp=0.05, hard_neg_ratio=0.2)

        # Training parameters
        patience = 20
        best_valid_acc = 0
        best_valid_loss = float('inf')
        counter = 0
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
        logs = {'epoch': [], 'train_loss': [], 'train_acc': [], 'valid_acc': []}

        # Training loop
        for epoch in range(100):  # Train for a maximum of 100 epochs
            train_loss, train_acc = train_contrastive_adversarial(
                contrast_model, train_loader, optimizer, criterion, device
            )
            valid_loss, valid_acc = validate(contrast_model, criterion, valid_loader, device)
            
            # Update learning rate
            scheduler.step(valid_acc)
            
            # Log records
            logs['epoch'].append(epoch)
            logs['train_loss'].append(train_loss)
            logs['train_acc'].append(train_acc)
            logs['valid_acc'].append(valid_acc)
            
            print(f"Epoch {epoch+1}/100:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
            print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2%}")
            
            # Early stopping check
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_valid_loss = valid_loss
                counter = 0
                # Save the best model
                torch.save(contrast_model.state_dict(), f"best_model_site_{site_name}.pth")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load the best model for testing
        best_model = ContrastiveModel(mlp_model, gnn_model).to(device)
        best_model.load_state_dict(torch.load(f"best_model_site_{site_name}.pth"))

        # Create test dataset
        test_paired_dataset = PairedDataset(
            (microbe_test_features, microbe_test_labels),
            (fmri_test_features, fmri_test_labels),
            train=False
        )
        test_loader = DataLoader(test_paired_dataset, batch_size=64, shuffle=False)

        # Perform testing
        test_loss, test_acc = validate(best_model, criterion, test_loader, device)
        print(f"Test accuracy for site {site_name}: {test_acc:.2%}")

        # Perform adversarial testing
        acc_original, acc_perturbed = test_adversarial(best_model, test_loader, device, criterion)
        print(f"Original test accuracy for site {site_name}: {acc_original:.2%}")
        print(f"Adversarial test accuracy for site {site_name}: {acc_perturbed:.2%}")

        # Store results
        all_site_results.append({
            'site': site_name,
            'test_acc': test_acc,
            'original_acc': acc_original,
            'perturbed_acc': acc_perturbed
        })

    # Calculate and print average results
    avg_test_acc = np.mean([r['test_acc'] for r in all_site_results])
    avg_original_acc = np.mean([r['original_acc'] for r in all_site_results])
    avg_perturbed_acc = np.mean([r['perturbed_acc'] for r in all_site_results])
    
    print("\n=== Overall Results ===")
    print(f"Average test accuracy: {avg_test_acc:.2%}")
    print(f"Average original test accuracy: {avg_original_acc:.2%}")
    print(f"Average adversarial test accuracy: {avg_perturbed_acc:.2%}")

if __name__ == "__main__":
    main()