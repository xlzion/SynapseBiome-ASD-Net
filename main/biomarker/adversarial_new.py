#对应contrastive_model1.pth

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
#torch.cuda.set_device(2) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class fMRIDataLoader:
    def __init__(self, file_path, graph_type, test_size=0.15, val_size=0.15, batch_size=32):
        self.file_path = file_path
        self.graph_type = graph_type
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.data_splits = self._load_and_split_data()
    
    def _load_and_split_data(self):
        graph_dataset = []
        labels = []
        
        with h5py.File(self.file_path, "r") as f:
            patients_group = f["/patients"]
            for subject_id in patients_group.keys():
                subject_group = patients_group[subject_id]
                if self.graph_type in subject_group:
                    triu_vector = subject_group[self.graph_type][:]
                    matrix = reconstruct_fc(triu_vector)
                    edge_index = self._get_brain_connectivity_edges(matrix)
                    #node_features = torch.FloatTensor(matrix).unsqueeze(1)  # (200,1)
                    '''graph_data = Data(
                        x=node_features,
                        edge_index=edge_index,
                        y=subject_group.attrs["y"]
                    )'''
                    label = torch.tensor(subject_group.attrs["y"], dtype=torch.long)
                    triu_vector = subject_group[self.graph_type][:]
                    matrix = reconstruct_fc(triu_vector)  # 二维矩阵（用于构建边）
                    flat_vector = matrix.flatten() 
                    graph_data = Data(
                        x=torch.FloatTensor(flat_vector),  # 存储展平后的40000维向量
                        edge_index=edge_index,
                        y=label
                    )
                    graph_dataset.append(graph_data)
                    labels.append(subject_group.attrs["y"])
        
        # 数据集划分
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            graph_dataset, labels, test_size=self.test_size, random_state=42
        )
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_val_data, train_val_labels, test_size=self.val_size/(1-self.test_size), random_state=42
        )
        
        return {
            "train": (train_data, train_labels),
            "valid": (val_data, val_labels),
            "test": (test_data, test_labels),
        }
    
    def _get_brain_connectivity_edges(self, matrix, threshold=0.3):
        """生成边索引（优化版本）"""
        # 创建全连接（考虑对称性）
        rows, cols = np.triu_indices_from(matrix, k=1)
        mask = matrix[rows, cols] > threshold
        edge_index = np.array([rows[mask], cols[mask]])
        
        # 添加反向边
        edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
        
        return torch.tensor(edge_index, dtype=torch.long)

    def _create_dataloader(self, data, labels):
        return DataLoader(
            data, 
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: Batch.from_data_list(batch)
        )
    
    def get_dataloaders(self):
        return {
            "train": self._create_dataloader(*self.data_splits["train"]),
            "valid": self._create_dataloader(*self.data_splits["valid"]),
            "test": self._create_dataloader(*self.data_splits["test"]),
        }

    def get_num_classes(self):
        return len(set(self.data_splits["train"][1]))

class fMRI3DGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.graph_builder = nn.Sequential(
            nn.Linear(40000, 200*200),
            nn.Sigmoid()
        )
        self.convs = nn.ModuleList([
            tg_nn.GATv2Conv(
                in_channels=16,
                out_channels=128,
                heads=8,
                dropout=config['dropout'],
                add_self_loops=False
            ),
            tg_nn.GATv2Conv(
                in_channels=128*8,
                out_channels=256,
                heads=4,
                dropout=config['dropout']
            ),
            tg_nn.GATv2Conv( # Last GAT layer
                in_channels=256*4,
                out_channels=512, # Output feature dimension before global pool
                heads=1,
                concat=False, # Important for single-head output before pooling
                dropout=config['dropout']
            )
        ])
        self.feature_enhancer = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 16),
            nn.LayerNorm(16)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), # Input matches last GAT layer's out_channels
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, config['num_classes'])
        )

    def build_graph(self, fc_matrix):
        # This function remains as you defined it in adversarial.py
        # It converts the 40000-dim fc_matrix to a batch of PyG Data objects
        # where x is the enhanced node features (16-dim) and edge_index is dynamic.
        batch_size = fc_matrix.size(0)
        adj = self.graph_builder(fc_matrix).view(batch_size, 200, 200).float()
        adj = (adj + adj.transpose(1,2)) / 2
        node_features_list = []
        edge_indices_list = []
        for b in range(batch_size):
            means = adj[b].mean(dim=1, keepdim=True)
            stds = adj[b].std(dim=1, keepdim=True)
            base_feat = torch.cat([means, stds + 1e-6], dim=1) # Added epsilon for std stability
            enhanced_feat = self.feature_enhancer(base_feat)
            threshold = torch.quantile(adj[b].flatten(), 0.75)
            mask = (adj[b] > threshold) # .float() not needed for .nonzero
            row, col = mask.nonzero(as_tuple=True) # Use as_tuple=True
            edge_index = torch.stack([row, col], dim=0)
            node_features_list.append(enhanced_feat)
            edge_indices_list.append(edge_index)
        
        # Ensure all tensors in Data are on the same device before Batch.from_data_list
        # This might be fc_matrix.device or a default device
        target_device = node_features_list[0].device if node_features_list else fc_matrix.device
        return Batch.from_data_list([
            Data(x=feat, edge_index=edge.to(target_device))
            for feat, edge in zip(node_features_list, edge_indices_list)
        ])

    # MODIFIED forward method for fMRI3DGNN
    def forward(self, raw_fc, return_attention_scores=False): # Added return_attention_scores flag
        if raw_fc.dim() != 2 or raw_fc.size(1) != 40000:
             raise ValueError(f"fMRI3DGNN input error: shape {raw_fc.shape}, expected [batch_size, 40000]")

        # Ensure raw_fc is on the same device as the model's parameters
        # This is important if device management is not handled consistently outside.
        # current_device = next(self.parameters()).device
        # raw_fc = raw_fc.to(current_device)

        batch_graph = self.build_graph(raw_fc)
        
        # Move graph data to the same device as raw_fc (which should be model's device)
        x = batch_graph.x.to(raw_fc.device)
        edge_index = batch_graph.edge_index.to(raw_fc.device)
        # batch_map is crucial for global_pool and for interpreting attention scores later
        batch_map = batch_graph.batch.to(raw_fc.device) 
        
        attention_scores_list = []

        for i, conv_layer in enumerate(self.convs):
            if x.size(1) != conv_layer.in_channels:
                raise ValueError(
                    f"Dim mismatch in fMRI3DGNN GAT layer {i+1}: Expected {conv_layer.in_channels}, got {x.size(1)}"
                )
            if return_attention_scores:
                # GATv2Conv returns (out_features, (edge_index, alpha_attention_coeffs))
                x, (edge_index_att, alpha) = conv_layer(x, edge_index, return_attention_weights=True)
                attention_scores_list.append({
                    "layer_index": i,
                    "edge_index_gat": edge_index_att, # Edge index used by this GAT layer
                    "attention_coeffs": alpha,      # Attention coefficients (num_edges_in_batch_graph, num_heads)
                    "batch_map_nodes": batch_map    # Node to graph mapping for the current batch_graph
                })
            else:
                x = conv_layer(x, edge_index)
            
            x = F.gelu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        x_pooled = tg_nn.global_mean_pool(x, batch_map) # Use batch_map from the graph
        
        if return_attention_scores:
            return x_pooled, attention_scores_list
        else:
            return x_pooled
    
    def classify(self, x): # x here is x_pooled from forward
        return self.classifier(x)

    # _generate_adaptive_edges, _adjust_model_parameters are not used in current forward path

# ... (load_pretrained_mlp and load_pretrained_gnn remain as in your adversarial.py)
# Ensure load_pretrained_gnn uses the GNN_CONFIG that matches the fMRI3DGNN structure
# and that the loaded state_dict is compatible.

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
    checkpoint = torch.load("/home/yangzongxian/xlz/ASD_GCN/main/down/fmri_gnn_best_lr.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)



# --- MODIFIED fMRI3DGNN ---
class fMRI3DGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.graph_builder = nn.Sequential(
            nn.Linear(40000, 200*200),
            nn.Sigmoid()
        )
        self.convs = nn.ModuleList([
            tg_nn.GATv2Conv(
                in_channels=16,
                out_channels=128,
                heads=8,
                dropout=config['dropout'],
                add_self_loops=False
            ),
            tg_nn.GATv2Conv(
                in_channels=128*8,
                out_channels=256,
                heads=4,
                dropout=config['dropout']
            ),
            tg_nn.GATv2Conv( # Last GAT layer
                in_channels=256*4,
                out_channels=512, # Output feature dimension before global pool
                heads=1,
                concat=False, # Important for single-head output before pooling
                dropout=config['dropout']
            )
        ])
        self.feature_enhancer = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 16),
            nn.LayerNorm(16)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), # Input matches last GAT layer's out_channels
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, config['num_classes'])
        )

    def build_graph(self, fc_matrix):
        # This function remains as you defined it in adversarial.py
        # It converts the 40000-dim fc_matrix to a batch of PyG Data objects
        # where x is the enhanced node features (16-dim) and edge_index is dynamic.
        batch_size = fc_matrix.size(0)
        adj = self.graph_builder(fc_matrix).view(batch_size, 200, 200).float()
        adj = (adj + adj.transpose(1,2)) / 2
        node_features_list = []
        edge_indices_list = []
        for b in range(batch_size):
            means = adj[b].mean(dim=1, keepdim=True)
            stds = adj[b].std(dim=1, keepdim=True)
            base_feat = torch.cat([means, stds + 1e-6], dim=1) # Added epsilon for std stability
            enhanced_feat = self.feature_enhancer(base_feat)
            threshold = torch.quantile(adj[b].flatten(), 0.75)
            mask = (adj[b] > threshold) # .float() not needed for .nonzero
            row, col = mask.nonzero(as_tuple=True) # Use as_tuple=True
            edge_index = torch.stack([row, col], dim=0)
            node_features_list.append(enhanced_feat)
            edge_indices_list.append(edge_index)
        
        # Ensure all tensors in Data are on the same device before Batch.from_data_list
        # This might be fc_matrix.device or a default device
        target_device = node_features_list[0].device if node_features_list else fc_matrix.device
        return Batch.from_data_list([
            Data(x=feat, edge_index=edge.to(target_device))
            for feat, edge in zip(node_features_list, edge_indices_list)
        ])

    # MODIFIED forward method for fMRI3DGNN
    def forward(self, raw_fc, return_attention_scores=False): # Added return_attention_scores flag
        if raw_fc.dim() != 2 or raw_fc.size(1) != 40000:
             raise ValueError(f"fMRI3DGNN input error: shape {raw_fc.shape}, expected [batch_size, 40000]")

        # Ensure raw_fc is on the same device as the model's parameters
        # This is important if device management is not handled consistently outside.
        # current_device = next(self.parameters()).device
        # raw_fc = raw_fc.to(current_device)

        batch_graph = self.build_graph(raw_fc)
        
        # Move graph data to the same device as raw_fc (which should be model's device)
        x = batch_graph.x.to(raw_fc.device)
        edge_index = batch_graph.edge_index.to(raw_fc.device)
        # batch_map is crucial for global_pool and for interpreting attention scores later
        batch_map = batch_graph.batch.to(raw_fc.device) 
        
        attention_scores_list = []

        for i, conv_layer in enumerate(self.convs):
            if x.size(1) != conv_layer.in_channels:
                raise ValueError(
                    f"Dim mismatch in fMRI3DGNN GAT layer {i+1}: Expected {conv_layer.in_channels}, got {x.size(1)}"
                )
            if return_attention_scores:
                # GATv2Conv returns (out_features, (edge_index, alpha_attention_coeffs))
                x, (edge_index_att, alpha) = conv_layer(x, edge_index, return_attention_weights=True)
                attention_scores_list.append({
                    "layer_index": i,
                    "edge_index_gat": edge_index_att, # Edge index used by this GAT layer
                    "attention_coeffs": alpha,      # Attention coefficients (num_edges_in_batch_graph, num_heads)
                    "batch_map_nodes": batch_map    # Node to graph mapping for the current batch_graph
                })
            else:
                x = conv_layer(x, edge_index)
            
            x = F.gelu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        x_pooled = tg_nn.global_mean_pool(x, batch_map) # Use batch_map from the graph
        
        if return_attention_scores:
            return x_pooled, attention_scores_list
        else:
            return x_pooled
    
    def classify(self, x): # x here is x_pooled from forward
        return self.classifier(x)

    # _generate_adaptive_edges, _adjust_model_parameters are not used in current forward path

# ... (load_pretrained_mlp and load_pretrained_gnn remain as in your adversarial.py)
# Ensure load_pretrained_gnn uses the GNN_CONFIG that matches the fMRI3DGNN structure
# and that the loaded state_dict is compatible.

# --- MODIFIED ContrastiveModel ---
class ContrastiveModel(nn.Module):
    def __init__(self, mlp_model, gnn_model, feat_dim=128):
        super().__init__()
        self.mlp = mlp_model
        self.gnn = gnn_model # This is an instance of the modified fMRI3DGNN

        # Freeze parameters (as in your original code)
        for param in self.mlp.parameters():
            param.requires_grad_(False)
        for param in self.gnn.parameters():
            param.requires_grad_(False)
        
        # Determine input dimensions for projection layers
        # For MLP: output of mlp.feature_extractor()
        # For GNN: output of gnn.forward() (which is x_pooled from fMRI3DGNN)
        # These need to be correct based on your actual SparseMLP and fMRI3DGNN output dims.
        # Example: Assuming SparseMLP's feature_extractor output is 512
        # and fMRI3DGNN's pooled output is 512.
        
        # A more robust way to get mlp_feat_dim would be to run a dummy input through mlp.feature_extractor
        # or have it as a property of SparseMLP.
        # Assuming mlp.classifier.in_features is the output of the feature_extractor part of SparseMLP
        # This depends on your SparseMLP's structure. If feature_extractor is nn.Sequential,
        # the in_features of the *next* layer (the classifier) is the out_features of feature_extractor.
        
        # If SparseMLP's structure is: self.feature_extractor -> self.classifier
        # then self.mlp.classifier.in_features is the output dim of self.mlp.feature_extractor
        mlp_intermediate_feat_dim = self.mlp.classifier.in_features # From SparseMLP
        
        # The gnn_feat from self.gnn() is the x_pooled, which has dimension config['gnn_output_dim'] (e.g., 512)
        # This becomes the input to gnn_proj
        # The gnn.classifier[0].in_features in your original code was for the GNN's *own* classifier,
        # not necessarily the dimension of gnn_feat passed to the projection head here.
        # The fMRI3DGNN.forward() returns x_pooled which has 'gnn_output_dim' (e.g. 512 from GAT config)
        gnn_intermediate_feat_dim = 512 # Should match fMRI3DGNN's pooled output dimension
                                       # This often corresponds to the out_channels of the last GAT layer if concat=False

        self.mlp_proj = nn.Sequential(
            nn.LayerNorm(mlp_intermediate_feat_dim),
            nn.Linear(mlp_intermediate_feat_dim, feat_dim),
            nn.GELU()
        )
        
        # print(f"ContrastiveModel: GNN feature dim for projection input: {gnn_intermediate_feat_dim}")
        self.gnn_proj = nn.Sequential(
            nn.LayerNorm(gnn_intermediate_feat_dim),
            nn.Linear(gnn_intermediate_feat_dim, feat_dim),
            nn.GELU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim*2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2) # Assuming 2 classes
        )

    # MODIFIED forward method for ContrastiveModel
    def forward(self, microbe_input, fmri_input, return_gnn_attention=False): # Added flag
        # Ensure inputs are on the same device as the model
        # current_device = next(self.parameters()).device
        # microbe_input = microbe_input.to(current_device)
        # fmri_input = fmri_input.to(current_device)

        if not hasattr(self.mlp, 'feature_extractor'):
             raise AttributeError("MLP model (SparseMLP) must have a 'feature_extractor' attribute/method.")
        mlp_feat = self.mlp.feature_extractor(microbe_input)
        
        gnn_attention_scores = None
        if return_gnn_attention:
            # gnn.forward now can return (x_pooled, attention_scores_list)
            gnn_feat, gnn_attention_scores = self.gnn(fmri_input, return_attention_scores=True)
        else:
            gnn_feat = self.gnn(fmri_input, return_attention_scores=False)
            
        h_microbe = F.normalize(self.mlp_proj(mlp_feat), dim=1)
        h_fmri = F.normalize(self.gnn_proj(gnn_feat), dim=1)
        
        combined = torch.cat([h_microbe, h_fmri], dim=1)
        final_logits = self.classifier(combined)
        
        if return_gnn_attention:
            return h_microbe, h_fmri, final_logits, gnn_attention_scores
        else:
            return h_microbe, h_fmri, final_logits
    

class LabelAwareContrastiveLoss(nn.Module):
    def __init__(self, temp=0.07, hard_neg_ratio=0.2):
        super().__init__()
        self.temp = temp
        self.hard_neg_ratio = hard_neg_ratio
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, h_microbe, h_fmri, labels):
        # 计算模态间相似度矩阵
        logits = torch.mm(h_microbe, h_fmri.T) / self.temp
        
        # 创建标签掩码矩阵
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        
       # 硬负样本挖掘
        neg_mask = ~label_mask
        neg_logits = logits * neg_mask.float()
        
        # 选择最难负样本
        k = int(self.hard_neg_ratio * neg_mask.sum(dim=1).float().mean())
        hard_neg_indices = neg_logits.topk(k, dim=1).indices
        
        # 构建增强后的目标矩阵
        targets = label_mask.float()
        targets.scatter_(1, hard_neg_indices, 0.5)  # 硬负样本部分权重
        
        # 对称损失计算
        loss = -torch.mean(
            F.log_softmax(logits, dim=1) * targets +
            F.log_softmax(logits.T, dim=1) * targets.T
        )
        
        return loss
    
class PairedDataset(Dataset):
    def __init__(self, microbe_data, fmri_data):
        """
        microbe_data: (features, labels)
        fmri_data: (features, labels)
        """
        self.microbe_features, self.microbe_labels = microbe_data
        self.fmri_features, self.fmri_labels = fmri_data
        
        # 建立双模态标签索引
        self.label_to_indices = {
            label: {
                'microbe': np.where(self.microbe_labels == label)[0],
                'fmri': np.where(self.fmri_labels == label)[0]
            }
            for label in np.unique(self.microbe_labels)
        }

    def __len__(self):
        return len(self.microbe_features)

    def __getitem__(self, idx):
        microbe_feat = self.microbe_features[idx]
        label = self.microbe_labels[idx]
        
        # 微生物组数据添加随机噪声
        microbe_feat = self.microbe_features[idx] + np.random.normal(0, 0.1, size=self.microbe_features[idx].shape)
        # 随机选择同类别fMRI样本
        fmri_idx = np.random.choice(self.label_to_indices[label]['fmri'])
        fmri_feat = self.fmri_features[fmri_idx]
        
        return {
            'microbe': torch.FloatTensor(microbe_feat),
            'fmri': torch.FloatTensor(fmri_feat),
            'label': torch.LongTensor([label])
        }
        
'''def train_contrastive(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    accuracies = []
    
    with tqdm(train_loader, desc="Training", unit="batch") as bar:
        for batch in bar:
            microbe = batch['microbe'].to(device)
            fmri = batch['fmri'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            optimizer.zero_grad()
            
            h_microbe, h_fmri, logits = model(microbe, fmri)
            
            # 计算对比损失
            contrast_loss = criterion(h_microbe, h_fmri, labels)
            
            # 计算分类损失
            cls_loss = F.cross_entropy(logits, labels)
            
            # 总损失
            loss = contrast_loss + 0.5 * cls_loss
            
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            
            total_loss += loss.item()
            accuracies.append(acc.item())
            
            bar.set_postfix({
                'loss': f"{total_loss/(bar.n+1):.4f}",
                'acc': f"{np.mean(accuracies):.2%}"
            })
    
    return total_loss / len(train_loader), np.mean(accuracies)'''
    
    
def train_contrastive_adversarial(model, train_loader, optimizer, criterion, device, epsilon=0.1):
    model.train()
    total_loss = 0.0
    accuracies = []
    with tqdm(train_loader, desc="Training", unit="batch") as bar:
        for batch in bar:
            microbe = batch['microbe'].to(device)
            fmri = batch['fmri'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            # 生成对抗性微生物组输入
            microbe.requires_grad = True
            h_m, h_f, logits = model(microbe, fmri)
            loss = criterion(h_m, h_f, labels) + 0.5 * F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, microbe)[0]
            delta = epsilon * torch.sign(grad)
            microbe_adv = microbe + delta
            microbe_adv = microbe_adv.detach()
            
            # 使用对抗性输入训练
            optimizer.zero_grad()
            h_m_adv, h_f, logits_adv = model(microbe_adv, fmri)
            loss_adv = criterion(h_m_adv, h_f, labels) + 0.5 * F.cross_entropy(logits_adv, labels)
            loss_adv.backward()
            optimizer.step()
            
            # 计算指标
            preds = torch.argmax(logits_adv, dim=1)
            acc = (preds == labels).float().mean()
            total_loss += loss_adv.item()
            accuracies.append(acc.item())
            bar.set_postfix({
                'loss': f"{total_loss/(bar.n+1):.4f}",
                'acc': f"{np.mean(accuracies):.2%}"
            })
    return total_loss / len(train_loader), np.mean(accuracies)

    
'''def validate(model, valid_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in valid_loader:
            microbe = batch['microbe'].to(device)
            fmri = batch['fmri'].to(device)
            labels = batch['label'].squeeze().to(device)
            _, _, logits = model(microbe, fmri)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy'''
 
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
    with torch.no_grad():
        for batch in test_loader:
            microbe = batch['microbe'].to(device)
            fmri = batch['fmri'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            # 原始预测
            _, _, logits_original = model(microbe, fmri)
            pred_original = torch.argmax(logits_original, dim=1)
            
            # 生成扰动后的微生物组输入
            microbe.requires_grad = True
            h_m, h_f, _ = model(microbe, fmri)
            loss = criterion(h_m, h_f, labels) + 0.5 * F.cross_entropy(logits_original, labels)
            grad = torch.autograd.grad(loss, microbe)[0]
            delta = epsilon * torch.sign(grad)
            microbe_perturbed = microbe + delta
            
            # 扰动后预测
            _, _, logits_perturbed = model(microbe_perturbed, fmri)
            pred_perturbed = torch.argmax(logits_perturbed, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_pred_original.extend(pred_original.cpu().numpy())
            all_pred_perturbed.extend(pred_perturbed.cpu().numpy())
    
    acc_original = accuracy_score(all_labels, all_pred_original)
    acc_perturbed = accuracy_score(all_labels, all_pred_perturbed)
    return acc_original, acc_perturbed
    
def reconstruct_fc(vector):
    """将上三角向量重建为对称矩阵"""
    # 创建空矩阵
    matrix = np.zeros((200, 200))
    # 提取上三角索引（不包括对角线）
    triu_indices = np.triu_indices(200, k=1)
    # 填充上三角
    matrix[triu_indices] = vector
    # 对称复制到下三角
    matrix = matrix + matrix.T - np.diag(matrix.diagonal())    
    return matrix



def extract_graph_features(loader):
    """从PyG数据加载器中提取全局特征"""
    all_features = []
    all_labels = []
    gnn_model.to(device)
    for batch in loader:
        batch = batch.to(device)
        raw_features = batch.x.view(batch.num_graphs, -1).cpu().numpy()  # 40000维
        all_features.append(raw_features)
        all_labels.append(batch.y.cpu().numpy())
    
    return np.concatenate(all_features), np.concatenate(all_labels)

def extract_microbe_features(loader):
    all_features = []
    all_labels = []
    for batch in loader:
        if not batch:  # 空批次检查
            continue
        features, labels = batch[0].numpy(), batch[1].numpy()
        if features.shape[0] == 0:  # 空数据检查
            continue
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

def expand_to_full_matrix(vector):
    """将19900维上三角向量转换为40000维全矩阵展平"""
    assert len(vector) == 19900, f"输入应为 19900 维，当前为 {len(vector)}"
    matrix = np.zeros((200, 200))
    triu_indices = np.triu_indices(200, k=1)
    matrix[triu_indices] = vector
    matrix = matrix + matrix.T  # 对称复制
    np.fill_diagonal(matrix, 1.0)  # 确保对角线为1
    return matrix.flatten()  # 返回40000维向量

def normalize_features(features, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
    return (features - mean) / std, mean, std

# 对fMRI数据特殊处理（与GNN预处理一致）
def process_fmri_features(features):
    """处理为GNN需要的40000维输入"""
    # 添加噪声
    noise = np.random.normal(scale=0.1, size=features.shape)
    noisy_features = np.clip(features + noise, -1, 1)
    
    # 转换为40000维
    expanded_features = np.array([expand_to_full_matrix(vec) for vec in noisy_features])
    assert expanded_features.shape[1] == 40000, "处理后的特征维度应为 40000"
    return expanded_features  # 返回(n_samples, 40000)

def get_accuracy_from_report(report_str):
    """从分类报告中安全提取准确率"""
    for line in report_str.split('\n'):
        if 'accuracy' in line:
            return float(line.split()[-1])
    raise ValueError("无法从报告中找到准确率信息")

# 加载预训练模型
mlp_model = load_pretrained_mlp()
gnn_model = load_pretrained_gnn()

# 创建对比模型
contrast_model = ContrastiveModel(mlp_model, gnn_model).to(device)

contrast_model.mlp.eval()
contrast_model.gnn.eval()

file_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/abide.hdf5"
graph_type = "cc200"
csv_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/microbe_data.csv"
biom_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/feature-table.biom"

fmri_loader = fMRIDataLoader(file_path = file_path, graph_type="cc200", batch_size=32)
microbe_loader = MicrobeDataLoader(csv_path = csv_path, biom_path=biom_path, batch_size=32)

fmri_train_loader = fmri_loader.get_dataloaders()["train"]
microbe_train_loader = microbe_loader.get_loaders()[0]

fmri_val_loader = fmri_loader.get_dataloaders()["valid"]
microbe_val_loader = microbe_loader.get_loaders()[1]

fmri_test_loader = fmri_loader.get_dataloaders()["test"]
microbe_test_loader = microbe_loader.get_loaders()[2]
 
 


fmri_train_features, fmri_train_labels = extract_graph_features(fmri_train_loader)
fmri_val_features, fmri_val_labels = extract_graph_features(fmri_val_loader)
fmri_test_features, fmri_test_labels = extract_graph_features(fmri_test_loader)

microbe_train_features, microbe_train_labels = extract_microbe_features(microbe_train_loader)
microbe_val_features, microbe_val_labels = extract_microbe_features(microbe_val_loader)
microbe_test_features, microbe_test_labels = extract_microbe_features(microbe_test_loader)

# 在特征提取后添加验证
print("微生物特征维度验证:", microbe_train_features.shape[1])
print("MLP输入层维度:", mlp_model.feature_extractor[0].in_features)
assert microbe_train_features.shape[1] == mlp_model.feature_extractor[0].in_features

print("fMRI特征维度验证:", fmri_train_features.shape[1]) #（19900）
print("GNN期望输入维度: 40000")


microbe_train_features, microbe_mean, microbe_std = normalize_features(microbe_train_features)
microbe_val_features = normalize_features(microbe_val_features, microbe_mean, microbe_std)[0]
microbe_test_features = normalize_features(microbe_test_features, microbe_mean, microbe_std)[0]

# 准备数据
microbe_data = (microbe_train_features, microbe_train_labels)
fmri_data = (fmri_train_features, fmri_train_labels)
paired_dataset = PairedDataset(microbe_data, fmri_data)
train_loader = DataLoader(paired_dataset, batch_size=64, shuffle=True)

# 创建微生物组锚点集（每类取10个样本）
microbe_anchors = []
for label in np.unique(microbe_train_labels):
    indices = np.where(microbe_train_labels == label)[0][:10]
    '''microbe_anchors.extend([{'microbe': microbe_train_features[i], 'label': label} 
                          for i in indices])'''
    for i in indices:
        # [2503] -> [1, 2503]
        microbe_anchors.append({
            'microbe': torch.FloatTensor(microbe_train_features[i]).unsqueeze(0),
            'label': label
        })

# 训练配置
optimizer = torch.optim.AdamW([
    {'params': contrast_model.mlp_proj.parameters()},
    {'params': contrast_model.gnn_proj.parameters()},
    {'params': contrast_model.classifier.parameters()}
], lr=1e-4, weight_decay=1e-5)
criterion = LabelAwareContrastiveLoss(temp=0.05, hard_neg_ratio=0.2)

valid_paired_dataset = PairedDataset(
    (microbe_val_features, microbe_val_labels),
    (fmri_val_features, fmri_val_labels)
)
valid_loader = DataLoader(valid_paired_dataset, batch_size=64, shuffle=False)

best_valid_acc = 0.0

test_paired_dataset = PairedDataset(
    (microbe_test_features, microbe_test_labels),
    (fmri_test_features, fmri_test_labels)
)
test_loader = DataLoader(test_paired_dataset, batch_size=64, shuffle=False)


patience = 20
best_valid_acc = 0
best_valid_loss = float('inf')
counter = 0

# 定义学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

# 初始化日志
logs = {'epoch': [], 'train_loss': [], 'train_acc': [], 'valid_acc': []}

for epoch in range(200):
    contrast_model.train()
    contrast_model.mlp.eval()
    contrast_model.gnn.eval()
    
    train_loss, train_acc = train_contrastive_adversarial(
        contrast_model, train_loader, optimizer, criterion, device
    )
   #valid_loss, valid_acc = validate(contrast_model, valid_loader, device)
    valid_loss, valid_acc = validate(contrast_model, criterion, valid_loader, device)
    
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2%}")
    
    # 更新学习率
    scheduler.step(valid_acc)
    
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(contrast_model.state_dict(), "/home/yangzongxian/xlz/ASD_GCN/main/biomarker/adversarial.pth")
        print(f"New best model saved with Valid Acc: {valid_acc:.2%}")
    
    # Early Stopping
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


pd.DataFrame(logs).to_csv("training_log1.csv", index=False)
print(f"Best validation accuracy: {best_valid_acc:.2%}")



best_model = ContrastiveModel(mlp_model, gnn_model).to(device)
best_model.load_state_dict(torch.load("/home/yangzongxian/xlz/ASD_GCN/main/biomarker/adversarial.pth"))

test_paired_dataset = PairedDataset(
    (microbe_test_features, microbe_test_labels),
    (fmri_test_features, fmri_test_labels)
)
test_loader = DataLoader(test_paired_dataset, batch_size=64, shuffle=False)

#test_acc = validate(best_model, test_loader, device)
test_loss, test_acc = validate(best_model, criterion, test_loader, device)
print(f"测试准确率（最佳模型）: {test_acc:.2%}")