import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from pathlib import Path
import sys
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent  
target_subdir = parent_dir / "pre_train"        
sys.path.append(str(target_subdir))
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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, confusion_matrix

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            nn.Linear(40000, 200*200),  # 将40000维输入转换为200x200矩阵
            nn.Sigmoid()
        )
        # 修改后的卷积层定义
        self.convs = nn.ModuleList([
            # 第一层GAT：输入特征维度需与增强后的节点特征匹配
            tg_nn.GATv2Conv(
                in_channels=16,  # 修改为特征增强后的维度
                out_channels=128,
                heads=8,
                dropout=config['dropout'],
                add_self_loops=False
            ),
            # 第二层GAT
            tg_nn.GATv2Conv(
                in_channels=128*8,  # 多头注意力的输出维度
                out_channels=256,
                heads=4,
                dropout=config['dropout']
            ),
            # 第三层GAT
            tg_nn.GATv2Conv(
                in_channels=256*4,
                out_channels=512,
                heads=1,
                dropout=config['dropout']
            )
        ])
        
        # 新增特征增强层
        self.feature_enhancer = nn.Sequential(
            nn.Linear(2, 8),  # 扩展节点特征维度
            nn.GELU(),
            nn.Linear(8, 16),
            nn.LayerNorm(16)
        )

        # 更新分类器输入维度
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, config['num_classes'])
        )
    def build_graph(self, fc_matrix):
        """修正后的图构建方法"""
        batch_size = fc_matrix.size(0)
        
        # 通过graph_builder生成邻接矩阵
        adj = self.graph_builder(fc_matrix).view(batch_size, 200, 200).float()
        adj = (adj + adj.transpose(1,2)) / 2  # 确保对称性

        # 生成增强节点特征（维度验证）
        node_features = []
        edge_indices = []
        for b in range(batch_size):
            # 基础统计特征
            
            means = adj[b].mean(dim=1, keepdim=True)  # (200,1)
            stds = adj[b].std(dim=1, keepdim=True)    # (200,1)
            base_feat = torch.cat([means, stds], dim=1)  # (200,2)
            
            # 特征增强（输出维度16）
            enhanced_feat = self.feature_enhancer(base_feat)  # (200,16)
            
            # 动态边生成（带阈值限制）
            assert adj[b].dtype == torch.float32, f"邻接矩阵数据类型错误: {adj[b].dtype}"
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
        """动态生成边连接"""
        # 空间约束
        spatial_dist = torch.cdist(self.spatial_emb.weight, self.spatial_emb.weight)
        
        # 特征相似性
        feat_sim = torch.mm(node_feat, node_feat.t())
        
        # 综合边权重
        combined = (feat_sim * (1 / (spatial_dist + 1e-6)))
        
        # 生成邻接矩阵
        adj = (combined > self.threshold).float()
        
        # 确保最小连接数
        topk = torch.topk(combined, self.k_neighbors, dim=1)
        adj[topk.indices] = 1.0
        
        return adj
   
    def forward(self, raw_fc):
        # 输入维度验证
        assert raw_fc.dim() == 2, f"输入应为二维张量，当前维度：{raw_fc.dim()}"
        assert raw_fc.size(1) == 40000, f"输入特征维度错误，期望40000，实际{raw_fc.size(1)}"
        
        # 构建动态图
        batch_graph = self.build_graph(raw_fc)
        
        # 图卷积处理
        x = batch_graph.x.to(raw_fc.device)
        edge_index = batch_graph.edge_index.to(raw_fc.device)
        
        for i, conv in enumerate(self.convs):
            # 维度适配检查
            assert x.size(1) == conv.in_channels, \
                f"第{i+1}层输入维度不匹配！期望{conv.in_channels}，实际{x.size(1)}"
            
            x = conv(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # 全局池化
        x = tg_nn.global_mean_pool(x, batch_graph.batch)
        return x
    
    def classify(self, x):
        return self.classifier(x)
    
    def _adjust_model_parameters(self, new_dim):
        """动态调整模型参数"""
        # 调整空间嵌入维度
        old_emb = self.spatial_emb
        self.spatial_emb = nn.Embedding(new_dim, 3)
        with torch.no_grad():
            min_dim = min(old_emb.weight.size(0), new_dim)
            self.spatial_emb.weight[:min_dim] = old_emb.weight[:min_dim]
        
        # 调整图卷积层输入维度
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
    checkpoint = torch.load("/home/yangzongxian/xlz/ASD_GCN/main/new/fmri_gnn.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)




class ContrastiveModel(nn.Module):
    def __init__(self, mlp_model, gnn_model, feat_dim=128):
        super().__init__()
        # 冻结预训练模型参数
        self.mlp = mlp_model
        self.gnn = gnn_model
        for param in mlp_model.parameters():
            param.requires_grad_(False)
        for param in gnn_model.parameters():
            param.requires_grad_(False)
        
        mlp_feat_dim = self.mlp.classifier.in_features  # 匹配MLP最终层输入
        gnn_feat_dim = self.gnn.classifier[0].in_features  # 匹配GNN最终层输入
        
        # 微生物组特征投影
        self.mlp_proj = nn.Sequential(
            nn.LayerNorm(mlp_feat_dim),
            nn.Linear(mlp_feat_dim, feat_dim),
            nn.GELU()
        )
        
        print(f"GNN feature dim: {gnn_feat_dim}")
        # fMRI特征投影
        self.gnn_proj = nn.Sequential(
            nn.LayerNorm(gnn_feat_dim),
            nn.Linear(gnn_feat_dim, feat_dim),
            nn.GELU()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim*2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, microbe_input, fmri_input):
        # 修改GNN输入处理
        with torch.no_grad():
            # 微生物特征提取
            #print(f"Microbe input shape: {microbe_input.shape}") #(64,2503)
            #print(f"fMRI input shape: {fmri_input.shape}")#(64,40000)
            mlp_feat = self.mlp.feature_extractor(microbe_input)
            gnn_feat = self.gnn(fmri_input)
            #print(f"MLP feature shape: {mlp_feat.shape}")#(64,512)
            #print(f"GNN feature shape: {gnn_feat.shape}")#(64,512)
            # fMRI处理为展平向量输入GNN
            #fmri_flatten = fmri_input.view(fmri_input.size(0), -1)  # 确保输入为(batch,40000)
            #gnn_feat = self.gnn(fmri_flatten)
            
        
        # 投影和分类保持不变
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
        
def train_contrastive(model, train_loader, optimizer, criterion, device):
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
    
    return total_loss / len(train_loader), np.mean(accuracies)

'''def test_fmri(model, test_loader, microbe_anchors, device):
    # 使用MLP的完整特征提取流程
    with torch.no_grad():
        anchor_features = []
        for anchor in microbe_anchors:
            #inputs = anchor['microbe'].unsqueeze(0).to(device)
            feat = model.mlp.feature_extractor(anchor['microbe'].to(device))
            #feat = model.mlp.feature_extractor(inputs)
            anchor_features.append(feat)
            #anchor_features.append(feat.squeeze(0))
        h_microbe = model.mlp_proj(torch.stack(anchor_features))
    
    # 使用GNN的完整处理流程
    with torch.no_grad():
        h_fmri = []
        for batch in test_loader:
            inputs = batch['fc'].to(device)
            features = model.gnn(inputs)  # 获取GNN特征
            h_fmri.append(model.gnn_proj(features))
        h_fmri = torch.cat(h_fmri)
    
    # 相似度计算
    sim_matrix = torch.mm(h_fmri, h_microbe.T)
    preds = torch.argmax(sim_matrix, dim=1)

    # 计算准确率
    accuracies = []
    for i, anchor in enumerate(microbe_anchors):
        mask = preds == i
        acc = accuracy_score(test_loader.dataset.labels[mask], anchor['label'])
        accuracies.append(acc)
    print(f"Accuracy: {np.mean(accuracies):.4f}")'''
 
'''def validate(model, valid_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in valid_loader:
            # 修正键名（根据PairedDataset的结构）
            microbe = batch['microbe'].to(device)
            fmri = batch['fmri'].to(device)
            labels = batch['label'].squeeze().to(device)  # 注意标签维度处理
            
            _, _, logits = model(microbe, fmri)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return classification_report(all_labels, all_preds)'''
    
def validate(model, loader, device, full_report=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Needed for AUC calculation

    with torch.no_grad():
        for batch in loader:
            microbe = batch['microbe'].to(device)
            fmri = batch['fmri'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            # Get the classification logits from the model
            _, _, logits = model(microbe, fmri)
            
            # Calculate probabilities for AUC
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Store probability of the positive class

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 计算sensitivity和specificity
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Ensure there are samples of both classes to compute AUC
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = float('nan') # AUC is not defined for a single class

    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

    if full_report:
        report_str = classification_report(
            all_labels, all_preds,
            target_names=["Healthy", "ASD"], # Adjust if your labels are different
            digits=4
        )
        return metrics, report_str
    
    return metrics
 
    
def test_fmri(model, test_loader, microbe_anchors, device):
    """
    Correctly performs retrieval-based evaluation by comparing embeddings
    from both processed modalities.
    """
    model.eval()

    # --- Step 1: Get all anchor embeddings (from microbe data) ---
    anchor_tensors = torch.stack([anchor['microbe'] for anchor in microbe_anchors]).to(device)
    anchor_labels = [anchor['label'] for anchor in microbe_anchors]

    with torch.no_grad():
        # Process all anchors in a single batch for efficiency
        mlp_feat = model.mlp.feature_extractor(anchor_tensors)
        h_microbe = F.normalize(model.mlp_proj(mlp_feat), dim=1)  # Shape: [num_anchors, 128]

    # --- Step 2: Get all test embeddings (from fMRI data) ---
    all_h_fmri = []
    all_fmri_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting fMRI Features for Retrieval"):
            fmri_data = batch.to(device)
            # Input to GNN is the flattened 40000-dim vector from the loader
            fmri_input = fmri_data.x.view(fmri_data.num_graphs, -1)

            # Process fMRI data through the GNN and its projection head
            gnn_feat = model.gnn(fmri_input)
            h_fmri = F.normalize(model.gnn_proj(gnn_feat), dim=1)  # Shape: [batch_size, 128]

            all_h_fmri.append(h_fmri)
            all_fmri_labels.append(fmri_data.y)

    all_h_fmri = torch.cat(all_h_fmri, dim=0)
    all_fmri_labels = torch.cat(all_fmri_labels, dim=0).cpu().numpy()

    # --- Step 3: Calculate similarity, predict, and evaluate ---
    # Similarity matrix between every fMRI sample and every microbe anchor
    sim_matrix = torch.mm(all_h_fmri, h_microbe.T)  # Shape: [num_test_samples, num_anchors]

    # For each fMRI sample, find the index of the most similar microbe anchor
    pred_anchor_indices = torch.argmax(sim_matrix, dim=1).cpu().numpy()

    # Get the predicted label from the anchor's known label
    predicted_labels = np.array(anchor_labels)[pred_anchor_indices]

    # Calculate final accuracy by comparing predicted labels to true fMRI labels
    accuracy = accuracy_score(all_fmri_labels, predicted_labels)
    return accuracy
    
def reconstruct_fc(vector):
    """将上三角向量重建为对称矩阵"""
    # 创建空矩阵
    matrix = np.zeros((200, 200))
    
    # 填充对角线
    #np.fill_diagonal(matrix, 1.0)
    
    # 提取上三角索引（不包括对角线）
    triu_indices = np.triu_indices(200, k=1)
    
    # 验证输入维度
    #assert len(vector) == triu_indices[0].size, "输入维度不匹配"
    
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
        # 使用GNN模型提取图特征
        #with torch.no_grad():
        #graph_features = gnn_model(batch.x.view(batch.num_graphs, -1))  # 输入展平的40000维
        #all_features.append(graph_features.cpu().numpy())
        #all_labels.append(batch.y.cpu().numpy())
        raw_features = batch.x.view(batch.num_graphs, -1).cpu().numpy()  # 保持40000维
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
'''def extract_features_labels(loader):
    """从数据加载器中提取所有特征和标签"""
    all_features = []
    all_labels = []
    for batch in loader:
        # fMRI数据格式特殊处理
        if isinstance(batch, dict):  # 适用于fMRI数据
            features = batch['fc'].numpy()
            labels = batch['labels'].numpy()
        else:  # 适用于微生物组数据
            features, labels = batch[0].numpy(), batch[1].numpy()
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)'''

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

print("\n--- Loading best model for final evaluation ---")
# Load the best model saved during training
mlp_model = load_pretrained_mlp()
gnn_model = load_pretrained_gnn()
best_model = ContrastiveModel(mlp_model, gnn_model).to(device)

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
 
 
#microbe_train_features, microbe_train_labels = extract_features_labels(microbe_loader.get_loaders()[0])
#microbe_val_features, microbe_val_labels = extract_features_labels(microbe_loader.get_loaders()[1])
#microbe_test_features, microbe_test_labels = extract_features_labels(microbe_loader.get_loaders()[2])   

#fmri_train_features, fmri_train_labels = extract_features_labels(fmri_loader.get_dataloaders()["train"])
#fmri_val_features, fmri_val_labels = extract_features_labels(fmri_loader.get_dataloaders()["valid"])
#fmri_test_features, fmri_test_labels = extract_features_labels(fmri_loader.get_dataloaders()["test"])


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
#fmri_train_processed = process_fmri_features(fmri_train_features)
print("fMRI特征维度验证:", fmri_train_features.shape[1]) #（19900）
print("GNN期望输入维度: 40000")
#fmri_val_processed = process_fmri_features(fmri_val_features)
#fmri_test_processed = process_fmri_features(fmri_test_features)

#assert fmri_val_processed.shape[1] == 40000, "验证集维度错误"
#assert fmri_test_processed.shape[1] == 40000, "测试集维度错误"
#assert fmri_train_processed.shape[1] == 19900, "处理后的特征维度错误"
#print("GNN期望输入维度: 40000")
#assert fmri_train_features.shape[1] == 200*200

microbe_train_features, microbe_mean, microbe_std = normalize_features(microbe_train_features)
microbe_val_features = normalize_features(microbe_val_features, microbe_mean, microbe_std)[0]
microbe_test_features = normalize_features(microbe_test_features, microbe_mean, microbe_std)[0]

#fmri_train_features = process_fmri_features(fmri_train_features)
#fmri_val_features = process_fmri_features(fmri_val_features)
#fmri_test_features = process_fmri_features(fmri_test_features)

# 准备数据
microbe_data = (microbe_train_features, microbe_train_labels)
fmri_data = (fmri_train_features, fmri_train_labels)
paired_dataset = PairedDataset(microbe_data, fmri_data)
train_loader = DataLoader(paired_dataset, batch_size=64, shuffle=True)

# 创建微生物组锚点集（每类取10个样本）
microbe_anchors = []
for label in np.unique(microbe_train_labels):
    indices = np.where(microbe_train_labels == label)[0][:10]
    for i in indices:
        # Store the raw 1D tensor. The function will handle batching.
        microbe_anchors.append({
            'microbe': torch.FloatTensor(microbe_train_features[i]),
            'label': label
        })

# 训练配置
optimizer = torch.optim.AdamW([
    {'params': best_model.mlp_proj.parameters()},
    {'params': best_model.gnn_proj.parameters()},
    {'params': best_model.classifier.parameters()}
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

# 训练循环
'''for epoch in range(200):
    train_loss, train_acc = train_contrastive(
        contrast_model, train_loader, optimizer, criterion, device
    )
    print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} |Train Acc: {train_acc:.2%}")
     
    valid_report = validate(contrast_model, valid_loader, device)
    
    valid_acc = validate(contrast_model, valid_loader, device)

    # 保存最佳模型
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(contrast_model.state_dict(), "best_model.pth")
    
    print(f"Epoch {epoch+1} | Valid Acc: {valid_acc:.2%}")'''
    
# 定义 Early Stopping 参数
patience = 20
best_valid_acc = 0
counter = 0

# 定义学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

# 初始化日志
logs = {'epoch': [], 'train_loss': [], 'train_acc': [], 'valid_acc': [], 'valid_sensitivity': [], 'valid_specificity': []}

for epoch in range(200):
    train_loss, train_acc = train_contrastive(
        best_model, train_loader, optimizer, criterion, device
    )
    valid_metrics = validate(best_model, valid_loader, device)

    # Extract the accuracy for saving the best model
    valid_acc = valid_metrics['accuracy']
    
    # 记录日志
    logs['epoch'].append(epoch + 1)
    logs['train_loss'].append(train_loss)
    logs['train_acc'].append(train_acc)
    logs['valid_acc'].append(valid_acc)
    logs['valid_sensitivity'].append(valid_metrics['sensitivity'])
    logs['valid_specificity'].append(valid_metrics['specificity'])
    
    print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
          f"Valid Acc: {valid_metrics['accuracy']:.2%} | Valid AUC: {valid_metrics['auc']:.4f} | "
          f"Valid F1: {valid_metrics['f1_score']:.4f} | "
          f"Sensitivity: {valid_metrics['sensitivity']:.4f} | Specificity: {valid_metrics['specificity']:.4f}")

    
    # 更新学习率
    #scheduler.step(valid_acc)
    
    # Early Stopping 逻辑
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(best_model.state_dict(), "best_model.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# 将日志保存为 CSV 文件
pd.DataFrame(logs).to_csv("training_log1.csv", index=False)
# 使用fMRI测试集评估
print(f"Best validation accuracy: {best_valid_acc:.2%}")

'''best_model = ContrastiveModel(mlp_model, gnn_model).to(device)

best_model.load_state_dict(torch.load("best_model.pth"))


test_acc = test_fmri(
    best_model,  
    fmri_test_loader,
    microbe_anchors,
    device
)
print(f"测试准确率（最佳模型）: {test_acc:.2%}")'''

'''test_report = test_fmri(
    contrast_model, 
    fmri_test_loader,
    microbe_anchors,
    device
)
print("Test Report:")
print(test_report)'''

 # 加载最佳模型

best_model.load_state_dict(torch.load("best_model.pth"))

# 创建测试集的DataLoader
test_paired_dataset = PairedDataset(
    (microbe_test_features, microbe_test_labels),
    (fmri_test_features, fmri_test_labels)
)
test_loader = DataLoader(test_paired_dataset, batch_size=64, shuffle=False)

# 直接使用validate函数测试准确率
'''test_acc = validate(best_model, test_loader, device)
print(f"测试准确率（最佳模型）: {test_acc:.2%}")'''


# --- Evaluation 1: Retrieval Accuracy (your original test) ---
print("\n--- Running Retrieval-Based Evaluation ---")
retrieval_acc = test_fmri(
    best_model,
    fmri_test_loader,
    microbe_anchors,
    device
)
print(f"Final Retrieval Accuracy (using microbe anchors): {retrieval_acc:.2%}")


# --- Evaluation 2: Standard Classification Metrics on Test Set ---
print("\n--- Running Standard Classification Evaluation on Test Set ---")
test_metrics, test_report = validate(
    best_model,
    test_loader,  # Use the paired test loader
    device,
    full_report=True
)

print(f"Final Test Accuracy:    {test_metrics['accuracy']:.4f}")
print(f"Final Test AUC:         {test_metrics['auc']:.4f}")
print(f"Final Test F1-score:    {test_metrics['f1_score']:.4f}")
print(f"Final Test Sensitivity: {test_metrics['sensitivity']:.4f}")
print(f"Final Test Specificity: {test_metrics['specificity']:.4f}")
print("\nFull Classification Report on Test Data:")
print(test_report)