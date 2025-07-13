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
torch.cuda.set_device(7) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LeaveOneSiteOutLoader:
    def __init__(self, file_path, graph_type, batch_size=32, site_name='UM'):
        self.file_path = file_path
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.site_name = site_name
        self.sites = self._get_sites()
        
    def _get_sites(self):
        with h5py.File(self.file_path, "r") as f:
            # 获取指定站点的数据集
            group_path = f"/experiments/{self.graph_type}_leavesiteout-{self.site_name}"
            if group_path in f:
                return list(f[group_path].keys())
            else:
                raise ValueError(f"找不到站点 {self.site_name} 的留一站点验证数据集")
    
    def get_site_data(self, test_site_idx):
        graph_dataset = []
        labels = []
        
        with h5py.File(self.file_path, "r") as f:
            # 获取实验组
            exp_group = f[f"/experiments/{self.graph_type}_leavesiteout-{self.site_name}"]
            test_site = self.sites[test_site_idx]
            
            # 获取已划分好的数据集ID
            test_subjects = [s.decode('utf-8') if isinstance(s, bytes) else s for s in exp_group[test_site]["test"][:]]
            train_subjects = [s.decode('utf-8') if isinstance(s, bytes) else s for s in exp_group[test_site]["train"][:]]
            val_subjects = [s.decode('utf-8') if isinstance(s, bytes) else s for s in exp_group[test_site]["valid"][:]]
            
            print(f"\n站点 {self.site_name} 的数据分布:")
            print(f"测试集: {len(test_subjects)} 个样本")
            print(f"训练集: {len(train_subjects)} 个样本")
            print(f"验证集: {len(val_subjects)} 个样本")
            
            # 处理所有受试者数据
            patients_group = f["/patients"]
            
            # 遍历patients组中的所有受试者
            for subject_id in patients_group.keys():
                subject_group = patients_group[subject_id]
                if self.graph_type in subject_group:
                    # 提取数据
                    triu_vector = subject_group[self.graph_type][:]
                    matrix = reconstruct_fc(triu_vector)
                    edge_index = self._get_brain_connectivity_edges(matrix)
                    label = torch.tensor(subject_group.attrs["y"], dtype=torch.long)
                    flat_vector = matrix.flatten()
                    
                    # 创建图数据
                    graph_data = Data(
                        x=torch.FloatTensor(flat_vector),
                        edge_index=edge_index,
                        y=label
                    )
                    
                    # 根据ID分配到对应数据集
                    if subject_id in test_subjects:
                        graph_dataset.append(("test", graph_data))
                        labels.append(("test", label))
                    elif subject_id in train_subjects:
                        graph_dataset.append(("train", graph_data))
                        labels.append(("train", label))
                    elif subject_id in val_subjects:
                        graph_dataset.append(("valid", graph_data))
                        labels.append(("valid", label))
            
            # 打印实际加载的样本数
            test_count = len([d for d in graph_dataset if d[0] == "test"])
            train_count = len([d for d in graph_dataset if d[0] == "train"])
            valid_count = len([d for d in graph_dataset if d[0] == "valid"])
            
            print("\n实际加载的样本数:")
            print(f"测试集: {test_count}/{len(test_subjects)} ({test_count/len(test_subjects):.2%})")
            print(f"训练集: {train_count}/{len(train_subjects)} ({train_count/len(train_subjects):.2%})")
            print(f"验证集: {valid_count}/{len(val_subjects)} ({valid_count/len(val_subjects):.2%})")
        
        # 组织数据
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
        if len(data) == 0:  # 如果数据集为空，返回None
            return None
        return DataLoader(
            data, 
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: Batch.from_data_list(batch)
        )
    
    def get_dataloaders(self, test_site_idx):
        data_splits = self.get_site_data(test_site_idx)
        return {
            "train": self._create_dataloader(*data_splits["train"]),
            "valid": self._create_dataloader(*data_splits["valid"]),
            "test": self._create_dataloader(*data_splits["test"]),
        }

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
    checkpoint = torch.load("/home/yangzongxian/xlz/ASD_GCN/main/down/fmri_gnn.pth", map_location=device)
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
       
        #print(f"Microbe input shape: {microbe_input.shape}") #(64,2503)
        #print(f"fMRI input shape: {fmri_input.shape}")#(64,40000)
        mlp_feat = self.mlp.feature_extractor(microbe_input)
        gnn_feat = self.gnn(fmri_input)
            
        
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
    def __init__(self, microbe_data, fmri_data, train=True):
        """
        microbe_data: (features, labels)
        fmri_data: (features, labels)
        """
        self.microbe_features, self.microbe_labels = microbe_data
        self.fmri_features, self.fmri_labels = fmri_data
        self.train = train
        
        # 建立双模态标签索引
        self.label_to_indices = {
            label: {
                'microbe': np.where(self.microbe_labels == label)[0],
                'fmri': np.where(self.fmri_labels == label)[0]
            }
            for label in np.unique(self.microbe_labels)
        }
        
        if not train:
            # 为测试集创建固定配对
            self.fmri_pairs = [None] * len(self.microbe_features)
            for label in np.unique(self.microbe_labels):
                microbe_indices = self.label_to_indices[label]['microbe']
                fmri_indices = self.label_to_indices[label]['fmri']
                
                # 确保fMRI样本足够
                if len(fmri_indices) < len(microbe_indices):
                    # 如果fMRI样本不足，循环使用现有样本
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
            # 训练集添加随机噪声
            microbe_feat += np.random.normal(0, 0.1, size=microbe_feat.shape)
            # 随机选择fMRI样本
            fmri_idx = np.random.choice(self.label_to_indices[label]['fmri'])
            fmri_feat = self.fmri_features[fmri_idx]
        else:
            # 测试集不添加噪声，使用固定配对
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
        
        # 原始预测
        with torch.no_grad():
            _, _, logits_original = model(microbe, fmri)
            pred_original = torch.argmax(logits_original, dim=1)
        
        # 生成扰动后的微生物组输入
        microbe = microbe.clone().detach().requires_grad_(True)
        h_m, h_f, _ = model(microbe, fmri)
        loss = criterion(h_m, h_f, labels) + 0.5 * F.cross_entropy(logits_original, labels)
        grad = torch.autograd.grad(loss, microbe)[0]
        delta = epsilon * torch.sign(grad)
        microbe_perturbed = microbe + delta
        
        # 扰动后预测
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
    #gnn_model.to(device)
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

def main():
    # 加载预训练模型
    mlp_model = load_pretrained_mlp()
    gnn_model = load_pretrained_gnn()

    file_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/abide.hdf5"
    graph_type = "cc200"
    csv_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/microbe_data.csv"
    biom_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/feature-table.biom"

    # 存储所有站点的结果
    all_site_results = []
    
    # 对每个站点进行训练和验证
    for site_idx in range(10):  # 10个站点
        print(f"\n训练站点 {site_idx + 1}/10")
        
        # 创建留一站点验证加载器
        fmri_loader = LeaveOneSiteOutLoader(file_path=file_path, graph_type="cc200", batch_size=32)
        microbe_loader = MicrobeDataLoader(csv_path=csv_path, biom_path=biom_path, batch_size=32)
        
        # 获取当前站点的数据加载器
        fmri_loaders = fmri_loader.get_dataloaders(site_idx)
        fmri_train_loader = fmri_loaders["train"]
        fmri_val_loader = fmri_loaders["valid"]
        fmri_test_loader = fmri_loaders["test"]
        
        microbe_train_loader = microbe_loader.get_loaders()[0]
        microbe_val_loader = microbe_loader.get_loaders()[1]
        microbe_test_loader = microbe_loader.get_loaders()[2]

        # 提取特征
        fmri_train_features, fmri_train_labels = extract_graph_features(fmri_train_loader)
        fmri_val_features, fmri_val_labels = extract_graph_features(fmri_val_loader)
        fmri_test_features, fmri_test_labels = extract_graph_features(fmri_test_loader)

        microbe_train_features, microbe_train_labels = extract_microbe_features(microbe_train_loader)
        microbe_val_features, microbe_val_labels = extract_microbe_features(microbe_val_loader)
        microbe_test_features, microbe_test_labels = extract_microbe_features(microbe_test_loader)

        # 特征归一化
        microbe_train_features, microbe_mean, microbe_std = normalize_features(microbe_train_features)
        microbe_val_features = normalize_features(microbe_val_features, microbe_mean, microbe_std)[0]
        microbe_test_features = normalize_features(microbe_test_features, microbe_mean, microbe_std)[0]

        # 创建对比模型
        contrast_model = ContrastiveModel(mlp_model, gnn_model).to(device)
        contrast_model.mlp.eval()
        contrast_model.gnn.eval()

        # 准备数据
        microbe_data = (microbe_train_features, microbe_train_labels)
        fmri_data = (fmri_train_features, fmri_train_labels)
        paired_dataset = PairedDataset(microbe_data, fmri_data)
        train_loader = DataLoader(paired_dataset, batch_size=64, shuffle=True)

        # 创建验证数据集
        valid_paired_dataset = PairedDataset(
            (microbe_val_features, microbe_val_labels),
            (fmri_val_features, fmri_val_labels)
        )
        valid_loader = DataLoader(valid_paired_dataset, batch_size=64, shuffle=False)

        # 训练配置
        optimizer = torch.optim.AdamW([
            {'params': contrast_model.mlp_proj.parameters()},
            {'params': contrast_model.gnn_proj.parameters()},
            {'params': contrast_model.classifier.parameters()}
        ], lr=1e-4, weight_decay=1e-5)
        criterion = LabelAwareContrastiveLoss(temp=0.05, hard_neg_ratio=0.2)

        # 训练参数
        patience = 20
        best_valid_acc = 0
        best_valid_loss = float('inf')
        counter = 0
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
        logs = {'epoch': [], 'train_loss': [], 'train_acc': [], 'valid_acc': []}

        # 训练循环
        for epoch in range(100):  # 最多训练100个epoch
            train_loss, train_acc = train_contrastive_adversarial(
                contrast_model, train_loader, optimizer, criterion, device
            )
            valid_loss, valid_acc = validate(contrast_model, criterion, valid_loader, device)
            
            # 更新学习率
            scheduler.step(valid_acc)
            
            # 记录日志
            logs['epoch'].append(epoch)
            logs['train_loss'].append(train_loss)
            logs['train_acc'].append(train_acc)
            logs['valid_acc'].append(valid_acc)
            
            print(f"Epoch {epoch+1}/100:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
            print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2%}")
            
            # 早停检查
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_valid_loss = valid_loss
                counter = 0
                # 保存最佳模型
                torch.save(contrast_model.state_dict(), f"best_model_site_{site_idx+1}.pth")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # 加载最佳模型进行测试
        best_model = ContrastiveModel(mlp_model, gnn_model).to(device)
        best_model.load_state_dict(torch.load(f"best_model_site_{site_idx+1}.pth"))

        # 创建测试数据集
        test_paired_dataset = PairedDataset(
            (microbe_test_features, microbe_test_labels),
            (fmri_test_features, fmri_test_labels),
            train=False
        )
        test_loader = DataLoader(test_paired_dataset, batch_size=64, shuffle=False)

        # 进行测试
        test_loss, test_acc = validate(best_model, criterion, test_loader, device)
        print(f"站点 {site_idx + 1} 测试准确率: {test_acc:.2%}")

        # 进行对抗测试
        acc_original, acc_perturbed = test_adversarial(best_model, test_loader, device, criterion)
        print(f"站点 {site_idx + 1} 原始测试准确率: {acc_original:.2%}")
        print(f"站点 {site_idx + 1} 对抗测试准确率: {acc_perturbed:.2%}")

        # 存储结果
        all_site_results.append({
            'site': site_idx + 1,
            'test_acc': test_acc,
            'original_acc': acc_original,
            'perturbed_acc': acc_perturbed
        })

    # 计算并打印平均结果
    avg_test_acc = np.mean([r['test_acc'] for r in all_site_results])
    avg_original_acc = np.mean([r['original_acc'] for r in all_site_results])
    avg_perturbed_acc = np.mean([r['perturbed_acc'] for r in all_site_results])
    
    print("\n=== 总体结果 ===")
    print(f"平均测试准确率: {avg_test_acc:.2%}")
    print(f"平均原始测试准确率: {avg_original_acc:.2%}")
    print(f"平均对抗测试准确率: {avg_perturbed_acc:.2%}")

if __name__ == "__main__":
    main()
