import torch
import numpy as np

#对应contrastive_model1.pth
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
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
from sklearn.model_selection import StratifiedKFold


# 设置设备
#torch.cuda.set_device(0) 
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class fMRIDataLoader:
    def __init__(self, file_path, graph_type, test_size=0.1, val_size=0.1, batch_size=32):
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
    checkpoint = torch.load("fmri_gnn.pth", map_location=device)
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
    
'''class PairedDataset(Dataset):
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
        }'''

class PairedDataset(Dataset):
    def __init__(self, microbe_data, fmri_data, train=True):
        self.microbe_features, self.microbe_labels = microbe_data
        self.fmri_features, self.fmri_labels = fmri_data
        self.train = train
        
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
        if self.train:
            # 仅在训练阶段添加噪声
            microbe_feat = microbe_feat + np.random.normal(0, 0.1, size=microbe_feat.shape)
        label = self.microbe_labels[idx]
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

def extract_biomarkers(model, data_loader, device, feature_type='microbe', top_k=50):
    """
    提取对ASD分类贡献最大的biomarker
    
    参数:
    - model: 训练好的模型
    - data_loader: 数据加载器
    - device: 计算设备
    - feature_type: 特征类型，'microbe'或'fmri'
    - top_k: 返回前k个重要特征
    
    返回:
    - top_indices: 重要特征的索引
    - importance_scores: 所有特征的重要性分数
    """
    # 确保模型处于训练模式，这样可以正确计算梯度
    model.train()
    
    # 初始化存储梯度的列表
    all_gradients = []
    
    # 遍历数据批次
    for batch in data_loader:
        # 将数据移到设备上
        microbe = batch['microbe'].to(device)
        fmri = batch['fmri'].to(device)
        labels = batch['label'].squeeze().to(device)
        
        # 根据特征类型选择计算梯度的输入
        if feature_type == 'microbe':
            # 确保microbe需要梯度
            microbe = microbe.clone().detach().requires_grad_(True)
            input_tensor = microbe
        else:  # fmri
            # 确保fmri需要梯度
            fmri = fmri.clone().detach().requires_grad_(True)
            input_tensor = fmri
        
        # 前向传播
        h_m, h_f, logits = model(microbe, fmri)
        loss = F.cross_entropy(logits, labels)
        
        # 清除之前的梯度
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        # 计算梯度
        try:
            # 使用retain_graph=True确保计算图不被释放
            grad = torch.autograd.grad(loss, input_tensor, retain_graph=True)[0]
            
            # 收集梯度
            all_gradients.append(grad.abs().mean(dim=0).cpu().numpy())
        except Exception as e:
            print(f"计算梯度时出错: {e}")
            # 如果出错，使用零梯度
            if feature_type == 'microbe':
                all_gradients.append(np.zeros(microbe.shape[1]))
            else:
                all_gradients.append(np.zeros(fmri.shape[1]))
    
    # 计算平均梯度
    mean_gradients = np.mean(all_gradients, axis=0)
    
    # 获取top-k重要特征
    top_indices = np.argsort(mean_gradients)[-top_k:]
    
    return top_indices, mean_gradients

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
 
 
all_fold_microbe_biomarkers = []
all_fold_fmri_biomarkers = []
all_fold_microbe_importance = []
all_fold_fmri_importance = []


fmri_train_features, fmri_train_labels = extract_graph_features(fmri_train_loader)
fmri_val_features, fmri_val_labels = extract_graph_features(fmri_val_loader)
fmri_test_features, fmri_test_labels = extract_graph_features(fmri_test_loader)

microbe_train_features, microbe_train_labels = extract_microbe_features(microbe_train_loader)
microbe_val_features, microbe_val_labels = extract_microbe_features(microbe_val_loader)
microbe_test_features, microbe_test_labels = extract_microbe_features(microbe_test_loader)

microbe_anchors = []
for label in np.unique(microbe_train_labels):
    indices = np.where(microbe_train_labels == label)[0][:10]
    for i in indices:
        # [2503] -> [1, 2503]
        microbe_anchors.append({
            'microbe': torch.FloatTensor(microbe_train_features[i]).unsqueeze(0),
            'label': label
        })

#10-fold
fmri_all_features = np.concatenate([fmri_train_features, fmri_val_features])
fmri_all_labels = np.concatenate([fmri_train_labels, fmri_val_labels])
microbe_all_features = np.concatenate([microbe_train_features, microbe_val_features])
microbe_all_labels = np.concatenate([microbe_train_labels, microbe_val_labels])

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_test_acc = []

for fold, (train_idx, val_idx) in enumerate(kf.split(microbe_all_features, microbe_all_labels)):
    print(f"\n=== Fold {fold+1}/10 ===")
    
    # 划分当前fold的数据
    microbe_train_f, microbe_val_f = microbe_all_features[train_idx], microbe_all_features[val_idx]
    microbe_l_train_f, microbe_l_val_f = microbe_all_labels[train_idx], microbe_all_labels[val_idx]
    fmri_train_f, fmri_val_f = fmri_all_features[train_idx], fmri_all_features[val_idx]
    fmri_l_train_f, fmri_l_val_f = fmri_all_labels[train_idx], fmri_all_labels[val_idx]

    # 标准化处理（使用训练fold的统计量）
    microbe_train_f, m_mean, m_std = normalize_features(microbe_train_f)
    microbe_val_f = (microbe_val_f - m_mean) / m_std
    
    def process_fold_fmri(features):
        """处理为GNN需要的40000维输入"""
        # 添加噪声
        noise = np.random.normal(scale=0.1, size=features.shape)
        noisy_features = np.clip(features + noise, -1, 1)
        return noisy_features  # 返回(n_samples, 40000)
    
    fmri_train_f = process_fold_fmri(fmri_train_f)
    fmri_val_f = process_fold_fmri(fmri_val_f)

    # 创建paired datasets
    train_dataset = PairedDataset(
        (microbe_train_f, microbe_l_train_f),
        (fmri_train_f, fmri_l_train_f),
        
    )
    val_dataset = PairedDataset(
        (microbe_val_f, microbe_l_val_f),
        (fmri_val_f, fmri_l_val_f),
        
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 初始化新模型（每个fold独立）
    model = ContrastiveModel(load_pretrained_mlp(), load_pretrained_gnn()).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.mlp_proj.parameters()},
        {'params': model.gnn_proj.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=1e-4, weight_decay=1e-5)
    criterion = LabelAwareContrastiveLoss(temp=0.05, hard_neg_ratio=0.2)
    
    # 训练循环
    best_acc = 0
    for epoch in range(100):
        # 训练步骤
        model.train()
        train_loss, train_acc = train_contrastive_adversarial(
            model, train_loader, optimizer, criterion, device
        )
        
        # 验证步骤
        val_loss, val_acc = validate(model, criterion, val_loader, device)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"fold_{fold}_best.pth")
            
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

    # 在测试集评估最佳模型
    test_dataset = PairedDataset(
        (normalize_features(microbe_test_features, m_mean, m_std)[0], microbe_test_labels),
        (process_fold_fmri(fmri_test_features), fmri_test_labels),
        
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model.load_state_dict(torch.load(f"fold_{fold}_best.pth"))
    print(f"提取Fold {fold+1}的微生物组biomarker...")
    microbe_indices, microbe_importance = extract_biomarkers(
        model, train_loader, device, feature_type='microbe', top_k=50
    )
    
    # 提取当前fold的fMRI biomarker
    print(f"提取Fold {fold+1}的fMRI biomarker...")
    fmri_indices, fmri_importance = extract_biomarkers(
        model, train_loader, device, feature_type='fmri', top_k=100
    )
    
    # 存储结果
    all_fold_microbe_biomarkers.append(microbe_indices)
    all_fold_fmri_biomarkers.append(fmri_indices)
    all_fold_microbe_importance.append(microbe_importance)
    all_fold_fmri_importance.append(fmri_importance)
    
    
    _, fold_acc = validate(model, criterion, test_loader, device)
    fold_test_acc.append(fold_acc)
    print(f"Fold {fold+1} Test Acc: {fold_acc:.2%}")

# 输出最终结果
print(f"\n10-fold Cross Validation Results:")
print(f"Average Test Accuracy: {np.mean(fold_test_acc):.2%} ± {np.std(fold_test_acc):.2%}")


# 分析所有fold的结果，找出在所有fold中都重要的特征
def find_consensus_biomarkers(all_fold_indices, threshold=0.7):
    """找出在所有fold中出现频率超过阈值的biomarker"""
    # 统计每个特征在所有fold中出现的次数
    feature_counts = {}
    for fold_indices in all_fold_indices:
        for idx in fold_indices:
            if idx in feature_counts:
                feature_counts[idx] += 1
            else:
                feature_counts[idx] = 1
    
    # 计算每个特征的出现频率
    n_folds = len(all_fold_indices)
    consensus_features = [idx for idx, count in feature_counts.items() 
                         if count / n_folds >= threshold]
    
    return consensus_features, feature_counts

# 找出共识biomarker
microbe_consensus, microbe_counts = find_consensus_biomarkers(all_fold_microbe_biomarkers)
fmri_consensus, fmri_counts = find_consensus_biomarkers(all_fold_fmri_biomarkers)

# 输出结果
print(f"\n=== Biomarker分析结果 ===")
print(f"微生物组共识biomarker数量: {len(microbe_consensus)}")
print(f"fMRI连接共识biomarker数量: {len(fmri_consensus)}")



# 微生物组biomarker重要性可视化
plt.figure(figsize=(12, 6))
top_microbe_indices = sorted(microbe_counts.items(), key=lambda x: x[1], reverse=True)[:20]
plt.bar([str(idx) for idx, _ in top_microbe_indices], [count for _, count in top_microbe_indices])
plt.title('Top 20 微生物组biomarker在所有fold中的出现频率')
plt.xlabel('特征索引')
plt.ylabel('出现频率')
plt.savefig('microbe_biomarkers.png')
plt.close()

# fMRI连接biomarker重要性可视化
plt.figure(figsize=(12, 6))
top_fmri_indices = sorted(fmri_counts.items(), key=lambda x: x[1], reverse=True)[:20]
plt.bar([str(idx) for idx, _ in top_fmri_indices], [count for _, count in top_fmri_indices])
plt.title('Top 20 fMRI连接biomarker在所有fold中的出现频率')
plt.xlabel('特征索引')
plt.ylabel('出现频率')
plt.savefig('fmri_biomarkers.png')
plt.close()


# 微生物组biomarker
microbe_df = pd.DataFrame({
    'feature_index': list(microbe_counts.keys()),
    'frequency': [count/len(all_fold_microbe_biomarkers) for count in microbe_counts.values()],
    'importance': [np.mean([imp[idx] for imp in all_fold_microbe_importance]) 
                  for idx in microbe_counts.keys()]
})
microbe_df = microbe_df.sort_values('importance', ascending=False)
microbe_df.to_csv('microbe_biomarkers.csv', index=False)

# fMRI连接biomarker
fmri_df = pd.DataFrame({
    'feature_index': list(fmri_counts.keys()),
    'frequency': [count/len(all_fold_fmri_biomarkers) for count in fmri_counts.values()],
    'importance': [np.mean([imp[idx] for imp in all_fold_fmri_importance]) 
                  for idx in fmri_counts.keys()]
})
fmri_df = fmri_df.sort_values('importance', ascending=False)
fmri_df.to_csv('fmri_biomarkers.csv', index=False)

def interpret_biomarkers(microbe_indices, fmri_indices, microbe_loader, file_path, graph_type):
    """解释biomarker的生物学意义"""
    # 获取微生物特征名称
    microbe_feature_names = microbe_loader.get_feature_names()
    
    # 获取fMRI连接名称
    fmri_connections = []
    with h5py.File(file_path, "r") as f:
        # 获取第一个受试者的连接信息作为参考
        first_subject = list(f["/patients"].keys())[0]
        if graph_type in f["/patients"][first_subject]:
            # 这里需要根据实际数据结构调整
            # 假设我们有某种方式获取连接名称
            pass
    
    # 创建解释结果
    microbe_explanations = {idx: microbe_feature_names[idx] for idx in microbe_indices}
    fmri_explanations = {idx: f"连接 {idx}" for idx in fmri_indices}
    
    return microbe_explanations, fmri_explanations

# 在主循环结束后调用
microbe_explanations, fmri_explanations = interpret_biomarkers(
    microbe_consensus, fmri_consensus, microbe_loader, file_path, graph_type
)

# 将解释结果添加到DataFrame中
microbe_df['feature_name'] = microbe_df['feature_index'].map(microbe_explanations)
fmri_df['connection_name'] = fmri_df['feature_index'].map(fmri_explanations)

# 保存带有解释的结果
microbe_df.to_csv('microbe_biomarkers_with_explanations.csv', index=False)
fmri_df.to_csv('fmri_biomarkers_with_explanations.csv', index=False)

def analyze_cross_modal_interactions(model, data_loader, device, microbe_indices, fmri_indices):
    """分析微生物组和fMRI特征之间的交互作用"""
    model.eval()
    interactions = np.zeros((len(microbe_indices), len(fmri_indices)))
    
    with torch.no_grad():
        for batch in data_loader:
            microbe = batch['microbe'].to(device)
            fmri = batch['fmri'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            # 计算梯度
            microbe.requires_grad = True
            fmri.requires_grad = True
            
            h_m, h_f, logits = model(microbe, fmri)
            loss = F.cross_entropy(logits, labels)
            
            # 计算交互梯度
            for i, m_idx in enumerate(microbe_indices):
                for j, f_idx in enumerate(fmri_indices):
                    grad_m = torch.autograd.grad(loss, microbe, create_graph=True)[0][:, m_idx]
                    grad_f = torch.autograd.grad(grad_m, fmri, create_graph=True)[0][:, f_idx]
                    interactions[i, j] += grad_f.abs().mean().item()
    
    # 归一化交互矩阵
    interactions = interactions / len(data_loader)
    
    return interactions

# 在主循环结束后调用
interaction_matrix = analyze_cross_modal_interactions(
    model, test_loader, device, microbe_consensus, fmri_consensus
)

# 可视化交互矩阵
plt.figure(figsize=(10, 8))
plt.imshow(interaction_matrix, cmap='viridis')
plt.colorbar(label='交互强度')
plt.title('微生物组和fMRI特征交互矩阵')
plt.xlabel('fMRI特征索引')
plt.ylabel('微生物组特征索引')
plt.savefig('cross_modal_interactions.png')
plt.close()

# 保存交互矩阵
np.save('cross_modal_interactions.npy', interaction_matrix)