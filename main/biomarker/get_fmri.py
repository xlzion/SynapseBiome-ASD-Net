from torch_geometric.data import DataLoader
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tg_nn
import h5py
import torch_geometric.transforms as T
from torch_geometric.data import Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graph_type = "cc200"

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
                if self.graph_type in subject_group and "y" in subject_group.attrs:
                    triu_vector = subject_group[self.graph_type][:]
                    matrix = reconstruct_fc(triu_vector)
                    # 生成40000维输入特征（展平整个矩阵）
                    node_features = matrix.flatten()  # 形状 (200x200=40000,)
                    edge_index = self._get_brain_connectivity_edges(matrix)
                    try:
                        label_value = subject_group.attrs["y"]
                        if isinstance(label_value, (int, np.number)):
                            graph_data = Data(
                                x=torch.FloatTensor(node_features).view(1, -1),  # 直接使用展平后的矩阵
                                edge_index=edge_index,
                                y=torch.tensor([label_value], dtype=torch.long)
                            )
                            graph_dataset.append(graph_data)
                            labels.append(label_value)
                    except KeyError:
                        pass
                        
                else:
                    print(f"Warning: Subject {subject_id} missing {self.graph_type} or label.")
        
        # Data splitting logic remains the same
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
            shuffle=True
        )

    '''def _create_dataloader(self, data, labels):
        return DataLoader(
            data, 
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: (
                torch.stack([d.x.view(1, 200, 2) for d in batch]),  # 处理特征
                torch.cat([d.y for d in batch])  # 正确拼接1维标签
            )
        )'''
    
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
        # 输入应为 [batch_size, 40000]
        if raw_fc.dim() == 1:
            raw_fc = raw_fc.unsqueeze(0)  # 添加批次维度 [1, 40000]
        assert raw_fc.dim() == 2, f"输入应为二维张量 [batch, 40000]，当前维度：{raw_fc.dim()}"
        batch_size = raw_fc.size(0)
        
        # 构建动态图
        batch_graph = self.build_graph(raw_fc)
        
        # 图卷积处理
        x = batch_graph.x  # [batch*200, 16] (经过特征增强)
        edge_index = batch_graph.edge_index
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, training=self.training)
        
        # 全局池化
        x = tg_nn.global_mean_pool(x, batch_graph.batch)  # [batch, hidden_dim]
        return self.classifier(x)
    
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

def get_top_rois(model, dataloader, device, top_k=100, roi_names=None):
    model.eval()
    gradients = []
    
    progress = tqdm(dataloader, desc="计算ROI重要性", unit="batch")
    
    try:
        for batch in progress:
            # 解包批次数据为 (input_features, labels)
            x_batch = batch.x.to(device)
            print("输入维度检查:", x_batch.shape) # [batch_size, 40000]
            x_batch = x_batch.requires_grad_(True)  # [batch_size, 40000]
            
            y_batch = batch.y  # [batch_size]
            
            
            # 前向传播
            outputs = model(x_batch)  # 确保输入是 [batch_size, 40000]
            
            # 梯度计算
            grads = torch.autograd.grad(
                outputs.sum(),
                x_batch,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )
            
            # 记录梯度
            gradients.append(grads[0].abs().mean(dim=0).cpu())
 
        # 合并梯度
        all_grads = torch.stack(gradients).mean(dim=0).numpy()
        
        # 转换为连接矩阵
        conn_matrix = all_grads.reshape(200, 200)
        roi_importance = conn_matrix.sum(0) + conn_matrix.sum(1)
        
        # 结果处理
        sorted_indices = np.argsort(roi_importance)[::-1]
        roi_names = roi_names or [f"ROI_{i+1:03d}" for i in range(200)]
        
        '''print(f"\nTop {top_k} fMRI ROIs:")
        for i in range(top_k):
            idx = sorted_indices[i]
            print(f"{i+1}. {roi_names[idx]} ({roi_importance[idx]:.4f})")
            
        return roi_importance, roi_names'''
        print(f"\nTop {top_k} fMRI ROIs (按索引显示):")
        for i in range(top_k):
            idx = sorted_indices[i]
            print(f"{i+1}. ROI_{idx+1:03d} 重要性分数: {roi_importance[idx]:.4f}")
            
        return roi_importance
    
    except Exception as e:
        print(f"分析失败: {str(e)}")
        return None, None

# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    # 加载预训练模型
    gnn_model = load_pretrained_gnn().to(device)
    file_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/abide.hdf5"
    graph_type = "cc200"
    fmri_loader = fMRIDataLoader(file_path = file_path, graph_type="cc200", batch_size=32)
    cc200_names = [
        "Precentral_L", "Precentral_R",  # 实际应使用完整的200个名称
        # ... 补充完整名称列表
    ]
    
    # 获取测试集数据加载器
    fmri_train_loader = fmri_loader.get_dataloaders()["train"]
    fmri_val_loader = fmri_loader.get_dataloaders()["valid"]
    fmri_test_loader = fmri_loader.get_dataloaders()["test"]
    
    # 运行分析
    importance_scores, roi_names = get_top_rois(
        model=gnn_model,
        dataloader=fmri_train_loader,
        device=device,
        top_k=100,
        roi_names=cc200_names
    )
    
    
    '''if importance_scores is not None:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance_scores)), importance_scores)
        plt.title("ROI Importance Distribution")
        plt.xlabel("ROI Index")
        plt.ylabel("Importance Score")
        plt.show()'''