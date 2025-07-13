import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import torch_geometric.nn as tg_nn
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score

file_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/abide.hdf5"
graph_type = "cc200"



def reconstruct_fc(vector):
    """将上三角向量重建为对称矩阵"""
    # 创建空矩阵
    matrix = np.zeros((200, 200))
    
    # 填充对角线
    np.fill_diagonal(matrix, 1.0)
    
    # 提取上三角索引（不包括对角线）
    triu_indices = np.triu_indices(200, k=1)
    
    # 验证输入维度
    assert len(vector) == triu_indices[0].size, "输入维度不匹配"
    
    # 填充上三角
    matrix[triu_indices] = vector
    
    # 对称复制到下三角
    matrix = matrix + matrix.T - np.diag(matrix.diagonal())
    
    return matrix

class FCReconstructor:
    @staticmethod
    def vector_to_matrix(vector):
        """专业化的矩阵重建类"""
        # 添加噪声增强
        noise = np.random.normal(scale=0.1, size=vector.shape)
        vector = np.clip(vector + noise, -1, 1)
        matrix = np.zeros((200, 200))
        triu_idx = np.triu_indices(200, k=1)
        
        # 高级数值稳定性处理
        vector = np.clip(vector, -1.0, 1.0)
        matrix[triu_idx] = vector
        matrix += matrix.T
        np.fill_diagonal(matrix, 1.0)
        return matrix



class fMRIDataLoader:
    def __init__(self, file_path, graph_type, test_size=0.15, val_size=0.15, batch_size=32):
        self.file_path = file_path
        self.graph_type = graph_type
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.data_splits = self._load_and_split_data()
        

    def _load_and_split_data(self):
        subjects = []
        labels = []
        
        with h5py.File(self.file_path, "r") as f:
            patients_group = f["/patients"]
            for subject_id in patients_group.keys():
                subject_group = patients_group[subject_id]
                if self.graph_type in subject_group:
                    # 加载原始向量
                    triu_vector = subject_group[self.graph_type][:]
                    
                    # 重建完整矩阵
                    fc_matrix = reconstruct_fc(triu_vector)
                    
                    # 展平为40000维
                    subjects.append(fc_matrix.flatten())
                    labels.append(subject_group.attrs["y"])
                    
        subjects = np.array(subjects)
        labels = np.array(labels)
        
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            subjects, labels, test_size=self.test_size, random_state=42
        )
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_val_data, train_val_labels, test_size=self.val_size / (1 - self.test_size), random_state=42
        )
        return {
            "train": (train_data, train_labels),
            "valid": (val_data, val_labels),
            "test": (test_data, test_labels),
        }

    def _create_dataloader(self, data, labels):
        
        dataset = []
        for fc, y in zip(data, labels):
            dataset.append({
                'x': torch.tensor(fc, dtype=torch.float32),
                'y': torch.tensor(y, dtype=torch.long)
            })
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)

    def get_dataloaders(self):
        return {
            "train": self._create_dataloader(*self.data_splits["train"]),
            "valid": self._create_dataloader(*self.data_splits["valid"]),
            "test": self._create_dataloader(*self.data_splits["test"]),
        }

    def get_input_dim(self):
        return self.data_splits["train"][0].shape[1]

    def get_num_classes(self):
        return len(set(self.data_splits["train"][1]))

    def _collate_fn(self, batch):
        """自定义批处理函数"""
        return {
            'fc': torch.stack([item['x'] for item in batch]),
            'labels': torch.stack([item['y'] for item in batch])
        }

class CustomDataset(Dataset):
    def __init__(self, h5_path, graph_type):
        self.h5_path = h5_path
        self.graph_type = graph_type
        self.subjects = []
        
        with h5py.File(h5_path, 'r') as f:
            for subj in f['/patients']:
                group = f[f'/patients/{subj}']
                if graph_type in group:
                    vector = group[graph_type][:]
                    matrix = FCReconstructor.vector_to_matrix(vector)
                    self.subjects.append({
                        'matrix': matrix,
                        'label': group.attrs['y']
                    })

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return {
            'fc': torch.FloatTensor(self.subjects[idx]['matrix'].flatten()),
            'label': torch.LongTensor([self.subjects[idx]['label']])
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


class MultiModalContrast(nn.Module):
    def __init__(self, mlp_model, gnn_model, feat_dim=128):
        super().__init__()
        # 微生物组特征提取器
        self.mlp_proj = nn.Sequential(
            mlp_model.feature_extractor,
            nn.Linear(512, feat_dim),  # 假设MLP最终输出512维
            nn.BatchNorm1d(feat_dim)
        )
        
        # fMRI特征提取器
        self.gnn_proj = nn.Sequential(
            gnn_model.convs,
            nn.Linear(128, feat_dim),  # 假设GNN输出128维
            nn.BatchNorm1d(feat_dim)
        )
        
        # 对抗判别器
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, microbe_input, fmri_input):
        # 特征提取
        h_microbe = F.normalize(self.mlp_proj(microbe_input), p=2, dim=1)
        h_fmri = F.normalize(self.gnn_proj(fmri_input), p=2, dim=1)
        
        # 对抗损失
        combined = torch.cat([h_microbe, h_fmri], dim=0)
        domain_pred = self.domain_classifier(combined.detach())  # 阻断梯度反传
        
        # 对比损失
        sim_matrix = torch.mm(h_microbe, h_fmri.t()) / 0.07  # 温度系数
        labels = torch.arange(h_microbe.size(0)).to(h_microbe.device)
        
        return {
            'domain_logits': domain_pred,
            'similarity': sim_matrix,
            'contrast_labels': labels
        }

def train_and_test_model(model, train_loader, valid_loader, test_loader, epochs=100, lr=1e-4, 
                        save_path="ASD_GCN/microbe/pre_train/fmri_gnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    '''optimizer = torch.optim.AdamW([
        {'params': model.convs.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr*0.1}
    ], weight_decay=1e-4)'''
    # 修改优化器定义
    optimizer = torch.optim.AdamW([ 
    {'params': model.convs.parameters()},
    {'params': model.classifier.parameters()},
    {'params': model.graph_builder.parameters()},  # 新增
    {'params': model.feature_enhancer.parameters()} # 新增
    ], lr=lr, weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        threshold=0.001,
        verbose=True
    )
    best_accuracy = 0.0
    scaler = torch.cuda.amp.GradScaler()

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_bar = tqdm(total=len(train_loader), 
                        desc=f"Epoch {epoch+1}/{epochs}",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = batch['fc'].to(device)
            targets = batch['labels'].to(device)
            
            # 混合精度训练修正
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                loss = criterion(logits, targets)
            
            # 统一反向传播流程
            scaler.scale(loss).backward()  # 使用scaler统一管理
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            epoch_bar.update(1)
            epoch_bar.set_postfix({
                'loss': f"{total_loss/(epoch_bar.n+1):.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        epoch_bar.close()

        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # 创建带统计信息的进度条
        val_bar = tqdm(
            valid_loader, 
            desc=f"验证 Epoch {epoch+1}/{epochs}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            postfix={"loss": "N/A", "acc": "0.00%"}
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_bar):
                inputs = batch['fc'].to(device)
                targets = batch['labels'].to(device)
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                
                # 统计指标
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(logits, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                # 实时更新进度条
                current_loss = val_loss / total
                current_acc = correct / total
                val_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{current_acc:.2%}"
                }, refresh=False)

        # 计算最终验证指标
        val_loss /= len(valid_loader.dataset)
        val_accuracy = correct / total

        # 更新学习率调度器
        scheduler.step(val_accuracy)

        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy
            }, save_path)
            val_bar.write(f"✨ 新最佳模型保存于epoch {epoch+1} | 验证准确率: {val_accuracy:.2%}")

        # 关闭进度条
        val_bar.close()

        # 打印epoch总结
        print(f"Epoch {epoch+1}/{epochs} | "
            f"训练损失: {total_loss/len(train_loader):.4f} | "
            f"验证损失: {val_loss:.4f} | "
            f"验证准确率: {val_accuracy:.2%}")
        
    print("\n正在加载最佳模型进行测试...")
    best_model = fMRI3DGNN(GNN_CONFIG).to(device)
    checkpoint = torch.load(save_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])

    # Updated call to evaluate_model
    test_metrics, report = evaluate_model(
        best_model, test_loader, device, full_report=True
    )

    # Print the results
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    print("\n详细分类报告:")
    print(report)

def evaluate_model(model, loader, device, full_report=False):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = [] # Store probabilities for AUC

    with torch.no_grad():
        for batch in loader:
            inputs = batch['fc'].to(device)
            targets = batch['labels'].to(device)

            logits = model(inputs)
            probs = torch.softmax(logits, dim=1) # Get probabilities
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Probability of the positive class

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # Ensure there are samples from both classes to calculate AUC
    if len(set(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_probs)
    else:
        auc = float('nan') # Not applicable if only one class is present

    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1
    }

    if full_report:
        report = classification_report(
            all_targets, all_preds,
            target_names=["Healthy", "ASD"], # Adjust based on your actual labels
            digits=4
        )
        return metrics, report

    return metrics

if __name__ == "__main__":
    # 修改配置参数
    GNN_CONFIG = {
    "gnn_layers": 3,          # 控制GAT层数
    "hidden_channels": 128,   # 每层隐藏单元数
    "num_classes": 2,
    "dropout": 0.4
    }
    dataset = CustomDataset(
        "/home/yangzongxian/xlz/ASD_GCN/main/data2/abide.hdf5",
        "cc200"
    )
    sample = dataset[0]
    print("重建后矩阵维度:", sample['fc'].shape) 
    # 数据加载
    fmri_loader = fMRIDataLoader(
        file_path=file_path,
        graph_type=graph_type,
        batch_size=16  
    )

    # 获取所有数据加载器
    loaders = fmri_loader.get_dataloaders()
    print(f"成功加载样本数: {len(fmri_loader.data_splits['train'][0])}")
    print(f"特征维度: {fmri_loader.get_input_dim()}")
    
    
    test_input = torch.randn(16, 40000)  # batch_size=16
    model = fMRI3DGNN(GNN_CONFIG)
    print("测试输入维度:", test_input.shape)
    output = model(test_input)
    print("测试输出维度:", output.shape)  # 应输出[16, 2]
    
    
    # 初始化GNN模型
    model = fMRI3DGNN(GNN_CONFIG)
    
    # 参数数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 启动训练+测试流程
    train_and_test_model(
        model=model,
        train_loader=loaders["train"],
        valid_loader=loaders["valid"],
        test_loader=loaders["test"],
        epochs=50,
        lr=3e-4,
        save_path="/home/yangzongxian/xlz/ASD_GCN/main/new/fmri_gnn.pth"
    )

    