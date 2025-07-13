import math
import biom
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from biom import load_table
import numpy as np
import logging
from datetime import datetime
import pdb
from tqdm import tqdm




class MicrobeDataLoader:
    def __init__(self, csv_path, biom_path, batch_size=32):
        try:
            self.batch_size = batch_size
            self.features_tensor, self.labels_tensor = self._load_from_files(csv_path, biom_path)
            
            # 添加数据验证
            assert len(self.features_tensor) > 0, "特征数据为空！"
            assert len(self.labels_tensor) > 0, "标签数据为空！"
            assert len(self.features_tensor) == len(self.labels_tensor), "特征与标签数量不匹配！"
            
            self._split_data()
            print("数据加载验证通过")
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise

 
    
    def _load_from_files(self, csv_path, biom_path):
        
        metadata = pd.read_csv(csv_path)
        #print(f"元数据加载成功，样本数: {len(metadata)}")
        
        table = load_table(biom_path)
        #print(f"BIOM表加载成功，原始样本数: {table.shape[1]}")

        sample_ids = set(table.ids(axis="sample"))
        metadata_ids = set(metadata["ID"])
        matched_ids = sample_ids & metadata_ids

        #print(f"匹配样本数: {len(matched_ids)}")
        unmatched = sample_ids - metadata_ids
        if unmatched:
            print(f"未匹配样本ID: {unmatched}")

        metadata = metadata[metadata["ID"].isin(matched_ids)].set_index("ID")
        filtered_table = table.filter(matched_ids, axis="sample", inplace=False)

        #if filtered_table.is_empty():
            #raise ValueError("过滤后的BIOM表为空，可能由于样本ID不匹配")

        # 转换为DataFrame并验证
        df = filtered_table.to_dataframe(dense=True).T  # 明确指定dense格式
        #print(f"特征DataFrame形状: {df.shape}")
        #print(f"前5行数据:\n{df.head()}")

        # 空值检查
        #if df.isnull().sum().sum() > 0:
            #raise ValueError("特征数据中存在空值")
        feature_data = self._add_feature_engineering(filtered_table)
        feature_data = feature_data.T

        feature_data = df.values.astype(np.float32)
        return torch.tensor(feature_data, dtype=torch.float32), torch.tensor(metadata['DX_GROUP'].values, dtype=torch.long)
    
    def _add_feature_engineering(self, table):
        # 示例特征工程：计算每个样本的总微生物数量
        data = table.matrix_data.T.toarray()
        return np.log1p(data) 
        


    def _split_data(self):
        try:
            train_features, test_features, train_labels, test_labels = train_test_split(
                self.features_tensor, self.labels_tensor, test_size=0.2, random_state=42
            )
            train_features, val_features, train_labels, val_labels = train_test_split(
                train_features, train_labels, test_size=0.1, random_state=42
            )

            self.train_dataset = TensorDataset(train_features, train_labels)
            self.val_dataset = TensorDataset(val_features, val_labels)
            self.test_dataset = TensorDataset(test_features, test_labels)
        except Exception as e:
            print(f"数据划分失败: {str(e)}")
            raise

    def get_loaders(self):
        return (
            DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4),
            DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4),
            DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        )
        
class SparseMLP(nn.Module):
    def __init__(self, input_dim, num_classes, 
                 hidden_dims=[1024, 512], 
                 dropout=0.5,
                 sparsity_lambda=0.01):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # 构建动态隐藏层
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        self.sparsity_lambda = sparsity_lambda  # L1正则化系数

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def l1_regularization(self):
        """计算稀疏性正则化项"""
        l1_reg = torch.tensor(0., device=next(self.parameters()).device)
        for param in self.feature_extractor.parameters():
            l1_reg += torch.norm(param, p=1)
        return self.sparsity_lambda * l1_reg
    


def train_sparse_mlp(model, train_loader, val_loader, epochs=10, lr=1e-4, 
                    device='cuda', save_path="/home/yangzongxian/xlz/ASD_GCN/main/down/sparse_mlp.pth"):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=True)

    # 初始化日志（根据实际配置补充）
    logger = logging.getLogger("SparseMLPTraining")
    
    best_accuracy = 0.0
    best_loss = float('inf')  # 关键初始化
    early_stop_counter = 0

    for epoch in range(epochs):
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_progress = tqdm(
            val_loader, 
            desc=f"验证 Epoch {epoch+1}/{epochs}", 
            unit="batch", 
            ncols=100, 
            ascii=False, 
            unit_scale=True,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 累计指标
                val_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                
                # 进度条更新
                val_progress.set_postfix({
                    'val_loss': val_loss/total,
                    'accuracy': correct/total
                }, refresh=False)
                val_progress.update()
        
        val_progress.close()
        avg_val_loss = val_loss / total  # 按样本数平均
        val_accuracy = correct / total
        
        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # 更新最佳模型
        if val_accuracy > best_accuracy or avg_val_loss < best_loss:
            best_accuracy = val_accuracy
            best_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved at epoch {epoch+1}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 10:
                logger.warning(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Training completed. Best Validation Accuracy: {best_accuracy:.4f}")
    
if __name__ == "__main__":
    # 数据加载
    csv_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/microbe_data.csv"
    biom_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/feature-table.biom"
    batch_size = 32

    data_loader = MicrobeDataLoader(csv_path, biom_path, batch_size)
    train_loader, val_loader, test_loader = data_loader.get_loaders()

    # 模型初始化
    feature_dim = data_loader.features_tensor.size(1)
    num_classes = len(torch.unique(data_loader.labels_tensor))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mlp_model = SparseMLP(
        input_dim=feature_dim,
        num_classes=num_classes,
        hidden_dims=[2048, 1024, 512],  # 更深的网络结构
        dropout=0.6,
        sparsity_lambda=0.05
    )

    # 训练过程
    train_sparse_mlp(
        mlp_model,
        train_loader,
        val_loader,
        epochs=100,  # 更长的训练轮次
        lr=3e-5,
        device=device,
        save_path="/home/yangzongxian/xlz/ASD_GCN/microbe/pre_train//home/yangzongxian/xlz/ASD_GCN/main/down/sparse_mlp.pth"
    )