# MLP.py

import math
import biom
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split # Keep this
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from biom import load_table
import numpy as np
import logging
from datetime import datetime
# import pdb # You can remove this if not used
from tqdm import tqdm

_MODEL_PARAMS = {
    "hidden_dim": 512,
    "nhead": 8,
    "num_layers": 3,
    "dropout": 0.2
}

# 训练超参数
_TRAIN_PARAMS = {
    "learning_rate": 1e-4,
    "patience": 10
}

class MicrobeDataLoader:
    def __init__(self, csv_path, biom_path, batch_size=32, config=None):
        try:
            self.batch_size = batch_size
            self.config = config if config is not None else {}
            
            # _load_from_files will now return 5 items
            (self.features_tensor, self.labels_tensor, 
             self.subject_ids_np, self.ages_tensor, self.genders_tensor) = self._load_from_files(csv_path, biom_path)

            assert len(self.features_tensor) > 0, "Microbe features data is empty!"
            assert len(self.labels_tensor) > 0, "Microbe labels data is empty!"
            assert len(self.features_tensor) == len(self.labels_tensor) == len(self.subject_ids_np) == \
                   len(self.ages_tensor) == len(self.genders_tensor), \
                   "Microbe features, labels, SIDs, ages, or genders have inconsistent sample counts."
            
            self._split_data() # This will now split all 5 data types
            print("MicrobeDataLoader: Data loading and splitting verified.")
        except Exception as e:
            print(f"MicrobeDataLoader: Data loading/splitting failed: {str(e)}")
            raise
 
    def _load_from_files(self, csv_path, biom_path):
        metadata_df = pd.read_csv(csv_path)
        biom_table = load_table(biom_path)
        
        biom_sample_ids = set(biom_table.ids(axis="sample"))
        metadata_sample_ids = set(metadata_df["ID"].astype(str)) # Ensure ID is string for matching

        if metadata_df["ID"].astype(str).duplicated().any():
            raise ValueError("Duplicate IDs found in microbe metadata CSV. 'ID' column must be unique.")
            
        matched_ids = sorted(list(biom_sample_ids & metadata_sample_ids)) # Sort for consistency
        if not matched_ids:
            raise ValueError("No common subject IDs found between BIOM table and microbe metadata CSV.")

        # Filter and reorder metadata to match the order of matched_ids
        metadata_filtered_df = metadata_df[metadata_df["ID"].astype(str).isin(matched_ids)].set_index("ID", drop=False)
        metadata_aligned_df = metadata_filtered_df.loc[matched_ids] # Reorder to matched_ids order

        # Filter BIOM table to matched_ids (and in that order)
        # Note: table.filter might not preserve order if ids are not in table's current order.
        # It's safer to extract all, convert to df, then reindex/filter.
        biom_df_full = biom_table.to_dataframe(dense=True).T # Samples as rows
        biom_df_aligned = biom_df_full.loc[matched_ids] # Filter and order by matched_ids

        features_np = biom_df_aligned.values.astype(np.float32)
        
        # Extract labels and metadata from the aligned metadata dataframe
        labels_np = metadata_aligned_df['DX_GROUP'].values.astype(np.int64)
        ages_np = metadata_aligned_df['AGE_AT_SCAN'].values.astype(np.float32)
        genders_np = metadata_aligned_df['SEX'].values.astype(np.int64)
        subject_ids_np_array = metadata_aligned_df['ID'].values.astype(str) # Keep as numpy array of strings

        return (torch.tensor(features_np, dtype=torch.float32),
                torch.tensor(labels_np, dtype=torch.long),
                subject_ids_np_array, # Return as NumPy array of strings
                torch.tensor(ages_np, dtype=torch.float32),
                torch.tensor(genders_np, dtype=torch.long))

    def _split_data(self):
        try:
            test_size = self.config.get('microbe_test_split_ratio', self.config.get('fMRI_test_split_ratio', 0.2))
            val_split_input = self.config.get('microbe_val_split_ratio', self.config.get('fMRI_val_split_ratio', 0.125))
            random_state = self.config.get('random_state_data', 42)
            val_size_of_train_val = val_split_input / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0

            if not (0 <= val_size_of_train_val < 1.0): # Allow 0 for no validation set
                 val_size_of_train_val = 0
                 print("Warning (Microbe): Effective validation split ratio for train_val pool is not valid or zero. Validation set will be empty.")

            indices = np.arange(len(self.features_tensor))
            
            train_val_idx, test_idx = train_test_split(
                indices, test_size=test_size, stratify=self.labels_tensor.cpu().numpy(), random_state=random_state
            )
            
            train_idx, val_idx = np.array([],dtype=int), np.array([],dtype=int) # Ensure they are int arrays
            if len(train_val_idx) > 0 and val_size_of_train_val > 0:
                labels_train_val_for_split = self.labels_tensor[train_val_idx].cpu().numpy()
                if len(np.unique(labels_train_val_for_split)) > 1:
                    train_idx, val_idx = train_test_split(
                        train_val_idx, test_size=val_size_of_train_val, stratify=labels_train_val_for_split, random_state=random_state
                    )
                else:
                    train_idx, val_idx = train_test_split(
                        train_val_idx, test_size=val_size_of_train_val, random_state=random_state # No stratify if only one class
                    )
            elif len(train_val_idx) > 0: # No validation split from train_val pool
                train_idx = train_val_idx
            
            # Store split data as attributes (for potential direct access to numpy arrays)
            self.train_features_np = self.features_tensor[train_idx].cpu().numpy()
            self.train_labels_np = self.labels_tensor[train_idx].cpu().numpy()
            self.train_ages_np = self.ages_tensor[train_idx].cpu().numpy()
            self.train_genders_np = self.genders_tensor[train_idx].cpu().numpy()
            self.train_sids_np = self.subject_ids_np[train_idx]

            self.val_features_np = self.features_tensor[val_idx].cpu().numpy()
            self.val_labels_np = self.labels_tensor[val_idx].cpu().numpy()
            self.val_ages_np = self.ages_tensor[val_idx].cpu().numpy()
            self.val_genders_np = self.genders_tensor[val_idx].cpu().numpy()
            self.val_sids_np = self.subject_ids_np[val_idx]

            self.test_features_np = self.features_tensor[test_idx].cpu().numpy()
            self.test_labels_np = self.labels_tensor[test_idx].cpu().numpy()
            self.test_ages_np = self.ages_tensor[test_idx].cpu().numpy()
            self.test_genders_np = self.genders_tensor[test_idx].cpu().numpy()
            self.test_sids_np = self.subject_ids_np[test_idx]

            # Create TensorDatasets - SIDs are not included here as they are strings
            # The extract_all_modal_data function will fetch SIDs from the _sids_np attributes
            self.train_dataset = TensorDataset(self.features_tensor[train_idx], self.labels_tensor[train_idx], self.ages_tensor[train_idx], self.genders_tensor[train_idx])
            self.val_dataset = TensorDataset(self.features_tensor[val_idx], self.labels_tensor[val_idx], self.ages_tensor[val_idx], self.genders_tensor[val_idx]) if len(val_idx) > 0 else TensorDataset(torch.empty(0, self.features_tensor.shape[1]), torch.empty(0, dtype=torch.long), torch.empty(0), torch.empty(0)) # Handle empty val_idx
            self.test_dataset = TensorDataset(self.features_tensor[test_idx], self.labels_tensor[test_idx], self.ages_tensor[test_idx], self.genders_tensor[test_idx])

        except Exception as e:
            print(f"MicrobeDataLoader: Data splitting failed: {str(e)}")
            raise

    def get_loaders(self):
        num_workers = self.config.get('num_workers', 0)
        # Ensure val_dataset is not empty before creating DataLoader, or handle appropriately
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers) if len(self.val_dataset) > 0 else None
        return (
            DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers),
            val_dl,
            DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
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
                    device='cuda', save_path="sparse_mlp.pth"):
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
            for inputs, labels, _, _ in val_loader:
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
 
 
 
 # (Keep SparseMLP and train_sparse_mlp as they are, or ensure they are robust)
# ... (rest of MLP.py: SparseMLP, train_sparse_mlp, if __name__ == "__main__": ...)
# Make sure to adjust the if __name__ == "__main__": part of MLP.py if you run it directly,
# as MicrobeDataLoader now expects a 'config' argument for splitting parameters.
# Example for standalone MLP.py execution:
# dummy_config = {'random_state_data': 42, 'fMRI_test_split_ratio': 0.2, 'fMRI_val_split_ratio': 0.1, 'num_workers': 0}
# data_loader = MicrobeDataLoader(csv_path, biom_path, batch_size, config=dummy_config)   
if __name__ == "__main__":
    # 数据加载
    csv_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/microbe_data.csv"
    biom_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/feature-table.biom"
    batch_size = 32
    dummy_config = {
        'random_state_data': 42,
        'fMRI_test_split_ratio': 0.2, # Used as fallback by MicrobeDataLoader
        'fMRI_val_split_ratio': 0.125, # Used as fallback (0.125 on remaining 0.8 is 0.1 of total)
        'num_workers': 0
    }
    #data_loader = MicrobeDataLoader(csv_path, biom_path, batch_size)
    data_loader = MicrobeDataLoader(csv_path, biom_path, batch_size, config=dummy_config)
    train_loader, val_loader, test_loader = data_loader.get_loaders()

    # 模型初始化
    feature_dim = data_loader.features_tensor.size(1)
    num_classes = len(torch.unique(data_loader.labels_tensor))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mlp_model = SparseMLP(
        input_dim=feature_dim,
        num_classes=num_classes,
        hidden_dims=[2048, 1024, 512],  
        dropout=0.6,
        sparsity_lambda=0.05
    )

    # 训练过程
    train_sparse_mlp(
        mlp_model,
        train_loader,
        val_loader,
        epochs=100, 
        lr=3e-5,
        device=device,
        save_path="/home/yangzongxian/xlz/ASD_GCN/main/new/sparse_mlp.pth"
    )