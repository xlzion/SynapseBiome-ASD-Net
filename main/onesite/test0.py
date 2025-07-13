from train import fMRI3DGNN,load_pretrained_gnn,load_pretrained_mlp,ContrastiveModel,LabelAwareContrastiveLoss,PairedDataset,validate,test_adversarial,reconstruct_fc,extract_graph_features,extract_microbe_features,expand_to_full_matrix,normalize_features,process_fmri_features,get_accuracy_from_report,LeaveOneSiteOutLoader
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



def main():
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

    # 创建留一站点验证加载器
    fmri_loader = LeaveOneSiteOutLoader(file_path=file_path, graph_type="cc200", batch_size=32)
    microbe_loader = MicrobeDataLoader(csv_path=csv_path, biom_path=biom_path, batch_size=32)

    # 存储所有站点的结果
    all_site_results = []
    
    # 对每个站点进行验证
    for site_idx in range(10):  # 10个站点
        print(f"\n验证站点 {site_idx + 1}/10")
        
        # 获取当前站点的数据加载器
        fmri_loaders = fmri_loader.get_dataloaders(site_idx)
        fmri_test_loader = fmri_loaders["test"]
        microbe_test_loader = microbe_loader.get_loaders()[2]

        # 提取特征
        fmri_test_features, fmri_test_labels = extract_graph_features(fmri_test_loader)
        microbe_test_features, microbe_test_labels = extract_microbe_features(microbe_test_loader)

        # 加载最佳模型
        best_model = ContrastiveModel(mlp_model, gnn_model).to(device)
        best_model.load_state_dict(torch.load("/home/yangzongxian/xlz/ASD_GCN/main/onesite/best_model_site_10.pth"))

        # 创建测试数据集
        test_paired_dataset = PairedDataset(
            (microbe_test_features, microbe_test_labels),
            (fmri_test_features, fmri_test_labels),
            train=False
        )
        test_loader = DataLoader(test_paired_dataset, batch_size=64, shuffle=False)

        # 创建损失函数
        criterion = LabelAwareContrastiveLoss(temp=0.05, hard_neg_ratio=0.2)

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

