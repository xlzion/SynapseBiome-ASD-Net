import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import os
import h5py
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import community  
import networkx as nx
import sys
from pathlib import Path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent  
target_subdir = parent_dir / "pre_train"        
sys.path.append(str(target_subdir))

from adversarial10 import (
    ContrastiveModel, 
    LabelAwareContrastiveLoss, 
    PairedDataset,
    load_pretrained_mlp,
    load_pretrained_gnn,
    fMRIDataLoader,
    MicrobeDataLoader,
    extract_graph_features,
    extract_microbe_features
)

from multimodal_biomarkers import (
    extract_multimodal_biomarkers,
    extract_multimodal_biomarkers_perturbation,
    extract_integrated_biomarkers,
    analyze_multimodal_feature_network,
    visualize_multimodal_network,
    analyze_microbe_roi_connections,
    main_multimodal_biomarker_analysis,
    extract_biomarkers_gradient,
    extract_biomarkers_integrated_gradients,
    extract_biomarkers_attention,
    extract_biomarkers_backprop,
    analyze_feature_interactions_efficient,
    extract_biomarkers_gradcam,
    extract_biomarkers_ensemble,
    main_multimodal_biomarker_analysis,
    extract_biomarkers,
    convert_biomarkers_to_roi_connections,
    index_to_matrix_coords,
    get_roi_names,
    visualize_microbe_roi_network,
    visualize_roi_network,
    pyg_to_custom_batch
)

# 设置设备
if torch.cuda.is_available():
    # 获取可用的 CUDA 设备数量
    num_devices = torch.cuda.device_count()
    if num_devices > 0:
        # 使用第一个可用设备
        torch.cuda.set_device(0)
    else:
        print("没有可用的 CUDA 设备，使用 CPU")
else:
    print("CUDA 不可用，使用 CPU")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置数据路径
file_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/abide.hdf5"
graph_type = "cc200"
csv_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/microbe_data.csv"
biom_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/feature-table.biom"
output_dir = "biomarkers_results"
os.makedirs(output_dir, exist_ok=True)
# 加载预训练模型
def load_pretrained_multimodal_model(model_path):
    """
    加载预训练的多模态模型
    
    参数:
    - model_path: 模型文件路径
    
    返回:
    - model: 加载好的模型
    """
    
    mlp_model = load_pretrained_mlp()
    gnn_model = load_pretrained_gnn()
    model = ContrastiveModel(mlp_model, gnn_model).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"成功加载模型: {model_path}")
    return model
    
if __name__ == "__main__":
    # 加载预训练模型
    model_path = "/home/yangzongxian/xlz/ASD_GCN/main/pre_train/contrastive_model.pth" 
    model = load_pretrained_multimodal_model(model_path)

    if model is None:
        print("无法加载预训练模型，请检查模型文件是否存在或格式是否正确")
        exit(1)

    # 加载数据
    print("加载数据...")
    fmri_loader = fMRIDataLoader(file_path = file_path, graph_type="cc200", batch_size=32)
    microbe_loader = MicrobeDataLoader(csv_path = csv_path, biom_path=biom_path, batch_size=32)
    
    pyg_train_loader = fmri_loader.get_dataloaders()["train"]
    
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

    full_paired_dataset = PairedDataset(
    microbe_data=(
        np.concatenate([microbe_train_features, microbe_val_features]),  # 微生物组特征
        np.concatenate([microbe_train_labels, microbe_val_labels])       # 微生物组标签
    ),
    fmri_data=(
        np.concatenate([fmri_train_features, fmri_val_features]),        # fMRI特征
        np.concatenate([fmri_train_labels, fmri_val_labels])             # fMRI标签
    ),
    train=False  # 明确指定是否为训练模式
)
        # 创建生物标志物分析专用数据加载器
    biomarker_loader = DataLoader(
            full_paired_dataset,
            batch_size=64,
            collate_fn=lambda batch: {
                'microbe': torch.stack([x['microbe'] for x in batch]),
                'fmri': torch.stack([x['fmri'] for x in batch]),
                'label': torch.stack([x['label'] for x in batch])
            }
        )





    roi_names = get_roi_names(file_path, graph_type=graph_type)
    print(f"获取了 {len(roi_names)} 个ROI名称")

    # 执行多模态生物标志物分析
    print("执行多模态生物标志物分析...")
    #main_multimodal_biomarker_analysis(model, microbe_val_loader, device, output_dir="biomarkers_results", top_k=50)
    main_multimodal_biomarker_analysis(
        model=model,
        data_loader=biomarker_loader,  # 使用配对数据加载器
        device=device,
        output_dir="final_biomarkers",
        top_k=50
    )
    # 提取微生物组生物标志物
    print("提取微生物组生物标志物...")
    microbe_indices, microbe_values = extract_biomarkers_ensemble(model, microbe_val_loader, device, feature_type='microbe', top_k=50)

    # 提取fMRI生物标志物
    print("提取fMRI生物标志物...")
    fmri_indices, fmri_values = extract_biomarkers_ensemble(model, fmri_val_loader, device, feature_type='fmri', top_k=50)

    # 保存微生物组生物标志物
    pd.DataFrame({
        'feature_index': microbe_indices,
        'importance': microbe_values
    }).to_csv(f"{output_dir}/microbe_biomarkers_ensemble.csv", index=False)

    # 保存fMRI生物标志物
    pd.DataFrame({
        'feature_index': fmri_indices,
        'importance': fmri_values
    }).to_csv(f"{output_dir}/fmri_biomarkers_ensemble.csv", index=False)

    # 将fMRI生物标志物转换为ROI连接
    print("将fMRI生物标志物转换为ROI连接...")
    roi_connections = convert_biomarkers_to_roi_connections(fmri_indices, roi_names)

    # 保存ROI连接
    pd.DataFrame({
        'roi1': [conn[0] for conn in roi_connections],
        'roi2': [conn[1] for conn in roi_connections],
        'importance': fmri_values
    }).to_csv(f"{output_dir}/roi_connections_ensemble.csv", index=False)

    # 方法2：使用集成梯度方法提取生物标志物
    print("方法2：使用集成梯度方法提取生物标志物...")
    microbe_indices_ig, microbe_values_ig = extract_biomarkers_integrated_gradients(model, microbe_val_loader, device, feature_type='microbe', top_k=50)
    fmri_indices_ig, fmri_values_ig = extract_biomarkers_integrated_gradients(model, fmri_val_loader, device, feature_type='fmri', top_k=50)

    # 保存微生物组生物标志物
    pd.DataFrame({
        'feature_index': microbe_indices_ig,
        'importance': microbe_values_ig
    }).to_csv(f"{output_dir}/microbe_biomarkers_ig.csv", index=False)

    # 保存fMRI生物标志物
    pd.DataFrame({
        'feature_index': fmri_indices_ig,
        'importance': fmri_values_ig
    }).to_csv(f"{output_dir}/fmri_biomarkers_ig.csv", index=False)

    # 方法3：使用反向传播方法提取生物标志物
    print("方法3：使用反向传播方法提取生物标志物...")
    microbe_indices_bp, microbe_values_bp = extract_biomarkers_backprop(model, microbe_val_loader, device, feature_type='microbe', top_k=50)
    fmri_indices_bp, fmri_values_bp = extract_biomarkers_backprop(model, fmri_val_loader, device, feature_type='fmri', top_k=50)

    # 保存微生物组生物标志物
    pd.DataFrame({
        'feature_index': microbe_indices_bp,
        'importance': microbe_values_bp
    }).to_csv(f"{output_dir}/microbe_biomarkers_bp.csv", index=False)

    # 保存fMRI生物标志物
    pd.DataFrame({
        'feature_index': fmri_indices_bp,
        'importance': fmri_values_bp
    }).to_csv(f"{output_dir}/fmri_biomarkers_bp.csv", index=False)

    # 方法4：分析特征交互
    print("方法4：分析特征交互...")
    microbe_interaction_indices, fmri_interaction_indices, interaction_values = analyze_feature_interactions_efficient(model, microbe_val_loader, device, top_k=50)

    # 保存特征交互
    pd.DataFrame({
        'microbe_index': microbe_interaction_indices,
        'fmri_index': fmri_interaction_indices,
        'importance': interaction_values
    }).to_csv(f"{output_dir}/feature_interactions.csv", index=False)

    # 将fMRI索引转换为ROI连接
    roi_interaction_connections = convert_biomarkers_to_roi_connections(fmri_interaction_indices, roi_names)

    # 保存ROI交互连接
    pd.DataFrame({
        'microbe_index': microbe_interaction_indices,
        'roi1': [conn[0] for conn in roi_interaction_connections],
        'roi2': [conn[1] for conn in roi_interaction_connections],
        'importance': interaction_values
    }).to_csv(f"{output_dir}/microbe_roi_interactions.csv", index=False)

    # 方法5：执行完整的多模态生物标志物分析
    print("方法5：执行完整的多模态生物标志物分析...")
    main_multimodal_biomarker_analysis(model, microbe_val_loader, device, output_dir=f"{output_dir}/complete_analysis", top_k=50)

    # 打印前10个重要的微生物组特征
    print("\n前10个重要的微生物组特征 (综合方法):")
    for i in range(min(10, len(microbe_indices))):
        print(f"{i+1}. 特征索引: {microbe_indices[i]}, 重要性: {microbe_values[i]:.6f}")

    # 打印前10个重要的ROI连接
    print("\n前10个重要的ROI连接 (综合方法):")
    for i in range(min(10, len(roi_connections))):
        print(f"{i+1}. {roi_connections[i][0]} - {roi_connections[i][1]}, 重要性: {fmri_values[i]:.6f}")



    # 可视化ROI连接
    visualize_roi_network(roi_connections, fmri_values, top_k=20, output_file=f"{output_dir}/roi_network_ensemble.png")

    print(f"\n生物标志物分析完成，结果保存在 {output_dir} 目录中")