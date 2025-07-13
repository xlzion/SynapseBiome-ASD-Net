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
)
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
import torch.nn.functional as F

def extract_biomarkers(model, data_loader, device, feature_type='microbe', top_k=20):
    """
    提取生物标志物
    
    参数:
    - model: 预训练模型
    - data_loader: 数据加载器
    - device: 计算设备
    - feature_type: 特征类型，'microbe'或'fmri'
    - top_k: 返回前k个重要特征
    
    返回:
    - indices: 重要特征索引
    - values: 重要特征值
    """
    model.eval()  # 设置为评估模式
    
    # 初始化结果列表
    all_gradients = []
    
    # 遍历数据
    for batch in data_loader:
        # 获取输入数据
        if feature_type == 'fmri':
            input_tensor = batch['fmri'].to(device)  
            other_tensor = torch.zeros(input_tensor.size(0), 2503)  # 微生物占位符
        else:
            input_tensor = batch['microbe'].to(device)
            other_tensor = torch.zeros(input_tensor.size(0), 40000) 
        
        # 确保输入张量需要梯度
        input_tensor.requires_grad_(True)
        
        # 前向传播
        h_microbe, h_fmri, logits = model(input_tensor, other_tensor)
        
        # 计算损失
        criterion = LabelAwareContrastiveLoss()
        loss = criterion(h_microbe, h_fmri, batch['label'].to(device))
        
        # 计算梯度
        grad = torch.autograd.grad(loss, input_tensor, retain_graph=True)[0]
        
        # 计算梯度绝对值
        grad_abs = torch.abs(grad).mean(dim=0)
        all_gradients.append(grad_abs)
    
    # 合并所有梯度
    all_gradients = torch.stack(all_gradients).mean(dim=0)
    
    # 获取top-k重要特征
    top_k_values, top_k_indices = torch.topk(all_gradients, top_k)
    
    return top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()

def index_to_matrix_coords(index, matrix_size=200):
    """
    将索引转换为矩阵坐标
    
    参数:
    - index: 索引
    - matrix_size: 矩阵大小
    
    返回:
    - row, col: 行和列坐标
    """
    row = index // matrix_size
    col = index % matrix_size
    return row, col

def pyg_to_custom_batch(batch):
    """将PyG的DataBatch转换为包含microbe和fmri键的字典"""
    return {
        'microbe': batch.x[:, :2503],  # 前2503维作为微生物组特征
        'fmri': batch.x[:, 2503:],     # 后40000维作为fMRI特征
        'label': batch.y
    }
    
def convert_biomarkers_to_roi_connections(biomarker_indices, roi_names, matrix_size=200):
    """
    将生物标志物索引转换为ROI连接
    
    参数:
    - biomarker_indices: 生物标志物索引
    - roi_names: ROI名称映射
    - matrix_size: 矩阵大小
    
    返回:
    - roi_connections: ROI连接列表
    """
    roi_connections = []
    
    for idx in biomarker_indices:
        row, col = index_to_matrix_coords(idx, matrix_size)
        
        # 获取ROI名称
        roi1_name = roi_names.get(row, f"ROI_{row}")
        roi2_name = roi_names.get(col, f"ROI_{col}")
        
        roi_connections.append((roi1_name, roi2_name))
    
    return roi_connections

def extract_multimodal_biomarkers(model, data_loader, device, top_k=20):
    """
    提取多模态生物标志物，同时考虑微生物组和fMRI特征及其相互作用
    
    参数:
    - model: 训练好的模型
    - data_loader: 数据加载器
    - device: 计算设备
    - top_k: 返回前k个重要特征
    
    返回:
    - microbe_indices: 重要微生物组特征索引
    - fmri_indices: 重要fMRI特征索引
    - interaction_scores: 特征交互重要性分数
    """
    model.train()
    
    # 初始化结果列表
    all_microbe_gradients = []
    all_fmri_gradients = []
    all_interaction_scores = []
    
    for batch in data_loader:
        # 获取输入数据
        microbe_input = batch['microbe'].to(device)
        fmri_input = batch['fmri'].to(device)
        
        # 确保输入张量需要梯度
        microbe_input.requires_grad_(True)
        fmri_input.requires_grad_(True)
        
        # 前向传播
        h_microbe, h_fmri, logits = model(microbe_input, fmri_input)
        
        # 计算损失
        criterion = LabelAwareContrastiveLoss()
        loss = criterion(h_microbe, h_fmri, batch['label'].to(device))
        
        # 计算对微生物组输入的梯度
        microbe_grad = torch.autograd.grad(loss, microbe_input, retain_graph=True)[0]
        microbe_grad_abs = torch.abs(microbe_grad).mean(dim=0)
        all_microbe_gradients.append(microbe_grad_abs)
        
        # 计算对fMRI输入的梯度
        fmri_grad = torch.autograd.grad(loss, fmri_input, retain_graph=True)[0]
        fmri_grad_abs = torch.abs(fmri_grad).mean(dim=0)
        all_fmri_gradients.append(fmri_grad_abs)
        
        # 计算特征交互重要性
        # 使用二阶导数或梯度乘积作为交互度量
        interaction_scores = torch.zeros(microbe_input.size(1), fmri_input.size(1))
        for i in range(microbe_input.size(1)):
            for j in range(fmri_input.size(1)):
                # 计算二阶导数作为交互度量
                interaction_scores[i, j] = torch.abs(
                    torch.autograd.grad(
                        microbe_grad[:, i].mean(), 
                        fmri_input, 
                        retain_graph=True
                    )[0][:, j].mean()
                )
        all_interaction_scores.append(interaction_scores)
    
    # 合并所有梯度
    all_microbe_gradients = torch.stack(all_microbe_gradients).mean(dim=0)
    all_fmri_gradients = torch.stack(all_fmri_gradients).mean(dim=0)
    all_interaction_scores = torch.stack(all_interaction_scores).mean(dim=0)
    
    # 获取top-k重要特征
    top_k_microbe_values, top_k_microbe_indices = torch.topk(all_microbe_gradients, top_k)
    top_k_fmri_values, top_k_fmri_indices = torch.topk(all_fmri_gradients, top_k)
    
    # 获取top-k重要交互
    top_k_interaction_values, top_k_interaction_indices = torch.topk(all_interaction_scores.flatten(), top_k)
    top_k_interaction_microbe = top_k_interaction_indices // fmri_input.size(1)
    top_k_interaction_fmri = top_k_interaction_indices % fmri_input.size(1)
    
    return {
        'microbe_indices': top_k_microbe_indices.cpu().numpy(),
        'microbe_values': top_k_microbe_values.cpu().numpy(),
        'fmri_indices': top_k_fmri_indices.cpu().numpy(),
        'fmri_values': top_k_fmri_values.cpu().numpy(),
        'interaction_microbe': top_k_interaction_microbe.cpu().numpy(),
        'interaction_fmri': top_k_interaction_fmri.cpu().numpy(),
        'interaction_values': top_k_interaction_values.cpu().numpy()
    }
    
def extract_multimodal_biomarkers_perturbation(model, data_loader, device, top_k=20, n_samples=100):
    """
    使用特征扰动方法提取多模态生物标志物
    
    参数:
    - model: 训练好的模型
    - data_loader: 数据加载器
    - device: 计算设备
    - top_k: 返回前k个重要特征
    - n_samples: 用于扰动的样本数量
    
    返回:
    - microbe_indices: 重要微生物组特征索引
    - fmri_indices: 重要fMRI特征索引
    - interaction_scores: 特征交互重要性分数
    """
    model.eval()
    
    # 初始化结果列表
    microbe_importance = torch.zeros(2503)  # 微生物组特征数量
    fmri_importance = torch.zeros(40000)    # fMRI特征数量
    interaction_importance = torch.zeros(2503, 40000)
    
    # 收集样本
    all_microbe = []
    all_fmri = []
    all_labels = []
    
    for batch in data_loader:
        all_microbe.append(batch['microbe'])
        all_fmri.append(batch['fmri'])
        all_labels.append(batch['label'])
    
    all_microbe = torch.cat(all_microbe, dim=0)
    all_fmri = torch.cat(all_fmri, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 随机选择样本
    indices = torch.randperm(len(all_microbe))[:n_samples]
    microbe_samples = all_microbe[indices].to(device)
    fmri_samples = all_fmri[indices].to(device)
    labels = all_labels[indices].to(device)
    
    # 基准预测
    with torch.no_grad():
        h_microbe, h_fmri, logits = model(microbe_samples, fmri_samples)
        baseline_pred = torch.argmax(logits, dim=1)
        baseline_conf = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
    
    # 扰动微生物组特征
    for i in range(2503):
        perturbed_microbe = microbe_samples.clone()
        perturbed_microbe[:, i] = torch.randn_like(perturbed_microbe[:, i])
        
        with torch.no_grad():
            h_microbe, h_fmri, logits = model(perturbed_microbe, fmri_samples)
            perturbed_pred = torch.argmax(logits, dim=1)
            perturbed_conf = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
        
        # 计算预测变化
        pred_change = (perturbed_pred != baseline_pred).float().mean()
        conf_change = (baseline_conf - perturbed_conf).mean()
        
        microbe_importance[i] = pred_change + conf_change
    
    # 扰动fMRI特征
    for i in range(40000):
        perturbed_fmri = fmri_samples.clone()
        perturbed_fmri[:, i] = torch.randn_like(perturbed_fmri[:, i])
        
        with torch.no_grad():
            h_microbe, h_fmri, logits = model(microbe_samples, perturbed_fmri)
            perturbed_pred = torch.argmax(logits, dim=1)
            perturbed_conf = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
        
        # 计算预测变化
        pred_change = (perturbed_pred != baseline_pred).float().mean()
        conf_change = (baseline_conf - perturbed_conf).mean()
        
        fmri_importance[i] = pred_change + conf_change
    
    # 扰动特征交互
    for i in range(2503):
        for j in range(40000):
            perturbed_microbe = microbe_samples.clone()
            perturbed_fmri = fmri_samples.clone()
            perturbed_microbe[:, i] = torch.randn_like(perturbed_microbe[:, i])
            perturbed_fmri[:, j] = torch.randn_like(perturbed_fmri[:, j])
            
            with torch.no_grad():
                h_microbe, h_fmri, logits = model(perturbed_microbe, perturbed_fmri)
                perturbed_pred = torch.argmax(logits, dim=1)
                perturbed_conf = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
            
            # 计算预测变化
            pred_change = (perturbed_pred != baseline_pred).float().mean()
            conf_change = (baseline_conf - perturbed_conf).mean()
            
            interaction_importance[i, j] = pred_change + conf_change
    
    # 获取top-k重要特征
    top_k_microbe_values, top_k_microbe_indices = torch.topk(microbe_importance, top_k)
    top_k_fmri_values, top_k_fmri_indices = torch.topk(fmri_importance, top_k)
    
    # 获取top-k重要交互
    top_k_interaction_values, top_k_interaction_indices = torch.topk(interaction_importance.flatten(), top_k)
    top_k_interaction_microbe = top_k_interaction_indices // 40000
    top_k_interaction_fmri = top_k_interaction_indices % 40000
    
    return {
        'microbe_indices': top_k_microbe_indices.cpu().numpy(),
        'microbe_values': top_k_microbe_values.cpu().numpy(),
        'fmri_indices': top_k_fmri_indices.cpu().numpy(),
        'fmri_values': top_k_fmri_values.cpu().numpy(),
        'interaction_microbe': top_k_interaction_microbe.cpu().numpy(),
        'interaction_fmri': top_k_interaction_fmri.cpu().numpy(),
        'interaction_values': top_k_interaction_values.cpu().numpy()
    }
    
def extract_integrated_biomarkers(model, data_loader, device, top_k=20, alpha=0.5):
    """
    集成两个模态的特征重要性，识别在两个模态中都重要的特征
    
    参数:
    - model: 训练好的模型
    - data_loader: 数据加载器
    - device: 计算设备
    - top_k: 返回前k个重要特征
    - alpha: 集成权重，控制两个模态的相对重要性
    
    返回:
    - microbe_indices: 重要微生物组特征索引
    - fmri_indices: 重要fMRI特征索引
    - integrated_indices: 集成后的重要特征索引
    """
    # 提取单个模态的生物标志物
    microbe_results = extract_biomarkers(model, data_loader, device, feature_type='microbe', top_k=100)
    fmri_results = extract_biomarkers(model, data_loader, device, feature_type='fmri', top_k=100)
    
    microbe_indices, microbe_values = microbe_results
    fmri_indices, fmri_values = fmri_results
    
    # 归一化重要性值
    microbe_values_norm = microbe_values / np.max(microbe_values)
    fmri_values_norm = fmri_values / np.max(fmri_values)
    
    # 创建特征重要性字典
    microbe_importance_dict = {idx: val for idx, val in zip(microbe_indices, microbe_values_norm)}
    fmri_importance_dict = {idx: val for idx, val in zip(fmri_indices, fmri_values_norm)}
    
    # 计算集成重要性
    integrated_importance = {}
    
    # 对于微生物组特征
    for idx in microbe_indices:
        integrated_importance[f"microbe_{idx}"] = alpha * microbe_importance_dict[idx]
    
    # 对于fMRI特征
    for idx in fmri_indices:
        integrated_importance[f"fmri_{idx}"] = (1 - alpha) * fmri_importance_dict[idx]
    
    # 按集成重要性排序
    sorted_features = sorted(integrated_importance.items(), key=lambda x: x[1], reverse=True)
    
    # 提取top-k特征
    top_k_features = sorted_features[:top_k]
    
    # 分离微生物组和fMRI特征
    microbe_top_k = []
    fmri_top_k = []
    
    for feature, importance in top_k_features:
        if feature.startswith("microbe_"):
            idx = int(feature.split("_")[1])
            microbe_top_k.append((idx, importance))
        else:
            idx = int(feature.split("_")[1])
            fmri_top_k.append((idx, importance))
    
    return {
        'microbe_top_k': microbe_top_k,
        'fmri_top_k': fmri_top_k,
        'integrated_top_k': top_k_features
    }


def analyze_multimodal_feature_network(model, data_loader, device, top_k=50):
    """
    构建和分析多模态特征交互网络
    
    参数:
    - model: 训练好的模型
    - data_loader: 数据加载器
    - device: 计算设备
    - top_k: 选择前k个重要特征构建网络
    
    返回:
    - G: 特征交互网络
    - communities: 特征社区
    - hub_features: 枢纽特征
    """
    import networkx as nx
    from community import community_louvain
    
    # 提取多模态生物标志物
    results = extract_multimodal_biomarkers(model, data_loader, device, top_k=top_k)
    
    microbe_indices = results['microbe_indices']
    fmri_indices = results['fmri_indices']
    interaction_microbe = results['interaction_microbe']
    interaction_fmri = results['interaction_fmri']
    interaction_values = results['interaction_values']
    
    # 创建特征交互网络
    G = nx.Graph()
    
    # 添加微生物组节点
    for i, idx in enumerate(microbe_indices):
        G.add_node(f"microbe_{idx}", type="microbe", importance=results['microbe_values'][i])
    
    # 添加fMRI节点
    for i, idx in enumerate(fmri_indices):
        G.add_node(f"fmri_{idx}", type="fmri", importance=results['fmri_values'][i])
    
    # 添加交互边
    for i in range(len(interaction_microbe)):
        microbe_idx = interaction_microbe[i]
        fmri_idx = interaction_fmri[i]
        weight = interaction_values[i]
        
        G.add_edge(f"microbe_{microbe_idx}", f"fmri_{fmri_idx}", weight=weight)
    
    # 检测社区
    communities = community_louvain.best_partition(G)
    
    # 识别枢纽特征
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # 综合中心性指标
    hub_scores = {}
    for node in G.nodes():
        hub_scores[node] = (
            0.4 * degree_centrality[node] + 
            0.3 * betweenness_centrality[node] + 
            0.3 * eigenvector_centrality[node]
        )
    
    # 获取top-k枢纽特征
    hub_features = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return G, communities, hub_features


def visualize_multimodal_network(G, communities, hub_features, output_file='multimodal_network.png'):
    """
    可视化多模态特征交互网络
    
    参数:
    - G: 特征交互网络
    - communities: 特征社区
    - hub_features: 枢纽特征
    - output_file: 输出文件名
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # 设置节点颜色
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]['type'] == 'microbe':
            node_colors.append('lightblue')
        else:
            node_colors.append('lightgreen')
    
    # 设置节点大小
    node_sizes = []
    for node in G.nodes():
        importance = G.nodes[node]['importance']
        node_sizes.append(300 + 1000 * importance)
    
    # 设置边权重
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # 绘制网络
    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # 绘制枢纽节点
    hub_nodes = [node for node, _ in hub_features[:10]]
    nx.draw_networkx_nodes(G, pos, nodelist=hub_nodes, node_color='red', 
                          node_size=[node_sizes[list(G.nodes()).index(node)] for node in hub_nodes], 
                          alpha=0.8)
    
    # 绘制标签
    labels = {node: node.split('_')[1] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title('Multimodal Feature Interaction Network', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"多模态特征交互网络已保存为 {output_file}")
    
def analyze_microbe_roi_connections(model, data_loader, device, roi_names, top_k=20):
    """
    分析微生物组特征与fMRI ROI连接的关系
    
    参数:
    - model: 训练好的模型
    - data_loader: 数据加载器
    - device: 计算设备
    - roi_names: ROI名称映射
    - top_k: 返回前k个重要连接
    
    返回:
    - microbe_roi_connections: 微生物组特征与ROI连接的关联
    """
    # 提取多模态生物标志物
    results = extract_multimodal_biomarkers(model, data_loader, device, top_k=100)
    
    microbe_indices = results['microbe_indices']
    fmri_indices = results['fmri_indices']
    interaction_microbe = results['interaction_microbe']
    interaction_fmri = results['interaction_fmri']
    interaction_values = results['interaction_values']
    
    # 将fMRI索引转换为ROI连接
    def index_to_roi_connection(idx, matrix_size=200):
        row, col = index_to_matrix_coords(idx, matrix_size)
        roi1_name = roi_names.get(row, f"ROI_{row}")
        roi2_name = roi_names.get(col, f"ROI_{col}")
        return (roi1_name, roi2_name)
    
    # 创建微生物组特征与ROI连接的关联
    microbe_roi_connections = []
    
    for i in range(len(interaction_microbe)):
        microbe_idx = interaction_microbe[i]
        fmri_idx = interaction_fmri[i]
        importance = interaction_values[i]
        
        roi_connection = index_to_roi_connection(fmri_idx)
        microbe_roi_connections.append((microbe_idx, roi_connection, importance))
    
    # 按重要性排序
    microbe_roi_connections = sorted(microbe_roi_connections, key=lambda x: x[2], reverse=True)
    
    # 提取top-k连接
    top_k_connections = microbe_roi_connections[:top_k]
    
    return top_k_connections


def extract_biomarkers_gradient(model, data_loader, device, feature_type='microbe', top_k=20):
    """
    使用梯度方法提取生物标志物
    
    参数:
    - model: 预训练模型
    - data_loader: 数据加载器
    - device: 计算设备
    - feature_type: 特征类型，'microbe'或'fmri'
    - top_k: 返回前k个重要特征
    
    返回:
    - indices: 重要特征索引
    - values: 重要特征值
    """
    try:
        model.eval()
        feature_importance = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 处理 PyTorch Geometric 的 DataBatch 类型
                if hasattr(batch, 'to'):
                    if feature_type == 'microbe':
                        microbe_input = batch.x[:, :2503].to(device)
                        fmri_input = batch.x[:, 2503:].to(device)
                        labels = batch.y.to(device).squeeze()  # 确保标签是一维的
                        input_tensor = microbe_input
                    else:  # fmri
                        microbe_input = batch.x[:, :2503].to(device)
                        fmri_input = batch.x[:, 2503:].to(device)
                        labels = batch.y.to(device).squeeze()  # 确保标签是一维的
                        input_tensor = fmri_input
                # 处理字典类型
                elif isinstance(batch, dict):
                    if feature_type == 'microbe':
                        microbe_input = batch['microbe'].to(device)
                        fmri_input = batch['fmri'].to(device)
                        labels = batch['label'].to(device).squeeze()  # 确保标签是一维的
                        input_tensor = microbe_input
                    else:  # fmri
                        microbe_input = batch['microbe'].to(device)
                        fmri_input = batch['fmri'].to(device)
                        labels = batch['label'].to(device).squeeze()  # 确保标签是一维的
                        input_tensor = fmri_input
                # 处理列表或元组类型
                elif isinstance(batch, (list, tuple)):
                    if feature_type == 'microbe':
                        input_tensor = batch[0].to(device)
                        other_tensor = torch.zeros((input_tensor.size(0), 40000), device=device)
                        labels = batch[1].to(device).squeeze() if len(batch) > 1 else None
                    else:  # fmri
                        input_tensor = batch[0].to(device)
                        other_tensor = torch.zeros((input_tensor.size(0), 2503), device=device)
                        labels = batch[1].to(device).squeeze() if len(batch) > 1 else None
                else:
                    print(f"未知的批次类型: {type(batch)}")
                    continue
                
                # 确保输入张量需要梯度
                input_tensor.requires_grad_(True)
                
                # 前向传播
                if feature_type == 'microbe':
                    h_microbe, h_fmri, logits = model(input_tensor, fmri_input)
                else:
                    h_microbe, h_fmri, logits = model(microbe_input, input_tensor)
                
                # 计算损失
                if labels is not None:
                    loss = F.cross_entropy(logits, labels)
                else:
                    loss = logits.mean()
                
                # 计算梯度
                loss.backward()
                
                # 计算特征重要性
                importance = input_tensor.grad.abs().mean(dim=0)
                feature_importance.append(importance.cpu().numpy())
        
        # 合并所有批次的结果
        if feature_importance:
            feature_importance = np.mean(feature_importance, axis=0)
            
            # 获取前 k 个重要特征
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            top_values = feature_importance[top_indices]
            
            return top_indices, top_values
        else:
            print("没有成功处理任何批次数据")
            return np.array([]), np.array([])
            
    except Exception as e:
        print(f"在梯度方法提取生物标志物时出错: {e}")
        # 返回空结果
        return np.array([]), np.array([])

def extract_biomarkers_integrated_gradients(model, data_loader, device, feature_type='microbe', top_k=20, steps=50):
    """
    使用积分梯度方法提取生物标志物
    
    参数:
    - model: 预训练模型
    - data_loader: 数据加载器
    - device: 计算设备
    - feature_type: 特征类型，'microbe'或'fmri'
    - top_k: 返回前k个重要特征
    - steps: 积分步数
    
    返回:
    - indices: 重要特征索引
    - values: 重要特征值
    """
    try:
        model.eval()
        feature_importance = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 处理 PyTorch Geometric 的 DataBatch 类型
                if hasattr(batch, 'to'):
                    if feature_type == 'microbe':
                        microbe_input = batch.x[:, :2503].to(device)
                        fmri_input = batch.x[:, 2503:].to(device)
                        labels = batch.y.to(device).squeeze()  # 确保标签是一维的
                        input_tensor = microbe_input
                    else:  # fmri
                        microbe_input = batch.x[:, :2503].to(device)
                        fmri_input = batch.x[:, 2503:].to(device)
                        labels = batch.y.to(device).squeeze()  # 确保标签是一维的
                        input_tensor = fmri_input
                # 处理字典类型
                elif isinstance(batch, dict):
                    if feature_type == 'microbe':
                        microbe_input = batch['microbe'].to(device)
                        fmri_input = batch['fmri'].to(device)
                        labels = batch['label'].to(device).squeeze()  # 确保标签是一维的
                        input_tensor = microbe_input
                    else:  # fmri
                        microbe_input = batch['microbe'].to(device)
                        fmri_input = batch['fmri'].to(device)
                        labels = batch['label'].to(device).squeeze()  # 确保标签是一维的
                        input_tensor = fmri_input
                # 处理列表或元组类型
                elif isinstance(batch, (list, tuple)):
                    if feature_type == 'microbe':
                        input_tensor = batch[0].to(device)
                        other_tensor = torch.zeros((input_tensor.size(0), 40000), device=device)
                        labels = batch[1].to(device).squeeze() if len(batch) > 1 else None
                    else:  # fmri
                        input_tensor = batch[0].to(device)
                        other_tensor = torch.zeros((input_tensor.size(0), 2503), device=device)
                        labels = batch[1].to(device).squeeze() if len(batch) > 1 else None
                else:
                    print(f"未知的批次类型: {type(batch)}")
                    continue
                
                # 创建基线输入
                baseline = torch.zeros_like(input_tensor)
                
                # 计算积分梯度
                alphas = torch.linspace(0, 1, steps, device=device)
                gradients = []
                
                for alpha in alphas:
                    interpolated = baseline + alpha * (input_tensor - baseline)
                    interpolated.requires_grad_(True)
                    
                    # 前向传播
                    if feature_type == 'microbe':
                        h_microbe, h_fmri, logits = model(interpolated, fmri_input)
                    else:
                        h_microbe, h_fmri, logits = model(microbe_input, interpolated)
                    
                    # 计算损失
                    if labels is not None:
                        loss = F.cross_entropy(logits, labels)
                    else:
                        loss = logits.mean()
                    
                    loss.backward()
                    gradients.append(interpolated.grad.abs())
                
                # 计算平均梯度
                avg_gradients = torch.stack(gradients).mean(dim=0)
                
                # 计算积分梯度
                integrated_gradients = (input_tensor - baseline) * avg_gradients
                
                # 计算特征重要性
                importance = integrated_gradients.abs().mean(dim=0)
                feature_importance.append(importance.cpu().numpy())
        
        # 合并所有批次的结果
        if feature_importance:
            feature_importance = np.mean(feature_importance, axis=0)
            
            # 获取前 k 个重要特征
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            top_values = feature_importance[top_indices]
            
            return top_indices, top_values
        else:
            print("没有成功处理任何批次数据")
            return np.array([]), np.array([])
            
    except Exception as e:
        print(f"在积分梯度方法提取生物标志物时出错: {e}")
        # 返回空结果
        return np.array([]), np.array([])

def extract_biomarkers_attention(model, data_loader, device, feature_type='microbe', top_k=20):
    """
    基于注意力机制的生物标志物提取方法
    
    参数:
    - model: 预训练模型
    - data_loader: 数据加载器
    - device: 计算设备
    - feature_type: 特征类型，'microbe'或'fmri'
    - top_k: 返回前k个重要特征
    
    返回:
    - indices: 重要特征索引
    - values: 重要特征值
    """
    model.eval()  # 设置为评估模式
    
    # 初始化结果列表
    all_attention_weights = []
    
    # 遍历数据
    for batch in data_loader:
        # 获取输入数据
        if feature_type == 'microbe':
            input_tensor = batch['microbe'].to(device)
            other_tensor = batch['fmri'].to(device)
        else:  # fmri
            input_tensor = batch['fmri'].to(device)
            other_tensor = batch['microbe'].to(device)
        
        # 前向传播并获取注意力权重
        # 注意：这里假设模型有get_attention_weights方法，需要根据实际模型结构修改
        h_microbe, h_fmri, logits, attention_weights = model(input_tensor, other_tensor, return_attention=True)
        
        # 提取特征对应的注意力权重
        if feature_type == 'microbe':
            feature_attention = attention_weights['microbe']
        else:  # fmri
            feature_attention = attention_weights['fmri']
        
        # 计算注意力权重平均值
        attention_mean = feature_attention.mean(dim=0)
        all_attention_weights.append(attention_mean)
    
    # 合并所有注意力权重
    all_attention_weights = torch.stack(all_attention_weights).mean(dim=0)
    
    # 获取top-k重要特征
    top_k_values, top_k_indices = torch.topk(all_attention_weights, top_k)
    
    return top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()

def extract_biomarkers_backprop(model, data_loader, device, feature_type='microbe', top_k=20):
    """
    基于反向传播的生物标志物提取方法
    
    参数:
    - model: 预训练模型
    - data_loader: 数据加载器
    - device: 计算设备
    - feature_type: 特征类型，'microbe'或'fmri'
    - top_k: 返回前k个重要特征
    
    返回:
    - indices: 重要特征索引
    - values: 重要特征值
    """
    model.eval()  # 设置为评估模式
    
    # 初始化结果列表
    all_saliency = []
    
    # 遍历数据
    for batch in data_loader:
        # 检查批次类型并相应处理
        if isinstance(batch, dict):
            # 如果批次是字典，直接访问
            if feature_type == 'microbe':
                input_tensor = batch['microbe'].to(device)
                other_tensor = batch['fmri'].to(device)
                labels = batch['label'].squeeze().to(device)
            else:  # fmri
                input_tensor = batch['fmri'].to(device)
                other_tensor = batch['microbe'].to(device)
                labels = batch['label'].squeeze().to(device)
        elif isinstance(batch, (list, tuple)):
            # 如果批次是列表或元组，假设第一个元素是特征，第二个元素是标签
            input_tensor = batch[0].to(device)
            # 创建一个空的输入，而不是 None
            if feature_type == 'microbe':
                other_tensor = torch.zeros((input_tensor.size(0), 40000), device=device)
            else:
                other_tensor = torch.zeros((input_tensor.size(0), 2503), device=device)
            labels = batch[1].squeeze().to(device)
        else:
            # 如果批次是其他类型，尝试直接使用
            input_tensor = batch.to(device)
            # 创建一个空的输入，而不是 None
            if feature_type == 'microbe':
                other_tensor = torch.zeros((input_tensor.size(0), 40000), device=device)
            else:
                other_tensor = torch.zeros((input_tensor.size(0), 2503), device=device)
            labels = None
        
        # 确保输入张量需要梯度
        input_tensor.requires_grad_(True)
        
        # 前向传播
        #h_microbe, h_fmri, logits = model(input_tensor, other_tensor)
        if feature_type == 'microbe':
            h_microbe, h_fmri, logits = model(input_tensor, other_tensor)
        else:  # fMRI处理
            h_microbe, h_fmri, logits = model(other_tensor, input_tensor)
        
        # 计算每个类别的得分
        scores = logits
        
        # 计算每个类别的得分对输入的偏导数
        saliency = torch.zeros_like(input_tensor)
        
        for i in range(scores.size(1)):
            # 计算第i个类别的得分对输入的偏导数
            score_i = scores[:, i]
            grad_i = torch.autograd.grad(score_i.sum(), input_tensor, retain_graph=True)[0]
            
            # 累加偏导数的绝对值
            saliency += torch.abs(grad_i)
        
        # 计算显著性图的平均值
        saliency_mean = saliency.mean(dim=0)
        all_saliency.append(saliency_mean)
    
    # 合并所有显著性图
    all_saliency = torch.stack(all_saliency).mean(dim=0)
    
    # 获取top-k重要特征
    top_k_values, top_k_indices = torch.topk(all_saliency, top_k)
    
    return top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()

def extract_biomarkers_gradcam(model, data_loader, device, feature_type='microbe', top_k=20):
    """
    基于Grad-CAM的生物标志物提取方法
    
    参数:
    - model: 预训练模型
    - data_loader: 数据加载器
    - device: 计算设备
    - feature_type: 特征类型，'microbe'或'fmri'
    - top_k: 返回前k个重要特征
    
    返回:
    - indices: 重要特征索引
    - values: 重要特征值
    """
    model.eval()  # 设置为评估模式
    
    # 初始化结果列表
    all_gradcam = []
    
    # 遍历数据
    for batch in data_loader:
        # 获取输入数据
        if feature_type == 'microbe':
            input_tensor = batch['microbe'].to(device)
            other_tensor = batch['fmri'].to(device)
        else:  # fmri
            input_tensor = batch['fmri'].to(device)
            other_tensor = batch['microbe'].to(device)
        
        # 确保输入张量需要梯度
        input_tensor.requires_grad_(True)
        
        # 前向传播
        h_microbe, h_fmri, logits = model(input_tensor, other_tensor)
        
        # 获取预测类别
        pred_class = torch.argmax(logits, dim=1)
        
        # 计算每个样本的Grad-CAM
        gradcam = torch.zeros_like(input_tensor)
        
        for i in range(len(pred_class)):
            # 计算第i个样本的Grad-CAM
            score_i = logits[i, pred_class[i]]
            grad_i = torch.autograd.grad(score_i, input_tensor, retain_graph=True)[0]
            
            # 计算权重
            weights = grad_i[i].mean(dim=0)
            
            # 计算Grad-CAM
            gradcam[i] = weights.unsqueeze(0) * input_tensor[i]
        
        # 计算Grad-CAM的平均值
        gradcam_mean = gradcam.mean(dim=0)
        all_gradcam.append(gradcam_mean)
    
    # 合并所有Grad-CAM
    all_gradcam = torch.stack(all_gradcam).mean(dim=0)
    
    # 获取top-k重要特征
    top_k_values, top_k_indices = torch.topk(all_gradcam, top_k)
    
    return top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()

def extract_biomarkers_ensemble(model, data_loader, device, feature_type='microbe', top_k=20):
    """
    综合多种方法的生物标志物提取
    
    参数:
    - model: 预训练模型
    - data_loader: 数据加载器
    - device: 计算设备
    - feature_type: 特征类型，'microbe'或'fmri'
    - top_k: 返回前k个重要特征
    
    返回:
    - indices: 重要特征索引
    - values: 重要特征值
    """
    try:
        # 使用多种方法提取生物标志物
        indices_gradient, values_gradient = extract_biomarkers_gradient(model, data_loader, device, feature_type, top_k)
        
        # 使用集成梯度方法
        indices_integrated, values_integrated = extract_biomarkers_integrated_gradients(model, data_loader, device, feature_type, top_k)
        
        # 使用反向传播方法
        indices_backprop, values_backprop = extract_biomarkers_backprop(model, data_loader, device, feature_type, top_k)
        
        # 合并结果
        all_indices = np.concatenate([indices_gradient, indices_integrated, indices_backprop])
        all_values = np.concatenate([values_gradient, values_integrated, values_backprop])
        
        # 计算每个特征的出现次数
        unique_indices, counts = np.unique(all_indices, return_counts=True)
        
        # 计算每个特征的平均重要性
        #unique_indices, inverse_indices = np.unique(all_indices, return_inverse=True)
        #counts = np.bincount(inverse_indices)
        #sums = np.bincount(inverse_indices, weights=all_values)
        #avg_values = sums / counts
        avg_values = all_values[np.argsort(all_indices)]
        
        # 获取前 k 个重要特征
        top_indices = np.argsort(avg_values)[-top_k:][::-1]
        top_values = avg_values[top_indices]
        
        return top_indices, top_values
    except Exception as e:
        print(f"在提取生物标志物时出错: {e}")
        # 如果出错，尝试使用单一方法
        try:
            print("尝试使用单一方法提取生物标志物...")
            indices, values = extract_biomarkers_gradient(model, data_loader, device, feature_type, top_k)
            return indices, values
        except Exception as e2:
            print(f"单一方法也失败: {e2}")
            # 返回空结果
            return np.array([]), np.array([])

# 获取ROI名称
def get_roi_names(file_path, graph_type="cc200"):
    roi_names = {}
    try:
        with h5py.File(file_path, "r") as f:
            if "roi_names" in f:
                roi_names = {int(k): v for k, v in f["roi_names"].items()}
            elif "atlas" in f and graph_type in f["atlas"]:
                roi_names = {int(k): v for k, v in f["atlas"][graph_type]["roi_names"].items()}
    except Exception as e:
        print(f"无法从数据文件中获取ROI名称: {e}")
    
    # 如果无法获取ROI名称，使用默认的ROI ID
    if not roi_names:
        roi_names = {i: f"ROI_{i}" for i in range(200)}
    
    return roi_names

def analyze_feature_interactions_efficient(model, data_loader, device, top_k=20, pre_filter=100):
    """
    高效的多模态特征交互分析方法
    
    参数:
    - model: 预训练模型
    - data_loader: 数据加载器
    - device: 计算设备
    - top_k: 返回前k个重要交互
    - pre_filter: 预筛选的特征数量
    
    返回:
    - microbe_indices: 重要微生物组特征索引
    - fmri_indices: 重要fMRI特征索引
    - interaction_values: 交互重要性值
    """
    # 预筛选重要特征
    microbe_indices, _ = extract_biomarkers_gradient(model, data_loader, device, feature_type='microbe', top_k=pre_filter)
    fmri_indices, _ = extract_biomarkers_gradient(model, data_loader, device, feature_type='fmri', top_k=pre_filter)
    
    # 初始化交互分数矩阵
    interaction_scores = torch.zeros(len(microbe_indices), len(fmri_indices))
    
    # 收集样本
    all_microbe = []
    all_fmri = []
    all_labels = []
    
    for batch in data_loader:
        all_microbe.append(batch['microbe'])
        all_fmri.append(batch['fmri'])
        all_labels.append(batch['label'])
    
    all_microbe = torch.cat(all_microbe, dim=0)
    all_fmri = torch.cat(all_fmri, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 随机选择样本
    indices = torch.randperm(len(all_microbe))[:100]
    microbe_samples = all_microbe[indices].to(device)
    fmri_samples = all_fmri[indices].to(device)
    labels = all_labels[indices].to(device)
    
    # 基准预测
    with torch.no_grad():
        h_microbe, h_fmri, logits = model(microbe_samples, fmri_samples)
        baseline_pred = torch.argmax(logits, dim=1)
        baseline_conf = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
    
    # 分析特征交互
    for i, microbe_idx in enumerate(microbe_indices):
        for j, fmri_idx in enumerate(fmri_indices):
            perturbed_microbe = microbe_samples.clone()
            perturbed_fmri = fmri_samples.clone()
            perturbed_microbe[:, microbe_idx] = torch.randn_like(perturbed_microbe[:, microbe_idx])
            perturbed_fmri[:, fmri_idx] = torch.randn_like(perturbed_fmri[:, fmri_idx])
            
            with torch.no_grad():
                h_microbe, h_fmri, logits = model(perturbed_microbe, perturbed_fmri)
                perturbed_pred = torch.argmax(logits, dim=1)
                perturbed_conf = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
            
            # 计算预测变化
            pred_change = (perturbed_pred != baseline_pred).float().mean()
            conf_change = (baseline_conf - perturbed_conf).mean()
            
            interaction_scores[i, j] = pred_change + conf_change
    
    # 获取top-k重要交互
    top_k_values, top_k_indices = torch.topk(interaction_scores.flatten(), min(top_k, len(microbe_indices) * len(fmri_indices)))
    top_k_microbe_idx = top_k_indices // len(fmri_indices)
    top_k_fmri_idx = top_k_indices % len(fmri_indices)
    
    # 转换回原始索引
    top_k_microbe = microbe_indices[top_k_microbe_idx]
    top_k_fmri = fmri_indices[top_k_fmri_idx]
    
    return top_k_microbe.cpu().numpy(), top_k_fmri.cpu().numpy(), top_k_values.cpu().numpy()

def convert_biomarkers_to_roi_connections(biomarker_indices, roi_names, matrix_size=200):
    """
    将生物标志物索引转换为ROI连接
    
    参数:
    - biomarker_indices: 生物标志物索引
    - roi_names: ROI名称映射
    - matrix_size: 矩阵大小
    
    返回:
    - roi_connections: ROI连接列表
    """
    roi_connections = []
    
    for idx in biomarker_indices:
        row, col = index_to_matrix_coords(idx, matrix_size)
        
        # 获取ROI名称
        roi1_name = roi_names.get(row, f"ROI_{row}")
        roi2_name = roi_names.get(col, f"ROI_{col}")
        
        roi_connections.append((roi1_name, roi2_name))
    
    return roi_connections

def visualize_roi_network(roi_connections, fmri_values, top_k=20, output_file='roi_network.png'):
    """
    可视化ROI连接网络
    
    参数:
    - roi_connections: 微生物组特征与ROI连接的关联
    - fmri_values: fMRI特征重要性值
    - top_k: 返回前k个重要连接
    - output_file: 输出文件名
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # 创建特征交互网络
    G = nx.Graph()
    
    # 添加节点
    for i, (roi1, roi2) in enumerate(roi_connections[:top_k]):
        G.add_edge(roi1, roi2)
    
    # 绘制网络
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=3000 * np.array(list(dict(G.degree()).values())), node_color='lightblue')
    plt.title('ROI Connection Network', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROI连接网络已保存为 {output_file}")

def visualize_microbe_roi_network(microbe_indices, roi_connections, interaction_values, top_k=20, output_file='microbe_roi_network.png'):
    """
    可视化微生物-脑区连接网络
    
    参数:
    - microbe_indices: 重要微生物组特征索引
    - roi_connections: 微生物组特征与ROI连接的关联
    - interaction_values: 交互重要性值
    - top_k: 返回前k个重要连接
    - output_file: 输出文件名
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # 创建特征交互网络
    G = nx.Graph()
    
    # 添加节点
    for i, microbe_idx in enumerate(microbe_indices[:top_k]):
        G.add_node(f"microbe_{microbe_idx}", type="microbe", importance=interaction_values[i])
    
    # 添加边
    for i, (roi1, roi2) in enumerate(roi_connections[:top_k]):
        G.add_edge(f"microbe_{microbe_indices[i]}", roi1)
        G.add_edge(f"microbe_{microbe_indices[i]}", roi2)
    
    # 绘制网络
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=3000 * np.array(list(dict(G.degree()).values())), node_color='lightblue')
    plt.title('Microbe-ROI Connection Network', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"微生物-脑区连接网络已保存为 {output_file}")

def get_roi_names(file_path, graph_type="cc200"):
    roi_names = {}
    try:
        with h5py.File(file_path, "r") as f:
            if "roi_names" in f:
                roi_names = {int(k): v for k, v in f["roi_names"].items()}
            elif "atlas" in f and graph_type in f["atlas"]:
                roi_names = {int(k): v for k, v in f["atlas"][graph_type]["roi_names"].items()}
    except Exception as e:
        print(f"无法从数据文件中获取ROI名称: {e}")
    
    # 如果无法获取ROI名称，使用默认的ROI ID
    if not roi_names:
        roi_names = {i: f"ROI_{i}" for i in range(200)}
    
    return roi_names
 
# 可视化ROI连接
def visualize_roi_network(roi_connections, importance_values, top_k=20, output_file='roi_network.png'):
    """
    可视化ROI连接网络
    
    参数:
    - roi_connections: ROI连接列表
    - importance_values: 重要性值
    - top_k: 只显示前k个最重要的连接
    - output_file: 输出文件名
    """
    
    
    # 创建图
    G = nx.Graph()
    
    # 添加边
    for i, ((roi1, roi2), importance) in enumerate(zip(roi_connections, importance_values)):
        if i >= top_k:
            break
        G.add_edge(roi1, roi2, weight=importance)
    
    # 绘制图
    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=700, alpha=0.8)
    
    # 绘制边
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, 
                          alpha=0.5, edge_color='blue')
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title('Top ROI Connections in ASD Biomarkers', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROI网络图已保存为 {output_file}") 

def create_biomarker_loader(microbe_features, fmri_features, labels, batch_size=64):
    """创建保证维度正确的数据加载器"""
    assert microbe_features.shape[0] == fmri_features.shape[0], "样本数量不匹配"
    assert microbe_features.shape[1] == 2503, f"微生物组特征维度错误：{microbe_features.shape}"
    assert fmri_features.shape[1] == 40000, f"fMRI特征维度错误：{fmri_features.shape}"
    
    dataset = PairedDataset(
        (microbe_features, labels),
        (fmri_features, labels),
        train=False  # 关闭数据增强
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: {
            'microbe': torch.stack([x['microbe'] for x in batch]),
            'fmri': torch.stack([x['fmri'] for x in batch]),
            'label': torch.stack([x['label'] for x in batch])
        }
    )
 
def main_multimodal_biomarker_analysis(model, data_loader, device, output_dir="biomarkers_results", top_k=50):
    """
    多模态生物标志物分析主函数
    
    参数:
    - model: 预训练模型
    - data_loader: 数据加载器
    - device: 计算设备
    - output_dir: 输出目录
    - top_k: 返回前k个重要特征
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    sample_batch = next(iter(data_loader))
    print(f"微生物组数据维度: {sample_batch['microbe'].shape}")  # 应为(batch,2503)
    print(f"fMRI数据维度: {sample_batch['fmri'].shape}")       # 应为(batch,40000)
    # 提取微生物组生物标志物
    print("提取微生物组生物标志物...")
    microbe_indices, microbe_values = extract_biomarkers_ensemble(model, data_loader, device, feature_type='microbe', top_k=top_k)
    
    # 提取 fMRI 生物标志物
    print("提取 fMRI 生物标志物...")
    fmri_indices, fmri_values = extract_biomarkers_ensemble(model, data_loader, device, feature_type='fmri', top_k=top_k)
    
    # 保存结果
    np.savetxt(os.path.join(output_dir, "microbe_biomarkers.csv"), np.column_stack((microbe_indices, microbe_values)), delimiter=",", header="index,importance", comments="")
    np.savetxt(os.path.join(output_dir, "fmri_biomarkers.csv"), np.column_stack((fmri_indices, fmri_values)), delimiter=",", header="index,importance", comments="")
    
    # 转换 fMRI 生物标志物为 ROI 连接
    '''roi_connections = convert_biomarkers_to_roi_connections(fmri_indices)
    
    # 保存 ROI 连接
    with open(os.path.join(output_dir, "roi_connections.csv"), "w") as f:
        f.write("roi1,roi2,importance\n")
        for i, (roi1, roi2) in enumerate(roi_connections):
            f.write(f"{roi1},{roi2},{fmri_values[i]}\n")
    
    # 可视化 ROI 连接
    visualize_roi_network(roi_connections, fmri_values, output_file=os.path.join(output_dir, "roi_network.png"))
    
    print(f"生物标志物分析完成，结果保存在 {output_dir} 目录中")'''