import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import logging
import h5py
import math
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, confusion_matrix
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch_geometric.nn as tg_nn
from torch_geometric.data import Data, Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add pre_train to path to import MicrobeDataLoader
#current_script_path = Path(__file__).resolve()
#parent_dir = current_script_path.parent.parent
#target_subdir = parent_dir / "pre_train"
#sys.path.append(str(target_subdir))
from MLP import MicrobeDataLoader, SparseMLP

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model and Data Handling Classes (Copied from original script) ---

def reconstruct_fc(vector):
    """将上三角向量重建为对称矩阵"""
    matrix = np.zeros((200, 200))
    triu_indices = np.triu_indices(200, k=1)
    matrix[triu_indices] = vector
    matrix = matrix + matrix.T - np.diag(matrix.diagonal())
    return matrix

class fMRIDataLoader:
    def __init__(self, file_path, graph_type):
        self.file_path = file_path
        self.graph_type = graph_type

    def load_all_data(self):
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
                    label = torch.tensor(subject_group.attrs["y"], dtype=torch.long)
                    flat_vector = matrix.flatten()
                    graph_data = Data(
                        x=torch.FloatTensor(flat_vector),
                        edge_index=edge_index,
                        y=label
                    )
                    graph_dataset.append(graph_data)
                    labels.append(subject_group.attrs["y"])
        
        features = np.array([data.x.numpy() for data in graph_dataset])
        labels = np.array(labels)
        return features, labels

    def _get_brain_connectivity_edges(self, matrix, threshold=0.3):
        rows, cols = np.triu_indices_from(matrix, k=1)
        mask = matrix[rows, cols] > threshold
        edge_index = np.array([rows[mask], cols[mask]])
        edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
        return torch.tensor(edge_index, dtype=torch.long)

class fMRI3DGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.graph_builder = nn.Sequential(
            nn.Linear(40000, 200*200),
            nn.Sigmoid()
        )
        self.convs = nn.ModuleList([
            tg_nn.GATv2Conv(
                in_channels=16,
                out_channels=128,
                heads=8,
                dropout=config['dropout'],
                add_self_loops=False
            ),
            tg_nn.GATv2Conv(
                in_channels=128*8,
                out_channels=256,
                heads=4,
                dropout=config['dropout']
            ),
            tg_nn.GATv2Conv(
                in_channels=256*4,
                out_channels=512,
                heads=1,
                dropout=config['dropout']
            )
        ])
        self.feature_enhancer = nn.Sequential(
            nn.Linear(2, 8),
            nn.GELU(),
            nn.Linear(8, 16),
            nn.LayerNorm(16)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, config['num_classes'])
        )
    def build_graph(self, fc_matrix):
        batch_size = fc_matrix.size(0)
        adj = self.graph_builder(fc_matrix).view(batch_size, 200, 200).float()
        adj = (adj + adj.transpose(1,2)) / 2
        node_features = []
        edge_indices = []
        for b in range(batch_size):
            means = adj[b].mean(dim=1, keepdim=True)
            stds = adj[b].std(dim=1, keepdim=True)
            base_feat = torch.cat([means, stds], dim=1)
            enhanced_feat = self.feature_enhancer(base_feat)
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

    def forward(self, raw_fc):
        assert raw_fc.dim() == 2 and raw_fc.size(1) == 40000
        batch_graph = self.build_graph(raw_fc)
        x = batch_graph.x.to(raw_fc.device)
        edge_index = batch_graph.edge_index.to(raw_fc.device)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        x = tg_nn.global_mean_pool(x, batch_graph.batch)
        return x

def load_pretrained_mlp():
    model = SparseMLP(
        input_dim=2503,
        num_classes=2,
        hidden_dims=[2048, 1024, 512],
        dropout=0.6,
        sparsity_lambda=0.05
    )
    model.load_state_dict(torch.load("/home/yangzongxian/xlz/ASD_GCN/main/down/sparse_mlp.pth"))
    return model

def load_pretrained_gnn():
    GNN_CONFIG = {"gnn_layers": 3, "hidden_channels": 128, "num_classes": 2, "dropout": 0.4}
    model = fMRI3DGNN(GNN_CONFIG)
    checkpoint = torch.load("/home/yangzongxian/xlz/ASD_GCN/main/new/fmri_gnn.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class ContrastiveModel(nn.Module):
    def __init__(self, mlp_model, gnn_model, feat_dim=128):
        super().__init__()
        self.mlp = mlp_model
        self.gnn = gnn_model
        for param in self.mlp.parameters(): param.requires_grad_(False)
        for param in self.gnn.parameters(): param.requires_grad_(False)
        mlp_feat_dim = self.mlp.classifier.in_features
        gnn_feat_dim = self.gnn.classifier[0].in_features
        self.mlp_proj = nn.Sequential(nn.LayerNorm(mlp_feat_dim), nn.Linear(mlp_feat_dim, feat_dim), nn.GELU())
        self.gnn_proj = nn.Sequential(nn.LayerNorm(gnn_feat_dim), nn.Linear(gnn_feat_dim, feat_dim), nn.GELU())
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim*2, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
        )

    def forward(self, microbe_input, fmri_input):
        with torch.no_grad():
            mlp_feat = self.mlp.feature_extractor(microbe_input)
            gnn_feat = self.gnn(fmri_input)
        h_microbe = F.normalize(self.mlp_proj(mlp_feat), dim=1)
        h_fmri = F.normalize(self.gnn_proj(gnn_feat), dim=1)
        combined = torch.cat([h_microbe, h_fmri], dim=1)
        return h_microbe, h_fmri, self.classifier(combined)

class LabelAwareContrastiveLoss(nn.Module):
    def __init__(self, temp=0.07, hard_neg_ratio=0.2):
        super().__init__()
        self.temp = temp
        self.hard_neg_ratio = hard_neg_ratio
    def forward(self, h_microbe, h_fmri, labels):
        logits = torch.mm(h_microbe, h_fmri.T) / self.temp
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        neg_mask = ~label_mask
        neg_logits = logits * neg_mask.float()
        k = int(self.hard_neg_ratio * neg_mask.sum(dim=1).float().mean())
        if k > 0:
            hard_neg_indices = neg_logits.topk(k, dim=1).indices
            targets = label_mask.float()
            targets.scatter_(1, hard_neg_indices, 0.5)
        else:
            targets = label_mask.float()
        loss = -torch.mean(F.log_softmax(logits, dim=1) * targets + F.log_softmax(logits.T, dim=1) * targets.T)
        return loss

class PairedDataset(Dataset):
    def __init__(self, microbe_data, fmri_data):
        self.microbe_features, self.microbe_labels = microbe_data
        self.fmri_features, self.fmri_labels = fmri_data
        self.label_to_indices = {
            label: {'microbe': np.where(self.microbe_labels == label)[0], 'fmri': np.where(self.fmri_labels == label)[0]}
            for label in np.unique(self.microbe_labels)
        }
    def __len__(self): return len(self.microbe_features)
    def __getitem__(self, idx):
        microbe_feat = self.microbe_features[idx]
        label = self.microbe_labels[idx]
        microbe_feat += np.random.normal(0, 0.1, size=microbe_feat.shape)
        fmri_indices = self.label_to_indices[label]['fmri']
        if len(fmri_indices) == 0: # Fallback if no matching label found
             fmri_idx = np.random.randint(0, len(self.fmri_features))
        else:
             fmri_idx = np.random.choice(fmri_indices)
        fmri_feat = self.fmri_features[fmri_idx]
        return {'microbe': torch.FloatTensor(microbe_feat), 'fmri': torch.FloatTensor(fmri_feat), 'label': torch.LongTensor([label])}

def normalize_features(features, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
    return (features - mean) / std, mean, std

def train_contrastive(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, accuracies = 0.0, []
    for batch in train_loader:
        microbe, fmri, labels = batch['microbe'].to(device), batch['fmri'].to(device), batch['label'].squeeze().to(device)
        optimizer.zero_grad()
        h_microbe, h_fmri, logits = model(microbe, fmri)
        contrast_loss = criterion(h_microbe, h_fmri, labels)
        cls_loss = F.cross_entropy(logits, labels)
        loss = contrast_loss + 0.5 * cls_loss
        loss.backward()
        optimizer.step()
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        total_loss += loss.item()
        accuracies.append(acc.item())
    return total_loss / len(train_loader), np.mean(accuracies)

def validate(model, loader, device, full_report=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            microbe, fmri, labels = batch['microbe'].to(device), batch['fmri'].to(device), batch['label'].squeeze().to(device)
            _, _, logits = model(microbe, fmri)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel() if len(np.unique(all_labels)) > 1 else (0,0,0,0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
    metrics = {'accuracy': accuracy, 'auc': auc, 'f1_score': f1, 'sensitivity': sensitivity, 'specificity': specificity}
    if full_report:
        report_str = classification_report(all_labels, all_preds, target_names=["Healthy", "ASD"], digits=4)
        return metrics, report_str
    return metrics


# --- Main Execution Logic for Hyperparameter Tuning and Cross-Validation ---

def main():
    # --- 1. Data Loading and Preparation ---
    print("--- Loading and preparing data ---")
    
    # Paths
    file_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/abide.hdf5"
    graph_type = "cc200"
    csv_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/microbe_data.csv"
    biom_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/feature-table.biom"

    # Load fMRI data
    fmri_loader = fMRIDataLoader(file_path=file_path, graph_type=graph_type)
    all_fmri_features, all_fmri_labels = fmri_loader.load_all_data()

    # Load microbe data
    microbe_loader = MicrobeDataLoader(csv_path=csv_path, biom_path=biom_path)
    all_microbe_features = microbe_loader.features_tensor.numpy()
    all_microbe_labels = microbe_loader.labels_tensor.numpy()

    print(f"Loaded {len(all_fmri_features)} fMRI samples and {len(all_microbe_features)} microbe samples.")

    # --- 2. Hyperparameter Grid Definition ---
    param_grid = {
        'lr': [1e-4, 5e-5],
        'weight_decay': [1e-5, 1e-6],
        'feat_dim': [128, 256],
        'temp': [0.05, 0.07],
        'hard_neg_ratio': [0.2, 0.3],
    }
    
    hyperparameter_results = []

    # --- 3. Outer Loop: Hyperparameter Search ---
    for params in ParameterGrid(param_grid):
        print(f"\n--- Testing Hyperparameters: {params} ---")
        
        fold_metrics = []
        
        # --- 4. Inner Loop: 10-Fold Cross-Validation ---
        # We perform stratified splitting on the microbe dataset, and mirror the splits for fMRI
        kfold_microbe = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        kfold_fmri = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        gen_microbe_split = kfold_microbe.split(all_microbe_features, all_microbe_labels)
        gen_fmri_split = kfold_fmri.split(all_fmri_features, all_fmri_labels)

        for fold in range(10):
            print(f"--- Fold {fold+1}/10 ---")
            
            # Get data splits for this fold
            train_idx_m, test_idx_m = next(gen_microbe_split)
            train_idx_f, test_idx_f = next(gen_fmri_split)
            
            # Further split training data into train and validation for early stopping
            train_idx_m, val_idx_m = train_test_split(train_idx_m, test_size=0.15, random_state=42, stratify=all_microbe_labels[train_idx_m])
            train_idx_f, val_idx_f = train_test_split(train_idx_f, test_size=0.15, random_state=42, stratify=all_fmri_labels[train_idx_f])

            # Normalize microbe features based on the training set of the fold
            microbe_train_feat_raw = all_microbe_features[train_idx_m]
            microbe_train_feat, microbe_mean, microbe_std = normalize_features(microbe_train_feat_raw)
            microbe_val_feat = normalize_features(all_microbe_features[val_idx_m], microbe_mean, microbe_std)[0]
            microbe_test_feat = normalize_features(all_microbe_features[test_idx_m], microbe_mean, microbe_std)[0]
            
            # Create Datasets and DataLoaders
            train_dataset = PairedDataset(
                (microbe_train_feat, all_microbe_labels[train_idx_m]),
                (all_fmri_features[train_idx_f], all_fmri_labels[train_idx_f])
            )
            val_dataset = PairedDataset(
                (microbe_val_feat, all_microbe_labels[val_idx_m]),
                (all_fmri_features[val_idx_f], all_fmri_labels[val_idx_f])
            )
            test_dataset = PairedDataset(
                (microbe_test_feat, all_microbe_labels[test_idx_m]),
                (all_fmri_features[test_idx_f], all_fmri_labels[test_idx_f])
            )

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # --- Model Initialization for the fold ---
            mlp_model = load_pretrained_mlp()
            gnn_model = load_pretrained_gnn()
            model = ContrastiveModel(mlp_model, gnn_model, feat_dim=params['feat_dim']).to(device)
            
            optimizer = torch.optim.AdamW([
                {'params': model.mlp_proj.parameters()},
                {'params': model.gnn_proj.parameters()},
                {'params': model.classifier.parameters()}
            ], lr=params['lr'], weight_decay=params['weight_decay'])
            criterion = LabelAwareContrastiveLoss(temp=params['temp'], hard_neg_ratio=params['hard_neg_ratio'])

            # --- Training with Early Stopping ---
            best_valid_acc = 0.0
            patience, counter = 20, 0
            
            for epoch in range(100): # Limit epochs for hyperparameter search
                train_loss, train_acc = train_contrastive(model, train_loader, optimizer, criterion, device)
                valid_metrics = validate(model, val_loader, device)
                valid_acc = valid_metrics['accuracy']

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    torch.save(model.state_dict(), f"best_model_fold_{fold+1}.pth")
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Load best model for this fold and evaluate on test set
            best_fold_model = ContrastiveModel(load_pretrained_mlp(), load_pretrained_gnn(), feat_dim=params['feat_dim']).to(device)
            best_fold_model.load_state_dict(torch.load(f"best_model_fold_{fold+1}.pth"))
            
            test_metrics = validate(best_fold_model, test_loader, device)
            fold_metrics.append(test_metrics)
            print(f"Fold {fold+1} Test Metrics: Accuracy={test_metrics['accuracy']:.4f}, AUC={test_metrics['auc']:.4f}")

        # --- Aggregate results for the current hyperparameter set ---
        avg_metrics = {key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0]}
        std_metrics = {key: np.std([m[key] for m in fold_metrics]) for key in fold_metrics[0]}
        
        print(f"\n--- Aggregated 10-Fold CV Results for params: {params} ---")
        for key in avg_metrics:
            print(f"Avg {key}: {avg_metrics[key]:.4f} (+/- {std_metrics[key]:.4f})")
        
        hyperparameter_results.append({'params': params, 'metrics': avg_metrics, 'std': std_metrics})

    # --- 5. Final Best Hyperparameter Selection ---
    best_result = max(hyperparameter_results, key=lambda x: x['metrics']['accuracy'])
    
    print("\n\n=======================================================")
    print("           Hyperparameter Tuning Finished            ")
    print("=======================================================")
    print(f"Best Hyperparameters found: {best_result['params']}")
    print("--- Best 10-Fold CV Performance ---")
    for key in best_result['metrics']:
        print(f"Avg {key}: {best_result['metrics'][key]:.4f} (+/- {best_result['std'][key]:.4f})")
    
    # Save results to a file
    results_df = pd.DataFrame([{**r['params'], **{'avg_'+k: v for k,v in r['metrics'].items()}} for r in hyperparameter_results])
    results_df.to_csv("hyperparameter_tuning_results.csv", index=False)
    print("\nHyperparameter tuning results saved to 'hyperparameter_tuning_results.csv'")


if __name__ == "__main__":
    main() 