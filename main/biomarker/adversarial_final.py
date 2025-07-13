import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import h5py
import pandas as pd 
from tqdm import tqdm
import torch.nn.functional as F
import torch_geometric.nn as tg_nn
from torch_geometric.data import Data, Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau


PRE_TRAIN_PATH = "/home/yangzongxian/xlz/ASD_GCN/main/pre_train"
if PRE_TRAIN_PATH not in sys.path:
    sys.path.append(PRE_TRAIN_PATH)

try:
    from MLP import MicrobeDataLoader, SparseMLP
except ImportError as e:
    print(f"Error importing from MLP.py (expected at {PRE_TRAIN_PATH}): {e}")
    raise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Utility Functions ---
def reconstruct_fc(vector, num_rois=200):
    matrix = np.zeros((num_rois, num_rois), dtype=np.float32)
    expected_len = num_rois * (num_rois - 1) // 2
    if len(vector) != expected_len:
        if len(vector) == 0 and expected_len > 0: return matrix
        raise ValueError(f"Input vector length {len(vector)} != expected {expected_len} for {num_rois} ROIs.")
    triu_indices = np.triu_indices(num_rois, k=1)
    matrix[triu_indices] = vector
    matrix = matrix + matrix.T
    return matrix

def normalize_features(features, mean=None, std=None):
    if not isinstance(features, np.ndarray): features = np.array(features)
    if features.size == 0: return features, mean, std
    if mean is None: mean = np.mean(features, axis=0)
    if std is None: std = np.std(features, axis=0)
    std_safe = np.where(std < 1e-8, 1e-8, std) # Avoid division by zero
    return (features - mean) / std_safe, mean, std

# --- Data Loaders (Using User's Provided fMRIDataLoader Structure) ---
class fMRIDataLoader:
    def __init__(self, file_path, graph_type, config):
        self.file_path = file_path
        self.graph_type = graph_type
        self.num_rois = config.get('num_rois', 200)
        self.test_size = config.get('fMRI_test_split_ratio', 0.15)
        val_split_input = config.get('fMRI_val_split_ratio', 0.15) # Original val_size from user's code
        self.val_size_of_train_val = val_split_input / (1.0 - self.test_size) if (1.0 - self.test_size) > 0 else 0.0
        self.batch_size = config.get('batch_size', 32)
        self.random_state = config.get('random_state_data', 42)
        self.data_splits = self._load_and_split_data()

    def _load_and_split_data(self):
        graph_dataset, labels_for_stratify = [], []
        expected_triu_len = self.num_rois * (self.num_rois - 1) // 2

        with h5py.File(self.file_path, "r") as f:
            patients_group = f["/patients"]
            for subject_id in patients_group.keys():
                subject_group = patients_group[subject_id]
                if self.graph_type in subject_group:
                    triu_vector = subject_group[self.graph_type][:]
                    if len(triu_vector) != expected_triu_len: continue
                    matrix = reconstruct_fc(triu_vector, self.num_rois)
                    label_val = subject_group.attrs["y"]
                    label = torch.tensor(label_val, dtype=torch.long)
                    flat_vector = matrix.flatten()
                    edge_index = self._get_brain_connectivity_edges(matrix) # From original
                    graph_data = Data(x=torch.FloatTensor(flat_vector), edge_index=edge_index, y=label)
                    graph_dataset.append(graph_data)
                    labels_for_stratify.append(label_val)
        
        if not graph_dataset: raise ValueError("No fMRI data loaded.")

        indices = np.arange(len(graph_dataset))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=self.test_size, stratify=labels_for_stratify, random_state=self.random_state)
        
        train_idx, val_idx = np.array([],dtype=int), np.array([],dtype=int)
        if len(train_val_idx) > 0:
            labels_train_val_for_split = np.array(labels_for_stratify)[train_val_idx]
            if self.val_size_of_train_val > 0 and self.val_size_of_train_val < 1.0 and len(np.unique(labels_train_val_for_split)) > 1:
                train_idx, val_idx = train_test_split(
                    train_val_idx, test_size=self.val_size_of_train_val, stratify=labels_train_val_for_split, random_state=self.random_state)
            else:
                train_idx = train_val_idx
                if self.val_size_of_train_val > 0 and self.val_size_of_train_val < 1.0:
                    print("Warning (fMRI): Val split from train_val pool issue. Val set may be empty.")

        return {
            "train": ([graph_dataset[i] for i in train_idx], [labels_for_stratify[i] for i in train_idx]),
            "valid": ([graph_dataset[i] for i in val_idx], [labels_for_stratify[i] for i in val_idx]),
            "test": ([graph_dataset[i] for i in test_idx], [labels_for_stratify[i] for i in test_idx]),
        }

    def _get_brain_connectivity_edges(self, matrix, threshold=0.3):
        rows, cols = np.triu_indices_from(matrix, k=1)
        mask = matrix[rows, cols] > threshold
        edge_index = np.array([rows[mask], cols[mask]])
        if edge_index.size > 0:
            edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
        return torch.tensor(edge_index, dtype=torch.long)

    def _create_dataloader(self, data_list, shuffle_flag):
        if not data_list: return DataLoader([], batch_size=self.batch_size)
        return DataLoader(data_list, batch_size=self.batch_size, shuffle=shuffle_flag,
                          collate_fn=lambda b: Batch.from_data_list(b), num_workers=config.get('num_workers',0))

    def get_dataloaders(self):
        return {
            "train": self._create_dataloader(self.data_splits["train"][0], True),
            "valid": self._create_dataloader(self.data_splits["valid"][0], False),
            "test": self._create_dataloader(self.data_splits["test"][0], False),
        }

# --- Model Definitions (Adapted for Attention & Config from your adversarial_new.py) ---
class fMRI3DGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_rois = config.get('num_rois', 200)
        self.input_flat_dim = self.num_rois * self.num_rois
        self.graph_builder = nn.Sequential(nn.Linear(self.input_flat_dim, self.input_flat_dim), nn.Sigmoid())
        self.convs = nn.ModuleList([
            tg_nn.GATv2Conv(16, 128, heads=8, dropout=config['gnn_dropout'], add_self_loops=False),
            tg_nn.GATv2Conv(128*8, 256, heads=4, dropout=config['gnn_dropout']),
            tg_nn.GATv2Conv(256*4, config['gnn_output_dim'], heads=1, concat=False, dropout=config['gnn_dropout'])])
        self.feature_enhancer = nn.Sequential(nn.Linear(2,8), nn.GELU(), nn.Linear(8,16), nn.LayerNorm(16))
        
        self.classifier = nn.Sequential( 
            nn.Linear(config['gnn_output_dim'], 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(config.get('gnn_classifier_dropout', 0.3)), nn.Linear(256, config['num_classes']))

    def build_graph(self, fc_matrix_flat):
        bs = fc_matrix_flat.size(0); dev = fc_matrix_flat.device
        adj_l = self.graph_builder(fc_matrix_flat).view(bs, self.num_rois, self.num_rois)
        adj_l = (adj_l + adj_l.transpose(1,2)) / 2
        n_fts_list, e_idxs_list = [], []
        for b in range(bs):
            adj_b = adj_l[b]
            m, s = adj_b.mean(dim=1, keepdim=True), adj_b.std(dim=1, keepdim=True)
            b_ft = torch.cat([m, s + 1e-6], dim=1)
            e_ft = self.feature_enhancer(b_ft)
            thr = torch.quantile(adj_b.flatten(), 0.75)
            r, c = (adj_b > thr).nonzero(as_tuple=True)
            e_idx = torch.stack([r, c], dim=0)
            n_fts_list.append(e_ft); e_idxs_list.append(e_idx)
        return Batch.from_data_list([Data(x=ft.to(dev), edge_index=ei.to(dev)) for ft,ei in zip(n_fts_list,e_idxs_list)])

    def forward(self, raw_fc_flat, return_attention_scores=False):
        if raw_fc_flat.shape[1]!=self.input_flat_dim: raise ValueError(f"fMRI3DGNN input dim {raw_fc_flat.shape[1]} != expected {self.input_flat_dim}")
        b_graph = self.build_graph(raw_fc_flat)
        x,e_idx,g_map = b_graph.x.to(raw_fc_flat.device),b_graph.edge_index.to(raw_fc_flat.device),b_graph.batch.to(raw_fc_flat.device)
        att_list = []
        for conv_idx, conv in enumerate(self.convs):
            if x.size(1) != conv.in_channels:
                 raise ValueError(f"GNN Layer {conv_idx+1} input dim {x.size(1)} != expected {conv.in_channels}")
            if return_attention_scores:
                x, (e_att, alpha) = conv(x,e_idx,return_attention_weights=True)
                att_list.append({"layer_index": conv_idx, "edge_index_gat":e_att, "attention_coeffs":alpha, "batch_map_nodes":g_map})
            else: x = conv(x,e_idx)
            x = F.gelu(F.dropout(x, p=config.get('gnn_gat_dropout', 0.3), training=self.training)) # Use config and module's training state
        x_pool = tg_nn.global_mean_pool(x, g_map)
        return (x_pool, att_list) if return_attention_scores else x_pool
    def classify(self, x_pool): return self.classifier_head(x_pool)

def load_pretrained_mlp(config):
    model = SparseMLP(
        input_dim=config['microbe_feature_dim'], num_classes=config['num_classes'],
        hidden_dims=config.get('mlp_hidden_dims', [2048, 1024, 512]), 
        dropout=config['mlp_dropout'],
        sparsity_lambda=config.get('mlp_sparsity_lambda', 0.05)
    )
    # Use path from config
    model.load_state_dict(torch.load(config['mlp_model_path'], map_location=device))
    return model.to(device)

def load_pretrained_gnn(config):
    model = fMRI3DGNN(config)
    checkpoint = torch.load(config['gnn_model_path'], map_location=device)
    state_dict_to_load = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict_to_load)
    return model.to(device)

class ContrastiveModel(nn.Module):
    def __init__(self, mlp_model, gnn_model, config):
        super().__init__()
        self.mlp, self.gnn = mlp_model, gnn_model
        feat_dim = config['contrastive_feat_dim']
        
        # Infer mlp_intermediate_feat_dim from SparseMLP structure
        # This assumes feature_extractor is nn.Sequential and last Linear layer's out_features is what we need
        mlp_hid_dims = config.get('mlp_hidden_dims', [2048, 1024, 512])
        mlp_intermediate_feat_dim = mlp_hid_dims[-1] if mlp_hid_dims else config['mlp_feat_extractor_out_dim']
        
        gnn_intermediate_feat_dim = config['gnn_output_dim']

        if config.get('freeze_base_models', True):
            for p in self.mlp.parameters(): p.requires_grad_(False)
            for p in self.gnn.parameters(): p.requires_grad_(False)
        
        self.mlp_proj=nn.Sequential(nn.LayerNorm(mlp_intermediate_feat_dim), nn.Linear(mlp_intermediate_feat_dim, feat_dim), nn.GELU())
        self.gnn_proj=nn.Sequential(nn.LayerNorm(gnn_intermediate_feat_dim), nn.Linear(gnn_intermediate_feat_dim, feat_dim), nn.GELU())
        self.classifier=nn.Sequential(nn.Linear(feat_dim*2, 256), nn.LayerNorm(256), nn.ReLU(), 
                                      nn.Dropout(config.get('contrastive_classifier_dropout', 0.3)), 
                                      nn.Linear(256, config['num_classes']))

    def forward(self, microbe_in, fmri_in, return_gnn_attention=False):
        if not hasattr(self.mlp,'feature_extractor'): raise AttributeError("MLP needs 'feature_extractor' method.")
        mlp_ft = self.mlp.feature_extractor(microbe_in) # Must be defined in SparseMLP
        gnn_att_scores = None
        if return_gnn_attention: gnn_ft,gnn_att_scores = self.gnn(fmri_in,return_attention_scores=True)
        else: gnn_ft = self.gnn(fmri_in,return_attention_scores=False)
        h_m,h_f=F.normalize(self.mlp_proj(mlp_ft),dim=1),F.normalize(self.gnn_proj(gnn_ft),dim=1)
        comb = torch.cat([h_m,h_f],dim=1); logits=self.classifier(comb)
        return (h_m,h_f,logits,gnn_att_scores) if return_gnn_attention else (h_m,h_f,logits)



class LabelAwareContrastiveLoss(nn.Module):
    def __init__(self, temp=0.07, hard_neg_ratio=0.2):
        super().__init__()
        self.temp = temp
        self.hr = hard_neg_ratio # Assuming hr is hard_neg_ratio

    def forward(self, h_m, h_f, lbls):
        if h_m.ndim != 2 or h_f.ndim != 2:
            raise ValueError(f"Embeddings must be 2D. Got h_m: {h_m.shape}, h_f: {h_f.shape}")
        if lbls.ndim != 1:
            lbls = lbls.squeeze()
        if h_m.shape[0] != lbls.shape[0] or h_f.shape[0] != lbls.shape[0]:
            raise ValueError(f"Batch size mismatch. h_m: {h_m.shape[0]}, h_f: {h_f.shape[0]}, labels: {lbls.shape[0]}")
        if h_m.shape[0] == 0:  # Handle empty batch
            return torch.tensor(0.0, device=h_m.device, requires_grad=True)

        logits = torch.mm(h_m, h_f.T) / self.temp
        label_mask = (lbls.unsqueeze(1) == lbls.unsqueeze(0)).float()
        neg_mask = 1.0 - label_mask
        num_negatives_per_sample = neg_mask.sum(dim=1)

        k_avg_float = self.hr * num_negatives_per_sample.float().mean()
        k_avg_val = int(k_avg_float.item()) if isinstance(k_avg_float, torch.Tensor) else int(k_avg_float)


        min_val_for_k_calculation = 0
        min_available_neg_strict_tensor = num_negatives_per_sample[num_negatives_per_sample > 0]
        if min_available_neg_strict_tensor.numel() > 0:
            min_val_for_k_calculation = min_available_neg_strict_tensor.min().item() # Get Python scalar
        # else, min_val_for_k_calculation remains 0 (Python int)

        # Now min_val_for_k_calculation is guaranteed to be a Python number
        actual_k = min(k_avg_val, int(min_val_for_k_calculation))
        actual_k = max(0, actual_k)  # Ensure k is a non-negative integer

        targets = label_mask.clone()
        if actual_k > 0:
            neg_logits_for_topk = logits.clone().detach()
            neg_logits_for_topk.masked_fill_(label_mask.bool(), -float('inf'))
            
            # Ensure k for topk is not larger than the dimension size
            k_for_topk = min(actual_k, neg_logits_for_topk.size(1))
            if k_for_topk > 0 : # If k is still positive after adjustments
                hard_neg_indices = neg_logits_for_topk.topk(k_for_topk, dim=1).indices
                targets.scatter_(1, hard_neg_indices, 0.5)  # Weight hard negatives

        log_softmax_logits = F.log_softmax(logits, dim=1)
        log_softmax_logits_t = F.log_softmax(logits.T, dim=1)
        
        loss1 = -(log_softmax_logits * targets).sum(dim=1).mean()
        loss2 = -(log_softmax_logits_t * targets.T).sum(dim=1).mean() # targets.T is correct for alignment
        
        return (loss1 + loss2) / 2.0

class PairedDataset(Dataset): # Updated to include train flag
    def __init__(self, microbe_data_tuple, fmri_data_tuple, train=True):
        self.m_fts_np, self.m_lbls_np = microbe_data_tuple
        self.f_fts_np, self.f_lbls_np = fmri_data_tuple
        self.train = train

        if self.m_fts_np.shape[0] != self.f_fts_np.shape[0] or \
           self.m_lbls_np.shape[0] != self.f_lbls_np.shape[0] or \
           self.m_fts_np.shape[0] != self.m_lbls_np.shape[0]:
            raise ValueError(f"PairedDataset: Sample/label count mismatch. "
                             f"M_fts:{self.m_fts_np.shape}, M_lbl:{self.m_lbls_np.shape}, "
                             f"F_fts:{self.f_fts_np.shape}, F_lbl:{self.f_lbls_np.shape}")
        
        if not np.array_equal(self.m_lbls_np, self.f_lbls_np) :
            print("Warning (PairedDataset): Microbe and fMRI labels are not identical element-wise. "
                  "This implies data is not perfectly aligned by subject AND label for pairing. "
                  "Pairing will use microbe label to find an fMRI sample with the *same label type*, "
                  "but it might not be the *same subject's* fMRI data if initial alignment was off.")

        self.f_idxs_by_lbl = {lbl_val: np.where(self.f_lbls_np == lbl_val)[0] 
                              for lbl_val in np.unique(self.f_lbls_np)}
    def __len__(self): return len(self.m_fts_np)
    def __getitem__(self,idx):
        m_ft_orig,lbl_val = self.m_fts_np[idx],self.m_lbls_np[idx]
        m_tensor = torch.from_numpy(m_ft_orig.astype(np.float32))
        if self.train: m_tensor += torch.randn_like(m_tensor) * 0.1 # Add noise only if training

        f_poss_idxs = self.f_idxs_by_lbl.get(lbl_val)
        if f_poss_idxs is None or not f_poss_idxs.size: # No fMRI samples for this specific label
            print(f"CRITICAL WARNING (PairedDataset): No fMRI sample found for label {lbl_val} (originating from microbe sample at index {idx}). "
                  "This means the fMRI dataset (passed to PairedDataset) doesn't contain any samples with this label. "
                  "Attempting to pick a random fMRI sample from ANY available label (HIGHLY SUBOPTIMAL, introduces label mismatch for pairing).")
            if len(self.f_fts_np) == 0: raise ValueError("No fMRI features available in PairedDataset for fallback.")
            chosen_fmri_idx = np.random.choice(len(self.f_fts_np)) # Fallback: random fMRI sample
        else:
            chosen_fmri_idx = np.random.choice(f_poss_idxs) # Random fMRI sample of the same class
        
        f_tensor = torch.from_numpy(self.f_fts_np[chosen_fmri_idx].astype(np.float32))
        return {'microbe':m_tensor,'fmri':f_tensor,'label':torch.LongTensor([lbl_val])}

# --- Feature Extraction Utilities ---
# Corrected extract_fmri_features_40k (replaces user's extract_graph_features)
def extract_fmri_features_40k(pyg_loader, expected_feature_dim=40000):
    all_features_list, all_labels_list = [], []
    if pyg_loader is None or (hasattr(pyg_loader, 'dataset') and len(pyg_loader.dataset) == 0):
        return np.array([]).reshape(0, expected_feature_dim), np.array([])

    for pyg_batch in pyg_loader:
        if pyg_batch.num_graphs == 0: continue # Skip empty batches from loader
            
        try: # Robust reshaping
            current_x_processed = pyg_batch.x.view(pyg_batch.num_graphs, -1)
        except RuntimeError as e:
            raise RuntimeError(f"Error reshaping pyg_batch.x (shape: {pyg_batch.x.shape}) "
                               f"with num_graphs {pyg_batch.num_graphs}. Original error: {e}")

        if current_x_processed.shape[1] != expected_feature_dim:
            raise ValueError(f"Reshaped fMRI features dim: {current_x_processed.shape[1]}, expected {expected_feature_dim}.")

        all_features_list.append(current_x_processed.cpu().numpy())
        all_labels_list.append(pyg_batch.y.cpu().numpy().reshape(-1)) # Ensure labels are 1D

    if not all_features_list: return np.array([]).reshape(0, expected_feature_dim), np.array([])
    return np.concatenate(all_features_list, axis=0), np.concatenate(all_labels_list, axis=0)

def extract_microbe_features_from_raw_loader(microbe_raw_loader, microbe_feature_dim): # As before
    all_features, all_labels = [], []
    if microbe_raw_loader is None or (hasattr(microbe_raw_loader, 'dataset') and len(microbe_raw_loader.dataset) == 0):
        return np.array([]).reshape(0, microbe_feature_dim), np.array([])
    for features_batch, labels_batch in microbe_raw_loader:
        all_features.append(features_batch.cpu().numpy())
        all_labels.append(labels_batch.cpu().numpy())
    if not all_features: return np.array([]).reshape(0, microbe_feature_dim), np.array([])
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

# --- Training, Validation, Test Functions (Using user's provided structure) ---
def train_contrastive_adversarial(model, train_loader, optimizer, criterion_contrastive, device, config):
    model.train()
    if config.get('freeze_base_models',True): # Set base models to eval if frozen
        if hasattr(model, 'mlp') and model.mlp is not None: model.mlp.eval()
        if hasattr(model, 'gnn') and model.gnn is not None: model.gnn.eval()

    total_loss_epoch, accuracies_epoch = 0.0, []
    classification_weight = config['classification_weight']
    epsilon = config['adversarial_epsilon']
    
    pbar = tqdm(train_loader, desc="Adv. Training", unit="batch", leave=False, ncols=100)
    for batch in pbar:
        microbe = batch['microbe'].to(device)
        fmri = batch['fmri'].to(device)
        labels = batch['label'].squeeze().to(device)

        microbe.requires_grad_(True)
        # Original forward pass for gradient calculation
        h_m, h_f, logits_orig, *_ = model(microbe, fmri) # Use *_ to handle optional attention return

        contrast_loss_val = criterion_contrastive(h_m, h_f, labels)
        cls_loss_val = F.cross_entropy(logits_orig, labels)
        combined_loss_for_grad = contrast_loss_val + classification_weight * cls_loss_val
        
        model.zero_grad() 
        if microbe.grad is not None: microbe.grad.zero_()

        grad_microbe = torch.autograd.grad(combined_loss_for_grad, microbe, retain_graph=False)[0]
        delta = epsilon * torch.sign(grad_microbe)
        microbe_adv = (microbe.detach() + delta).detach() # Detach original microbe before adding delta

        optimizer.zero_grad()
        h_m_adv, h_f_adv, logits_adv, *_ = model(microbe_adv, fmri) # fmri is original
        
        contrast_loss_adv = criterion_contrastive(h_m_adv, h_f_adv, labels)
        cls_loss_adv = F.cross_entropy(logits_adv, labels)
        final_loss = contrast_loss_adv + classification_weight * cls_loss_adv
        
        final_loss.backward()
        optimizer.step()

        total_loss_epoch += final_loss.item()
        preds = torch.argmax(logits_adv, dim=1)
        acc_batch = (preds == labels).float().mean().item()
        accuracies_epoch.append(acc_batch)
        pbar.set_postfix_str(f"Loss: {final_loss.item():.3f}, Acc: {acc_batch:.2%}")
        
    avg_loss = total_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0
    avg_acc = np.mean(accuracies_epoch) if accuracies_epoch else 0
    # For returning detailed losses if needed by logger (match previous full version)
    # For now, returning only total loss and accuracy as per user's original train_contrastive_adversarial
    return avg_loss, avg_acc


def validate_or_test(model, data_loader, criterion_contrastive, device, config, mode="Validating"): # Renamed from validate
    model.eval()
    total_loss_epoch, all_labels_list, all_preds_list = 0.0, [], []
    classification_weight = config['classification_weight']

    if data_loader is None or len(data_loader) == 0: # Handle empty dataloader
        print(f"Warning: {mode} data_loader is empty. Returning 0 loss/acc.")
        return 0.0, 0.0

    pbar = tqdm(data_loader, desc=mode, unit="batch", leave=False, ncols=100)
    with torch.no_grad():
        for batch in pbar:
            microbe = batch['microbe'].to(device)
            fmri = batch['fmri'].to(device)
            labels = batch['label'].squeeze().to(device)

            h_m, h_f, logits, *_ = model(microbe, fmri) # Use *_ for optional attention
            contrast_loss_val = criterion_contrastive(h_m, h_f, labels)
            cls_loss_val = F.cross_entropy(logits, labels)
            final_loss = contrast_loss_val + classification_weight * cls_loss_val
            
            total_loss_epoch += final_loss.item()
            preds = torch.argmax(logits, dim=1)
            all_labels_list.extend(labels.cpu().numpy())
            all_preds_list.extend(preds.cpu().numpy())
            pbar.set_postfix_str(f"Loss: {final_loss.item():.3f}")

    avg_loss = total_loss_epoch / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = accuracy_score(all_labels_list, all_preds_list) if all_labels_list else 0
    # Match return signature of user's original validate function: avg_loss, accuracy
    return avg_loss, accuracy


# --- Main 10-Fold CV Workflow ---
def run_10fold_cv_evaluation(config):
    print("--- Starting 10-Fold Cross-Validation for Hyperparameter Evaluation ---")
    print(f"Using configuration: {config}")

    # 1. Initialize DataLoaders (these perform internal train/val/test splits)
    fmri_data_source = fMRIDataLoader(config['fmri_hdf5_path'], config['fmri_graph_type'], config)
    # This MicrobeDataLoader is from MLP.py
    microbe_data_source = MicrobeDataLoader(config['microbe_csv_path'], config['microbe_biom_path'], config['batch_size'])

    # 2. Get the PyTorch DataLoaders for each pre-defined split
    fmri_split_loaders = fmri_data_source.get_dataloaders() # dict: {"train": Dataloader, ...}
    microbe_split_loaders = microbe_data_source.get_loaders() # tuple: (train_DL, val_DL, test_DL)

    # 3. Extract features and labels from these DataLoaders into NumPy arrays
    print("Extracting features from pre-defined data loader splits...")
    fmri_train_np, fmri_train_labels_np = extract_fmri_features_40k(fmri_split_loaders["train"], config['num_rois']**2)
    fmri_val_np,   fmri_val_labels_np   = extract_fmri_features_40k(fmri_split_loaders["valid"], config['num_rois']**2)
    fmri_test_np,  fmri_test_labels_np  = extract_fmri_features_40k(fmri_split_loaders["test"],  config['num_rois']**2)

    microbe_train_np, microbe_train_labels_np = extract_microbe_features_from_raw_loader(microbe_split_loaders[0], config['microbe_feature_dim'])
    microbe_val_np,   microbe_val_labels_np   = extract_microbe_features_from_raw_loader(microbe_split_loaders[1], config['microbe_feature_dim'])
    microbe_test_np,  microbe_test_labels_np  = extract_microbe_features_from_raw_loader(microbe_split_loaders[2], config['microbe_feature_dim'])

    # --- Critical Data Alignment Checks (based on sample counts per split) ---
    print("\n--- Verifying Data Alignment from DataLoaders' Pre-defined Splits ---")
    error_messages = []
    if fmri_train_np.shape[0] != microbe_train_np.shape[0]:
        error_messages.append(f"Train split sample count mismatch! fMRI: {fmri_train_np.shape[0]}, Microbe: {microbe_train_np.shape[0]}")
    if fmri_val_np.shape[0] != microbe_val_np.shape[0] and (fmri_val_np.size > 0 or microbe_val_np.size > 0) : # Only error if one is non-empty and they mismatch
        error_messages.append(f"Validation split sample count mismatch! fMRI: {fmri_val_np.shape[0]}, Microbe: {microbe_val_np.shape[0]}")
    if fmri_test_np.shape[0] != microbe_test_np.shape[0] and (fmri_test_np.size > 0 or microbe_test_np.size > 0):
        error_messages.append(f"Test split sample count mismatch! fMRI: {fmri_test_np.shape[0]}, Microbe: {microbe_test_np.shape[0]}")
    
    if error_messages:
        for msg in error_messages: print(f"CRITICAL ERROR: {msg}")
        print("Your fMRIDataLoader and MicrobeDataLoader are not producing aligned splits. "
              "You MUST ensure they process the same subjects in the same order for each split, "
              "or implement robust subject ID-based alignment before forming PairedDataset.")
        # Option: raise ValueError here, or try to proceed with smallest common N if that's desired (risky)
        # For now, we'll proceed, PairedDataset might error or produce misaligned pairs if counts differ for a given split.
        # The CV loop will use the length of the microbe data for its main loop.

    # 4. Prepare CV pool (Train + Val from the pre-defined splits)
    # We proceed by concatenating, assuming user will fix upstream if counts mismatch significantly.
    # If counts mismatch, PairedDataset will likely use the length of the first modality (microbe) passed to it.
    cv_pool_fmri_list, cv_pool_microbe_list, cv_pool_labels_list = [], [], []

    # Use train split data
    if fmri_train_np.shape[0] > 0 and microbe_train_np.shape[0] > 0 :
        # Take the minimum length if counts mismatched, and warn user
        min_len_train = min(fmri_train_np.shape[0], microbe_train_np.shape[0])
        if fmri_train_np.shape[0] != microbe_train_np.shape[0]:
            print(f"Warning: Truncating train split to {min_len_train} samples due to count mismatch.")
        cv_pool_fmri_list.append(fmri_train_np[:min_len_train])
        cv_pool_microbe_list.append(microbe_train_np[:min_len_train])
        cv_pool_labels_list.append(fmri_train_labels_np[:min_len_train]) # Using fMRI labels as primary for pool
    
    # Use validation split data
    if fmri_val_np.shape[0] > 0 and microbe_val_np.shape[0] > 0:
        min_len_val = min(fmri_val_np.shape[0], microbe_val_np.shape[0])
        if fmri_val_np.shape[0] != microbe_val_np.shape[0]:
            print(f"Warning: Truncating val split to {min_len_val} samples due to count mismatch.")
        cv_pool_fmri_list.append(fmri_val_np[:min_len_val])
        cv_pool_microbe_list.append(microbe_val_np[:min_len_val])
        cv_pool_labels_list.append(fmri_val_labels_np[:min_len_val]) # Using fMRI labels

    if not cv_pool_fmri_list:
        raise ValueError("Cannot form CV pool: No usable aligned training or validation data.")

    cv_pool_fmri = np.concatenate(cv_pool_fmri_list, axis=0)
    cv_pool_microbe = np.concatenate(cv_pool_microbe_list, axis=0)
    cv_pool_labels = np.concatenate(cv_pool_labels_list, axis=0)
    
    print(f"CV pool size (Train+Val splits, potentially truncated): {cv_pool_fmri.shape[0]} samples.")
    if cv_pool_fmri.shape[0] == 0: print("CV pool is empty after processing! Cannot proceed."); return
    
    n_splits_actual = min(config['n_splits_cv'], cv_pool_fmri.shape[0])
    if n_splits_actual < 2 and cv_pool_fmri.shape[0] > 0 : n_splits_actual = 2 # Need at least 2 for KFold
    if n_splits_actual != config['n_splits_cv']:
        print(f"Adjusted n_splits_cv to {n_splits_actual} due to CV pool size.")
    if cv_pool_fmri.shape[0] < 2 : # Cannot do CV if less than 2 samples
         print(f"CV pool size is {cv_pool_fmri.shape[0]}, too small for K-Fold CV. Aborting CV.")
         return

    # 5. Prepare External Test Set
    ext_test_fmri, ext_test_microbe, ext_test_labels = np.array([]), np.array([]), np.array([])
    if fmri_test_np.shape[0] > 0 and microbe_test_np.shape[0] > 0:
        min_len_test = min(fmri_test_np.shape[0], microbe_test_np.shape[0])
        if fmri_test_np.shape[0] != microbe_test_np.shape[0]:
             print(f"Warning: Truncating test split to {min_len_test} samples due to count mismatch.")
        ext_test_fmri = fmri_test_np[:min_len_test]
        ext_test_microbe = microbe_test_np[:min_len_test]
        ext_test_labels = fmri_test_labels_np[:min_len_test] # Using fMRI labels
        print(f"External test set size (potentially truncated): {ext_test_fmri.shape[0]} samples.")
    else:
        print("Warning: External test set is empty or mismatched. Test evaluation will be skipped.")

    # 6. Perform K-Fold Cross-Validation
    kf = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=config['random_state_cv'])
    fold_test_accuracies, fold_best_val_accuracies = [], []
    can_stratify = len(np.unique(cv_pool_labels)) > 1 and len(cv_pool_labels) >= n_splits_actual

    # The rest of the run_10fold_cv_evaluation loop remains the same as the comprehensive version from my previous response
    # This includes: splitting data for the fold, normalizing microbe, creating PairedDataset and DataLoaders,
    # model initialization, optimizer, criterion, scheduler, training loop with early stopping,
    # and evaluation on the external test set.
    # For brevity, the full loop is not repeated here but should be taken from the last complete script.

    for fold, (train_idx, val_idx) in enumerate(kf.split(cv_pool_microbe, cv_pool_labels) if can_stratify else kf.split(cv_pool_microbe)):
        if not can_stratify and fold == 0: print("Warning: Cannot stratify CV folds.")
        print(f"\n=== Fold {fold + 1}/{n_splits_actual} ===")
        
        microbe_train_fold, labels_train_fold = cv_pool_microbe[train_idx], cv_pool_labels[train_idx]
        microbe_val_fold, labels_val_fold = cv_pool_microbe[val_idx], cv_pool_labels[val_idx]
        fmri_train_fold, fmri_val_fold = cv_pool_fmri[train_idx], cv_pool_fmri[val_idx]

        microbe_train_fold, m_mean, m_std = normalize_features(microbe_train_fold)
        microbe_val_fold, _, _ = normalize_features(microbe_val_fold, m_mean, m_std)

        train_ds_fold = PairedDataset((microbe_train_fold, labels_train_fold), (fmri_train_fold, labels_train_fold), train=True)
        # Handle cases where val_idx might be empty if n_splits_actual is small or data is very limited
        val_ds_fold = PairedDataset((microbe_val_fold, labels_val_fold), (fmri_val_fold, labels_val_fold), train=False) if len(val_idx)>0 else None
        
        train_dl_fold = DataLoader(train_ds_fold, batch_size=config['batch_size'], shuffle=True, num_workers=config.get('num_workers',0))
        val_dl_fold = DataLoader(val_ds_fold, batch_size=config['batch_size'], shuffle=False, num_workers=config.get('num_workers',0)) if val_ds_fold else None

        mlp_base = load_pretrained_mlp(config)
        gnn_base = load_pretrained_gnn(config)
        model_fold = ContrastiveModel(mlp_base, gnn_base, config).to(device)
        
        opt_params = [p for p in model_fold.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(opt_params if opt_params else model_fold.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        criterion_con = LabelAwareContrastiveLoss(config['contrastive_temp'], config['contrastive_hard_neg_ratio'])
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=config['lr_patience'], verbose=False)

        best_val_acc_fold, es_counter = 0.0, 0
        save_path = os.path.join(config['model_save_dir'], f"contrastive_fold_{fold+1}_best.pth")

        for epoch in range(config['num_epochs']):
            # Use the full train_contrastive_adversarial function logic here
            tr_loss, tr_acc = train_contrastive_adversarial(model_fold,train_dl_fold,optimizer,criterion_con,device,config) # Simplified return from placeholder
            
            val_acc = 0.0
            if val_dl_fold and len(val_dl_fold.dataset) > 0 : # Check if val_dl_fold has items
                val_loss, val_acc_current = validate_or_test(model_fold,val_dl_fold,criterion_con,device,config,"Validating") # User's validate returns (loss, acc)
                val_acc = val_acc_current
                scheduler.step(val_acc)
                print(f"E{epoch+1:03d}|TrL:{tr_loss:.3f},TrAcc:{tr_acc:.1%}|ValL:{val_loss:.3f},ValAcc:{val_acc:.1%}|LR:{optimizer.param_groups[0]['lr']:.0e}")
            else:
                print(f"E{epoch+1:03d}|TrL:{tr_loss:.3f},TrAcc:{tr_acc:.1%}| (No validation set for this fold/config)")

            if val_acc > best_val_acc_fold:
                best_val_acc_fold = val_acc
                torch.save(model_fold.state_dict(), save_path)
                es_counter = 0
            elif val_dl_fold and len(val_dl_fold.dataset) > 0 :
                es_counter += 1
                if es_counter >= config['early_stopping_patience']: print(f"Early stopping."); break
            elif not (val_dl_fold and len(val_dl_fold.dataset) > 0) and epoch == config['num_epochs']-1: # No val, save last epoch
                 torch.save(model_fold.state_dict(), save_path)
                 print("No val set, saved model from last epoch.")

        fold_best_val_accuracies.append(best_val_acc_fold)

        if ext_test_labels.size > 0 and os.path.exists(save_path):
            print(f"Testing Fold {fold+1} (Best Val Acc: {best_val_acc_fold:.2%}) on external test set ({ext_test_labels.shape[0]} samples)...")
            microbe_ext_test_norm, _, _ = normalize_features(ext_test_microbe, m_mean, m_std)
            test_ds_fold = PairedDataset((microbe_ext_test_norm, ext_test_labels), (ext_test_fmri, ext_test_labels), train=False)
            test_dl_fold = DataLoader(test_ds_fold, batch_size=config['batch_size'], shuffle=False, num_workers=config.get('num_workers',0))
            
            eval_model = ContrastiveModel(load_pretrained_mlp(config), load_pretrained_gnn(config), config).to(device)
            eval_model.load_state_dict(torch.load(save_path))
            # Use the user's validate function which returns (loss, acc)
            _, tst_acc = validate_or_test(eval_model,test_dl_fold,criterion_con,device,config,"Testing") 
            fold_test_accuracies.append(tst_acc)
            print(f"Fold {fold+1} Test Acc: {tst_acc:.1%}")
        else:
            print(f"Fold {fold+1}: External test set empty or model not saved. Skipping test.")

    print("\n\n--- Overall 10-Fold CV Results ---")
    if fold_best_val_accuracies: print(f"Mean Best Val Acc: {np.mean(fold_best_val_accuracies):.2%} ± {np.std(fold_best_val_accuracies):.2%}")
    if fold_test_accuracies: print(f"Mean Test Acc: {np.mean(fold_test_accuracies):.2%} ± {np.std(fold_test_accuracies):.2%}")
    else: print("No external test accuracies recorded.")


if __name__ == "__main__":
    config = {
        'fmri_hdf5_path': "/home/yangzongxian/xlz/ASD_GCN/main/data2/abide.hdf5",
        'fmri_graph_type': "cc200",
        'microbe_csv_path': "/home/yangzongxian/xlz/ASD_GCN/main/data2/microbe_data.csv",
        'microbe_biom_path': "/home/yangzongxian/xlz/ASD_GCN/main/data2/feature-table.biom",
        'mlp_model_path': "/home/yangzongxian/xlz/ASD_GCN/main/down/sparse_mlp.pth",
        'gnn_model_path': "/home/yangzongxian/xlz/ASD_GCN/main/down/fmri_gnn_best_lr.pth",
        
        'num_rois': 200, 'batch_size': 32, 'num_classes': 2,
        'microbe_feature_dim': 2503,
        'gnn_dropout': 0.4, 'mlp_dropout': 0.6, 'gnn_classifier_dropout': 0.3, 'gnn_gat_dropout': 0.3,
        'contrastive_classifier_dropout': 0.3,
        'gnn_output_dim': 512, 'mlp_feat_extractor_out_dim': 512, # Check SparseMLP output for this
        'mlp_hidden_dims': [2048, 1024, 512], 'mlp_sparsity_lambda': 0.05,
        'contrastive_feat_dim': 128, 'freeze_base_models': True,
        
        'learning_rate': 1e-4, 'weight_decay': 1e-5,
        'contrastive_temp': 0.07, 'contrastive_hard_neg_ratio': 0.2,
        'classification_weight': 0.5, 'adversarial_epsilon': 0.01,
        
        'num_epochs': 100, 'lr_patience': 7, 'early_stopping_patience': 15,
        
        'n_splits_cv': 10, 'random_state_cv': 42,
        'random_state_data': 42,
        # These are for fMRIDataLoader's internal splitting, used to get initial train/val/test DataLoaders
        'fMRI_test_split_ratio': 0.15, 
        'fMRI_val_split_ratio': 0.15, # Original val_size parameter for fMRIDataLoader

        'model_save_dir': "./cv_models_adversarial_final_v4",
        'num_workers': 0, # Set > 0 for parallel data loading
    }
    os.makedirs(config['model_save_dir'], exist_ok=True)

    # Ensure your MLP.py's MicrobeDataLoader loads all data initially for alignment,
    # and its get_loaders() provides the splits used below.
    # The script now relies on these predefined splits being alignable by sample count.
    run_10fold_cv_evaluation(config)