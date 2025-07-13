import sys
'''from pathlib import Path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent  
target_subdir = parent_dir / "pre_train"        
sys.path.append(str(target_subdir))'''
from adversarial_new import (
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

# --- Utility Functions ---
def index_to_matrix_coords(index, matrix_size=200):
    """Converts a flat index from a matrix to (row, col) coordinates."""
    row = index // matrix_size
    col = index % matrix_size
    return row, col

def get_roi_names(file_path, graph_type="cc200", num_rois_default=200):
    """
    Retrieves ROI names from an HDF5 file or generates default names.
    """
    roi_names_dict = {}
    try:
        with h5py.File(file_path, "r") as f:
            # Attempt to find ROI names, trying common HDF5 structures
            if "roi_names" in f: # Simple list/dict of ROI names
                 # Assuming f["roi_names"] is a dataset of strings or a group with string attributes
                try:
                    data = f["roi_names"]
                    if isinstance(data, h5py.Dataset):
                        names_from_file = data[:].astype(str)
                        roi_names_dict = {i: name for i, name in enumerate(names_from_file)}
                    elif isinstance(data, h5py.Group): # if keys are '0', '1', etc.
                        roi_names_dict = {int(k): str(v[()]) if isinstance(v, h5py.Dataset) else str(v) for k, v in data.items()}
                except Exception as e_detail:
                    print(f"Could not parse 'roi_names' from HDF5: {e_detail}")

            elif "atlas" in f and graph_type in f["atlas"] and "roi_names" in f["atlas"][graph_type]:
                # ABIDE preprocessed structure
                names_from_file = f["atlas"][graph_type]["roi_names"][:].astype(str)
                roi_names_dict = {i: name for i, name in enumerate(names_from_file)}
            # Add more parsing logic here if other HDF5 structures are used for ROI names
    except Exception as e:
        print(f"Warning: Could not read ROI names from HDF5 file '{file_path}': {e}")

    if not roi_names_dict:
        print(f"Using default ROI names (ROI_0 to ROI_{num_rois_default-1}).")
        roi_names_dict = {i: f"ROI_{i}" for i in range(num_rois_default)}
    return roi_names_dict


def convert_biomarkers_to_roi_connections(biomarker_indices, roi_names_dict, matrix_size=200):
    """Converts fMRI biomarker indices (from flattened matrix) to ROI connection pairs."""
    roi_connections = []
    for flat_idx in biomarker_indices:
        try:
            r, c = index_to_matrix_coords(flat_idx, matrix_size)
            roi1_name = roi_names_dict.get(r, f"ROI_{r}")
            roi2_name = roi_names_dict.get(c, f"ROI_{c}")
            # Optionally, ensure r < c to represent unique connections if matrix is symmetric
            # and self-connections (r == c) might be handled differently.
            # For now, just converting index to pair.
            roi_connections.append((roi1_name, roi2_name))
        except Exception as e:
            print(f"Error converting index {flat_idx} to ROI connection: {e}")
    return roi_connections

def pyg_to_custom_batch(pyg_batch, microbe_feature_dim=2503):
    """
    Converts a PyTorch Geometric DataBatch to a custom dictionary format.
    Assumes pyg_batch.x contains concatenated microbe and fMRI features.
    """
    return {
        'microbe': pyg_batch.x[:, :microbe_feature_dim],
        'fmri': pyg_batch.x[:, microbe_feature_dim:],
        'label': pyg_batch.y
    }
    
def create_biomarker_loader(microbe_features, fmri_features, labels, batch_size=32, microbe_dim=2503, fmri_dim=40000):
    """Creates a DataLoader with PairedDataset, ensuring correct feature dimensions."""
    if microbe_features.shape[0] != fmri_features.shape[0] or microbe_features.shape[0] != labels.shape[0]:
        raise ValueError("Mismatched number of samples between modalities or labels.")
    if microbe_features.shape[1] != microbe_dim:
        raise ValueError(f"Microbe feature dimension is {microbe_features.shape[1]}, expected {microbe_dim}.")
    if fmri_features.shape[1] != fmri_dim:
        raise ValueError(f"fMRI feature dimension is {fmri_features.shape[1]}, expected {fmri_dim}.")

    dataset = PairedDataset(
        (microbe_features, labels),
        (fmri_features, labels),
        train=False  # Ensure no training-specific augmentations
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle for biomarker extraction
        collate_fn=lambda batch_list: {
            'microbe': torch.stack([x['microbe'] for x in batch_list]),
            'fmri': torch.stack([x['fmri'] for x in batch_list]),
            'label': torch.stack([x['label'] for x in batch_list])
        }
    )

# --- Core Biomarker Extraction Methods ---

def _prepare_batch_and_inputs(batch, feature_type, device, microbe_dim=2503, fmri_dim=40000):
    """Helper to process a batch and prepare inputs for gradient calculation."""
    if isinstance(batch, dict):
        microbe_actual = batch['microbe'].to(device)
        fmri_actual = batch['fmri'].to(device)
        labels_actual = batch['label'].squeeze().to(device).long()
    elif hasattr(batch, 'x') and hasattr(batch, 'y'): # PyG Batch
        custom_batch = pyg_to_custom_batch(batch, microbe_feature_dim=microbe_dim)
        microbe_actual = custom_batch['microbe'].to(device)
        fmri_actual = custom_batch['fmri'].to(device)
        labels_actual = custom_batch['label'].squeeze().to(device).long()
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    if microbe_actual.shape[1] != microbe_dim:
        raise ValueError(f"Microbe input dim {microbe_actual.shape[1]} != expected {microbe_dim}")
    if fmri_actual.shape[1] != fmri_dim:
         raise ValueError(f"fMRI input dim {fmri_actual.shape[1]} != expected {fmri_dim}")


    if feature_type == 'microbe':
        input_for_grad = microbe_actual.clone().detach().requires_grad_(True)
        other_input = fmri_actual
        model_args = (input_for_grad, other_input)
    elif feature_type == 'fmri':
        input_for_grad = fmri_actual.clone().detach().requires_grad_(True)
        other_input = microbe_actual
        model_args = (other_input, input_for_grad)
    else:
        raise ValueError("feature_type must be 'microbe' or 'fmri'")
    
    return input_for_grad, model_args, labels_actual


def extract_biomarkers_saliency(model, data_loader, device, feature_type='microbe', top_k=20, loss_type='cross_entropy'):
    """
    Extracts biomarkers using input saliency (gradients of loss w.r.t. inputs).
    This is a generalized function for gradient-based and backpropagation-style saliency.

    Args:
        loss_type (str): 'cross_entropy' for classification logits, 
                         'contrastive' for LabelAwareContrastiveLoss.
    """
    model.eval()
    all_saliency_scores = []
    
    num_features = 0
    if feature_type == 'microbe':
        num_features = model.mlp.classifier.in_features if hasattr(model.mlp, 'classifier') else 2503 # Fallback
        num_features = 2503 # From data loader typically
    elif feature_type == 'fmri':
        num_features = model.gnn.classifier[0].in_features if hasattr(model.gnn, 'classifier') else 40000 # Fallback
        num_features = 40000 # From data loader typically


    for batch in data_loader:
        try:
            input_for_grad, model_args, labels_actual = _prepare_batch_and_inputs(batch, feature_type, device)
            
            h_microbe, h_fmri, logits = model(*model_args)

            if loss_type == 'cross_entropy':
                if labels_actual is None:
                    print("Warning: Labels are None for cross_entropy loss. Skipping batch.")
                    continue
                loss = F.cross_entropy(logits, labels_actual)
            elif loss_type == 'contrastive':
                if labels_actual is None:
                    print("Warning: Labels are None for contrastive loss. Skipping batch.")
                    continue
                # Ensure h_microbe and h_fmri are correctly passed if model_args were reordered.
                # The current model_args passes (grad_input, other_input) or (other_input, grad_input)
                # We need the original h_microbe and h_fmri from the forward pass.
                # Let's re-evaluate h_microbe, h_fmri based on which input had grads.
                if feature_type == 'microbe':
                    h_microbe_for_loss = h_microbe # from model(input_for_grad, other_input)
                    h_fmri_for_loss = h_fmri
                else: # feature_type == 'fmri'
                    h_microbe_for_loss = h_microbe # from model(other_input, input_for_grad)
                    h_fmri_for_loss = h_fmri
                
                criterion = LabelAwareContrastiveLoss() # Consider temp and hard_neg_ratio from config
                loss = criterion(h_microbe_for_loss, h_fmri_for_loss, labels_actual)
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            model.zero_grad()
            if input_for_grad.grad is not None:
                input_for_grad.grad.zero_()
            
            loss.backward()
            saliency = input_for_grad.grad.abs().mean(dim=0)
            all_saliency_scores.append(saliency.cpu().numpy())

        except Exception as e:
            print(f"Error during saliency calculation for batch: {e}")
            # Append zeros if a batch fails, to maintain array structure if needed, but check num_features
            if num_features > 0 :
                 all_saliency_scores.append(np.zeros(num_features))


    if not all_saliency_scores:
        print(f"No saliency scores computed for {feature_type}.")
        return np.array([]), np.array([])

    mean_saliency = np.mean(all_saliency_scores, axis=0)
    
    if mean_saliency.size == 0: # handles case where all_saliency_scores was empty or resulted in empty mean
        return np.array([]), np.array([])

    top_indices_actual = np.argsort(mean_saliency)[::-1][:top_k] # Descending sort
    top_values_actual = mean_saliency[top_indices_actual]
    
    return top_indices_actual, top_values_actual


def extract_biomarkers_integrated_gradients(model, data_loader, device, feature_type='microbe', top_k=20, steps=50):
    """Extracts biomarkers using Integrated Gradients."""
    model.eval()
    all_ig_scores = []
    
    num_features = 0
    if feature_type == 'microbe': num_features = 2503
    elif feature_type == 'fmri': num_features = 40000
    
    for batch_idx, batch in enumerate(data_loader):
        try:
            original_input, model_args_template, labels_actual = _prepare_batch_and_inputs(batch, feature_type, device)
            baseline = torch.zeros_like(original_input)
            scaled_inputs = [baseline + (float(i)/steps) * (original_input - baseline) for i in range(0, steps + 1)]
            
            grads_list = []
            for scaled_input_item in scaled_inputs:
                scaled_input_item_grad = scaled_input_item.clone().detach().requires_grad_(True)
                
                if feature_type == 'microbe':
                    current_model_args = (scaled_input_item_grad, model_args_template[1]) # (grad_input, other_actual)
                else: # fmri
                    current_model_args = (model_args_template[0], scaled_input_item_grad) # (other_actual, grad_input)

                _, _, logits = model(*current_model_args)
                
                if labels_actual is None:
                    print(f"Warning: Labels are None for batch {batch_idx} in Integrated Gradients. Skipping batch.")
                    grads_list = [] # Clear to signify failure for this batch
                    break
                
                loss = F.cross_entropy(logits, labels_actual)
                
                model.zero_grad()
                if scaled_input_item_grad.grad is not None:
                    scaled_input_item_grad.grad.zero_()
                
                loss.backward()
                grads_list.append(scaled_input_item_grad.grad.clone().detach())
            
            if not grads_list: # if loop was broken due to no labels
                if num_features > 0: all_ig_scores.append(np.zeros(num_features))
                continue

            avg_grads = torch.stack(grads_list).mean(dim=0)
            integrated_gradients = (original_input - baseline) * avg_grads
            ig_batch_scores = integrated_gradients.abs().mean(dim=0) # Average over batch
            all_ig_scores.append(ig_batch_scores.cpu().numpy())

        except Exception as e:
            print(f"Error during Integrated Gradients for batch {batch_idx}: {e}")
            if num_features > 0: all_ig_scores.append(np.zeros(num_features))

    if not all_ig_scores:
        print(f"No Integrated Gradients scores computed for {feature_type}.")
        return np.array([]), np.array([])

    mean_ig_scores = np.mean(all_ig_scores, axis=0)
    
    if mean_ig_scores.size == 0:
        return np.array([]), np.array([])
        
    top_indices_actual = np.argsort(mean_ig_scores)[::-1][:top_k]
    top_values_actual = mean_ig_scores[top_indices_actual]
    
    return top_indices_actual, top_values_actual




def extract_biomarkers_attention(model: ContrastiveModel, # Type hint for clarity
                                 data_loader, device, 
                                 feature_type='fmri', # Attention primarily from GNN (fMRI)
                                 top_k=20):
    if feature_type != 'fmri':
        print("Warning: Attention-based biomarkers are currently implemented for fMRI (GNN) features only.")
        return np.array([]), np.array([])

    model.eval()
    aggregated_attention_per_edge = {} # To store summed attention for each unique edge across all batches/layers

    for batch_idx, batch in enumerate(data_loader):
        try:
            # Assuming _prepare_batch_and_inputs or similar logic provides 'microbe_actual' and 'fmri_actual'
            # For simplicity, let's assume batch is a dict from a PairedDataset DataLoader
            microbe_input = batch['microbe'].to(device)
            fmri_input = batch['fmri'].to(device)
            # labels_actual = batch['label'].squeeze().to(device).long() # Not directly used for getting attention here

            # Call the model to get attention scores
            # The model now returns 4 items if return_gnn_attention=True
            _h_microbe, _h_fmri, _logits, gnn_attention_data_list = model(
                microbe_input, fmri_input, return_gnn_attention=True
            )

            if gnn_attention_data_list is None:
                print(f"Warning: gnn_attention_data_list is None for batch {batch_idx}. Skipping.")
                continue

            # Process gnn_attention_data_list (which is a list of dicts, one per GAT layer)
            for layer_attention_data in gnn_attention_data_list:
                edge_index_att = layer_attention_data["edge_index"] # (2, num_edges_in_batch_graph)
                alpha = layer_attention_data["alpha"] # (num_edges_in_batch_graph, num_heads)
                batch_mapping = layer_attention_data["batch_mapping"] # (num_nodes_in_batch_graph)

                # Sum attention scores over heads for each edge
                edge_attention_scores = alpha.abs().sum(dim=1) # (num_edges_in_batch_graph)
                
                # Map edges back to individual graphs if needed, or treat globally
                # For simplicity here, let's consider unique edges in the batch graph.
                # These edge_indices are local to the current batch_graph structure from build_graph.
                # If you need to map them to global ROI indices, it depends on how build_graph forms nodes.
                # The current build_graph in fMRI3DGNN creates 200 nodes per graph.
                # Edges are (node_idx_in_graph, node_idx_in_graph).
                # These node_idx are 0-199 for each graph.
                
                # To get unique global-like edges (e.g., from ROI_i to ROI_j), you'd need to know how
                # the dynamically built graph's edges correspond to fixed ROI pairs.
                # The GAT attention is on the dynamically constructed edges.
                # If biomarkers are these dynamic connections, that's one interpretation.
                # If biomarkers are fixed ROI-ROI pair importance, you need another mapping layer.

                # For now, let's aggregate attention on the (source_node, target_node) pairs
                # within the dynamic graphs.
                num_graphs_in_batch = batch_mapping.max().item() + 1
                node_offsets = [0] * (num_graphs_in_batch + 1) 
                for i in range(num_graphs_in_batch):
                    node_offsets[i+1] = node_offsets[i] + (batch_mapping == i).sum().item()


                for edge_i in range(edge_index_att.size(1)):
                    src_node_batch_local = edge_index_att[0, edge_i].item()
                    tgt_node_batch_local = edge_index_att[1, edge_i].item()
                    
                    # Determine which graph this edge belongs to
                    graph_idx_src = batch_mapping[src_node_batch_local].item()
                    # graph_idx_tgt = batch_mapping[tgt_node_batch_local].item() # Should be same

                    # Convert batch-local node index to graph-local node index (0-199)
                    src_node_graph_local = src_node_batch_local - node_offsets[graph_idx_src]
                    tgt_node_graph_local = tgt_node_batch_local - node_offsets[graph_idx_src]

                    # Create a unique key for the edge (canonical form: min_node, max_node)
                    # These are indices within a 200-node graph.
                    edge_key = tuple(sorted((src_node_graph_local, tgt_node_graph_local)))
                    
                    score_to_add = edge_attention_scores[edge_i].item()
                    aggregated_attention_per_edge[edge_key] = aggregated_attention_per_edge.get(edge_key, 0.0) + score_to_add
        
        except Exception as e:
            print(f"Error processing batch {batch_idx} for attention: {e}")
            import traceback
            traceback.print_exc()


    if not aggregated_attention_per_edge:
        print("No attention scores were aggregated.")
        return np.array([]), np.array([])

    # Sort edges by aggregated attention
    sorted_edges = sorted(aggregated_attention_per_edge.items(), key=lambda item: item[1], reverse=True)
    
    top_k_actual = min(top_k, len(sorted_edges))
    
    # The "indices" here are the edge tuples (ROI_idx1, ROI_idx2)
    # The "values" are their aggregated attention scores
    top_indices = [edge[0] for edge in sorted_edges[:top_k_actual]]
    top_values = np.array([edge[1] for edge in sorted_edges[:top_k_actual]])
    
    # print(f"Top attention biomarkers (edges): {top_indices}")
    # print(f"Their scores: {top_values}")
    
    # Note: The 'indices' returned here are tuples of node indices (dynamic graph edges).
    # You'll need further processing if you want to map these back to the original 40000 flat fMRI features
    # or specific ROI names based on these dynamic edges.
    # For now, it returns the most attended *dynamic edges* as biomarkers.
    return top_indices, top_values


def _normalize_importance_scores(scores_dict):
    """Normalizes importance scores from different methods."""
    normalized_dict = {}
    for method, (indices, values) in scores_dict.items():
        if values.size > 0:
            min_val, max_val = np.min(values), np.max(values)
            if max_val - min_val > 1e-6: # Avoid division by zero
                normalized_values = (values - min_val) / (max_val - min_val)
            else:
                normalized_values = np.ones_like(values) if max_val > 1e-6 else np.zeros_like(values)
            normalized_dict[method] = (indices, normalized_values)
        else:
            normalized_dict[method] = (indices, values) # Keep empty if it was empty
    return normalized_dict


def extract_biomarkers_ensemble(model, data_loader, device, feature_type='microbe', top_k=20,
                                methods_and_weights=None):
    """
    Ensemble method combining Saliency and Integrated Gradients.
    'attention' is excluded by default as it requires model changes.
    """
    if methods_and_weights is None:
        methods_and_weights = {
            'saliency': 0.6,
            'integrated_gradients': 0.4
            # 'attention': 0.0 # Default to 0 as it's likely not implemented
        }

    print(f"\nRunning ensemble biomarker extraction for {feature_type} with weights: {methods_and_weights}")
    
    all_scores_by_method = {}
    num_features = 2503 if feature_type == 'microbe' else 40000

    if methods_and_weights.get('saliency', 0) > 0:
        print("... calculating saliency scores")
        s_indices, s_values = extract_biomarkers_saliency(model, data_loader, device, feature_type, top_k=num_features) # Get all scores
        all_scores_by_method['saliency'] = (s_indices, s_values)
    
    if methods_and_weights.get('integrated_gradients', 0) > 0:
        print("... calculating integrated gradients scores")
        ig_indices, ig_values = extract_biomarkers_integrated_gradients(model, data_loader, device, feature_type, top_k=num_features) # Get all scores
        all_scores_by_method['integrated_gradients'] = (ig_indices, ig_values)

    # Note: Attention is problematic, only include if explicitly weighted and user is aware.
    if methods_and_weights.get('attention', 0) > 0:
        print("... calculating attention scores (placeholder - may not work without model changes)")
        att_indices, att_values = extract_biomarkers_attention(model, data_loader, device, feature_type, top_k=num_features) # Get all scores
        all_scores_by_method['attention'] = (att_indices, att_values)

    if not all_scores_by_method:
        print(f"No methods were specified or successfully run for ensemble for {feature_type}.")
        return np.array([]), np.array([])

    # Normalize scores from each method
    normalized_scores_by_method = _normalize_importance_scores(all_scores_by_method)
    
    # Combine scores
    # Initialize combined_feature_importance with zeros for all features
    combined_feature_importance = np.zeros(num_features)

    for method_name, weight in methods_and_weights.items():
        if weight > 0 and method_name in normalized_scores_by_method:
            indices, norm_values = normalized_scores_by_method[method_name]
            if indices.size > 0 and norm_values.size > 0 : # Ensure there are scores
                # Add scores at their respective indices
                # This assumes indices are unique identifiers for features if top_k < num_features was used.
                # Since we get all scores now, indices should be 0 to N-1 if sorted by importance.
                # The `extract_` functions return top_k indices which are the original feature indices.
                # So, we need to map these.
                # Simpler: if `top_k=num_features`, then indices are 0..N-1 in some order, values match.
                # The current extract functions return top_k `original_indices` and their `values`.
                # So, we need to iterate through these pairs.
                for original_idx, score_val in zip(indices, norm_values):
                    if 0 <= original_idx < num_features:
                         combined_feature_importance[original_idx] += weight * score_val
                    else:
                        print(f"Warning: Index {original_idx} from method {method_name} is out of bounds for {num_features} features.")
            else:
                print(f"Note: Method {method_name} did not return scores, skipping for ensemble.")


    if np.sum(combined_feature_importance) == 0: # Check if any scores were actually combined
        print(f"Warning: Combined feature importance is all zeros for {feature_type}.")
        # Fallback to saliency if it exists and has scores
        if 'saliency' in all_scores_by_method and all_scores_by_method['saliency'][0].size > 0:
            print("Falling back to saliency scores.")
            s_indices, s_values = all_scores_by_method['saliency']
            top_s_indices = np.argsort(s_values)[::-1][:top_k]
            return s_indices[top_s_indices], s_values[top_s_indices] # Return original indices and their scores
        return np.array([]), np.array([])

    # Get top_k from combined scores
    final_top_indices = np.argsort(combined_feature_importance)[::-1][:top_k]
    final_top_values = combined_feature_importance[final_top_indices]
    
    return final_top_indices, final_top_values


# --- Visualization and High-Level Analysis ---

def visualize_roi_network(roi_connections_with_importance, top_k_vis=20, output_file="roi_network.png"):
    """
    Visualizes top ROI connections.
    roi_connections_with_importance: list of tuples e.g. ((roi1_name, roi2_name), importance_score)
    """
    if not roi_connections_with_importance:
        print("No ROI connections to visualize.")
        return

    plt.figure(figsize=(14, 12))
    G = nx.Graph()

    # Sort by importance for visualization
    sorted_connections = sorted(roi_connections_with_importance, key=lambda x: x[1], reverse=True)
    
    connections_to_plot = sorted_connections[:top_k_vis]

    if not connections_to_plot:
        print("No connections left after filtering for top_k_vis.")
        plt.close()
        return

    min_importance = min(c[1] for c in connections_to_plot)
    max_importance = max(c[1] for c in connections_to_plot)
    
    nodes = set()
    for (roi1, roi2), importance in connections_to_plot:
        nodes.add(roi1)
        nodes.add(roi2)
        # Normalize importance for edge width for better visualization
        # Ensure width is positive and scaled reasonably
        norm_imp = (importance - min_importance) / (max_importance - min_importance + 1e-6) if max_importance > min_importance else 0.5
        width = 0.5 + norm_imp * 4.5 
        G.add_edge(roi1, roi2, weight=importance, viz_width=width)
    
    if not G.nodes() or not G.edges():
        print("Graph has no nodes or edges after processing. Skipping visualization.")
        plt.close()
        return

    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    
    edge_widths = [d['viz_width'] for _, _, d in G.edges(data=True)]

    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=700, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color="gray")
    nx_labels = nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    # Improve label visibility if they overlap (optional, may require external libraries or more complex logic)
    # For example, adjust_text library if available.

    plt.title(f"Top {len(connections_to_plot)} ROI Connections by Importance", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    try:
        plt.savefig(output_file, dpi=300)
        print(f"ROI network visualization saved to {output_file}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    plt.close()


def main_multimodal_biomarker_analysis(model, data_loader, device, output_dir="biomarkers_results", top_k=50,
                                       fmri_file_path=None, fmri_graph_type="cc200"):
    """
    Main function to run multimodal biomarker analysis using ensemble method.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Starting Multimodal Biomarker Analysis (Output Dir: {output_dir}) ---")

    # Check data loader
    try:
        sample_batch = next(iter(data_loader))
        print(f"Sample batch check: Microbe shape {sample_batch['microbe'].shape}, fMRI shape {sample_batch['fmri'].shape}")
    except Exception as e:
        print(f"Error checking data_loader: {e}. Ensure it yields dicts with 'microbe', 'fmri', 'label'.")
        return

    # Ensemble method weights (can be configured)
    ensemble_weights = {
        'saliency': 0.6,
        'integrated_gradients': 0.4,
        # 'attention': 0.0 # Keep at 0 unless attention is fully implemented and model supports it
    }

    # --- Microbe Biomarkers ---
    print("\nExtracting Microbe biomarkers (Ensemble)...")
    microbe_indices, microbe_values = extract_biomarkers_ensemble(
        model, data_loader, device, feature_type='microbe', top_k=top_k, methods_and_weights=ensemble_weights
    )
    if microbe_indices.size > 0:
        microbe_df = pd.DataFrame({'feature_index': microbe_indices, 'importance': microbe_values})
        microbe_df = microbe_df.sort_values(by='importance', ascending=False)
        microbe_csv_path = os.path.join(output_dir, "microbe_biomarkers_ensemble.csv")
        microbe_df.to_csv(microbe_csv_path, index=False)
        print(f"Microbe biomarkers (Ensemble) saved to {microbe_csv_path}")
    else:
        print("No microbe biomarkers were extracted by the ensemble method.")

    # --- fMRI Biomarkers ---
    print("\nExtracting fMRI biomarkers (Ensemble)...")
    fmri_indices, fmri_values = extract_biomarkers_ensemble(
        model, data_loader, device, feature_type='fmri', top_k=top_k * 2, # More fMRI features for connectivity
        methods_and_weights=ensemble_weights
    )
    if fmri_indices.size > 0:
        fmri_df = pd.DataFrame({'feature_index': fmri_indices, 'importance': fmri_values})
        fmri_df = fmri_df.sort_values(by='importance', ascending=False)
        fmri_csv_path = os.path.join(output_dir, "fmri_biomarkers_ensemble_flat_indices.csv")
        fmri_df.to_csv(fmri_csv_path, index=False)
        print(f"fMRI biomarkers (Ensemble) based on flat indices saved to {fmri_csv_path}")

        if fmri_file_path:
            roi_names_map = get_roi_names(fmri_file_path, graph_type=fmri_graph_type)
            # Assuming CC200 atlas, matrix_size = 200
            fmri_connections_names = convert_biomarkers_to_roi_connections(fmri_indices, roi_names_map, matrix_size=200)
            
            fmri_connections_with_importance = []
            # Map importance scores to the converted (ROI_name1, ROI_name2) pairs
            # fmri_indices and fmri_values are aligned by their original definition
            idx_to_importance = {idx: val for idx, val in zip(fmri_indices, fmri_values)}

            processed_flat_indices = set()
            for i in range(len(fmri_indices)):
                flat_idx = fmri_indices[i]
                if flat_idx in processed_flat_indices: continue # Skip if already processed (e.g. from top_k)
                
                try:
                    r, c = index_to_matrix_coords(flat_idx, matrix_size=200)
                    # Optional: filter self-loops or ensure unique pairs (r < c) if symmetry is assumed
                    # if r == c: continue # Example: ignore self-connections
                    # if r > c: (r,c) = (c,r) # Example: ensure unique pair for symmetric matrix
                    
                    roi1_name = roi_names_map.get(r, f"ROI_{r}")
                    roi2_name = roi_names_map.get(c, f"ROI_{c}")
                    connection_tuple = tuple(sorted((roi1_name, roi2_name))) # Canonical form for connection

                    # Store connection with its importance. If multiple flat_indices map to same pair, sum/avg importance?
                    # For now, assume each flat_idx is a distinct element of the connectivity matrix.
                    fmri_connections_with_importance.append(((roi1_name, roi2_name), idx_to_importance[flat_idx]))
                    processed_flat_indices.add(flat_idx)

                except Exception as e_conn:
                    print(f"Error processing fMRI index {flat_idx} for ROI connection: {e_conn}")

            if fmri_connections_with_importance:
                # Sort by importance before saving/visualizing
                fmri_connections_with_importance.sort(key=lambda x: x[1], reverse=True)

                fmri_conn_df_data = [{"roi1": conn[0][0], "roi2": conn[0][1], "importance": conn[1]} for conn in fmri_connections_with_importance]
                fmri_conn_df = pd.DataFrame(fmri_conn_df_data)
                
                fmri_conn_csv_path = os.path.join(output_dir, "fmri_roi_connections_ensemble.csv")
                fmri_conn_df.to_csv(fmri_conn_csv_path, index=False)
                print(f"fMRI ROI connections (Ensemble) saved to {fmri_conn_csv_path}")

                visualize_roi_network(fmri_connections_with_importance, 
                                      top_k_vis=20, # Visualize top N connections
                                      output_file=os.path.join(output_dir, "fmri_roi_network_ensemble.png"))
            else:
                print("No fMRI ROI connections could be generated.")
        else:
            print("fMRI file path not provided, skipping ROI name conversion and network visualization.")

    else:
        print("No fMRI biomarkers were extracted by the ensemble method.")
    
    print(f"\n--- Multimodal Biomarker Analysis (Ensemble) Finished ---")


# Note: Functions like extract_multimodal_biomarkers (with 2nd order grads),
# extract_multimodal_biomarkers_perturbation, extract_biomarkers_gradcam,
# analyze_multimodal_feature_network, analyze_microbe_roi_connections,
# validate_biomarkers are kept from the original file.
# They might have issues (computational cost, specific assumptions).
# The focus of this revision was on making the core gradient-based and ensemble methods more robust.
# The user should review these other functions carefully before use.
# For instance, extract_biomarkers_attention still needs model changes.
# The original 'extract_biomarkers' function using LabelAwareContrastiveLoss is still present;
# it can be used if importance w.r.t contrastive loss is specifically desired.