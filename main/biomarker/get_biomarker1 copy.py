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
import multimodal_biomarkers as mmb



# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define output directory
output_dir_main = "biomarker_analysis_results"
os.makedirs(output_dir_main, exist_ok=True)


# --- Main Script ---
if __name__ == "__main__":
    # Define data paths 
    fmri_hdf5_path = "/Users/xlzion/Desktop/ASD/ASD_GCN/main/data/abide.hdf5"
    fmri_graph_type = "cc200"
    microbe_csv_path = "/Users/xlzion/Desktop/ASD/ASD_GCN/main/data/microbe_data.csv"
    microbe_biom_path = "/Users/xlzion/Desktop/ASD/ASD_GCN/main/data/feature-table.biom"
    
    # Define model path 
    trained_model_path = "/Users/xlzion/Desktop/ASD/ASD_GCN/main/contrastive1.pth"

    # 1. Load Pre-trained Multimodal Model
    print(f"Loading pre-trained multimodal model from: {trained_model_path}")
    try:
        # These load_pretrained_X functions are from adversarial10.py
        # Their internal paths must be correct.
        mlp_base = load_pretrained_mlp().to(device)
        gnn_base = load_pretrained_gnn().to(device)
        
        model = ContrastiveModel(mlp_base, gnn_base).to(device)
        model.load_state_dict(torch.load(trained_model_path, map_location=device))
        model.eval()
        print("Multimodal model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {trained_model_path}. Please check.")
        exit(1)
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit(1)

    # 2. Load and Prepare Data for Biomarker Analysis
    print("\nLoading and preparing data for biomarker analysis...")
    try:
        # Using fMRIDataLoader and MicrobeDataLoader from adversarial10.py
        fmri_data_loader_obj = fMRIDataLoader(file_path=fmri_hdf5_path, graph_type=fmri_graph_type, batch_size=32)
        microbe_data_loader_obj = MicrobeDataLoader(csv_path=microbe_csv_path, biom_path=microbe_biom_path, batch_size=32)

        # Using combined train + validation data for biomarker analysis pool
        # Data loaders from adversarial10 give PyG batches (fMRI) or (tensor,label) (microbe)
        fmri_train_loader_raw = fmri_data_loader_obj.get_dataloaders()["train"]
        microbe_train_loader_raw = microbe_data_loader_obj.get_loaders()[0]
        fmri_val_loader_raw = fmri_data_loader_obj.get_dataloaders()["valid"]
        microbe_val_loader_raw = microbe_data_loader_obj.get_loaders()[1]

        # Extracting features (numpy arrays)
        # Ensure extract_graph_features from adversarial10.py is correct (it had a minor gnn_model.to(device) issue)
        fmri_train_features_np, fmri_train_labels_np = extract_graph_features(fmri_train_loader_raw)
        fmri_val_features_np, fmri_val_labels_np = extract_graph_features(fmri_val_loader_raw)
        microbe_train_features_np, microbe_train_labels_np = extract_microbe_features(microbe_train_loader_raw)
        microbe_val_features_np, microbe_val_labels_np = extract_microbe_features(microbe_val_loader_raw)

        combined_microbe_features_np = np.concatenate([microbe_train_features_np, microbe_val_features_np])
        combined_microbe_labels_np = np.concatenate([microbe_train_labels_np, microbe_val_labels_np])
        combined_fmri_features_np = np.concatenate([fmri_train_features_np, fmri_val_features_np])
        # Assuming fMRI labels are consistent with microbe labels after pairing
        
        print(f"Combined microbe features shape: {combined_microbe_features_np.shape}")
        print(f"Combined fMRI features shape: {combined_fmri_features_np.shape}")
        print(f"Combined labels shape: {combined_microbe_labels_np.shape}")


        # Create the specific DataLoader format needed by multimodal_biomarkers.py functions
        # using the utility from multimodal_biomarkers.py
        biomarker_analysis_loader = mmb.create_biomarker_loader(
            combined_microbe_features_np,
            combined_fmri_features_np,
            combined_microbe_labels_np, # Assuming these are the paired labels
            batch_size=32 # Or your preferred batch size for inference
        )
        print("Data loaded and prepared into final biomarker_analysis_loader.")
    except Exception as e:
        print(f"Error loading or preparing data: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # 3. Run Biomarker Extraction using main analysis function from the module
    # This function internally calls the ensemble method.
    print("\n--- Running Main Multimodal Biomarker Analysis (Ensemble based) ---")
    mmb.main_multimodal_biomarker_analysis(
        model=model,
        data_loader=biomarker_analysis_loader,
        device=device,
        output_dir=os.path.join(output_dir_main, "ensemble_results"),
        top_k=50, # Top-k features to report
        fmri_file_path=fmri_hdf5_path, # Pass path for ROI name loading
        fmri_graph_type=fmri_graph_type
    )

    # 4. Optionally, run specific biomarker methods if desired for comparison
    # Example: Saliency (Gradient-based)
    print("\n--- Running Specific Biomarker Method: Saliency (Gradient-based) ---")
    saliency_output_dir = os.path.join(output_dir_main, "saliency_results")
    os.makedirs(saliency_output_dir, exist_ok=True)
    
    top_k_specific = 50
    microbe_indices_sal, microbe_values_sal = mmb.extract_biomarkers_saliency(
        model, biomarker_analysis_loader, device, feature_type='microbe', top_k=top_k_specific, loss_type='cross_entropy'
    )
    if microbe_indices_sal.size > 0:
        pd.DataFrame({'feature_index': microbe_indices_sal, 'importance': microbe_values_sal}).to_csv(
            os.path.join(saliency_output_dir, "microbe_biomarkers_saliency.csv"), index=False)
        print(f"Microbe Saliency biomarkers saved to {saliency_output_dir}")

    fmri_indices_sal, fmri_values_sal = mmb.extract_biomarkers_saliency(
        model, biomarker_analysis_loader, device, feature_type='fmri', top_k=top_k_specific * 2, loss_type='cross_entropy'
    )
    if fmri_indices_sal.size > 0:
        pd.DataFrame({'feature_index': fmri_indices_sal, 'importance': fmri_values_sal}).to_csv(
            os.path.join(saliency_output_dir, "fmri_biomarkers_saliency_flat.csv"), index=False)
        print(f"fMRI Saliency biomarkers (flat indices) saved to {saliency_output_dir}")
        
        roi_names_map_sal = mmb.get_roi_names(fmri_hdf5_path, graph_type=fmri_graph_type)
        fmri_connections_names_sal = mmb.convert_biomarkers_to_roi_connections(fmri_indices_sal, roi_names_map_sal)
        # Create list of ((roi1, roi2), importance) for visualization
        fmri_connections_with_importance_sal = []
        idx_to_importance_sal = {idx: val for idx, val in zip(fmri_indices_sal, fmri_values_sal)}

        for i in range(len(fmri_indices_sal)):
            flat_idx = fmri_indices_sal[i]
            try:
                r, c = mmb.index_to_matrix_coords(flat_idx, matrix_size=200)
                roi1_name = roi_names_map_sal.get(r, f"ROI_{r}")
                roi2_name = roi_names_map_sal.get(c, f"ROI_{c}")
                fmri_connections_with_importance_sal.append(((roi1_name, roi2_name), idx_to_importance_sal[flat_idx]))
            except Exception as e_conn_sal:
                 print(f"Error processing fMRI index {flat_idx} for ROI connection (saliency): {e_conn_sal}")
        
        if fmri_connections_with_importance_sal:
            fmri_connections_with_importance_sal.sort(key=lambda x: x[1], reverse=True)
            fmri_conn_df_sal_data = [{"roi1": conn[0][0], "roi2": conn[0][1], "importance": conn[1]} for conn in fmri_connections_with_importance_sal]
            pd.DataFrame(fmri_conn_df_sal_data).to_csv(os.path.join(saliency_output_dir, "fmri_roi_connections_saliency.csv"), index=False)
            mmb.visualize_roi_network(fmri_connections_with_importance_sal, top_k_vis=20,
                                   output_file=os.path.join(saliency_output_dir, "fmri_roi_network_saliency.png"))


    # Example: Integrated Gradients
    print("\n--- Running Specific Biomarker Method: Integrated Gradients ---")
    ig_output_dir = os.path.join(output_dir_main, "integrated_gradients_results")
    os.makedirs(ig_output_dir, exist_ok=True)
    
    microbe_indices_ig, microbe_values_ig = mmb.extract_biomarkers_integrated_gradients(
        model, biomarker_analysis_loader, device, feature_type='microbe', top_k=top_k_specific
    )
    if microbe_indices_ig.size > 0:
        pd.DataFrame({'feature_index': microbe_indices_ig, 'importance': microbe_values_ig}).to_csv(
            os.path.join(ig_output_dir, "microbe_biomarkers_ig.csv"), index=False)
        print(f"Microbe Integrated Gradients biomarkers saved to {ig_output_dir}")

    # (Add fMRI IG extraction, conversion, and visualization similarly if desired)


    # Example: Attention-based (will print warning and return empty if model not modified)
    print("\n--- Attempting Specific Biomarker Method: Attention-based ---")
    attention_output_dir = os.path.join(output_dir_main, "attention_results")
    os.makedirs(attention_output_dir, exist_ok=True)
    microbe_indices_att, microbe_values_att = mmb.extract_biomarkers_attention(
        model, biomarker_analysis_loader, device, feature_type='microbe', top_k=top_k_specific
    )
    if microbe_indices_att.size > 0:
        pd.DataFrame({'feature_index': microbe_indices_att, 'importance': microbe_values_att}).to_csv(
            os.path.join(attention_output_dir, "microbe_biomarkers_attention.csv"), index=False)
        print(f"Microbe Attention-based biomarkers saved to {attention_output_dir} (if any were produced).")
    else:
        print("No microbe attention-based biomarkers produced (as expected if model not modified).")


    print(f"\n\nAll requested biomarker analyses complete. Results are in '{output_dir_main}'.")