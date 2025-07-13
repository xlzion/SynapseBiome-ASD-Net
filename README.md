# SynapseBiome ASD-Net: A Multimodal Deep Learning Approach for Autism Spectrum Disorder Diagnosis

This project presents **SynapseBiome ASD-Net**, a multimodal deep learning framework for Autism Spectrum Disorder (ASD) diagnosis. The model leverages a novel approach by integrating functional Magnetic Resonance Imaging (fMRI) data with gut microbiome data to identify robust biomarkers and improve diagnostic accuracy.

## Project Overview

The core of this project is a **Contrastive Learning Model** that learns a shared embedding space for fMRI and microbiome data. It uses a label-aware contrastive loss to pull representations of the same class from different modalities closer together, while pushing dissimilar classes apart. This encourages the model to learn modality-invariant features that are discriminative for ASD.

### Key Architectural Features:

-   **Microbiome Branch**: A Sparse MLP (`SparseMLP`) processes the high-dimensional microbiome feature vectors.
-   **fMRI Branch (`fMRI3DGNN`)**: This is a key innovation of the model. Instead of using a static graph, it features:
    -   A **Dynamic Graph Builder**: For each subject, a unique brain connectivity graph is dynamically constructed from the flattened 40,000-dimensional fMRI connectivity vector.
    -   **Node Feature Enhancement**: Node features are derived from the statistics (mean, std) of the adjacency matrix and enhanced through a small feed-forward network.
    -   **Graph Attention Layers**: The dynamically built graphs are processed using a series of Graph Attention v2 (`GATv2Conv`) layers to learn contextual representations of brain regions.
-   **Contrastive Head**: The outputs from both the MLP and GNN branches are projected into a shared latent space where the `LabelAwareContrastiveLoss` is calculated.
-   **Classification Head**: A final classifier operates on the concatenated features from both modalities to predict the diagnosis.

### Biomarker Discovery

After training, the model can be analyzed to discover potential biomarkers. The `main/biomarker/multimodal_biomarkers1.py` script provides a comprehensive suite for this purpose, including methods like:
-   Saliency Maps (Input Gradients)
-   Integrated Gradients
-   Ensemble methods combining multiple importance scores.
-   Visualization of the most salient ROI-to-ROI connections.

## How to Use

The project contains code for both training the model and for subsequent biomarker analysis.

1.  **Training (`adversarial_new.py`)**:
    -   Configure the data paths in `fMRIDataLoader` and `MicrobeDataLoader`.
    -   Instantiate the `ContrastiveModel`.
    -   Use the `train_contrastive_adversarial` function to train the model.

2.  **Biomarker Analysis (`multimodal_biomarkers1.py`)**:
    -   Load a pre-trained `ContrastiveModel`.
    -   Create a `DataLoader` with your test data using the utility functions.
    -   Run the `main_multimodal_biomarker_analysis` function to extract and save feature importance scores and visualizations.

```python
# Conceptual Snippet for Biomarker Analysis
# from main.biomarker.multimodal_biomarkers1 import main_multimodal_biomarker_analysis, create_biomarker_loader
# from main.biomarker.adversarial_new import ContrastiveModel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1. Load your pre-trained model
# model = ContrastiveModel(...) # Initialize with pre-trained MLP and GNN
# model.load_state_dict(torch.load('path/to/contrastive_model.pth'))
# model.to(device)
# model.eval()

# # 2. Load your data
# # microbe_features, fmri_features, labels should be torch.Tensor objects
# data_loader = create_biomarker_loader(microbe_features, fmri_features, labels)

# # 3. Run analysis
# main_multimodal_biomarker_analysis(
#     model=model,
#     data_loader=data_loader,
#     device=device,
#     output_dir="biomarker_results",
#     fmri_file_path="path/to/your/fmri_data.h5" # For ROI name mapping
# )
```

### Dependencies

-   Python 3.x
-   PyTorch
-   PyTorch Geometric (`torch_geometric`)
-   NumPy
-   Pandas
-   scikit-learn
-   h5py
-   Matplotlib
-   NetworkX
-   Seaborn
-   BIOM-Format (`biom-format`)

## Output

The biomarker analysis script will generate several files in the specified output directory:
-   `microbe_biomarkers_ensemble.csv`: Top microbe features and their importance scores.
-   `fmri_roi_connections_ensemble.csv`: Top fMRI biomarkers mapped to ROI-ROI connection pairs.
-   `fmri_roi_network_ensemble.png`: A network visualization of the most significant ROI connections. 