"""
Utility functions for data processing.

This module provides utility functions for data preprocessing, including
connectivity computation, phenotype loading, and data formatting.
"""

import pandas as pd
import numpy as np
import numpy.ma as ma
from typing import Dict, List, Optional, Union
import logging


def compute_connectivity(functional: np.ndarray) -> np.ndarray:
    """
    Compute functional connectivity matrix from time series data.
    
    Args:
        functional: Time series data [n_timepoints, n_rois]
        
    Returns:
        connectivity: Flattened connectivity matrix
    """
    with np.errstate(invalid="ignore"):
        # Compute correlation matrix
        corr = np.nan_to_num(np.corrcoef(functional))
        
        # Create mask for upper triangle (excluding diagonal)
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        
        # Return flattened lower triangle
        return ma.masked_where(m, corr).compressed()


def load_phenotypes(pheno_path: str) -> pd.DataFrame:
    """
    Load phenotype data from CSV file.
    
    Args:
        pheno_path: Path to phenotype CSV file
        
    Returns:
        pheno: Phenotype DataFrame
    """
    try:
        pheno = pd.read_csv(pheno_path)
        
        # Clean up column names
        pheno.columns = pheno.columns.str.strip()
        
        # Convert DX_GROUP to binary (1: ASD, 2: TD)
        if 'DX_GROUP' in pheno.columns:
            pheno['DX_GROUP'] = pheno['DX_GROUP'].map({1: 1, 2: 0})  # 1: ASD, 0: TD
        
        return pheno
    except Exception as e:
        logging.error(f"Error loading phenotype data: {e}")
        raise


def format_config(template: str, config: Dict) -> str:
    """
    Format string template with configuration values.
    
    Args:
        template: String template with placeholders
        config: Configuration dictionary
        
    Returns:
        formatted_string: Formatted string
    """
    try:
        return template.format(**config)
    except KeyError as e:
        logging.error(f"Missing configuration key: {e}")
        raise


def run_progress(func, items: List, message: str = "Processing", jobs: int = 1):
    """
    Run function with progress tracking.
    
    Args:
        func: Function to run
        items: List of items to process
        message: Progress message
        jobs: Number of parallel jobs
        
    Yields:
        result: Function result
    """
    total = len(items)
    
    for i, item in enumerate(items):
        try:
            result = func(item)
            yield result
            
            if (i + 1) % 10 == 0:
                progress = 100 * (i + 1) / total
                logging.info(f"{message}: {progress:.1f}% ({i+1}/{total})")
                
        except Exception as e:
            logging.warning(f"Error processing item {item}: {e}")
            continue


def hdf5_handler(filepath: str, mode: str = 'r'):
    """
    Create HDF5 file handler.
    
    Args:
        filepath: Path to HDF5 file
        mode: File mode ('r', 'w', 'a')
        
    Returns:
        hdf5: HDF5 file object
    """
    import h5py
    return h5py.File(filepath, mode)


def normalize_data(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize data using specified method.
    
    Args:
        data: Input data
        method: Normalization method ('zscore', 'minmax', 'robust')
        
    Returns:
        normalized_data: Normalized data
    """
    if method == 'zscore':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        return scaler.fit_transform(data)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_data_splits(data: np.ndarray, labels: np.ndarray, 
                      train_ratio: float = 0.7, val_ratio: float = 0.15,
                      random_state: int = 42) -> tuple:
    """
    Create train/validation/test splits.
    
    Args:
        data: Input data
        labels: Labels
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        random_state: Random seed
        
    Returns:
        splits: Tuple of (train_data, val_data, test_data, train_labels, val_labels, test_labels)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        data, labels, test_size=1-train_ratio-val_ratio, 
        random_state=random_state, stratify=labels
    )
    
    # Second split: train vs val
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, 
        test_size=val_ratio/(train_ratio+val_ratio),
        random_state=random_state, stratify=train_val_labels
    )
    
    return (train_data, val_data, test_data, 
            train_labels, val_labels, test_labels)


def extract_roi_features(functional_data: np.ndarray, 
                        feature_type: str = 'connectivity') -> np.ndarray:
    """
    Extract ROI-based features from functional data.
    
    Args:
        functional_data: Functional time series data
        feature_type: Type of features to extract
        
    Returns:
        features: Extracted features
    """
    if feature_type == 'connectivity':
        return compute_connectivity(functional_data)
    
    elif feature_type == 'correlation':
        corr = np.corrcoef(functional_data)
        return corr[np.triu_indices_from(corr, k=1)]
    
    elif feature_type == 'partial_correlation':
        from scipy import stats
        # Compute partial correlations
        n_rois = functional_data.shape[1]
        partial_corr = np.zeros((n_rois, n_rois))
        
        for i in range(n_rois):
            for j in range(n_rois):
                if i != j:
                    # Remove other ROIs for partial correlation
                    other_rois = [k for k in range(n_rois) if k not in [i, j]]
                    if other_rois:
                        partial_corr[i, j] = stats.pearsonr(
                            functional_data[:, i], 
                            functional_data[:, j]
                        )[0]
        
        return partial_corr[np.triu_indices_from(partial_corr, k=1)]
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def validate_data_quality(data: np.ndarray, labels: np.ndarray, 
                         min_samples: int = 10) -> bool:
    """
    Validate data quality.
    
    Args:
        data: Input data
        labels: Labels
        min_samples: Minimum samples per class
        
    Returns:
        is_valid: Whether data meets quality criteria
    """
    # Check for sufficient samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    if np.any(counts < min_samples):
        logging.warning(f"Insufficient samples per class: {dict(zip(unique_labels, counts))}")
        return False
    
    # Check for NaN values
    if np.any(np.isnan(data)):
        logging.warning("Data contains NaN values")
        return False
    
    # Check for infinite values
    if np.any(np.isinf(data)):
        logging.warning("Data contains infinite values")
        return False
    
    return True


def get_dataset_info(data_path: str) -> Dict:
    """
    Get information about dataset.
    
    Args:
        data_path: Path to dataset file
        
    Returns:
        info: Dataset information dictionary
    """
    import h5py
    
    info = {}
    
    with h5py.File(data_path, 'r') as f:
        # Get dataset structure
        info['structure'] = {}
        def get_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                info['structure'][name] = {
                    'shape': obj.shape,
                    'dtype': str(obj.dtype)
                }
            elif isinstance(obj, h5py.Group):
                info['structure'][name] = 'group'
        
        f.visititems(get_structure)
        
        # Get number of patients
        if 'patients' in f:
            info['n_patients'] = len(f['patients'])
        
        # Get experiments
        if 'experiments' in f:
            info['experiments'] = list(f['experiments'].keys())
    
    return info 