"""
Data loaders for SynapseBiome ASD-Net.

This module provides data loaders for both ABIDE I and ABIDE II datasets,
supporting multimodal data loading and preprocessing.
"""

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import h5py
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path


class ABIDEDataset(data.Dataset):
    """
    Base dataset class for ABIDE data.
    """
    
    def __init__(self, data_path: str, split: str = 'train', 
                 transform=None, target_transform=None):
        """
        Initialize ABIDE dataset.
        
        Args:
            data_path: Path to HDF5 data file
            split: Data split ('train', 'val', 'test')
            transform: Data transformation
            target_transform: Target transformation
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load data from HDF5 file."""
        with h5py.File(self.data_path, 'r') as f:
            # Get split indices
            split_group = f['experiments']['cc200_whole']['0']
            self.indices = [idx.decode('utf-8') for idx in split_group[self.split]]
            
            # Load patient data
            self.patients = f['patients']
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        patient_id = self.indices[idx]
        patient_data = self.patients[patient_id]
        
        # Get functional data
        functional = patient_data['cc200'][:]
        
        # Get label
        label = patient_data.attrs['y']
        
        # Apply transformations
        if self.transform:
            functional = self.transform(functional)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return functional, label


class MultimodalDataset(data.Dataset):
    """
    Multimodal dataset for ABIDE I and ABIDE II data.
    """
    
    def __init__(self, abide1_path: str, abide2_fmri_path: str, 
                 abide2_microbiome_path: str, split: str = 'train',
                 transform=None, target_transform=None):
        """
        Initialize multimodal dataset.
        
        Args:
            abide1_path: Path to ABIDE I HDF5 file
            abide2_fmri_path: Path to ABIDE II fMRI data
            abide2_microbiome_path: Path to ABIDE II microbiome data
            split: Data split ('train', 'val', 'test')
            transform: Data transformation
            target_transform: Target transformation
        """
        self.abide1_path = abide1_path
        self.abide2_fmri_path = abide2_fmri_path
        self.abide2_microbiome_path = abide2_microbiome_path
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load multimodal data."""
        # Load ABIDE I data
        self.abide1_dataset = ABIDEDataset(
            self.abide1_path, self.split, self.transform, self.target_transform
        )
        
        # Load ABIDE II data
        self.abide2_fmri = np.load(self.abide2_fmri_path)
        self.abide2_microbiome = np.load(self.abide2_microbiome_path)
        
        # Align data (this would need proper implementation based on your data structure)
        self._align_data()
    
    def _align_data(self):
        """Align ABIDE I and ABIDE II data."""
        # This is a placeholder - actual implementation would depend on data structure
        pass
    
    def __len__(self):
        return len(self.abide1_dataset)
    
    def __getitem__(self, idx):
        # Get ABIDE I data
        abide1_fmri, abide1_label = self.abide1_dataset[idx]
        
        # Get ABIDE II data (assuming aligned indices)
        abide2_fmri = torch.tensor(self.abide2_fmri[idx], dtype=torch.float32)
        abide2_microbiome = torch.tensor(self.abide2_microbiome[idx], dtype=torch.float32)
        
        return {
            'abide1_fmri': abide1_fmri,
            'abide2_fmri': abide2_fmri,
            'abide2_microbiome': abide2_microbiome,
            'label': abide1_label
        }


class MultimodalDataLoader:
    """
    Data loader for multimodal ABIDE data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize multimodal data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_abide1_loaders(self, data_path: str, batch_size: int = 32,
                          num_workers: int = 4, shuffle: bool = True) -> Tuple:
        """
        Get ABIDE I data loaders.
        
        Args:
            data_path: Path to ABIDE I HDF5 file
            batch_size: Batch size
            num_workers: Number of worker processes
            shuffle: Whether to shuffle data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = ABIDEDataset(data_path, 'train')
        val_dataset = ABIDEDataset(data_path, 'val')
        test_dataset = ABIDEDataset(data_path, 'test')
        
        # Create data loaders
        train_loader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_multimodal_loaders(self, abide1_path: str, abide2_fmri_path: str,
                              abide2_microbiome_path: str, batch_size: int = 32,
                              num_workers: int = 4, shuffle: bool = True) -> Tuple:
        """
        Get multimodal data loaders.
        
        Args:
            abide1_path: Path to ABIDE I HDF5 file
            abide2_fmri_path: Path to ABIDE II fMRI data
            abide2_microbiome_path: Path to ABIDE II microbiome data
            batch_size: Batch size
            num_workers: Number of worker processes
            shuffle: Whether to shuffle data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = MultimodalDataset(
            abide1_path, abide2_fmri_path, abide2_microbiome_path, 'train'
        )
        val_dataset = MultimodalDataset(
            abide1_path, abide2_fmri_path, abide2_microbiome_path, 'val'
        )
        test_dataset = MultimodalDataset(
            abide1_path, abide2_fmri_path, abide2_microbiome_path, 'test'
        )
        
        # Create data loaders
        train_loader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


class ABIDEDataLoader:
    """
    Unified data loader for ABIDE datasets.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize ABIDE data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_loaders(self, dataset_type: str = 'abide1', **kwargs) -> Tuple:
        """
        Get data loaders for specified dataset type.
        
        Args:
            dataset_type: Dataset type ('abide1', 'abide2', 'multimodal')
            **kwargs: Additional arguments
            
        Returns:
            Tuple of data loaders
        """
        if dataset_type == 'abide1':
            return self._get_abide1_loaders(**kwargs)
        elif dataset_type == 'abide2':
            return self._get_abide2_loaders(**kwargs)
        elif dataset_type == 'multimodal':
            return self._get_multimodal_loaders(**kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _get_abide1_loaders(self, data_path: str, **kwargs) -> Tuple:
        """Get ABIDE I data loaders."""
        multimodal_loader = MultimodalDataLoader(self.config)
        return multimodal_loader.get_abide1_loaders(data_path, **kwargs)
    
    def _get_abide2_loaders(self, fmri_path: str, microbiome_path: str, **kwargs) -> Tuple:
        """Get ABIDE II data loaders."""
        # This would implement ABIDE II specific loading
        # For now, return placeholders
        self.logger.info("ABIDE II data loading not yet implemented")
        return None, None, None
    
    def _get_multimodal_loaders(self, abide1_path: str, abide2_fmri_path: str,
                               abide2_microbiome_path: str, **kwargs) -> Tuple:
        """Get multimodal data loaders."""
        multimodal_loader = MultimodalDataLoader(self.config)
        return multimodal_loader.get_multimodal_loaders(
            abide1_path, abide2_fmri_path, abide2_microbiome_path, **kwargs
        ) 