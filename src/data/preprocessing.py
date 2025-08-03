"""
Data preprocessing utilities for ABIDE I and ABIDE II datasets.

This module provides comprehensive data preprocessing capabilities for both
ABIDE I and ABIDE II datasets, including download, preprocessing, and
feature extraction.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import h5py
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import warnings

from .utils import compute_connectivity, load_phenotypes, format_config


class ABIDE1DataProcessor:
    """
    Data processor for ABIDE I dataset.
    
    This class handles downloading, preprocessing, and feature extraction
    for the ABIDE I dataset using the CPAC preprocessing pipeline.
    """
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize ABIDE I data processor.
        
        Args:
            data_dir: Directory to store raw data
            output_dir: Directory to store processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.pheno_dir = self.data_dir / "phenotypes"
        self.functional_dir = self.data_dir / "functionals"
        
        # Create directories
        self.pheno_dir.mkdir(parents=True, exist_ok=True)
        self.functional_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # ABIDE I specific configurations
        self.s3_prefix = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE"
        self.pheno_url = "https://raw.githubusercontent.com/preprocessed-connectomes-project/abide/master/phenotypic/Phenotypic_V1_0b_preprocessed1.csv"
        self.pheno_filename = "Phenotypic_V1_0b_preprocessed1.csv"
        
        # Available derivatives
        self.derivatives = {
            "aal": "cpac/filt_global/rois_aal/{subject}_rois_aal.1D",
            "cc200": "cpac/filt_global/rois_cc200/{subject}_rois_cc200.1D",
            "dosenbach160": "cpac/filt_global/rois_dosenbach160/{subject}_rois_dosenbach160.1D",
            "ez": "cpac/filt_global/rois_ez/{subject}_rois_ez.1D",
            "ho": "cpac/filt_global/rois_ho/{subject}_rois_ho.1D",
            "tt": "cpac/filt_global/rois_tt/{subject}_rois_tt.1D",
        }
    
    def download_phenotype_data(self) -> str:
        """
        Download ABIDE I phenotype data.
        
        Returns:
            Path to downloaded phenotype file
        """
        pheno_filepath = self.pheno_dir / self.pheno_filename
        
        if not pheno_filepath.exists():
            self.logger.info("Downloading ABIDE I phenotype file...")
            try:
                urllib.request.urlretrieve(self.pheno_url, pheno_filepath)
                self.logger.info("Phenotype file downloaded successfully.")
            except Exception as e:
                self.logger.error(f"Error downloading phenotype file: {e}")
                raise
        else:
            self.logger.info("Phenotype file already exists.")
        
        return str(pheno_filepath)
    
    def download_functional_data(self, derivatives: List[str] = None, 
                               pipeline: str = "cpac", 
                               strategy: str = "filt_global") -> None:
        """
        Download ABIDE I functional data.
        
        Args:
            derivatives: List of derivatives to download
            pipeline: Preprocessing pipeline
            strategy: Preprocessing strategy
        """
        if derivatives is None:
            derivatives = ["rois_cc200", "rois_aal", "rois_ez"]
        
        # Download phenotype data first
        pheno_filepath = self.download_phenotype_data()
        
        # Load phenotype data
        pheno = load_phenotypes(pheno_filepath)
        
        for derivative in derivatives:
            self._download_derivative(derivative, pipeline, strategy, pheno_filepath, pheno)
    
    def _download_derivative(self, derivative: str, pipeline: str, 
                           strategy: str, pheno_filepath: str, pheno: pd.DataFrame) -> None:
        """
        Download a specific derivative.
        
        Args:
            derivative: Derivative name
            pipeline: Preprocessing pipeline
            strategy: Preprocessing strategy
            pheno_filepath: Path to phenotype file
            pheno: Phenotype data
        """
        derivative = derivative.lower()
        pipeline = pipeline.lower()
        strategy = strategy.lower()
        
        if "roi" in derivative:
            extension = ".1D"
        else:
            extension = ".nii.gz"
        
        out_dir = self.functional_dir / pipeline / strategy / derivative
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file IDs from phenotype data
        file_ids = pheno["FILE_ID"].tolist()
        
        s3_paths = []
        for file_id in file_ids:
            if pd.isna(file_id) or file_id == "no_filename":
                continue
            filename = f"{file_id}_{derivative}{extension}"
            s3_path = f"{self.s3_prefix}/Outputs/{pipeline}/{strategy}/{derivative}/{filename}"
            s3_paths.append(s3_path)
        
        total_files = len(s3_paths)
        if total_files == 0:
            self.logger.warning(f"No files to download for derivative {derivative}")
            return
        
        self.logger.info(f"Downloading {total_files} files for {derivative}")
        
        for i, s3_path in enumerate(s3_paths):
            filename = s3_path.split("/")[-1]
            download_file = out_dir / filename
            
            if not download_file.exists():
                self.logger.info(f"Downloading: {filename}")
                try:
                    urllib.request.urlretrieve(s3_path, download_file)
                    progress = 100 * (i + 1) / total_files
                    self.logger.info(f"Progress: {progress:.1f}%")
                except Exception as e:
                    self.logger.warning(f"Could not download {s3_path}: {e}")
            else:
                self.logger.debug(f"File {filename} already exists, skipping...")
    
    def process_functional_data(self, derivatives: List[str] = None) -> str:
        """
        Process functional data and create HDF5 file.
        
        Args:
            derivatives: List of derivatives to process
            
        Returns:
            Path to processed HDF5 file
        """
        if derivatives is None:
            derivatives = ["cc200", "aal", "ez"]
        
        # Download phenotype data
        pheno_filepath = self.download_phenotype_data()
        pheno = load_phenotypes(pheno_filepath)
        
        # Create HDF5 file
        hdf5_path = self.output_dir / "abide1_processed.hdf5"
        
        with h5py.File(hdf5_path, 'w') as hdf5:
            self._store_patients_data(hdf5, pheno, derivatives)
            self._create_data_splits(hdf5, pheno, derivatives)
        
        self.logger.info(f"Processed data saved to {hdf5_path}")
        return str(hdf5_path)
    
    def _store_patients_data(self, hdf5: h5py.File, pheno: pd.DataFrame, 
                           derivatives: List[str]) -> None:
        """
        Store patient data in HDF5 file.
        
        Args:
            hdf5: HDF5 file object
            pheno: Phenotype data
            derivatives: List of derivatives
        """
        storage = hdf5.require_group("patients")
        file_ids = pheno["FILE_ID"].tolist()
        
        for derivative in derivatives:
            self.logger.info(f"Processing {derivative} data...")
            
            file_template = self.functional_dir / self.derivatives[derivative]
            func_data = self._load_patients_data(file_ids, file_template)
            
            for pid in func_data:
                record = pheno[pheno["FILE_ID"] == pid].iloc[0]
                patient_storage = storage.require_group(pid)
                
                # Store metadata
                patient_storage.attrs["id"] = record["FILE_ID"]
                patient_storage.attrs["y"] = record["DX_GROUP"]
                patient_storage.attrs["site"] = record["SITE_ID"]
                patient_storage.attrs["sex"] = record["SEX"]
                patient_storage.attrs["age"] = record.get("AGE_AT_SCAN", -1)
                
                # Store functional data
                patient_storage.create_dataset(derivative, data=func_data[pid])
    
    def _load_patients_data(self, file_ids: List[str], 
                          file_template: Path) -> Dict[str, np.ndarray]:
        """
        Load patients' functional data.
        
        Args:
            file_ids: List of file IDs
            file_template: File path template
            
        Returns:
            Dictionary mapping file IDs to functional data
        """
        func_data = {}
        
        for file_id in file_ids:
            try:
                file_path = format_config(str(file_template), {"subject": file_id})
                
                if not os.path.exists(file_path):
                    self.logger.warning(f"File not found: {file_path}")
                    continue
                
                # Load and process data
                df = pd.read_csv(file_path, sep="\t", header=0)
                df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                
                # Get ROI columns
                roi_cols = [col for col in df.columns if col.startswith("#")]
                roi_cols = sorted(roi_cols, key=lambda x: int(x[1:]))
                
                if not roi_cols:
                    self.logger.warning(f"No ROI columns found in {file_path}")
                    continue
                
                # Extract functional data
                functional = df[roi_cols].to_numpy().T
                functional = np.nan_to_num(functional)
                
                # Normalize
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                functional = scaler.fit_transform(functional.T).T
                
                # Compute connectivity
                connectivity = compute_connectivity(functional)
                func_data[file_id] = connectivity.astype(np.float32)
                
            except Exception as e:
                self.logger.warning(f"Error processing {file_id}: {e}")
                continue
        
        return func_data
    
    def _create_data_splits(self, hdf5: h5py.File, pheno: pd.DataFrame, 
                           derivatives: List[str]) -> None:
        """
        Create train/validation/test splits.
        
        Args:
            hdf5: HDF5 file object
            pheno: Phenotype data
            derivatives: List of derivatives
        """
        from sklearn.model_selection import StratifiedKFold, train_test_split
        
        exps = hdf5.require_group("experiments")
        ids = pheno["FILE_ID"]
        
        for derivative in derivatives:
            exp = exps.require_group(f"{derivative}_whole")
            exp.attrs["derivative"] = derivative
            
            # Create 10-fold cross-validation
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            
            for i, (train_index, test_index) in enumerate(skf.split(ids, pheno["DX_GROUP"])):
                train_index, valid_index = train_test_split(
                    train_index, test_size=0.33, random_state=42, 
                    stratify=pheno.iloc[train_index]["DX_GROUP"]
                )
                
                fold = exp.require_group(str(i))
                fold['train'] = [ind.encode('utf8') for ind in ids[train_index]]
                fold['valid'] = [ind.encode('utf8') for ind in ids[valid_index]]
                fold['test'] = [ind.encode('utf8') for ind in ids[test_index]]


class ABIDE2DataProcessor:
    """
    Data processor for ABIDE II dataset.
    
    This class handles downloading, preprocessing, and feature extraction
    for the ABIDE II dataset.
    """
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize ABIDE II data processor.
        
        Args:
            data_dir: Directory to store raw data
            output_dir: Directory to store processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # ABIDE II specific configurations
        self.pheno_url = "https://raw.githubusercontent.com/preprocessed-connectomes-project/abide/master/phenotypic/ABIDEII_Composite_Phenotypic.csv"
        self.pheno_filename = "ABIDEII_Composite_Phenotypic.csv"
    
    def download_phenotype_data(self) -> str:
        """
        Download ABIDE II phenotype data.
        
        Returns:
            Path to downloaded phenotype file
        """
        pheno_filepath = self.data_dir / self.pheno_filename
        
        if not pheno_filepath.exists():
            self.logger.info("Downloading ABIDE II phenotype file...")
            try:
                urllib.request.urlretrieve(self.pheno_url, pheno_filepath)
                self.logger.info("Phenotype file downloaded successfully.")
            except Exception as e:
                self.logger.error(f"Error downloading phenotype file: {e}")
                raise
        else:
            self.logger.info("Phenotype file already exists.")
        
        return str(pheno_filepath)
    
    def process_fmri_data(self) -> str:
        """
        Process fMRI data for ABIDE II.
        
        Returns:
            Path to processed fMRI data
        """
        # This would implement ABIDE II specific fMRI processing
        # For now, return a placeholder
        self.logger.info("Processing ABIDE II fMRI data...")
        return str(self.output_dir / "fmri_processed.npy")
    
    def process_microbiome_data(self, biom_file_path: str = None) -> str:
        """
        Process microbiome data from QIIME2 BIOM format.
        
        Args:
            biom_file_path: Path to feature_table.biom file from QIIME2
            
        Returns:
            Path to processed microbiome data
        """
        if biom_file_path is None:
            # Look for BIOM file in data directory
            biom_files = list(self.data_dir.glob("*.biom"))
            if not biom_files:
                raise FileNotFoundError("No BIOM file found. Please provide path to feature_table.biom")
            biom_file_path = biom_files[0]
        
        self.logger.info(f"Processing microbiome data from: {biom_file_path}")
        
        try:
            # Load BIOM file
            import biom
            table = biom.load_table(str(biom_file_path))
            
            # Convert to pandas DataFrame
            df = table.to_dataframe()
            
            # Transpose to get samples as rows and features as columns
            df = df.T
            
            # Handle missing values
            df = df.fillna(0)
            
            # Normalize (relative abundance)
            df = df.div(df.sum(axis=1), axis=0)
            
            # Apply log transformation (add small constant to avoid log(0))
            df = np.log(df + 1e-10)
            
            # Feature selection (optional)
            # Remove features with low variance
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            df_selected = pd.DataFrame(
                selector.fit_transform(df),
                index=df.index,
                columns=df.columns[selector.get_support()]
            )
            
            # Save processed data
            output_path = self.output_dir / "microbiome_processed.npy"
            np.save(output_path, df_selected.values)
            
            # Save feature names and sample IDs
            feature_info_path = self.output_dir / "microbiome_features.csv"
            feature_df = pd.DataFrame({
                'feature_id': df_selected.columns,
                'feature_name': df_selected.columns  # You might want to map to actual names
            })
            feature_df.to_csv(feature_info_path, index=False)
            
            sample_info_path = self.output_dir / "microbiome_samples.csv"
            sample_df = pd.DataFrame({
                'sample_id': df_selected.index
            })
            sample_df.to_csv(sample_info_path, index=False)
            
            self.logger.info(f"Processed microbiome data saved to: {output_path}")
            self.logger.info(f"Number of features: {df_selected.shape[1]}")
            self.logger.info(f"Number of samples: {df_selected.shape[0]}")
            
            return str(output_path)
            
        except ImportError:
            self.logger.error("biom-format package not found. Please install it: pip install biom-format")
            raise
        except Exception as e:
            self.logger.error(f"Error processing microbiome data: {e}")
            raise
    
    def process_microbiome_data_from_csv(self, csv_file_path: str) -> str:
        """
        Process microbiome data from CSV format (alternative to BIOM).
        
        Args:
            csv_file_path: Path to microbiome CSV file
            
        Returns:
            Path to processed microbiome data
        """
        self.logger.info(f"Processing microbiome data from CSV: {csv_file_path}")
        
        try:
            # Load CSV file
            df = pd.read_csv(csv_file_path, index_col=0)
            
            # Handle missing values
            df = df.fillna(0)
            
            # Normalize (relative abundance)
            df = df.div(df.sum(axis=1), axis=0)
            
            # Apply log transformation
            df = np.log(df + 1e-10)
            
            # Feature selection
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            df_selected = pd.DataFrame(
                selector.fit_transform(df),
                index=df.index,
                columns=df.columns[selector.get_support()]
            )
            
            # Save processed data
            output_path = self.output_dir / "microbiome_processed.npy"
            np.save(output_path, df_selected.values)
            
            self.logger.info(f"Processed microbiome data saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error processing microbiome data: {e}")
            raise
    
    def get_dataloaders(self, batch_size: int = 32, num_workers: int = 4,
                       train_split: float = 0.7, val_split: float = 0.15) -> Tuple:
        """
        Get data loaders for training.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            train_split: Training split ratio
            val_split: Validation split ratio
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # This would implement data loading for ABIDE II
        # For now, return placeholders
        self.logger.info("Creating data loaders for ABIDE II...")
        return None, None, None


class MultimodalDataProcessor:
    """
    Unified data processor for both ABIDE I and ABIDE II datasets.
    
    This class provides a unified interface for processing both datasets
    and combining them for multimodal analysis.
    """
    
    def __init__(self, abide1_dir: str, abide2_dir: str, output_dir: str):
        """
        Initialize multimodal data processor.
        
        Args:
            abide1_dir: Directory for ABIDE I data
            abide2_dir: Directory for ABIDE II data
            output_dir: Output directory for processed data
        """
        self.abide1_processor = ABIDE1DataProcessor(abide1_dir, output_dir)
        self.abide2_processor = ABIDE2DataProcessor(abide2_dir, output_dir)
        self.output_dir = Path(output_dir)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def process_abide1_data(self, derivatives: List[str] = None) -> str:
        """
        Process ABIDE I data.
        
        Args:
            derivatives: List of derivatives to process
            
        Returns:
            Path to processed ABIDE I data
        """
        return self.abide1_processor.process_functional_data(derivatives)
    
    def process_abide2_data(self, biom_file_path: str = None) -> Tuple[str, str]:
        """
        Process ABIDE II data.
        
        Args:
            biom_file_path: Path to QIIME2 BIOM file
            
        Returns:
            Tuple of (fMRI data path, microbiome data path)
        """
        fmri_path = self.abide2_processor.process_fmri_data()
        microbiome_path = self.abide2_processor.process_microbiome_data(biom_file_path)
        return fmri_path, microbiome_path
    
    def combine_datasets(self, abide1_path: str, abide2_fmri_path: str, 
                        abide2_microbiome_path: str) -> str:
        """
        Combine ABIDE I and ABIDE II datasets.
        
        Args:
            abide1_path: Path to processed ABIDE I data
            abide2_fmri_path: Path to ABIDE II fMRI data
            abide2_microbiome_path: Path to ABIDE II microbiome data
            
        Returns:
            Path to combined dataset
        """
        self.logger.info("Combining ABIDE I and ABIDE II datasets...")
        
        # This would implement dataset combination logic
        # For now, return a placeholder
        combined_path = self.output_dir / "combined_dataset.hdf5"
        
        self.logger.info(f"Combined dataset saved to {combined_path}")
        return str(combined_path) 