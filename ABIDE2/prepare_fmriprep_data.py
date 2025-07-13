#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABIDE II fMRIPrep Data Preparation

This script processes downloaded fMRIPrep data. For each subject, it:
1. Extracts ROI timeseries using a given NIfTI atlas.
2. Computes a Pearson correlation connectivity matrix.
3. Stores the flattened matrix and subject metadata in an HDF5 file.

Usage:
  prepare_fmriprep_data.py <data_dir> --atlas_name=NAME --atlas_file=PATH [--pheno_file=PATH]
  prepare_fmriprep_data.py (-h | --help)

Options:
  -h --help                   Show this screen.
  <data_dir>                  Path to the directory where ABIDE II sites data is stored.
  --atlas_name=NAME           Name of the atlas for HDF5 storage (e.g., rois_cc200). [default: rois_cc200]
  --atlas_file=PATH           Path to the NIfTI atlas file (e.g., /path/to/cc200.nii.gz).
  --pheno_file=PATH           Path to the ABIDE II Phenotype CSV file. [default: ./data/phenotypes/ABIDEII_Composite_Phenotypic.csv]
"""
import numpy as np
import pandas as pd
import os
import random
import glob
from docopt import docopt
from functools import partial
from nilearn.maskers import NiftiLabelsMasker
from utils import (load_phenotypes, run_progress, hdf5_handler)

def compute_connectivity(timeseries):
    """Computes the Pearson correlation matrix and returns the flattened upper triangle."""
    with np.errstate(invalid="ignore"):
        # Timeseries shape is (n_timepoints, n_rois). We want to correlate ROIs.
        # np.corrcoef by default expects each row to be a variable (ROI).
        # So we transpose the timeseries matrix.
        corr_matrix = np.nan_to_num(np.corrcoef(timeseries.T))
        # Create a mask for the upper triangle (excluding the diagonal)
        upper_triangle_mask = np.triu_indices(corr_matrix.shape[0], k=1)
        # Flatten the upper triangle
        return corr_matrix[upper_triangle_mask]

def load_patient_fmriprep(subject_info, data_dir, atlas_file):
    """
    Loads a patient's fMRIPrep data, extracts timeseries, and computes connectivity.
    
    Args:
        subject_info (tuple): A tuple containing (subject_id, site_id).
        data_dir (str): The root directory containing the downloaded site data.
        atlas_file (str): Path to the NIfTI atlas file.

    Returns:
        tuple: A tuple (subject_id, connectivity_matrix) or (subject_id, None) on failure.
    """
    subject_id, site_id = subject_info
    
    # Construct the search path for the subject's functional file
    subject_dir = os.path.join(data_dir, site_id, 'fmriprep_functional')
    
    if not os.path.isdir(subject_dir):
        # This can happen if a site from the phenotype file was not downloaded
        # print(f"Info: Directory not found for site {site_id}. Skipping subjects from this site.")
        return subject_id, None

    # Use glob to find the file, handling potential variations in BIDS filenames
    search_pattern = os.path.join(subject_dir, f"sub-{subject_id}*_desc-preproc_bold.nii.gz")
    found_files = glob.glob(search_pattern)

    if not found_files:
        # print(f"Warning: File not found for subject {subject_id} in {site_id}. Skipping.")
        return subject_id, None
    
    filepath = found_files[0]
    if len(found_files) > 1:
        print(f"Warning: Found multiple functional files for subject {subject_id}. Using first one: {filepath}")

    try:
        # Create a masker object to apply the atlas to the fMRI data.
        # This object will extract one time series for each region in the atlas.
        # standardize=True performs z-scoring on the timeseries.
        masker = NiftiLabelsMasker(labels_img=atlas_file, standardize=True, memory='nilearn_cache', verbose=0)
        
        # masker.fit_transform does all the work:
        # 1. Loads the fMRI NIfTI data
        # 2. Resamples the atlas to match the fMRI data's space
        # 3. Extracts the mean signal from each atlas region
        timeseries = masker.fit_transform(filepath)
        
        # Check if timeseries extraction was successful
        if timeseries.shape[0] == 0:
             print(f"Warning: Timeseries extraction yielded 0 timepoints for subject {subject_id}. Skipping.")
             return subject_id, None

        # Compute connectivity from the extracted timeseries
        connectivity_matrix = compute_connectivity(timeseries)
        
        return subject_id, connectivity_matrix.astype(np.float32)

    except Exception as e:
        print(f"Warning: Could not process subject {subject_id}. Error: {e}")
        return subject_id, None

def load_patients_to_file(hdf5, pheno, atlas_name, atlas_file, data_dir):
    """Main function to load patient data and store it in HDF5."""
    storage = hdf5.require_group("patients")
    
    # Create a list of tuples (SUB_ID, SITE_ID) to process
    subjects_to_process = list(zip(pheno["SUB_ID"].astype(str), pheno["SITE_ID"]))

    # Create a partial function with the fixed arguments for multiprocessing
    partial_load_patient = partial(
        load_patient_fmriprep,
        data_dir=data_dir,
        atlas_file=atlas_file
    )

    # Run the processing with a progress bar
    msg = "Processing {current} of {total} subjects"
    # Filter out subjects that failed to process (returned None)
    results = dict(filter(lambda item: item[1] is not None, run_progress(partial_load_patient, subjects_to_process, message=msg)))
    
    print(f"\nSuccessfully processed {len(results)} subjects.")
    
    # Store results in the HDF5 file
    print("Storing processed data in HDF5 file...")
    for pid, func_data in results.items():
        # Find the corresponding record in the phenotype dataframe
        record = pheno[pheno["SUB_ID"] == int(pid)].iloc[0]
        
        # Create a group for the patient, named by their ID
        patient_storage = storage.require_group(str(pid))
        
        # Store metadata as attributes
        patient_storage.attrs["id"] = str(record["SUB_ID"])
        patient_storage.attrs["y"] = record["DX_GROUP"]
        patient_storage.attrs["site"] = record["SITE_ID"]
        patient_storage.attrs["sex"] = record["SEX"]
        
        # Store the connectivity data in a dataset named after the atlas
        patient_storage.create_dataset(atlas_name, data=func_data)
        
if __name__ == "__main__":
    random.seed(19)
    np.random.seed(19)

    arguments = docopt(__doc__)

    data_dir = arguments['<data_dir>']
    pheno_path = arguments['--pheno_file']
    atlas_name = arguments['--atlas_name']
    atlas_file = arguments['--atlas_file']

    if not atlas_file or not os.path.exists(atlas_file):
        print("\nError: You must provide a valid path to an atlas file using --atlas_file.")
        print("Example: --atlas_file=/path/to/your/cc200_atlas.nii.gz\n")
        exit(1)
        
    if not os.path.isdir(data_dir):
        print(f"\nError: The data directory '{data_dir}' does not exist.")
        print("This should be the directory containing the downloaded site folders.\n")
        exit(1)

    pheno = load_phenotypes(pheno_path)
    # Ensure SUB_ID is integer for matching later
    pheno['SUB_ID'] = pheno['SUB_ID'].astype(int)

    hdf5 = hdf5_handler(bytes("./data/abide2_fmriprep.hdf5", encoding="utf8"), 'a')

    # Check if data for this atlas already exists before starting the whole process
    # We check the first subject's record as a proxy.
    first_subject_id = str(pheno["SUB_ID"].iloc[0])
    if "patients" in hdf5 and first_subject_id in hdf5["patients"] and atlas_name in hdf5["patients"][first_subject_id]:
        print(f"Data for atlas '{atlas_name}' already exists in the HDF5 file. Nothing to do.")
    else:
        print(f"Data for atlas '{atlas_name}' not found. Starting processing...")
        load_patients_to_file(hdf5, pheno, atlas_name, atlas_file, data_dir)

    print("\n--- HDF5 preparation complete ---")
    print(f"Data saved in: ./data/abide2_fmriprep.hdf5")