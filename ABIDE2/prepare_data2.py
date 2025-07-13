#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABIDE II Data Preparation
Usage:
  prepare_data2.py [--folds=N] [--whole] [<derivative> ...]
  prepare_data2.py (-h | --help)

Options:
  -h --help           Show this screen
  --folds=N           Number of folds [default: 10]
  --whole             Prepare data of the whole dataset
  derivative          Derivatives to process (e.g. rois_cc200) [default: rois_cc200]
"""
import numpy as np
import pandas as pd
import os
import random
from docopt import docopt
from functools import partial
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
# Assuming 'utils.py' exists with the required functions
from utils import (load_phenotypes, format_config, run_progress, hdf5_handler)


def compute_connectivity(functional):
    """Computes the Pearson correlation matrix and returns the flattened upper triangle."""
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(functional))
        # Mask the upper triangle and diagonal
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = np.ma.masked_where(mask == 1, mask)
        return np.ma.masked_where(m, corr).compressed()


def load_patient(subject_id, tmpl):
    """Loads and processes a single patient's timeseries data."""
    filepath = format_config(tmpl, {"subject": subject_id})
    try:
        df = pd.read_csv(filepath, sep=r'\s+', header=None) # Use regex for whitespace
        df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        
        # Create ROI names based on the number of columns
        rois = ["#" + str(y) for y in range(df.shape[1])]
        df.columns = rois

        functional = np.nan_to_num(df[rois].to_numpy().T).tolist()
        # Scale each ROI's timeseries
        functional = preprocessing.scale(functional, axis=1)
        functional = compute_connectivity(functional)
        functional = functional.astype(np.float32)

        return subject_id, functional
    except FileNotFoundError:
        print(f"Warning: File not found for subject {subject_id} at {filepath}. Skipping.")
        return subject_id, None
    except Exception as e:
        print(f"Warning: Could not process subject {subject_id}. Error: {e}")
        return subject_id, None


def load_patients(subjs, tmpl, jobs=1):
    """Loads a list of patients in parallel."""
    partial_load_patient = partial(load_patient, tmpl=tmpl)
    msg = "Processing {current} of {total}"
    # Filter out subjects that failed to load (returned None)
    results = dict(filter(lambda item: item[1] is not None, run_progress(partial_load_patient, subjs, message=msg, jobs=jobs)))
    return results


def prepare_folds(hdf5, pheno, derivatives, experiment):
    """Prepares and stores cross-validation folds in the HDF5 file."""
    exps = hdf5.require_group("experiments")
    # Use SUB_ID for ABIDE II
    ids = pheno["SUB_ID"]

    for derivative in derivatives:
        exp = exps.require_group(format_config(
            experiment, {"derivative": derivative}
        ))
        exp.attrs["derivative"] = derivative

        # Use STRAT for stratification if available, otherwise DX_GROUP
        strata_col = 'STRAT' if 'STRAT' in pheno.columns else 'DX_GROUP'
        skf = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=19)

        for i, (train_index, test_index) in enumerate(skf.split(ids, pheno[strata_col])):
            train_index, valid_index = train_test_split(train_index, test_size=0.33, random_state=19, stratify=pheno.iloc[train_index][strata_col])
            
            fold = exp.require_group(str(i))
            fold['train'] = [str(ind).encode('utf8') for ind in ids.iloc[train_index]]
            fold['valid'] = [str(indv).encode('utf8') for indv in ids.iloc[valid_index]]
            fold["test"] = [str(indt).encode('utf8') for indt in ids.iloc[test_index]]


def load_patients_to_file(hdf5, pheno, derivatives):
    """Main function to load patient data and store it in HDF5."""
    # This path should match the output_dir from download_abide2.py
    download_root = "./data/functionals_abide2"
    derivatives_path = {
        "rois_aal": "cpac/filt_global/rois_aal/{subject}_rois_aal.1D",
        "rois_cc200": "cpac/filt_global/rois_cc200/{subject}_rois_cc200.1D",
        "rois_dosenbach160": "cpac/filt_global/rois_dosenbach160/{subject}_rois_dosenbach160.1D",
        "rois_ez": "cpac/filt_global/rois_ez/{subject}_rois_ez.1D",
        "rois_ho": "cpac/filt_global/rois_ho/{subject}_rois_ho.1D",
        "rois_tt": "cpac/filt_global/rois_tt/{subject}_rois_tt.1D",
    }
    
    storage = hdf5.require_group("patients")
    subject_ids = pheno["SUB_ID"].tolist()

    for derivative in derivatives:
        if derivative not in derivatives_path:
            print(f"Error: Derivative '{derivative}' not defined in derivatives_path. Skipping.")
            continue
            
        file_template = os.path.join(download_root, derivatives_path[derivative])
        func_data = load_patients(subject_ids, tmpl=file_template)

        for pid in func_data:
            # Match patient ID (pid) with SUB_ID in the phenotype dataframe
            record = pheno[pheno["SUB_ID"] == int(pid)].iloc[0]
            patient_storage = storage.require_group(str(pid))
            patient_storage.attrs["id"] = str(record["SUB_ID"])
            patient_storage.attrs["y"] = record["DX_GROUP"]
            patient_storage.attrs["site"] = record["SITE_ID"]
            patient_storage.attrs["sex"] = record["SEX"]
            patient_storage.create_dataset(derivative, data=func_data[pid])
  

if __name__ == "__main__":
    random.seed(19)
    np.random.seed(19)

    arguments = docopt(__doc__)

    folds = int(arguments["--folds"])
    # *** IMPORTANT: Path to the ABIDE II phenotype file ***
    pheno_path = "./data/phenotypes/ABIDEII_Composite_Phenotypic.csv"
    pheno = load_phenotypes(pheno_path)

    # *** IMPORTANT: Name of the output HDF5 file ***
    hdf5 = hdf5_handler(bytes("./data/abide2.hdf5", encoding="utf8"), 'a')

    valid_derivatives = list(derivatives_path.keys())
    derivatives = arguments["<derivative>"]
    if not derivatives:
        derivatives = ['rois_cc200']
    
    derivatives = [d for d in derivatives if d in valid_derivatives]

    if "patients" not in hdf5 or not all(d in hdf5["patients"].get(str(pheno["SUB_ID"].iloc[0]), {}) for d in derivatives):
         load_patients_to_file(hdf5, pheno, derivatives)

    if arguments["--whole"]:
        print("Preparing whole dataset")
        prepare_folds(hdf5, pheno, derivatives, experiment="{derivative}_whole")