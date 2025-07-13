#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABIDE II fMRIPrep All Sites Downloader

This script reads the ABIDE II phenotype file to identify all available sites,
then downloads the corresponding fMRIPrep data for every subject in every site.
It cross-references a master list of all fMRIPrep files on S3 to ensure only
existing files are downloaded.

Usage:
  download_by_site.py <phenotype_csv> <s3_file_list> <output_dir>
  download_by_site.py (-h | --help)

Options:
  -h --help                   Show this screen.
  <phenotype_csv>             Path to the ABIDE II phenotype CSV file.
  <s3_file_list>              Path to the COMPLETE list of S3 fmriprep files.
  <output_dir>                Local directory to download the data to.
"""

from docopt import docopt
import boto3
import os
import pandas as pd
import re
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError

BUCKET_NAME = 'fcp-indi'

def download_single_site(site_id, all_pheno_df, all_s3_keys, output_dir, bucket):
    """
    Downloads all fMRIPrep files for a single site.
    This function expects the full phenotype dataframe and S3 key list to be pre-loaded.
    """
    print(f"\n--- Processing site: {site_id} ---")

    # 1. Filter the main phenotype dataframe to find subjects for the target site
    site_pheno = all_pheno_df[all_pheno_df['SITE_ID'] == site_id]
    subject_ids = site_pheno['SUB_ID'].astype(str).tolist()
    print(f"Found {len(subject_ids)} subjects in phenotype file.")

    # 2. Filter the master S3 list to get only the files for this site's subjects
    keys_to_download = []
    for s3_key in all_s3_keys:
        # Optimization: Only download the main pre-processed functional file.
        if "func" not in s3_key or not s3_key.endswith('_desc-preproc_bold.nii.gz'):
            continue
            
        match = re.search(r'sub-(\d+)', s3_key)
        if match:
            # We need to be careful with ID formats. Pheno SUB_ID is like '29006'.
            # BIDS ID in the path is also '29006', but can have leading zeros.
            # We'll treat them as integers for robust matching.
            try:
                key_sub_id_int = int(match.group(1))
                pheno_sub_ids_int = [int(sid) for sid in subject_ids]
                if key_sub_id_int in pheno_sub_ids_int:
                    keys_to_download.append(s3_key)
            except ValueError:
                continue # Ignore if subject IDs are not clean integers
    
    total_files = len(keys_to_download)
    if total_files == 0:
        print(f"Warning: Found 0 matching files in the S3 list for site {site_id}. Skipping.")
        return
        
    print(f"Found {total_files} matching functional files (`...desc-preproc_bold.nii.gz`) to download for site '{site_id}'.")

    # 3. Download the filtered list of files
    local_site_dir = os.path.join(output_dir, site_id, 'fmriprep_functional')
    os.makedirs(local_site_dir, exist_ok=True)
    
    success_count = 0
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    for idx, s3_key in enumerate(keys_to_download):
        filename = os.path.basename(s3_key)
        local_path = os.path.join(local_site_dir, filename)

        if not os.path.exists(local_path):
            try:
                print(f"({idx+1}/{total_files}) Downloading: {filename}")
                s3_client.download_file(bucket, s3_key, local_path)
                success_count += 1
            except ClientError as e:
                print(f"({idx+1}/{total_files}) FAILED to download {s3_key}. Error: {e}")
        else:
            print(f"({idx+1}/{total_files}) Already downloaded: {filename}")
            success_count += 1
            
    print(f"Successfully processed {success_count}/{total_files} files for site {site_id}.")
    print(f"Data is located in: {local_site_dir}")

def main():
    """Main function."""
    arguments = docopt(__doc__)
    pheno_csv = arguments['<phenotype_csv>']
    s3_file_list = arguments['<s3_file_list>']
    output_dir = arguments['<output_dir>']

    # 1. Load phenotype CSV and S3 file list ONCE
    try:
        print(f"Loading phenotype data from {pheno_csv}...")
        pheno_df = pd.read_csv(pheno_csv, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: Phenotype file not found at '{pheno_csv}'")
        return

    try:
        print(f"Loading S3 file list from {s3_file_list}...")
        with open(s3_file_list, 'r') as f:
            all_s3_keys = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: S3 file list not found at '{s3_file_list}'")
        return

    # 2. Get all unique site IDs from the phenotype file
    all_sites = sorted(pheno_df['SITE_ID'].unique().tolist())
    print(f"\nFound {len(all_sites)} sites in total. Preparing to download for all.")
    
    # 3. Loop through each site and download its data
    for site_id in all_sites:
        download_single_site(site_id, pheno_df, all_s3_keys, output_dir, BUCKET_NAME)

    print("\n--- All site download processes complete ---")

if __name__ == '__main__':
    main()