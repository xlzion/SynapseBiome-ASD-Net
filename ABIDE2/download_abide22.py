#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ABIDE II fMRIPrep Downloader - Direct from S3 File List

This script downloads fMRIPrep pre-processed functional data from the ABIDE II dataset.
Instead of relying on phenotypic files to construct paths, it reads a pre-generated
list of files that are confirmed to exist on the S3 bucket.

This approach bypasses issues with mismatched subject IDs between phenotypic files
and BIDS-formatted file paths.

Usage:
  download_abide22.py <s3_file_list> <output_dir> [--max_subjects=N]
  download_abide22.py (-h | --help)

Options:
  -h --help                   Show this screen.
  <s3_file_list>              Path to the text file containing the list of S3 file keys.
                              (e.g., the 's3_fmriprep_file_list.txt' we generated).
  <output_dir>                Local directory to download the data to.
  --max_subjects=N            Maximum number of subjects to download (for testing).
"""

from docopt import docopt
import boto3
import os
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError
import re

# --- Global Configuration ---
BUCKET_NAME = 'fcp-indi'

def download_from_list(s3_file_list_path, local_dir, max_subjects, bucket):
    """
    Downloads files directly from a list of S3 keys.
    """
    print(f"\n--- Reading S3 keys from {s3_file_list_path} ---")

    try:
        with open(s3_file_list_path, 'r') as f:
            all_keys = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: S3 file list not found at '{s3_file_list_path}'")
        print("Please run the 'list_s3_files.py' script first to generate this file.")
        return

    # Filter for the specific functional pre-processed files we want
    target_keys = [
        key for key in all_keys 
        if 'ses-1/func' in key and key.endswith('_desc-preproc_bold.nii.gz')
    ]
    
    if not target_keys:
        print("Could not find any target functional files ('...desc-preproc_bold.nii.gz') in the list.")
        return

    if max_subjects:
        # We need to be careful here, as one subject can have multiple runs.
        # We'll group by subject ID and then take the requested number of subjects.
        subjects = {}
        for key in target_keys:
            match = re.search(r'sub-(\d+)', key)
            if match:
                sub_id = match.group(1)
                if sub_id not in subjects:
                    subjects[sub_id] = []
                subjects[sub_id].append(key)
        
        selected_subject_ids = sorted(subjects.keys())[:int(max_subjects)]
        keys_to_download = []
        for sub_id in selected_subject_ids:
            keys_to_download.extend(subjects[sub_id])
        print(f"Found {len(subjects)} unique subjects. Will download for {len(selected_subject_ids)} of them.")
    else:
        keys_to_download = target_keys

    total_files = len(keys_to_download)
    print(f"Found {total_files} files to download.")
    
    local_func_dir = os.path.join(local_dir, 'fmriprep_functional')
    os.makedirs(local_func_dir, exist_ok=True)
    success_count = 0
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    for idx, s3_key in enumerate(keys_to_download):
        filename = os.path.basename(s3_key)
        local_path = os.path.join(local_func_dir, filename)

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
    
    print(f"\nSuccessfully processed {success_count}/{total_files} files.")

def main():
    """Main function to parse arguments and initiate download."""
    arguments = docopt(__doc__)
    s3_list_file = arguments['<s3_file_list>']
    output_dir = arguments['<output_dir>']
    max_subjects = arguments.get('--max_subjects')

    download_from_list(s3_list_file, output_dir, max_subjects, BUCKET_NAME)

    print("\n--- Download process complete ---")

if __name__ == '__main__':
    main()
