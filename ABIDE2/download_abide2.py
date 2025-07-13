# download_abide2.py
import argparse
import os
import pandas as pd
import boto3
import urllib.request
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download ABIDE II data from the public S3 bucket."
    )
    parser.add_argument(
        "--phenotypic_file",
        type=str,
        default="ABIDEII_Composite_Phenotypic.csv",
        help="Path to the ABIDE II composite phenotypic CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the downloaded files.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="raw",
        choices=["raw", "fmriprep"],
        help="Data processing pipeline to download. 'raw' for unprocessed BIDS data, 'fmriprep' for preprocessed outputs.",
    )
    parser.add_argument(
        "--derivative",
        type=str,
        default="func",
        choices=["func", "anat"],
        help="Type of imaging data to download ('func' or 'anat').",
    )
    parser.add_argument(
        "--dx_group",
        type=str,
        default="all",
        choices=["ASD", "TC", "all"],
        help="Diagnostic group to download: 'ASD' (Autism), 'TC' (Typical Control), or 'all'.",
    )
    parser.add_argument(
        "--min_age",
        type=float,
        default=None,
        help="Minimum age of subjects to include.",
    )
    parser.add_argument(
        "--max_age",
        type=float,
        default=None,
        help="Maximum age of subjects to include.",
    )
    parser.add_argument(
        "--sex",
        type=str,
        default="all",
        choices=["M", "F", "all"],
        help="Sex of subjects to include: 'M' (Male), 'F' (Female), or 'all'.",
    )
    return parser.parse_args()

def construct_s3_key(pipeline, derivative, site_id, sub_id):
    """Constructs the S3 object key based on the specified pipeline and derivative."""
    if pipeline == "raw":
        # Raw data is in RawDataBIDS, and path doesn't include site_id
        if derivative == "func":
            return f"data/Projects/ABIDE2/RawDataBIDS/sub-{sub_id}/ses-1/func/sub-{sub_id}_ses-1_task-rest_run-1_bold.nii.gz"
        elif derivative == "anat":
            return f"data/Projects/ABIDE2/RawDataBIDS/sub-{sub_id}/ses-1/anat/sub-{sub_id}_ses-1_T1w.nii.gz"
    elif pipeline == "fmriprep":
        # Note: fmriprep paths might vary slightly. This is a common structure.
        if derivative == "func":
            return f"data/Projects/ABIDE2/Outputs/fmriprep/sub-{sub_id}/ses-1/func/sub-{sub_id}_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        elif derivative == "anat":
            return f"data/Projects/ABIDE2/Outputs/fmriprep/sub-{sub_id}/ses-1/anat/sub-{sub_id}_ses-1_desc-preproc_T1w.nii.gz"
    
    # Return None if the combination is not supported
    return None

def download_subject_data(s3_client, bucket_name, s3_key, local_path):
    """Downloads a single file from S3 if it doesn't already exist locally."""
    if os.path.exists(local_path):
        # print(f"File already exists: {local_path}")
        return True, "Exists"

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True, "Success"
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False, "NotFound"
        else:
            return False, f"ClientError: {e}"
    except Exception as e:
        return False, f"Exception: {e}"

def main():
    """Main function to filter subjects and orchestrate downloads."""
    args = parse_arguments()

    # --- Phenotype File Handling ---
    pheno_path = args.phenotypic_file
    
    # If the provided path (or default filename) doesn't exist, check default dir or download.
    if not os.path.exists(pheno_path):
        default_pheno_dir = "data/phenotypes"
        default_pheno_path = os.path.join(default_pheno_dir, "ABIDEII_Composite_Phenotypic.csv")
        
        if os.path.exists(default_pheno_path):
            print(f"Phenotypic file not found at '{pheno_path}', but found at default location '{default_pheno_path}'.")
            pheno_path = default_pheno_path
        else:
            print(f"Phenotypic file not found. Attempting to download to '{default_pheno_path}'...")
            os.makedirs(default_pheno_dir, exist_ok=True)
            pheno_url = 'https://raw.githubusercontent.com/preprocessed-connectomes-project/abide/master/phenotypic/ABIDEII_Composite_Phenotypic.csv'
            try:
                urllib.request.urlretrieve(pheno_url, default_pheno_path)
                print("Download successful.")
                pheno_path = default_pheno_path
            except Exception as e:
                print(f"FATAL: Failed to download phenotypic file: {e}")
                return

    # --- Load and Filter Phenotypic Data ---
    try:
        pheno_df = pd.read_csv(pheno_path, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: Phenotypic file not found at '{pheno_path}'")
        return

    # Apply filters
    filtered_df = pheno_df.copy()
    # Clean column names to handle potential extra spacing
    filtered_df.columns = [col.strip() for col in filtered_df.columns]

    if args.dx_group != "all":
        dx_code = 1 if args.dx_group == "ASD" else 2
        filtered_df = filtered_df[filtered_df['DX_GROUP'] == dx_code]
    
    if args.min_age is not None:
        filtered_df = filtered_df[filtered_df['AGE_AT_SCAN'] >= args.min_age]
 
    if args.max_age is not None:
        filtered_df = filtered_df[filtered_df['AGE_AT_SCAN'] <= args.max_age]
         
    if args.sex != "all":
        sex_code = 1 if args.sex == "M" else 2
        filtered_df = filtered_df[filtered_df['SEX'] == sex_code]
 
    if filtered_df.empty:
        print("No subjects found matching the specified criteria.")
        return
        
    print(f"Found {len(filtered_df)} subjects matching criteria. Starting download...")

    # --- S3 Download Logic ---
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket_name = 'fcp-indi'
    
    success_count = 0
    not_found_count = 0
    exist_count = 0
    error_count = 0
    
    # Use tqdm for a progress bar
    for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0], desc="Downloading files"):
        site_id = row['SITE_ID']
        sub_id = row['SUB_ID']
        
        s3_key = construct_s3_key(args.pipeline, args.derivative, site_id, sub_id)
        
        if s3_key is None:
            print(f"Warning: Unsupported pipeline/derivative combination for subject {sub_id}. Skipping.")
            error_count += 1
            continue
            
        local_path = os.path.join(args.output_dir, s3_key.split('Projects/ABIDE2/')[-1])
        
        success, status = download_subject_data(s3_client, bucket_name, s3_key, local_path)
        
        if success:
            if status == "Success":
                success_count += 1
            elif status == "Exists":
                exist_count += 1
        else:
            if status == "NotFound":
                not_found_count += 1
            else:
                print(f"Error downloading {s3_key}: {status}")
                error_count += 1

    print("\n--- Download Summary ---")
    print(f"Successfully downloaded: {success_count}")
    print(f"Already existed: {exist_count}")
    print(f"Files not found on S3 (404): {not_found_count}")
    print(f"Other errors: {error_count}")
    print("------------------------")

if __name__ == "__main__":
    main()