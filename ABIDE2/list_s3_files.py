import boto3
from botocore import UNSIGNED
from botocore.client import Config

def list_s3_prefix(bucket_name, prefix, output_file):
    """Lists all objects in an S3 prefix and saves them to a file."""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator('list_objects_v2')
    
    count = 0
    print(f"Listing files from bucket '{bucket_name}' with prefix '{prefix}'...")
    
    try:
        with open(output_file, 'w') as f:
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, MaxKeys=200) # Limit for a quick sample
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        f.write(obj['Key'] + '\n')
                        count += 1
                else:
                    print("No files found in the specified prefix.")
                    break
            # We only need a sample, so we break after the first page.
            if count > 0:
                 print(f"SUCCESS: Wrote a sample of {count} file paths to '{output_file}'.")
            else:
                 print("Could not find any files matching the prefix.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    BUCKET = 'fcp-indi'
    # Now let's explore the contents of the 'fmriprep' directory
    PREFIX_TO_LIST = 'data/Projects/ABIDE2/Outputs/fmriprep/'
    OUTPUT = 's3_fmriprep_file_list.txt'
    
    list_s3_prefix(BUCKET, PREFIX_TO_LIST, OUTPUT)
