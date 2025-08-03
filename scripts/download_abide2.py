#!/usr/bin/env python3
"""
ABIDE II Data Download Script

This script downloads and preprocesses ABIDE II dataset for SynapseBiome ASD-Net.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import ABIDE2DataProcessor
import logging


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('abide2_download.log')
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download ABIDE II dataset")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ABIDE2",
        help="Directory to store ABIDE II data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/ABIDE2",
        help="Directory to store processed data"
    )
    
    parser.add_argument(
        "--biom_file",
        type=str,
        default=None,
        help="Path to microbiome BIOM file (optional)"
    )
    
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="Path to microbiome CSV file (optional)"
    )
    
    parser.add_argument(
        "--process_only",
        action="store_true",
        help="Only process data, skip download"
    )
    
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only download data, skip processing"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ABIDE II data download and processing...")
    
    # Initialize processor
    processor = ABIDE2DataProcessor(args.data_dir, args.output_dir)
    
    try:
        if not args.process_only:
            logger.info("Downloading ABIDE II data...")
            # Note: ABIDE II data download requires manual setup
            # as it involves S3 access and large file downloads
            logger.warning("ABIDE II data download requires manual setup.")
            logger.info("Please ensure ABIDE II data is available in the data directory.")
        
        if not args.download_only:
            logger.info("Processing ABIDE II data...")
            
            # Process fMRI data
            fmri_path = processor.process_fmri_data()
            logger.info(f"fMRI processing completed! Data saved to: {fmri_path}")
            
            # Process microbiome data
            if args.biom_file:
                microbiome_path = processor.process_microbiome_data(args.biom_file)
                logger.info(f"Microbiome processing completed! Data saved to: {microbiome_path}")
            elif args.csv_file:
                microbiome_path = processor.process_microbiome_data_from_csv(args.csv_file)
                logger.info(f"Microbiome processing completed! Data saved to: {microbiome_path}")
            else:
                logger.info("No microbiome data provided. Skipping microbiome processing.")
        
        logger.info("ABIDE II data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ABIDE II data preparation: {e}")
        raise


if __name__ == "__main__":
    main() 