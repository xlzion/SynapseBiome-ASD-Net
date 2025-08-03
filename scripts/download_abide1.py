#!/usr/bin/env python3
"""
ABIDE I Data Download Script

This script downloads and preprocesses ABIDE I dataset for SynapseBiome ASD-Net.
Based on the MADE-for-ASD project's download approach.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import ABIDE1DataProcessor
import logging


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('abide1_download.log')
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download ABIDE I dataset")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ABIDE1",
        help="Directory to store ABIDE I data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/ABIDE1",
        help="Directory to store processed data"
    )
    
    parser.add_argument(
        "--derivatives",
        nargs="+",
        default=["rois_cc200", "rois_aal", "rois_ez"],
        help="Derivatives to download"
    )
    
    parser.add_argument(
        "--pipeline",
        type=str,
        default="cpac",
        help="Preprocessing pipeline"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="filt_global",
        help="Preprocessing strategy"
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
    
    logger.info("Starting ABIDE I data download and processing...")
    
    # Initialize processor
    processor = ABIDE1DataProcessor(args.data_dir, args.output_dir)
    
    try:
        if not args.process_only:
            logger.info("Downloading ABIDE I data...")
            processor.download_functional_data(
                derivatives=args.derivatives,
                pipeline=args.pipeline,
                strategy=args.strategy
            )
            logger.info("Download completed successfully!")
        
        if not args.download_only:
            logger.info("Processing ABIDE I data...")
            # Convert derivative names (remove 'rois_' prefix)
            process_derivatives = [d.replace('rois_', '') for d in args.derivatives]
            hdf5_path = processor.process_functional_data(process_derivatives)
            logger.info(f"Processing completed! Data saved to: {hdf5_path}")
        
        logger.info("ABIDE I data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ABIDE I data preparation: {e}")
        raise


if __name__ == "__main__":
    main() 