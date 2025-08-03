#!/usr/bin/env python3
"""
fMRI Data Preprocessing Script

This script preprocesses fMRI data for SynapseBiome ASD-Net.
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
            logging.FileHandler('fmri_preprocessing.log')
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess fMRI data")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw fMRI data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store preprocessed data"
    )
    
    parser.add_argument(
        "--pipeline",
        type=str,
        default="fmriprep",
        choices=["fmriprep", "cpac", "custom"],
        help="Preprocessing pipeline to use"
    )
    
    parser.add_argument(
        "--atlas",
        type=str,
        default="aal",
        choices=["aal", "cc200", "ez", "ho", "tt"],
        help="Atlas for ROI extraction"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="filt_global",
        choices=["filt_global", "filt_noglobal", "nofilt_global", "nofilt_noglobal"],
        help="Preprocessing strategy"
    )
    
    parser.add_argument(
        "--smooth",
        type=float,
        default=6.0,
        help="Smoothing FWHM in mm"
    )
    
    parser.add_argument(
        "--tr",
        type=float,
        default=2.0,
        help="Repetition time in seconds"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting fMRI data preprocessing...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Pipeline: {args.pipeline}")
    logger.info(f"Atlas: {args.atlas}")
    logger.info(f"Strategy: {args.strategy}")
    
    # Initialize processor
    processor = ABIDE2DataProcessor(args.input_dir, args.output_dir)
    
    try:
        # Process fMRI data
        logger.info("Processing fMRI data...")
        fmri_path = processor.process_fmri_data()
        logger.info(f"fMRI preprocessing completed! Data saved to: {fmri_path}")
        
        logger.info("fMRI preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during fMRI preprocessing: {e}")
        raise


if __name__ == "__main__":
    main() 