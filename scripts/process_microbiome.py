#!/usr/bin/env python3
"""
Microbiome Data Processing Script

This script processes microbiome data from QIIME2 BIOM format for SynapseBiome ASD-Net.
It handles feature table.biom files and converts them to the format required by the model.
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import ABIDE2DataProcessor
import yaml


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('microbiome_processing.log')
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process microbiome data from QIIME2")
    
    parser.add_argument(
        "--biom_file",
        type=str,
        required=True,
        help="Path to feature_table.biom file from QIIME2"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/microbiome",
        help="Output directory for processed data"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data configuration file"
    )
    
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=0.01,
        help="Variance threshold for feature selection"
    )
    
    parser.add_argument(
        "--normalization",
        type=str,
        default="relative_abundance",
        choices=["relative_abundance", "clr", "none"],
        help="Normalization method"
    )
    
    parser.add_argument(
        "--transformation",
        type=str,
        default="log",
        choices=["log", "sqrt", "none"],
        help="Data transformation method"
    )
    
    parser.add_argument(
        "--feature_selection",
        action="store_true",
        help="Apply feature selection"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_microbiome_data(biom_file: str, output_dir: str, config: dict, args) -> str:
    """
    Process microbiome data from BIOM format.
    
    Args:
        biom_file: Path to BIOM file
        output_dir: Output directory
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Path to processed data
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = ABIDE2DataProcessor(str(output_path.parent), str(output_path))
    
    # Process microbiome data
    logger.info(f"Processing microbiome data from: {biom_file}")
    
    try:
        # Load BIOM file
        import biom
        table = biom.load_table(biom_file)
        
        # Convert to pandas DataFrame
        df = table.to_dataframe()
        
        # Transpose to get samples as rows and features as columns
        df = df.T
        
        logger.info(f"Original data shape: {df.shape}")
        
        # Handle missing values
        df = df.fillna(0)
        
        # Apply normalization
        if args.normalization == "relative_abundance":
            df = df.div(df.sum(axis=1), axis=0)
            logger.info("Applied relative abundance normalization")
        elif args.normalization == "clr":
            # Centered log ratio transformation
            df = df.div(df.sum(axis=1), axis=0)
            df = np.log(df + 1e-10)
            df = df - df.mean(axis=0)
            logger.info("Applied CLR transformation")
        
        # Apply transformation
        if args.transformation == "log":
            df = np.log(df + 1e-10)
            logger.info("Applied log transformation")
        elif args.transformation == "sqrt":
            df = np.sqrt(df)
            logger.info("Applied square root transformation")
        
        # Feature selection
        if args.feature_selection:
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=args.variance_threshold)
            df_selected = pd.DataFrame(
                selector.fit_transform(df),
                index=df.index,
                columns=df.columns[selector.get_support()]
            )
            df = df_selected
            logger.info(f"Feature selection applied. New shape: {df.shape}")
        
        # Save processed data
        output_file = output_path / "microbiome_processed.npy"
        np.save(output_file, df.values)
        
        # Save metadata
        feature_info_path = output_path / "microbiome_features.csv"
        feature_df = pd.DataFrame({
            'feature_id': df.columns,
            'feature_name': df.columns,
            'mean_abundance': df.mean(),
            'std_abundance': df.std()
        })
        feature_df.to_csv(feature_info_path, index=False)
        
        sample_info_path = output_path / "microbiome_samples.csv"
        sample_df = pd.DataFrame({
            'sample_id': df.index,
            'total_reads': df.sum(axis=1),
            'n_features': (df > 0).sum(axis=1)
        })
        sample_df.to_csv(sample_info_path, index=False)
        
        # Save processing summary
        summary_path = output_path / "processing_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Microbiome Data Processing Summary\n")
            f.write("==================================\n\n")
            f.write(f"Input file: {biom_file}\n")
            f.write(f"Output file: {output_file}\n")
            f.write(f"Data shape: {df.shape}\n")
            f.write(f"Normalization: {args.normalization}\n")
            f.write(f"Transformation: {args.transformation}\n")
            f.write(f"Feature selection: {args.feature_selection}\n")
            f.write(f"Variance threshold: {args.variance_threshold}\n\n")
            f.write("Feature statistics:\n")
            f.write(f"  Mean abundance range: {df.mean().min():.6f} - {df.mean().max():.6f}\n")
            f.write(f"  Standard deviation range: {df.std().min():.6f} - {df.std().max():.6f}\n")
            f.write(f"  Zero features: {(df == 0).sum().sum()}\n")
            f.write(f"  Non-zero features: {(df > 0).sum().sum()}\n")
        
        logger.info(f"Processed microbiome data saved to: {output_file}")
        logger.info(f"Number of features: {df.shape[1]}")
        logger.info(f"Number of samples: {df.shape[0]}")
        
        return str(output_file)
        
    except ImportError:
        logger.error("biom-format package not found. Please install it: pip install biom-format")
        raise
    except Exception as e:
        logger.error(f"Error processing microbiome data: {e}")
        raise


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info("Starting microbiome data processing...")
    
    try:
        # Process microbiome data
        output_file = process_microbiome_data(
            args.biom_file, 
            args.output_dir, 
            config, 
            args
        )
        
        logger.info("Microbiome data processing completed successfully!")
        logger.info(f"Output file: {output_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("MICROBIOME DATA PROCESSING COMPLETED")
        print("="*50)
        print(f"Input file: {args.biom_file}")
        print(f"Output file: {output_file}")
        print(f"Output directory: {args.output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error during microbiome data processing: {e}")
        raise


if __name__ == "__main__":
    main() 