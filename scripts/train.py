#!/usr/bin/env python3
"""
Main training script for SynapseBiome ASD-Net.

This script provides a command-line interface for training the multimodal
ASD diagnosis model with various configurations.
"""

import argparse
import yaml
import torch
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import SynapseBiomeASDNet
from src.training import ContrastiveTrainer
from src.data import ABIDE2DataProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SynapseBiome ASD-Net")
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Path to configuration file"
    )
    
    # Data
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True,
        help="Path to data directory"
    )
    
    # Training
    parser.add_argument(
        "--phases", 
        nargs="+", 
        default=None,
        help="Training phases to execute (pretrain, contrastive, finetune)"
    )
    
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Hardware
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    
    parser.add_argument(
        "--gpu_ids", 
        nargs="+", 
        type=int, 
        default=[0],
        help="GPU IDs to use"
    )
    
    # Output
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs",
        help="Output directory"
    )
    
    return parser.parse_args()


def setup_device(device: str, gpu_ids: list) -> torch.device:
    """Setup device for training."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        # Set GPU device
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f"cuda:{gpu_ids[0]}")
        
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_data_loaders(config: dict, data_dir: str):
    """Setup data loaders."""
    # Initialize data processor
    processor = ABIDE2DataProcessor(
        data_dir=data_dir,
        output_dir=os.path.join(config['logging']['output_dir'], 'processed_data')
    )
    
    # Process data if needed
    if not os.path.exists(processor.output_dir):
        print("Processing data...")
        processor.process_fmri_data()
        processor.process_microbiome_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = processor.get_dataloaders(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split']
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    config['logging']['output_dir'] = args.output_dir
    config['hardware']['device'] = args.device
    config['hardware']['gpu_ids'] = args.gpu_ids
    
    # Setup device
    device = setup_device(args.device, args.gpu_ids)
    
    # Setup reproducibility
    if config['reproducibility']['deterministic']:
        torch.manual_seed(config['reproducibility']['seed'])
        torch.cuda.manual_seed(config['reproducibility']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = config['reproducibility']['benchmark']
    
    # Setup data loaders
    print("Setting up data loaders...")
    train_loader, val_loader, test_loader = setup_data_loaders(config, args.data_dir)
    
    # Initialize model
    print("Initializing model...")
    model = SynapseBiomeASDNet(config['model'])
    model = model.to(device)
    
    # Print model summary
    summary = model.get_model_summary()
    print(f"Model Summary: {summary}")
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = ContrastiveTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    print("Starting training...")
    trainer.train(phases=args.phases)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    # Save final results
    results_path = os.path.join(config['logging']['results_dir'], 'test_results.yaml')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        yaml.dump(test_metrics, f)
    
    print(f"Training completed! Test results saved to: {results_path}")
    print(f"Test Metrics: {test_metrics}")


if __name__ == "__main__":
    main() 