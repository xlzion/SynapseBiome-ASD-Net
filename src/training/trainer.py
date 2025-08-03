"""
Training utilities for SynapseBiome ASD-Net.

This module provides the main training loop and utilities for training
the multimodal ASD diagnosis model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import yaml
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import logging
from datetime import datetime

from ..models import SynapseBiomeASDNet
from .utils import EarlyStopping, MetricTracker, LearningRateScheduler


class ContrastiveTrainer:
    """
    Trainer class for SynapseBiome ASD-Net.
    
    This class handles the complete training pipeline including pretraining,
    contrastive learning, and fine-tuning phases.
    """
    
    def __init__(self, config: Dict, model: SynapseBiomeASDNet,
                 train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            model: SynapseBiome ASD-Net model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup logging
        self._setup_logging()
        
        # Setup tensorboard
        self._setup_tensorboard()
        
        # Initialize training components
        self._initialize_training_components()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            log_dir = os.path.join(self.config['logging']['output_dir'], 'logs')
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            )
            file_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        if self.config['logging']['tensorboard']:
            tb_dir = os.path.join(
                self.config['logging']['tensorboard_dir'],
                f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            self.writer = SummaryWriter(tb_dir)
        else:
            self.writer = None
    
    def _initialize_training_components(self):
        """Initialize optimizer, scheduler, and other training components."""
        # Optimizer
        if self.config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['phases'][0]['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['phases'][0]['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['training']['optimizer']}")
        
        # Learning rate scheduler
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            self.config['training']['scheduler'],
            self.config['training']['warmup_epochs']
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping_patience'],
            min_delta=1e-4
        )
        
        # Metric tracker
        self.metric_tracker = MetricTracker()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.config['hardware']['mixed_precision'] else None
    
    def train(self, phases: Optional[List[str]] = None):
        """
        Train the model through specified phases.
        
        Args:
            phases: List of training phases to execute
        """
        if phases is None:
            phases = [phase['name'] for phase in self.config['training']['phases']]
        
        self.logger.info(f"Starting training with phases: {phases}")
        
        for phase_name in phases:
            self.logger.info(f"Starting {phase_name} phase")
            self._train_phase(phase_name)
            
            # Save phase checkpoint
            self._save_checkpoint(f"{phase_name}_final")
        
        self.logger.info("Training completed!")
    
    def _train_phase(self, phase_name: str):
        """Train a specific phase."""
        # Get phase configuration
        phase_config = next(
            (phase for phase in self.config['training']['phases'] if phase['name'] == phase_name),
            None
        )
        if phase_config is None:
            raise ValueError(f"Phase {phase_name} not found in configuration")
        
        # Set model training mode
        self.model.set_training_mode(phase_name)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = phase_config['learning_rate']
        
        # Training loop
        for epoch in range(phase_config['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self._train_epoch()
            
            # Validate epoch
            val_metrics = self._validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, phase_name)
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging']['save_interval'] == 0:
                self._save_checkpoint(f"{phase_name}_epoch_{epoch+1}")
            
            # Early stopping
            if self.early_stopping(val_metrics['total_loss']):
                self.logger.info(f"Early stopping triggered in {phase_name} phase")
                break
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metric_tracker.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} - Training")
        
        for batch_idx, (microbe_data, fmri_data, labels) in enumerate(pbar):
            # Move data to device
            microbe_data = microbe_data.to(self.device)
            fmri_data = fmri_data.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = self.model.compute_loss(microbe_data, fmri_data, labels)
                    total_loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                loss_dict = self.model.compute_loss(microbe_data, fmri_data, labels)
                total_loss = loss_dict['total_loss']
                
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Update metrics
            self.metric_tracker.update(loss_dict)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        return self.metric_tracker.get_metrics()
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        self.metric_tracker.reset()
        
        with torch.no_grad():
            for microbe_data, fmri_data, labels in self.val_loader:
                # Move data to device
                microbe_data = microbe_data.to(self.device)
                fmri_data = fmri_data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss_dict = self.model.compute_loss(microbe_data, fmri_data, labels)
                else:
                    loss_dict = self.model.compute_loss(microbe_data, fmri_data, labels)
                
                # Update metrics
                self.metric_tracker.update(loss_dict)
        
        return self.metric_tracker.get_metrics()
    
    def _log_metrics(self, train_metrics: Dict[str, float], 
                    val_metrics: Dict[str, float], phase_name: str):
        """Log training and validation metrics."""
        # Console logging
        self.logger.info(
            f"{phase_name} - Epoch {self.current_epoch + 1}: "
            f"Train Loss: {train_metrics['total_loss']:.4f}, "
            f"Val Loss: {val_metrics['total_loss']:.4f}"
        )
        
        # TensorBoard logging
        if self.writer is not None:
            for metric_name, value in train_metrics.items():
                self.writer.add_scalar(
                    f'{phase_name}/train/{metric_name}',
                    value,
                    self.current_epoch
                )
            
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(
                    f'{phase_name}/val/{metric_name}',
                    value,
                    self.current_epoch
                )
            
            # Learning rate
            self.writer.add_scalar(
                f'{phase_name}/lr',
                self.optimizer.param_groups[0]['lr'],
                self.current_epoch
            )
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config['logging']['model_dir'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'{name}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metric = checkpoint['best_val_metric']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            test_metrics: Dictionary of test metrics
        """
        self.model.eval()
        self.metric_tracker.reset()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for microbe_data, fmri_data, labels in test_loader:
                # Move data to device
                microbe_data = microbe_data.to(self.device)
                fmri_data = fmri_data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(microbe_data, fmri_data)
                else:
                    logits = self.model(microbe_data, fmri_data)
                
                # Store predictions and targets
                predictions.extend(logits.argmax(dim=1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        # Compute additional metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary')
        
        test_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        self.logger.info(f"Test Results: {test_metrics}")
        
        return test_metrics 