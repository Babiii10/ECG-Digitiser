#!/usr/bin/env python3
"""
Training script for PhysioNet Challenge 2025
Complete training pipeline with all preprocessing
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src_2025.data.dataset import ChagasECGDataset, create_dataloaders
from src_2025.data.preprocessing import ECGPreprocessor, ECGAugmentation
from src_2025.models.resnet1d import (
    create_resnet1d_small,
    create_resnet1d_medium,
    create_resnet1d_large,
    create_seresnet1d
)
from src_2025.models.losses import create_loss_function
from src_2025.utils.metrics import MetricsTracker, print_metrics, compute_tpr_at_top_k_percent


class Trainer:
    """Main trainer class"""

    def __init__(self, config):
        """
        Initialize trainer

        Args:
            config: Dictionary with training configuration
        """
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Create model
        self.model = self._create_model(config['model_type'])
        self.model = self.model.to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")

        # Create loss function
        self.criterion = create_loss_function(
            config['loss_type'],
            pos_weight=config.get('pos_weight', None),
            **config.get('loss_params', {})
        )

        # Create optimizer
        self.optimizer = self._create_optimizer(
            config['optimizer'],
            config['learning_rate'],
            config.get('weight_decay', 0.0)
        )

        # Create scheduler
        self.scheduler = self._create_scheduler(
            config.get('scheduler', 'cosine'),
            config['epochs']
        )

        # Training state
        self.best_tpr = 0.0
        self.best_epoch = 0
        self.train_history = []
        self.val_history = []

    def _create_model(self, model_type):
        """Create model based on type"""
        if model_type == 'resnet1d_small':
            return create_resnet1d_small()
        elif model_type == 'resnet1d_medium':
            return create_resnet1d_medium()
        elif model_type == 'resnet1d_large':
            return create_resnet1d_large()
        elif model_type == 'seresnet1d':
            return create_seresnet1d()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_optimizer(self, optimizer_type, lr, weight_decay):
        """Create optimizer"""
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def _create_scheduler(self, scheduler_type, epochs):
        """Create learning rate scheduler"""
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=epochs//3, gamma=0.1)
        elif scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5)
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        tracker = MetricsTracker()

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (signals, labels, demographics) in enumerate(pbar):
            # Move to device
            signals = signals.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(signals)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

            self.optimizer.step()

            # Track metrics
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            tracker.update(labels.cpu().numpy(), probs, loss.item())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        return tracker.compute()

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        tracker = MetricsTracker()

        with torch.no_grad():
            for signals, labels, demographics in tqdm(val_loader, desc="Validation"):
                # Move to device
                signals = signals.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                # Forward pass
                outputs = self.model(signals)

                # Compute loss
                loss = self.criterion(outputs, labels)

                # Track metrics
                probs = torch.sigmoid(outputs).cpu().numpy()
                tracker.update(labels.cpu().numpy(), probs, loss.item())

        return tracker.compute()

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }

        # Save latest
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
            torch.save(self.model.state_dict(), self.output_dir / 'model_best.pth')

    def train(self, train_loader, val_loader):
        """
        Main training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)

        start_time = time.time()

        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            print("-" * 60)

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate(val_loader)
            self.val_history.append(val_metrics)

            # Print metrics
            print_metrics(train_metrics, prefix="Train")
            print_metrics(val_metrics, prefix="Val")

            # Check if best model
            val_tpr = val_metrics['tpr_at_5pct']
            is_best = val_tpr > self.best_tpr

            if is_best:
                self.best_tpr = val_tpr
                self.best_epoch = epoch
                print(f"\n🏆 New best TPR@5%: {self.best_tpr:.4f}")

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_tpr)
                else:
                    self.scheduler.step()

            # Print current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")

        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("Training complete!")
        print(f"Time elapsed: {elapsed_time/60:.2f} minutes")
        print(f"Best TPR@5%: {self.best_tpr:.4f} (epoch {self.best_epoch})")
        print("="*60)

        # Save training history
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_tpr': self.best_tpr,
            'best_epoch': self.best_epoch
        }

        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train ECG model for Chagas detection')

    # Data arguments
    parser.add_argument('--data_folder', type=str, required=True,
                      help='Path to training data folder')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Path to output directory')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='resnet1d_medium',
                      choices=['resnet1d_small', 'resnet1d_medium', 'resnet1d_large', 'seresnet1d'],
                      help='Model architecture')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                      choices=['adam', 'adamw', 'sgd'],
                      help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                      choices=['cosine', 'step', 'reduce_on_plateau', 'none'],
                      help='Learning rate scheduler')
    parser.add_argument('--loss_type', type=str, default='focal',
                      choices=['bce', 'weighted_bce', 'focal', 'ranking', 'tpr', 'combined'],
                      help='Loss function type')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='Gradient clipping (0 to disable)')

    # Data arguments
    parser.add_argument('--train_ratio', type=float, default=0.8,
                      help='Train/val split ratio')
    parser.add_argument('--target_length', type=int, default=5000,
                      help='Target signal length')
    parser.add_argument('--target_fs', type=int, default=500,
                      help='Target sampling frequency')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create config
    config = vars(args)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        data_folder=args.data_folder,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        target_length=args.target_length,
        target_fs=args.target_fs,
        random_seed=args.seed
    )

    # Create trainer
    trainer = Trainer(config)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
