"""
Training Script for Improved DGMR

This module provides the training logic for the Improved DGMR model
using PyTorch Lightning.

Key features:
1. PyTorch Lightning integration
2. Automatic mixed precision training
3. Checkpointing and logging
4. Validation metrics (CSI, CRPS)
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pysteps.dgmr.core.improved_loss import (
    ExtendedBalancedLoss,
    ImprovedDGMRLoss
)
from pysteps.dgmr.core.improved_generator import (
    ImprovedDGMRGenerator,
    DGMRDiscriminator
)


class ImprovedDGMRModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Improved DGMR

    Parameters
    ----------
    input_frames : int
        Number of input frames
    output_frames : int
        Number of output frames
    hidden_dim : int
        Hidden dimension
    learning_rate : float
        Learning rate
    lambda_reconstruction : float
        Weight for reconstruction loss
    lambda_gan : float
        Weight for GAN loss
    threshold_high : float
        High precipitation threshold
    weight_high : float
        Weight for high-intensity precipitation
    """

    def __init__(
        self,
        input_frames: int = 12,
        output_frames: int = 24,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        lambda_reconstruction: float = 1.0,
        lambda_gan: float = 0.1,
        threshold_high: float = 5.0,
        weight_high: float = 3.0
    ):
        super().__init__()

        self.save_hyperparameters()

        # Models
        self.generator = ImprovedDGMRGenerator(
            input_frames=input_frames,
            output_frames=output_frames,
            hidden_dim=hidden_dim,
            use_attention=True,
            use_convlstm=True
        )

        self.discriminator = DGMRDiscriminator(
            input_channels=output_frames
        )

        # Loss function
        self.loss_fn = ImprovedDGMRLoss(
            lambda_reconstruction=lambda_reconstruction,
            lambda_gan=lambda_gan,
            threshold_high=threshold_high,
            weight_high=weight_high
        )

        # Metrics tracking
        self.train_g_loss = 0.0
        self.train_d_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.generator(x)

    def training_step(self, batch: tuple, batch_idx: int, optimizer_idx: int):
        """Training step"""
        x, y = batch
        batch_size = x.shape[0]

        # Train Discriminator
        if optimizer_idx == 0:
            # Generate fake samples
            with torch.no_grad():
                fake = self.generator(x)

            # Discriminator outputs
            real_logits = self.discriminator(y)
            fake_logits = self.discriminator(fake.detach())

            # Discriminator loss
            d_loss = self.loss_fn.discriminator_loss(real_logits, fake_logits)

            # Log
            self.log('train/d_loss', d_loss, prog_bar=True)
            self.train_d_loss = d_loss

            return d_loss

        # Train Generator
        elif optimizer_idx == 1:
            # Generate fake samples
            fake = self.generator(x)

            # Discriminator output for fake samples
            fake_logits = self.discriminator(fake)

            # Generator loss
            g_loss, recon_loss, gan_loss = self.loss_fn.generator_loss(
                fake, y, fake_logits
            )

            # Log
            self.log('train/g_loss', g_loss, prog_bar=True)
            self.log('train/recon_loss', recon_loss)
            self.log('train/gan_loss', gan_loss)
            self.train_g_loss = g_loss

            return g_loss

    def validation_step(self, batch: tuple, batch_idx: int):
        """Validation step"""
        x, y = batch

        # Generate
        with torch.no_grad():
            fake = self.generator(x)

        # Compute metrics for different thresholds
        thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        for threshold in thresholds:
            csi = self._compute_csi(fake, y, threshold)
            self.log(f'val/csi_{threshold}mm', csi, prog_bar=(threshold == 5.0))

        # High-intensity CSI (primary metric)
        csi_high = self._compute_csi(fake, y, threshold=5.0)
        self.log('val/csi_high', csi_high, prog_bar=True)

        return csi_high

    def _compute_csi(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Compute Critical Success Index (CSI)

        CSI = hits / (hits + misses + false alarms)

        Parameters
        ----------
        pred : torch.Tensor
            Predicted precipitation [B, T, H, W]
        target : torch.Tensor
            Ground truth [B, T, H, W]
        threshold : float
            Precipitation threshold

        Returns
        -------
        csi : torch.Tensor
            CSI value (scalar)
        """
        # Binary classification
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()

        # Compute hits, misses, false alarms
        hits = (pred_binary * target_binary).sum()
        misses = ((1 - pred_binary) * target_binary).sum()
        false_alarms = (pred_binary * (1 - target_binary)).sum()

        # CSI
        csi = hits / (hits + misses + false_alarms + 1e-8)

        return csi

    def configure_optimizers(self):
        """Configure optimizers"""
        # Generator optimizer
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.999)
        )

        # Discriminator optimizer
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.999)
        )

        return [opt_d, opt_g]


def train_improved_dgmr(
    train_loader,
    val_loader,
    config: Dict[str, Any]
):
    """
    Train Improved DGMR model

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    config : dict
        Training configuration

    Returns
    -------
    trainer : pl.Trainer
        Trained model trainer
    """
    # Create model
    model = ImprovedDGMRModule(**config)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val/csi_high',  # Primary metric: high-intensity CSI
        mode='max',
        save_top_k=3,
        filename='improved-dgmr-{epoch:02d}-{val/csi_high:.4f}',
        save_last=True
    )

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val/csi_high',
        patience=15,
        mode='max',
        verbose=True
    )

    # Logger
    logger = TensorBoardLogger(
        'logs/improved_dgmr',
        name='default'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 100),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        logger=logger,
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        log_every_n_steps=10,
        check_val_every_n_epoch=1
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    return trainer


if __name__ == "__main__":
    import argparse
    from .data.datamodule import DGMRDataModule
    import glob

    parser = argparse.ArgumentParser(description='Train Improved DGMR')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--input_frames', type=int, default=12,
                        help='Number of input frames')
    parser.add_argument('--output_frames', type=int, default=24,
                        help='Number of output frames')
    parser.add_argument('--threshold_high', type=float, default=5.0,
                        help='High precipitation threshold (mm/h)')
    parser.add_argument('--weight_high', type=float, default=3.0,
                        help='Weight for high-intensity loss')

    args = parser.parse_args()

    # Create data module
    train_files = sorted(glob.glob(f"{args.train_path}/*.h5"))
    val_files = sorted(glob.glob(f"{args.val_path}/*.h5"))

    print(f"Found {len(train_files)} training files")
    print(f"Found {len(val_files)} validation files")

    dm = DGMRDataModule(
        train_files=train_files,
        val_files=val_files,
        batch_size=args.batch_size,
        num_workers=4,
        input_frames=args.input_frames,
        output_frames=args.output_frames
    )
    dm.setup()

    # Training config
    config = {
        'input_frames': args.input_frames,
        'output_frames': args.output_frames,
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.learning_rate,
        'max_epochs': args.max_epochs,
        'threshold_high': args.threshold_high,
        'weight_high': args.weight_high
    }

    print("\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Train
    print("\nStarting training...")
    trainer = train_improved_dgmr(
        dm.train_dataloader(),
        dm.val_dataloader(),
        config
    )

    print("\nTraining complete!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
