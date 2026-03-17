"""
Improved Loss Functions for DGMR

This module implements the extended balanced loss function and related
loss functions for training DGMR with improved performance on
high-intensity precipitation events.

Key improvements:
1. Weighted loss for high precipitation intensities
2. Spatial structure preservation (gradient loss)
3. Probability matching for distribution consistency
4. Combined generator and discriminator losses

Reference:
Improving Precipitation Nowcasting for High-Intensity Events Using Deep
Generative Models with Balanced Loss and Temperature Data
AMS AI for Earth Systems, 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtendedBalancedLoss(nn.Module):
    """
    Extended Balanced Loss Function for High-Intensity Precipitation

    This loss function addresses the imbalance between frequent low-intensity
    and rare high-intensity precipitation events by assigning higher weights
    to high-intensity pixels.

    Key features:
    1. Intensity-based weighting
    2. Spatial gradient preservation
    3. Probability matching (optional)
    4. Flexible threshold configuration

    Parameters
    ----------
    threshold_low : float, default=0.5
        Low precipitation threshold (mm/h) for distinguishing light rain
    threshold_high : float, default=5.0
        High precipitation threshold (mm/h) for heavy rain
    weight_high : float, default=3.0
        Weight multiplier for high-intensity precipitation
    weight_spatial : float, default=0.5
        Weight for spatial structure (gradient) loss
    use_prob_matching : bool, default=True
        Whether to use probability matching loss

    Examples
    --------
    >>> loss_fn = ExtendedBalancedLoss(threshold_high=5.0, weight_high=3.0)
    >>> pred = torch.randn(4, 24, 256, 256)  # [B, T, H, W]
    >>> target = torch.rand(4, 24, 256, 256)
    >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        threshold_low=0.5,
        threshold_high=5.0,
        weight_high=3.0,
        weight_spatial=0.5,
        use_prob_matching=True
    ):
        super().__init__()
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.weight_high = weight_high
        self.weight_spatial = weight_spatial
        self.use_prob_matching = use_prob_matching

    def forward(self, pred, target):
        """
        Compute extended balanced loss

        Parameters
        ----------
        pred : torch.Tensor
            Predicted precipitation [B, T, H, W]
        target : torch.Tensor
            Ground truth precipitation [B, T, H, W]

        Returns
        -------
        loss : torch.Tensor
            Computed loss value (scalar)
        """
        # 1. Base MSE loss
        mse_loss = F.mse_loss(pred, target, reduction='none')

        # 2. Intensity-based weighting
        # Create masks for different intensity ranges
        high_precip_mask = (target >= self.threshold_high).float()
        medium_precip_mask = (
            (target >= self.threshold_low) &
            (target < self.threshold_high)
        ).float()
        low_precip_mask = (target < self.threshold_low).float()

        # Weighted loss with emphasis on high-intensity events
        weighted_loss = (
            self.weight_high * high_precip_mask * mse_loss +
            2.0 * medium_precip_mask * mse_loss +
            1.0 * low_precip_mask * mse_loss
        )

        # 3. Spatial structure preservation (gradient loss)
        if self.weight_spatial > 0:
            # Horizontal gradient
            grad_pred_x = torch.abs(pred[..., :, :-1] - pred[..., :, 1:])
            grad_target_x = torch.abs(target[..., :, :-1] - target[..., :, 1:])

            # Vertical gradient
            grad_pred_y = torch.abs(pred[..., :-1, :] - pred[..., 1:, :])
            grad_target_y = torch.abs(target[..., :-1, :] - target[..., 1:, :])

            gradient_loss = (
                F.mse_loss(grad_pred_x, grad_target_x) +
                F.mse_loss(grad_pred_y, grad_target_y)
            )
        else:
            gradient_loss = torch.tensor(0.0, device=pred.device)

        # 4. Probability matching (optional)
        # Ensures predicted distribution matches target distribution
        if self.use_prob_matching:
            # Flatten and sort values
            pred_flat = pred.flatten()
            target_flat = target.flatten()

            # Subsample for efficiency (use 10% of pixels)
            n_samples = min(len(pred_flat), len(target_flat)) // 10

            if n_samples > 0:
                indices = torch.randperm(len(pred_flat), device=pred.device)[:n_samples]
                pred_sorted = torch.sort(pred_flat[indices])[0]
                target_sorted = torch.sort(target_flat[indices])[0]
                pm_loss = F.mse_loss(pred_sorted, target_sorted)
            else:
                pm_loss = torch.tensor(0.0, device=pred.device)
        else:
            pm_loss = torch.tensor(0.0, device=pred.device)

        # Combine all losses
        total_loss = (
            weighted_loss.mean() +
            self.weight_spatial * gradient_loss +
            0.1 * pm_loss
        )

        return total_loss


class ImprovedDGMRLoss(nn.Module):
    """
    Complete Loss Function for Improved DGMR

    Combines reconstruction loss (with balanced weighting) and adversarial
    loss for both generator and discriminator training.

    Parameters
    ----------
    lambda_reconstruction : float, default=1.0
        Weight for reconstruction loss
    lambda_gan : float, default=0.1
        Weight for GAN loss
    threshold_high : float, default=5.0
        High precipitation threshold for balanced loss
    weight_high : float, default=3.0
        Weight for high-intensity precipitation

    Examples
    --------
    >>> loss_fn = ImprovedDGMRLoss(lambda_reconstruction=1.0, lambda_gan=0.1)
    >>> # Generator training
    >>> fake_samples = model(input)
    >>> fake_output = discriminator(fake_samples)
    >>> g_loss, recon, gan = loss_fn.generator_loss(fake_samples, real_samples, fake_output)
    """

    def __init__(
        self,
        lambda_reconstruction=1.0,
        lambda_gan=0.1,
        threshold_high=5.0,
        weight_high=3.0
    ):
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_gan = lambda_gan

        # Balanced reconstruction loss
        self.reconstruction_loss = ExtendedBalancedLoss(
            threshold_high=threshold_high,
            weight_high=weight_high
        )

        # Binary cross-entropy for GAN loss
        self.bce_loss = nn.BCEWithLogitsLoss()

    def generator_loss(self, fake_samples, real_samples, fake_disc_output):
        """
        Compute generator loss

        The generator aims to:
        1. Reconstruct realistic precipitation fields (reconstruction loss)
        2. Fool the discriminator (GAN loss)

        Parameters
        ----------
        fake_samples : torch.Tensor
            Generated precipitation [B, T, H, W]
        real_samples : torch.Tensor
            Ground truth precipitation [B, T, H, W]
        fake_disc_output : torch.Tensor
            Discriminator output for fake samples [B, 1]

        Returns
        -------
        total_loss : torch.Tensor
            Combined generator loss
        recon_loss : torch.Tensor
            Reconstruction loss component
        gan_loss : torch.Tensor
            GAN loss component
        """
        # 1. Reconstruction loss with balanced weighting
        recon_loss = self.reconstruction_loss(fake_samples, real_samples)

        # 2. GAN loss (want discriminator to think it's real)
        gan_loss = self.bce_loss(
            fake_disc_output,
            torch.ones_like(fake_disc_output)
        )

        # Combine with learned weights
        total_loss = (
            self.lambda_reconstruction * recon_loss +
            self.lambda_gan * gan_loss
        )

        return total_loss, recon_loss, gan_loss

    def discriminator_loss(self, real_disc_output, fake_disc_output):
        """
        Compute discriminator loss

        The discriminator aims to:
        1. Correctly identify real samples (real_loss)
        2. Correctly identify fake samples (fake_loss)

        Parameters
        ----------
        real_disc_output : torch.Tensor
            Discriminator output for real samples [B, 1]
        fake_disc_output : torch.Tensor
            Discriminator output for fake samples [B, 1]

        Returns
        -------
        loss : torch.Tensor
            Combined discriminator loss
        """
        # Real samples: want output to be 1
        real_loss = self.bce_loss(
            real_disc_output,
            torch.ones_like(real_disc_output)
        )

        # Fake samples: want output to be 0
        fake_loss = self.bce_loss(
            fake_disc_output,
            torch.zeros_like(fake_disc_output)
        )

        return real_loss + fake_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance

    Focuses training on hard-to-predict examples by down-weighting
    easy examples.

    Parameters
    ----------
    alpha : float, default=0.25
        Weighting factor for rare class
    gamma : float, default=2.0
        Focusing parameter (higher = more focus on hard examples)
    threshold : float, default=0.5
        Threshold for binary classification

    Reference
    ---------
    Lin et al. (2017). Focal Loss for Dense Object Detection.
    IEEE Transactions on Pattern Analysis and Machine Intelligence.
    """

    def __init__(self, alpha=0.25, gamma=2.0, threshold=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold

    def forward(self, pred, target):
        """
        Compute focal loss

        Parameters
        ----------
        pred : torch.Tensor
            Predicted precipitation [B, T, H, W]
        target : torch.Tensor
            Ground truth precipitation [B, T, H, W]

        Returns
        -------
        loss : torch.Tensor
            Focal loss value
        """
        # Binary classification: above/below threshold
        pred_binary = (pred >= self.threshold).float()
        target_binary = (target >= self.threshold).float()

        # BCE
        bce = F.binary_cross_entropy(
            pred_binary,
            target_binary,
            reduction='none'
        )

        # Probability of correct class
        p_t = target_binary * pred_binary + (1 - target_binary) * (1 - pred_binary)

        # Focal weight
        alpha_t = self.alpha * target_binary + (1 - self.alpha) * (1 - target_binary)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Apply weight
        loss = (focal_weight * bce).mean()

        return loss


class CRPSLoss(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS) Loss

    CRPS is a proper scoring rule for probabilistic forecasts that measures
    the integrated squared difference between the forecast and target CDFs.

    Parameters
    ----------
    threshold : float, default=0.5
        Precipitation threshold for computation

    Reference
    ---------
    Gneiting et al. (2005). Calibrated Probabilistic Forecasting Using
    Ensemble Model Output Statistics and Minimum CRPS Estimation.
    Monthly Weather Review.
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred_ensemble, target):
        """
        Compute CRPS loss

        Parameters
        ----------
        pred_ensemble : torch.Tensor
            Ensemble predictions [B, N, T, H, W] where N is ensemble size
        target : torch.Tensor
            Ground truth [B, T, H, W]

        Returns
        -------
        loss : torch.Tensor
            CRPS value (scalar)
        """
        # CRPS = E[(X - Y)^2] where X is forecast, Y is observation
        # For ensemble forecast: CRPS = mean((f_i - y)^2) - 0.5 * mean((f_i - f_j)^2)

        # First term: mean squared error
        mse = torch.mean((pred_ensemble - target.unsqueeze(1)) ** 2, dim=1)

        # Second term: mean squared difference between ensemble members
        # Expand for pairwise comparison
        B, N = pred_ensemble.shape[:2]
        ensemble_diff = []

        for i in range(N):
            for j in range(i + 1, N):
                diff = (pred_ensemble[:, i] - pred_ensemble[:, j]) ** 2
                ensemble_diff.append(diff)

        if ensemble_diff:
            ensemble_var = 2.0 * torch.mean(torch.stack(ensemble_diff, dim=0), dim=0) / (N * (N - 1))
        else:
            ensemble_var = torch.zeros_like(mse)

        crps = mse - 0.5 * ensemble_var

        return crps.mean()
