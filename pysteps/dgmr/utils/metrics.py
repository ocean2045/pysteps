"""
Evaluation Metrics for Improved DGMR

This module provides evaluation metrics for assessing precipitation
nowcasting performance, including CSI, CRPS, and other skill scores.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def compute_csi(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute Critical Success Index (CSI)

    CSI = hits / (hits + misses + false_alarms)

    Parameters
    ----------
    pred : np.ndarray
        Predicted precipitation [T, H, W] or [B, T, H, W]
    target : np.ndarray
        Ground truth precipitation (same shape as pred)
    threshold : float, default=0.5
        Precipitation threshold (mm/h)

    Returns
    -------
    csi : float
        CSI value

    Examples
    --------
    >>> pred = np.random.rand(24, 256, 256) * 10
    >>> target = np.random.rand(24, 256, 256) * 10
    >>> csi = compute_csi(pred, target, threshold=1.0)
    """
    # Ensure numpy arrays
    pred = np.asarray(pred)
    target = np.asarray(target)

    # Binary classification
    pred_binary = (pred > threshold).astype(float)
    target_binary = (target > threshold).astype(float)

    # Compute hits, misses, false alarms
    hits = (pred_binary * target_binary).sum()
    misses = ((1 - pred_binary) * target_binary).sum()
    false_alarms = (pred_binary * (1 - target_binary)).sum()

    # CSI
    csi = hits / (hits + misses + false_alarms + 1e-8)

    return float(csi)


def compute_pod(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute Probability of Detection (POD)

    POD = hits / (hits + misses)

    Parameters
    ----------
    pred : np.ndarray
        Predicted precipitation
    target : np.ndarray
        Ground truth precipitation
    threshold : float
        Precipitation threshold

    Returns
    -------
    pod : float
        POD value (range: 0-1)
    """
    pred = np.asarray(pred)
    target = np.asarray(target)

    pred_binary = (pred > threshold).astype(float)
    target_binary = (target > threshold).astype(float)

    hits = (pred_binary * target_binary).sum()
    misses = ((1 - pred_binary) * target_binary).sum()

    pod = hits / (hits + misses + 1e-8)

    return float(pod)


def compute_far(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute False Alarm Rate (FAR)

    FAR = false_alarms / (hits + false_alarms)

    Parameters
    ----------
    pred : np.ndarray
        Predicted precipitation
    target : np.ndarray
        Ground truth precipitation
    threshold : float
        Precipitation threshold

    Returns
    -------
    far : float
        FAR value (range: 0-1, lower is better)
    """
    pred = np.asarray(pred)
    target = np.asarray(target)

    pred_binary = (pred > threshold).astype(float)
    target_binary = (target > threshold).astype(float)

    hits = (pred_binary * target_binary).sum()
    false_alarms = (pred_binary * (1 - target_binary)).sum()

    far = false_alarms / (hits + false_alarms + 1e-8)

    return float(far)


def compute_bias(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute Frequency Bias

    Bias = (hits + false_alarms) / (hits + misses)

    Values > 1 indicate overforecasting
    Values < 1 indicate underforecasting

    Parameters
    ----------
    pred : np.ndarray
        Predicted precipitation
    target : np.ndarray
        Ground truth precipitation
    threshold : float
        Precipitation threshold

    Returns
    -------
    bias : float
        Bias value
    """
    pred = np.asarray(pred)
    target = np.asarray(target)

    pred_binary = (pred > threshold).astype(float)
    target_binary = (target > threshold).astype(float)

    hits = (pred_binary * target_binary).sum()
    false_alarms = (pred_binary * (1 - target_binary)).sum()
    misses = ((1 - pred_binary) * target_binary).sum()

    bias = (hits + false_alarms) / (hits + misses + 1e-8)

    return float(bias)


def compute_mse(
    pred: np.ndarray,
    target: np.ndarray
) -> float:
    """
    Compute Mean Squared Error

    Parameters
    ----------
    pred : np.ndarray
        Predicted precipitation
    target : np.ndarray
        Ground truth precipitation

    Returns
    -------
    mse : float
        MSE value
    """
    pred = np.asarray(pred)
    target = np.asarray(target)

    mse = np.mean((pred - target) ** 2)

    return float(mse)


def compute_mae(
    pred: np.ndarray,
    target: np.ndarray
) -> float:
    """
    Compute Mean Absolute Error

    Parameters
    ----------
    pred : np.ndarray
        Predicted precipitation
    target : np.ndarray
        Ground truth precipitation

    Returns
    -------
    mae : float
        MAE value
    """
    pred = np.asarray(pred)
    target = np.asarray(target)

    mae = np.mean(np.abs(pred - target))

    return float(mae)


def compute_crps(
    pred_ensemble: np.ndarray,
    target: np.ndarray
) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS)

    CRPS measures the integrated squared difference between the forecast
    and target cumulative distribution functions.

    Parameters
    ----------
    pred_ensemble : np.ndarray
        Ensemble predictions [N, T, H, W] or [N, B, T, H, W]
        where N is the number of ensemble members
    target : np.ndarray
        Ground truth [T, H, W] or [B, T, H, W]

    Returns
    -------
    crps : float
        CRPS value (lower is better)

    Reference
    ---------
    Gneiting et al. (2005). Calibrated Probabilistic Forecasting Using
    Ensemble Model Output Statistics and Minimum CRPS Estimation.
    Monthly Weather Review.
    """
    pred_ensemble = np.asarray(pred_ensemble)
    target = np.asarray(target)

    # If target has no batch dimension, add it
    if pred_ensemble.ndim == 4:  # [N, T, H, W]
        target = target[np.newaxis, ...]

    # CRPS = E[(X - Y)^2] - 0.5 * E[(X - X')^2]
    # where X, X' are independent ensemble members, Y is observation

    # First term: mean squared error
    mse = np.mean((pred_ensemble - target) ** 2)

    # Second term: mean squared difference between ensemble members
    N = pred_ensemble.shape[0]
    if N > 1:
        ensemble_diff = []
        for i in range(N):
            for j in range(i + 1, N):
                diff = (pred_ensemble[i] - pred_ensemble[j]) ** 2
                ensemble_diff.append(diff)

        ensemble_var = 2.0 * np.mean(ensemble_diff) / (N * (N - 1))
    else:
        ensemble_var = 0.0

    crps = mse - 0.5 * ensemble_var

    return float(crps)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for precipitation nowcasting

    Parameters
    ----------
    thresholds : list of float, default=[0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 10.0]
        Precipitation thresholds to evaluate
    """

    def __init__(
        self,
        thresholds: List[float] = [0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 10.0]
    ):
        self.thresholds = thresholds

    def evaluate(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        lead_times: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation

        Parameters
        ----------
        pred : np.ndarray
            Predicted precipitation [B, T, H, W] or [T, H, W]
        target : np.ndarray
            Ground truth precipitation (same shape as pred)
        lead_times : list of int, optional
            Lead times in minutes for each time step

        Returns
        -------
        results : dict
            Dictionary of metric names and values
        """
        results = {}

        # CSI for all thresholds
        for thresh in self.thresholds:
            csi = compute_csi(pred, target, threshold=thresh)
            results[f'csi_{thresh}mm'] = csi

            pod = compute_pod(pred, target, threshold=thresh)
            results[f'pod_{thresh}mm'] = pod

            far = compute_far(pred, target, threshold=thresh)
            results[f'far_{thresh}mm'] = far

            bias = compute_bias(pred, target, threshold=thresh)
            results[f'bias_{thresh}mm'] = bias

        # MSE and MAE
        results['mse'] = compute_mse(pred, target)
        results['mae'] = compute_mae(pred, target)

        # Lead-time specific metrics
        if lead_times is not None and pred.ndim == 4:
            for i, lead_time in enumerate(lead_times):
                # Per-time-step CSI at 5mm threshold
                csi_t = compute_csi(pred[:, i], target[:, i], threshold=5.0)
                results[f'csi_5mm_lead{lead_time}min'] = csi_t

        return results

    def print_summary(self, results: Dict[str, float]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        # CSI table
        print("\nCSI by Precipitation Threshold:")
        print("-" * 60)
        print(f"{'Threshold (mm/h)':<20} {'CSI':<15} {'POD':<15} {'FAR':<15}")
        print("-" * 60)

        for thresh in self.thresholds:
            csi = results.get(f'csi_{thresh}mm', 0.0)
            pod = results.get(f'pod_{thresh}mm', 0.0)
            far = results.get(f'far_{thresh}mm', 0.0)

            print(f"{thresh:<20.1f} {csi:<15.4f} {pod:<15.4f} {far:<15.4f}")

        print("-" * 60)

        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  MSE:  {results['mse']:.4f}")
        print(f"  MAE:  {results['mae']:.4f}")

        # High-intensity performance
        high_csi = results.get('csi_5.0mm', 0.0)
        very_high_csi = results.get('csi_10.0mm', 0.0)

        print(f"\nHigh-Intensity Performance:")
        print(f"  CSI at 5mm/h:  {high_csi:.4f}")
        print(f"  CSI at 10mm/h: {very_high_csi:.4f}")

        print("="*60 + "\n")


def evaluate_improved_dgmr(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    thresholds: List[float] = [0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 10.0]
) -> Dict[str, float]:
    """
    Evaluate Improved DGMR model on a dataset

    Parameters
    ----------
    model : torch.nn.Module
        Trained DGMR model
    data_loader : DataLoader
        Test data loader
    device : str
        Device to run evaluation on
    thresholds : list of float
        Precipitation thresholds to evaluate

    Returns
    -------
    results : dict
        Evaluation results
    """
    model.eval()
    model.to(device)

    evaluator = EvaluationMetrics(thresholds=thresholds)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            # Move to device
            x = x.to(device)
            y = y.to(device)

            # Generate predictions
            pred = model(x)

            # Move to CPU and convert to numpy
            pred = pred.cpu().numpy()
            target = y.cpu().numpy()

            all_preds.append(pred)
            all_targets.append(target)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Evaluate
    results = evaluator.evaluate(all_preds, all_targets)

    return results


if __name__ == "__main__":
    # Test evaluation metrics
    print("Testing evaluation metrics...")

    # Create synthetic data
    pred = np.random.rand(4, 24, 128, 128) * 10  # 4 samples, 24 frames
    target = np.random.rand(4, 24, 128, 128) * 10

    # Evaluate
    evaluator = EvaluationMetrics()
    results = evaluator.evaluate(pred, target)

    # Print summary
    evaluator.print_summary(results)

    print("Evaluation metrics test passed!")
