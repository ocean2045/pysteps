"""
Data Module for Improved DGMR Training

This module provides data loading and preprocessing functionality for
training the Improved DGMR model with radar precipitation data.

Supports:
- OdimH5 format radar data (DWD, KNMI datasets)
- Multi-frame sequence loading
- Data augmentation
- PyTorch DataLoader integration
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import glob
import warnings

# Try to import pysteps for OdimH5 loading
try:
    from pysteps.io import import_odim_h5
    from pysteps.utils import conversion
    HAS_PYSTEPS = True
except ImportError:
    HAS_PYSTEPS = False
    warnings.warn("pysteps not available. Some features will be limited.")


class PrecipitationSequenceDataset(Dataset):
    """
    Precipitation Sequence Dataset for DGMR Training

    Loads radar data and creates input-output sequences for training.

    Parameters
    ----------
    data_files : list of str
        List of data file paths (OdimH5 format)
    input_frames : int, default=12
        Number of input frames (historical context)
    output_frames : int, default=24
        Number of output frames (to predict)
    threshold : float, default=0.1
        Minimum precipitation threshold (mm/h)
    max_precip : float, default=100.0
        Maximum precipitation value for clipping (mm/h)
    augment : bool, default=True
        Whether to apply data augmentation
    normalize : bool, default=True
        Whether to normalize precipitation values

    Examples
    --------
    >>> files = glob.glob("data/*.h5")
    >>> dataset = PrecipitationSequenceDataset(
    ...     files,
    ...     input_frames=12,
    ...     output_frames=24
    ... )
    >>> x, y = dataset[0]
    >>> print(x.shape)  # [12, H, W]
    >>> print(y.shape)  # [24, H, W]
    """

    def __init__(
        self,
        data_files: List[str],
        input_frames: int = 12,
        output_frames: int = 24,
        threshold: float = 0.1,
        max_precip: float = 100.0,
        augment: bool = True,
        normalize: bool = True
    ):
        self.data_files = data_files
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.threshold = threshold
        self.max_precip = max_precip
        self.augment = augment
        self.normalize = normalize

        # Load all data
        print(f"Loading data from {len(data_files)} files...")
        self.data = self._load_all_data()
        print(f"Loaded {len(self.data)} frames")

        # Compute statistics
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)

    def _load_all_data(self) -> np.ndarray:
        """Load and concatenate all radar data files"""
        all_data = []

        for file_path in self.data_files:
            try:
                if HAS_PYSTEPS:
                    # Use pysteps to load OdimH5
                    precip, _, _ = import_odim_h5(file_path)

                    # Convert to rain rate (mm/h)
                    precip, _ = conversion.to_rainrate(precip)

                    # Handle missing values
                    precip = np.nan_to_num(precip, nan=0.0)
                else:
                    # Fallback: load with h5py
                    import h5py
                    with h5py.File(file_path, 'r') as f:
                        # Try common paths
                        if 'dataset1/data1/data' in f:
                            precip = f['dataset1/data1/data'][:]
                        elif 'data' in f:
                            precip = f['data'][:]
                        else:
                            # Last resort
                            precip = np.array(f[list(f.keys())[0]])

                    # Simple dBZ to mm/h conversion
                    precip = self._dbz_to_mmh(precip)

                # Apply threshold
                precip[precip < self.threshold] = 0.0

                # Clip to max value
                precip = np.clip(precip, 0, self.max_precip)

                all_data.append(precip)

            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue

        if not all_data:
            raise ValueError("No data files could be loaded successfully")

        # Concatenate along time axis
        full_sequence = np.concatenate(all_data, axis=0)

        return full_sequence

    @staticmethod
    def _dbz_to_mmh(dbz: np.ndarray) -> np.ndarray:
        """
        Convert dBZ to mm/h using standard formula

        Parameters
        ----------
        dbz : np.ndarray
            Reflectivity in dBZ

        Returns
        -------
        precip : np.ndarray
            Precipitation rate in mm/h
        """
        # Z = 10^(dBZ/10)
        # R = (Z/200)^(1/1.6)  (Marshall-Palmer relation)

        # Clip dBZ to reasonable range
        dbz_clipped = np.clip(dbz, -10, 70)

        # Convert to mm/h
        Z = 10 ** (dbz_clipped / 10.0)
        R = (Z / 200.0) ** (1.0 / 1.6)

        return R

    def __len__(self) -> int:
        """Number of possible sequences in the dataset"""
        return len(self.data) - self.input_frames - self.output_frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample

        Returns
        -------
        x : torch.Tensor
            Input frames [input_frames, H, W]
        y : torch.Tensor
            Target frames [output_frames, H, W]
        """
        # Extract sequence
        sequence = self.data[idx:idx + self.input_frames + self.output_frames]

        # Normalize if needed
        if self.normalize:
            sequence = (sequence - self.mean) / (self.std + 1e-8)
        else:
            # Scale to [0, 1]
            sequence = np.clip(sequence / self.max_precip, 0, 1)

        # Split input and output
        x = sequence[:self.input_frames]
        y = sequence[self.input_frames:]

        # Apply augmentation
        if self.augment and np.random.rand() > 0.5:
            x, y = self._augment(x, y)

        # Convert to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y

    def _augment(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation

        Augmentations:
        - Random horizontal flip
        - Random vertical flip
        - Random 90-degree rotations
        """
        # Horizontal flip
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2)
            y = np.flip(y, axis=2)

        # Vertical flip
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)

        # Random rotation (0, 90, 180, 270 degrees)
        if np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            x = np.rot90(x, k, axes=(1, 2))
            y = np.rot90(y, k, axes=(1, 2))

        return x, y


class DGMRDataModule:
    """
    Data Module for DGMR Training

    Wraps dataset and provides train/val dataloaders.

    Parameters
    ----------
    train_files : list of str
        Training data file paths
    val_files : list of str
        Validation data file paths
    batch_size : int, default=4
        Batch size for training
    num_workers : int, default=4
        Number of data loading workers
    input_frames : int, default=12
        Number of input frames
    output_frames : int, default=24
        Number of output frames

    Examples
    --------
    >>> train_files = glob.glob("data/train/*.h5")
    >>> val_files = glob.glob("data/val/*.h5")
    >>> dm = DGMRDataModule(train_files, val_files, batch_size=4)
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()
    >>> for x, y in train_loader:
    ...     print(x.shape)  # [B, 12, H, W]
    ...     print(y.shape)  # [B, 24, H, W]
    ...     break
    """

    def __init__(
        self,
        train_files: List[str],
        val_files: List[str],
        batch_size: int = 4,
        num_workers: int = 4,
        input_frames: int = 12,
        output_frames: int = 24
    ):
        self.train_files = train_files
        self.val_files = val_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_frames = input_frames
        self.output_frames = output_frames

    def setup(self):
        """Setup datasets"""
        print("Setting up datasets...")

        self.train_dataset = PrecipitationSequenceDataset(
            self.train_files,
            input_frames=self.input_frames,
            output_frames=self.output_frames,
            augment=True
        )

        self.val_dataset = PrecipitationSequenceDataset(
            self.val_files,
            input_frames=self.input_frames,
            output_frames=self.output_frames,
            augment=False
        )

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )


def create_dataloaders_from_config(config: dict):
    """
    Create dataloaders from configuration dictionary

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - train_data_path: str
        - val_data_path: str
        - batch_size: int
        - num_workers: int
        - input_frames: int
        - output_frames: int

    Returns
    -------
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    """
    # Get file lists
    train_files = sorted(glob.glob(f"{config['train_data_path']}/*.h5"))
    val_files = sorted(glob.glob(f"{config['val_data_path']}/*.h5"))

    print(f"Found {len(train_files)} training files")
    print(f"Found {len(val_files)} validation files")

    # Create data module
    dm = DGMRDataModule(
        train_files=train_files,
        val_files=val_files,
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 4),
        input_frames=config.get('input_frames', 12),
        output_frames=config.get('output_frames', 24)
    )

    dm.setup()

    return dm.train_dataloader(), dm.val_dataloader()


if __name__ == "__main__":
    # Test the data module
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Create dataloaders
    config = {
        'train_data_path': args.train_path,
        'val_data_path': args.val_path,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'input_frames': 12,
        'output_frames': 24
    }

    train_loader, val_loader = create_dataloaders_from_config(config)

    # Test iteration
    print("\nTesting data loading...")
    for x, y in train_loader:
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
        print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
        break

    print("\nData module test passed!")
