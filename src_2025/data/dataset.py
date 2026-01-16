#!/usr/bin/env python3
"""
PyTorch Dataset for ECG signals - PhysioNet Challenge 2025
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

# Add parent directory to path to import helper_code
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    from src.utils import helper_code
except ImportError:
    # Fallback if structure is different
    import helper_code

from src_2025.data.preprocessing import ECGPreprocessor, ECGAugmentation


class ChagasECGDataset(Dataset):
    """PyTorch Dataset for Chagas disease detection from ECG"""

    def __init__(self,
                 data_folder,
                 preprocessor=None,
                 augmenter=None,
                 target_length=5000,
                 target_fs=500,
                 augment=False,
                 cache_preprocessed=True):
        """
        Initialize dataset

        Args:
            data_folder: Path to folder containing WFDB records
            preprocessor: ECGPreprocessor instance
            augmenter: ECGAugmentation instance
            target_length: Target signal length in samples
            target_fs: Target sampling frequency
            augment: Whether to apply augmentation
            cache_preprocessed: Whether to cache preprocessed signals in memory
        """
        self.data_folder = data_folder
        self.target_length = target_length
        self.target_fs = target_fs
        self.augment = augment
        self.cache_preprocessed = cache_preprocessed

        # Initialize preprocessor
        if preprocessor is None:
            self.preprocessor = ECGPreprocessor(target_fs=target_fs)
        else:
            self.preprocessor = preprocessor

        # Initialize augmenter
        if augmenter is None:
            self.augmenter = ECGAugmentation()
        else:
            self.augmenter = augmenter

        # Find all records
        print(f"Loading records from {data_folder}...")
        self.records = helper_code.find_records(data_folder)
        print(f"Found {len(self.records)} records")

        # Load labels and filter valid records
        self.valid_records = []
        self.labels = []
        self.demographics = []

        for record in self.records:
            try:
                # Load header
                header = helper_code.load_header(record)

                # Get label
                label = helper_code.get_labels_from_header(header)

                # Check if label exists and is valid
                if label is not None and len(label) > 0:
                    chagas_label = int(label[0]) if isinstance(label[0], (int, float, str)) else 0

                    # Get demographics
                    age = helper_code.get_age(header)
                    sex = helper_code.get_sex(header)

                    self.valid_records.append(record)
                    self.labels.append(chagas_label)
                    self.demographics.append({
                        'age': age if age is not None else -1,
                        'sex': sex if sex is not None else 'Unknown'
                    })

            except Exception as e:
                warnings.warn(f"Error loading record {record}: {e}")
                continue

        print(f"Valid records with labels: {len(self.valid_records)}")

        if len(self.valid_records) == 0:
            raise ValueError("No valid records found with labels!")

        # Calculate class distribution
        positive_count = sum(self.labels)
        negative_count = len(self.labels) - positive_count
        print(f"Class distribution: Positive={positive_count}, Negative={negative_count}")
        print(f"Positive rate: {positive_count / len(self.labels) * 100:.2f}%")

        # Cache for preprocessed signals
        self.cache = {} if cache_preprocessed else None

    def __len__(self):
        return len(self.valid_records)

    def __getitem__(self, idx):
        """
        Get item by index

        Returns:
            signal: Preprocessed ECG signal (n_samples, n_leads)
            label: Chagas disease label (0 or 1)
            demographics: Dictionary with age and sex
        """
        record = self.valid_records[idx]
        label = self.labels[idx]
        demographics = self.demographics[idx]

        # Check cache first
        if self.cache is not None and idx in self.cache:
            signal = self.cache[idx].copy()
        else:
            # Load signal
            signal, fields = helper_code.load_signals(record)

            if signal is None:
                # Return zero signal if loading fails
                signal = np.zeros((self.target_length, 12), dtype=np.float32)
            else:
                # Get sampling frequency
                header = helper_code.load_header(record)
                fs = helper_code.get_sampling_frequency(header)

                # Handle NaN values
                if np.isnan(signal).any():
                    signal = np.nan_to_num(signal, nan=0.0)

                # Ensure we have 12 leads
                if signal.shape[1] < 12:
                    # Pad with zeros
                    padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
                    signal = np.hstack([signal, padding])
                elif signal.shape[1] > 12:
                    # Take first 12 leads
                    signal = signal[:, :12]

                # Preprocess
                signal = self.preprocessor.preprocess(
                    signal,
                    fs_original=fs,
                    target_length=self.target_length
                )

            # Cache preprocessed signal
            if self.cache is not None:
                self.cache[idx] = signal.copy()

        # Apply augmentation if training
        if self.augment:
            signal = self.augmenter.augment(signal, prob=0.5)

        # Convert to tensor (channels first: [n_leads, n_samples])
        signal_tensor = torch.from_numpy(signal.T).float()  # Transpose to (12, 5000)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return signal_tensor, label_tensor, demographics


def create_dataloaders(data_folder,
                      batch_size=32,
                      train_ratio=0.8,
                      num_workers=4,
                      target_length=5000,
                      target_fs=500,
                      random_seed=42):
    """
    Create train and validation dataloaders

    Args:
        data_folder: Path to data folder
        batch_size: Batch size
        train_ratio: Ratio of training data
        num_workers: Number of workers for DataLoader
        target_length: Target signal length
        target_fs: Target sampling frequency
        random_seed: Random seed for splitting

    Returns:
        train_loader, val_loader
    """
    # Create full dataset (no augmentation for splitting)
    full_dataset = ChagasECGDataset(
        data_folder=data_folder,
        target_length=target_length,
        target_fs=target_fs,
        augment=False,
        cache_preprocessed=True
    )

    # Split indices
    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)

    indices = np.arange(n_total)
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create train dataset with augmentation
    train_dataset = ChagasECGDataset(
        data_folder=data_folder,
        target_length=target_length,
        target_fs=target_fs,
        augment=True,
        cache_preprocessed=True
    )

    # Create validation dataset without augmentation
    val_dataset = ChagasECGDataset(
        data_folder=data_folder,
        target_length=target_length,
        target_fs=target_fs,
        augment=False,
        cache_preprocessed=True
    )

    # Create subset datasets
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Create balanced sampler for training (handle class imbalance)
    train_labels = [full_dataset.labels[i] for i in train_indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]

    from torch.utils.data import WeightedRandomSampler
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing Chagas ECG Dataset...")

    # This would need actual data to run
    # Example usage:
    # dataset = ChagasECGDataset(
    #     data_folder="data_2025/SaMi-Trop",
    #     target_length=5000,
    #     target_fs=500,
    #     augment=True
    # )
    #
    # print(f"Dataset size: {len(dataset)}")
    #
    # # Get a sample
    # signal, label, demographics = dataset[0]
    # print(f"Signal shape: {signal.shape}")
    # print(f"Label: {label}")
    # print(f"Demographics: {demographics}")

    print("✓ Dataset module ready!")
