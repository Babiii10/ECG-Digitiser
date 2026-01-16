#!/usr/bin/env python3
"""
Prediction script for PhysioNet Challenge 2025
Generate predictions on test data
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.utils import helper_code
except ImportError:
    import helper_code

from src_2025.data.dataset import ChagasECGDataset
from src_2025.data.preprocessing import ECGPreprocessor, ECGAugmentation
from src_2025.models.resnet1d import (
    create_resnet1d_small,
    create_resnet1d_medium,
    create_resnet1d_large,
    create_seresnet1d
)


class Predictor:
    """Prediction class with TTA support"""

    def __init__(self, model_path, config_path=None, device='cuda'):
        """
        Initialize predictor

        Args:
            model_path: Path to model checkpoint
            config_path: Path to config file (optional)
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Use defaults
            self.config = {
                'model_type': 'resnet1d_medium',
                'target_length': 5000,
                'target_fs': 500
            }

        # Load model
        self.model = self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")

    def _load_model(self, model_path):
        """Load model from checkpoint"""
        # Create model
        model_type = self.config.get('model_type', 'resnet1d_medium')

        if model_type == 'resnet1d_small':
            model = create_resnet1d_small()
        elif model_type == 'resnet1d_medium':
            model = create_resnet1d_medium()
        elif model_type == 'resnet1d_large':
            model = create_resnet1d_large()
        elif model_type == 'seresnet1d':
            model = create_seresnet1d()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        return model

    def predict_single(self, signal_tensor):
        """
        Predict on a single signal

        Args:
            signal_tensor: Signal tensor (1, n_leads, n_samples)

        Returns:
            Probability [0, 1]
        """
        with torch.no_grad():
            signal_tensor = signal_tensor.to(self.device)
            output = self.model(signal_tensor)
            prob = torch.sigmoid(output).cpu().item()

        return prob

    def predict_with_tta(self, signal_tensor, n_augmentations=5):
        """
        Predict with Test-Time Augmentation

        Args:
            signal_tensor: Signal tensor (1, n_leads, n_samples)
            n_augmentations: Number of augmentations

        Returns:
            Average probability [0, 1]
        """
        # Original prediction
        probs = [self.predict_single(signal_tensor)]

        # Augmented predictions
        augmenter = ECGAugmentation(
            time_warp_sigma=0.1,
            amplitude_scale_range=(0.95, 1.05),
            noise_level=0.005,
            time_shift_range=20,
            lead_dropout_prob=0.05
        )

        signal_np = signal_tensor.squeeze(0).cpu().numpy().T  # (n_samples, n_leads)

        for _ in range(n_augmentations):
            # Augment
            augmented = augmenter.augment(signal_np, prob=0.5)

            # Convert back to tensor
            aug_tensor = torch.from_numpy(augmented.T).float().unsqueeze(0)

            # Predict
            prob = self.predict_single(aug_tensor)
            probs.append(prob)

        # Return average
        return np.mean(probs)

    def predict_dataset(self, data_folder, use_tta=False, n_augmentations=5):
        """
        Predict on entire dataset

        Args:
            data_folder: Path to data folder
            use_tta: Whether to use test-time augmentation
            n_augmentations: Number of augmentations for TTA

        Returns:
            Dictionary with record names and predictions
        """
        # Create dataset (no augmentation, no labels required)
        preprocessor = ECGPreprocessor(
            target_fs=self.config.get('target_fs', 500)
        )

        # Find all records
        records = helper_code.find_records(data_folder)
        print(f"Found {len(records)} records")

        results = {}

        for record in tqdm(records, desc="Predicting"):
            try:
                # Load signal
                signal, fields = helper_code.load_signals(record)

                if signal is None:
                    results[record] = 0.0  # Default prediction
                    continue

                # Get sampling frequency
                header = helper_code.load_header(record)
                fs = helper_code.get_sampling_frequency(header)

                # Handle NaN values
                if np.isnan(signal).any():
                    signal = np.nan_to_num(signal, nan=0.0)

                # Ensure we have 12 leads
                if signal.shape[1] < 12:
                    padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
                    signal = np.hstack([signal, padding])
                elif signal.shape[1] > 12:
                    signal = signal[:, :12]

                # Preprocess
                signal_processed = preprocessor.preprocess(
                    signal,
                    fs_original=fs,
                    target_length=self.config.get('target_length', 5000)
                )

                # Convert to tensor (channels first)
                signal_tensor = torch.from_numpy(signal_processed.T).float().unsqueeze(0)

                # Predict
                if use_tta:
                    prob = self.predict_with_tta(signal_tensor, n_augmentations)
                else:
                    prob = self.predict_single(signal_tensor)

                results[record] = prob

            except Exception as e:
                print(f"Error processing {record}: {e}")
                results[record] = 0.0

        return results


def main():
    parser = argparse.ArgumentParser(description='Predict Chagas disease from ECG')

    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data_folder', type=str, required=True,
                      help='Path to test data folder')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to output CSV file')
    parser.add_argument('--config_path', type=str, default=None,
                      help='Path to config file')

    parser.add_argument('--use_tta', action='store_true',
                      help='Use test-time augmentation')
    parser.add_argument('--n_augmentations', type=int, default=5,
                      help='Number of augmentations for TTA')

    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to use')

    args = parser.parse_args()

    # Create predictor
    predictor = Predictor(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )

    # Predict
    print(f"\nPredicting on {args.data_folder}...")
    results = predictor.predict_dataset(
        data_folder=args.data_folder,
        use_tta=args.use_tta,
        n_augmentations=args.n_augmentations
    )

    # Save results
    df = pd.DataFrame([
        {'record': record, 'probability': prob}
        for record, prob in results.items()
    ])

    df = df.sort_values('probability', ascending=False)
    df.to_csv(args.output_file, index=False)

    print(f"\nPredictions saved to {args.output_file}")
    print(f"Total records: {len(results)}")
    print(f"Mean probability: {df['probability'].mean():.4f}")
    print(f"Median probability: {df['probability'].median():.4f}")

    # Show top 10
    print("\nTop 10 predictions:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
