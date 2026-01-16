#!/usr/bin/env python3
"""
Preprocessing module for ECG signals - PhysioNet Challenge 2025
Handles filtering, normalization, artifact removal, and resampling
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


class ECGPreprocessor:
    """Complete preprocessing pipeline for ECG signals"""

    def __init__(self,
                 target_fs=500,
                 lowcut=0.5,
                 highcut=40,
                 notch_freq=60,
                 normalize=True):
        """
        Initialize ECG preprocessor

        Args:
            target_fs: Target sampling frequency (Hz)
            lowcut: Lowcut frequency for bandpass filter (Hz)
            highcut: Highcut frequency for bandpass filter (Hz)
            notch_freq: Notch filter frequency for powerline interference (Hz)
            normalize: Whether to normalize signals
        """
        self.target_fs = target_fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.normalize = normalize

    def bandpass_filter(self, ecg_signal, fs, order=4):
        """
        Apply bandpass filter to remove baseline wander and high-frequency noise

        Args:
            ecg_signal: Input signal (n_samples, n_leads)
            fs: Sampling frequency
            order: Filter order

        Returns:
            Filtered signal
        """
        nyq = 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq

        # Ensure frequencies are valid
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))

        if low >= high:
            return ecg_signal

        b, a = signal.butter(order, [low, high], btype='band')

        # Apply filter to each lead
        filtered = np.zeros_like(ecg_signal)
        for i in range(ecg_signal.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, ecg_signal[:, i])

        return filtered

    def notch_filter(self, ecg_signal, fs, quality_factor=30):
        """
        Apply notch filter to remove powerline interference (50/60 Hz)

        Args:
            ecg_signal: Input signal (n_samples, n_leads)
            fs: Sampling frequency
            quality_factor: Quality factor of notch filter

        Returns:
            Filtered signal
        """
        nyq = 0.5 * fs
        freq = self.notch_freq / nyq

        if freq >= 1.0 or freq <= 0:
            return ecg_signal

        b, a = signal.iirnotch(freq, quality_factor)

        # Apply filter to each lead
        filtered = np.zeros_like(ecg_signal)
        for i in range(ecg_signal.shape[1]):
            filtered[:, i] = signal.filtfilt(b, a, ecg_signal[:, i])

        return filtered

    def remove_baseline_wander(self, ecg_signal, window_size=200):
        """
        Remove baseline wander using median filter

        Args:
            ecg_signal: Input signal (n_samples, n_leads)
            window_size: Window size for median filter

        Returns:
            Signal with baseline removed
        """
        corrected = np.zeros_like(ecg_signal)

        for i in range(ecg_signal.shape[1]):
            baseline = signal.medfilt(ecg_signal[:, i], kernel_size=window_size | 1)  # Ensure odd
            corrected[:, i] = ecg_signal[:, i] - baseline

        return corrected

    def detect_and_remove_artifacts(self, ecg_signal, threshold=5.0):
        """
        Detect and remove artifacts using amplitude threshold

        Args:
            ecg_signal: Input signal (n_samples, n_leads)
            threshold: Z-score threshold for artifact detection

        Returns:
            Signal with artifacts removed (interpolated)
        """
        cleaned = ecg_signal.copy()

        for i in range(ecg_signal.shape[1]):
            lead_signal = ecg_signal[:, i]

            # Calculate z-scores
            mean = np.mean(lead_signal)
            std = np.std(lead_signal)

            if std < 1e-6:
                continue

            z_scores = np.abs((lead_signal - mean) / std)

            # Find artifacts
            artifact_mask = z_scores > threshold

            if artifact_mask.sum() > 0 and artifact_mask.sum() < len(lead_signal) * 0.5:
                # Interpolate artifacts
                valid_indices = np.where(~artifact_mask)[0]
                artifact_indices = np.where(artifact_mask)[0]

                if len(valid_indices) > 1:
                    f = interp1d(valid_indices, lead_signal[valid_indices],
                                kind='linear', fill_value='extrapolate')
                    cleaned[artifact_indices, i] = f(artifact_indices)

        return cleaned

    def resample_signal(self, ecg_signal, fs_original):
        """
        Resample signal to target frequency

        Args:
            ecg_signal: Input signal (n_samples, n_leads)
            fs_original: Original sampling frequency

        Returns:
            Resampled signal
        """
        if fs_original == self.target_fs:
            return ecg_signal

        num_samples_new = int(len(ecg_signal) * self.target_fs / fs_original)
        resampled = signal.resample(ecg_signal, num_samples_new, axis=0)

        return resampled

    def normalize_per_lead(self, ecg_signal):
        """
        Normalize each lead independently (z-score normalization)

        Args:
            ecg_signal: Input signal (n_samples, n_leads)

        Returns:
            Normalized signal
        """
        normalized = np.zeros_like(ecg_signal)

        for i in range(ecg_signal.shape[1]):
            mean = np.mean(ecg_signal[:, i])
            std = np.std(ecg_signal[:, i])

            if std > 1e-6:
                normalized[:, i] = (ecg_signal[:, i] - mean) / std
            else:
                normalized[:, i] = ecg_signal[:, i] - mean

        return normalized

    def pad_or_truncate(self, ecg_signal, target_length=5000):
        """
        Pad or truncate signal to fixed length

        Args:
            ecg_signal: Input signal (n_samples, n_leads)
            target_length: Target number of samples

        Returns:
            Signal with fixed length
        """
        current_length = ecg_signal.shape[0]

        if current_length == target_length:
            return ecg_signal

        elif current_length < target_length:
            # Pad with zeros
            pad_length = target_length - current_length
            padding = np.zeros((pad_length, ecg_signal.shape[1]))
            return np.vstack([ecg_signal, padding])

        else:
            # Truncate from center
            start = (current_length - target_length) // 2
            return ecg_signal[start:start+target_length, :]

    def preprocess(self, ecg_signal, fs_original, target_length=5000):
        """
        Complete preprocessing pipeline

        Args:
            ecg_signal: Input signal (n_samples, n_leads)
            fs_original: Original sampling frequency
            target_length: Target signal length in samples

        Returns:
            Preprocessed signal
        """
        # Step 1: Bandpass filter
        processed = self.bandpass_filter(ecg_signal, fs_original)

        # Step 2: Notch filter (remove powerline interference)
        processed = self.notch_filter(processed, fs_original)

        # Step 3: Remove baseline wander
        processed = self.remove_baseline_wander(processed)

        # Step 4: Detect and remove artifacts
        processed = self.detect_and_remove_artifacts(processed)

        # Step 5: Resample to target frequency
        processed = self.resample_signal(processed, fs_original)

        # Step 6: Normalize per lead
        if self.normalize:
            processed = self.normalize_per_lead(processed)

        # Step 7: Pad or truncate to fixed length
        processed = self.pad_or_truncate(processed, target_length)

        return processed


class ECGAugmentation:
    """Data augmentation for ECG signals"""

    def __init__(self,
                 time_warp_sigma=0.2,
                 amplitude_scale_range=(0.9, 1.1),
                 noise_level=0.01,
                 time_shift_range=50,
                 lead_dropout_prob=0.1):
        """
        Initialize ECG augmentation

        Args:
            time_warp_sigma: Sigma for time warping
            amplitude_scale_range: Range for amplitude scaling
            noise_level: Standard deviation of Gaussian noise
            time_shift_range: Range for time shifting (samples)
            lead_dropout_prob: Probability of dropping a lead
        """
        self.time_warp_sigma = time_warp_sigma
        self.amplitude_scale_range = amplitude_scale_range
        self.noise_level = noise_level
        self.time_shift_range = time_shift_range
        self.lead_dropout_prob = lead_dropout_prob

    def time_warp(self, ecg_signal):
        """
        Apply time warping (temporal distortion)
        """
        from scipy.ndimage import map_coordinates

        time_steps = np.arange(len(ecg_signal))
        warp = np.random.normal(0, self.time_warp_sigma, len(ecg_signal)).cumsum()
        warped_time = time_steps + warp
        warped_time = np.clip(warped_time, 0, len(ecg_signal) - 1)

        warped = np.zeros_like(ecg_signal)
        for i in range(ecg_signal.shape[1]):
            warped[:, i] = map_coordinates(ecg_signal[:, i], [warped_time],
                                          order=1, mode='nearest')

        return warped

    def amplitude_scale(self, ecg_signal):
        """
        Scale amplitude randomly
        """
        scale = np.random.uniform(*self.amplitude_scale_range)
        return ecg_signal * scale

    def add_gaussian_noise(self, ecg_signal):
        """
        Add Gaussian noise
        """
        noise = np.random.normal(0, self.noise_level, ecg_signal.shape)
        return ecg_signal + noise

    def time_shift(self, ecg_signal):
        """
        Shift signal in time
        """
        shift = np.random.randint(-self.time_shift_range, self.time_shift_range)
        return np.roll(ecg_signal, shift, axis=0)

    def lead_dropout(self, ecg_signal):
        """
        Randomly drop (zero out) some leads
        """
        mask = np.random.random(ecg_signal.shape[1]) > self.lead_dropout_prob
        augmented = ecg_signal.copy()
        augmented[:, ~mask] = 0
        return augmented

    def baseline_wander_augmentation(self, ecg_signal, amplitude=0.05, freq=0.3):
        """
        Add realistic baseline wander
        """
        fs = 500  # Assumed sampling frequency
        t = np.arange(len(ecg_signal)) / fs
        wander = amplitude * np.sin(2 * np.pi * freq * t)
        return ecg_signal + wander[:, np.newaxis]

    def augment(self, ecg_signal, prob=0.5):
        """
        Apply random augmentations

        Args:
            ecg_signal: Input signal (n_samples, n_leads)
            prob: Probability of applying each augmentation

        Returns:
            Augmented signal
        """
        augmented = ecg_signal.copy()

        if np.random.random() < prob:
            augmented = self.time_warp(augmented)

        if np.random.random() < prob:
            augmented = self.amplitude_scale(augmented)

        if np.random.random() < prob:
            augmented = self.add_gaussian_noise(augmented)

        if np.random.random() < prob:
            augmented = self.time_shift(augmented)

        if np.random.random() < prob * 0.5:  # Less frequent
            augmented = self.lead_dropout(augmented)

        if np.random.random() < prob * 0.5:  # Less frequent
            augmented = self.baseline_wander_augmentation(augmented)

        return augmented


if __name__ == "__main__":
    # Test preprocessing
    print("Testing ECG Preprocessor...")

    # Create dummy ECG signal (10 seconds at 400 Hz, 12 leads)
    fs = 400
    duration = 10
    n_samples = fs * duration
    n_leads = 12

    dummy_signal = np.random.randn(n_samples, n_leads) * 0.5

    # Add some artifacts
    dummy_signal[1000:1020, 0] = 10  # Spike artifact

    # Preprocess
    preprocessor = ECGPreprocessor(target_fs=500)
    processed = preprocessor.preprocess(dummy_signal, fs_original=fs, target_length=5000)

    print(f"Original shape: {dummy_signal.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"Processed mean: {processed.mean():.4f}, std: {processed.std():.4f}")

    # Test augmentation
    print("\nTesting ECG Augmentation...")
    augmenter = ECGAugmentation()
    augmented = augmenter.augment(processed, prob=0.5)

    print(f"Augmented shape: {augmented.shape}")
    print("✓ Preprocessing and augmentation working correctly!")
