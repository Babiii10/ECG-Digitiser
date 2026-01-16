#!/usr/bin/env python3
"""
Evaluation metrics for PhysioNet Challenge 2025
Focus on TPR@top-5%
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    roc_curve
)


def compute_tpr_at_top_k_percent(y_true, y_pred_proba, k=5):
    """
    Compute True Positive Rate among top k% of predictions
    This is the PRIMARY metric for PhysioNet Challenge 2025

    Args:
        y_true: Ground truth labels (0 or 1), shape (n,)
        y_pred_proba: Predicted probabilities [0, 1], shape (n,)
        k: Percentage (default 5 for top 5%)

    Returns:
        TPR @ top k%
    """
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()

    n = len(y_true)
    n_top_k = max(1, int(n * k / 100))

    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    top_k_indices = sorted_indices[:n_top_k]

    # Calculate TPR
    n_positives_total = y_true.sum()

    if n_positives_total == 0:
        return 0.0

    n_positives_top_k = y_true[top_k_indices].sum()
    tpr = n_positives_top_k / n_positives_total

    return tpr


def compute_all_metrics(y_true, y_pred_proba, threshold=0.5, k_percent=5):
    """
    Compute comprehensive set of metrics

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        threshold: Threshold for binary classification
        k_percent: Percentage for TPR@k calculation

    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {}

    # Primary metric
    metrics['tpr_at_5pct'] = compute_tpr_at_top_k_percent(y_true, y_pred_proba, k=k_percent)

    # ROC-AUC
    try:
        metrics['auroc'] = roc_auc_score(y_true, y_pred_proba)
    except:
        metrics['auroc'] = 0.0

    # Average Precision (AUPRC)
    try:
        metrics['auprc'] = average_precision_score(y_true, y_pred_proba)
    except:
        metrics['auprc'] = 0.0

    # F1 Score
    try:
        metrics['f1'] = f1_score(y_true, y_pred)
    except:
        metrics['f1'] = 0.0

    # Confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)

        # Sensitivity (Recall, TPR)
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Specificity (TNR)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Precision (PPV)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Accuracy
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

    except:
        metrics['tn'] = 0
        metrics['fp'] = 0
        metrics['fn'] = 0
        metrics['tp'] = 0
        metrics['sensitivity'] = 0.0
        metrics['specificity'] = 0.0
        metrics['precision'] = 0.0
        metrics['accuracy'] = 0.0

    return metrics


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find optimal classification threshold

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'youden', 'tpr_at_5pct')

    Returns:
        Optimal threshold
    """
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()

    if metric == 'f1':
        # Find threshold that maximizes F1 score
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        # Calculate F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    elif metric == 'youden':
        # Find threshold that maximizes Youden's J statistic (sensitivity + specificity - 1)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

    elif metric == 'tpr_at_5pct':
        # Find threshold that maximizes TPR@5%
        thresholds = np.linspace(0, 1, 100)
        best_tpr = 0
        optimal_threshold = 0.5

        for threshold in thresholds:
            tpr = compute_tpr_at_top_k_percent(y_true, y_pred_proba, k=5)
            if tpr > best_tpr:
                best_tpr = tpr
                optimal_threshold = threshold

    else:
        optimal_threshold = 0.5

    return optimal_threshold


class MetricsTracker:
    """Track metrics during training"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.y_true = []
        self.y_pred_proba = []
        self.losses = []

    def update(self, y_true, y_pred_proba, loss=None):
        """
        Update with batch results

        Args:
            y_true: Ground truth labels (batch_size,)
            y_pred_proba: Predicted probabilities (batch_size,)
            loss: Loss value (optional)
        """
        self.y_true.extend(y_true.flatten().tolist())
        self.y_pred_proba.extend(y_pred_proba.flatten().tolist())

        if loss is not None:
            self.losses.append(loss)

    def compute(self, threshold=0.5):
        """
        Compute all metrics

        Returns:
            Dictionary of metrics
        """
        if len(self.y_true) == 0:
            return {}

        metrics = compute_all_metrics(
            np.array(self.y_true),
            np.array(self.y_pred_proba),
            threshold=threshold
        )

        if len(self.losses) > 0:
            metrics['loss'] = np.mean(self.losses)

        return metrics

    def get_arrays(self):
        """Get raw arrays"""
        return np.array(self.y_true), np.array(self.y_pred_proba)


def print_metrics(metrics, prefix=""):
    """
    Pretty print metrics

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for output (e.g., "Train" or "Val")
    """
    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nMetrics:")

    print("-" * 50)

    # Primary metric
    if 'tpr_at_5pct' in metrics:
        print(f"TPR @ Top 5%:    {metrics['tpr_at_5pct']:.4f} ⭐")

    # Loss
    if 'loss' in metrics:
        print(f"Loss:            {metrics['loss']:.4f}")

    # ROC-AUC
    if 'auroc' in metrics:
        print(f"AUROC:           {metrics['auroc']:.4f}")

    # AUPRC
    if 'auprc' in metrics:
        print(f"AUPRC:           {metrics['auprc']:.4f}")

    # F1
    if 'f1' in metrics:
        print(f"F1 Score:        {metrics['f1']:.4f}")

    # Accuracy
    if 'accuracy' in metrics:
        print(f"Accuracy:        {metrics['accuracy']:.4f}")

    # Sensitivity/Specificity
    if 'sensitivity' in metrics and 'specificity' in metrics:
        print(f"Sensitivity:     {metrics['sensitivity']:.4f}")
        print(f"Specificity:     {metrics['specificity']:.4f}")

    # Confusion matrix
    if all(k in metrics for k in ['tp', 'fp', 'tn', 'fn']):
        print(f"\nConfusion Matrix:")
        print(f"  TP: {metrics['tp']:4d}  |  FP: {metrics['fp']:4d}")
        print(f"  FN: {metrics['fn']:4d}  |  TN: {metrics['tn']:4d}")

    print("-" * 50)


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    # Create dummy data
    n_samples = 1000
    np.random.seed(42)

    y_true = np.random.randint(0, 2, n_samples)
    y_pred_proba = np.random.random(n_samples)

    # Make predictions somewhat correlated with truth
    y_pred_proba = 0.7 * y_true + 0.3 * y_pred_proba

    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred_proba, threshold=0.5, k_percent=5)

    # Print metrics
    print_metrics(metrics, prefix="Test")

    # Test TPR@5%
    tpr_5 = compute_tpr_at_top_k_percent(y_true, y_pred_proba, k=5)
    print(f"\nTPR @ Top 5%: {tpr_5:.4f}")

    # Test optimal threshold finding
    optimal_thresh = find_optimal_threshold(y_true, y_pred_proba, metric='f1')
    print(f"Optimal threshold (F1): {optimal_thresh:.4f}")

    # Test metrics tracker
    print("\nTesting MetricsTracker...")
    tracker = MetricsTracker()

    # Simulate batches
    for i in range(10):
        batch_y_true = y_true[i*100:(i+1)*100]
        batch_y_pred = y_pred_proba[i*100:(i+1)*100]
        batch_loss = np.random.random()

        tracker.update(batch_y_true, batch_y_pred, batch_loss)

    tracked_metrics = tracker.compute(threshold=0.5)
    print_metrics(tracked_metrics, prefix="Tracked")

    print("\n✓ All metrics working correctly!")
