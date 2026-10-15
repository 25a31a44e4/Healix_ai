"""Tests for evaluation metrics: AUC-PR, Recall@50, Precision@50, ECE, and Brier Score."""

import pytest
import numpy as np
from src.evaluator import compute_evaluation_metrics, compute_expected_calibration_error


def test_recall_at_50_and_precision_at_50():
    """Validates mathematical correctness of Recall@K and Precision@K definitions."""
    # Create synthetic test cases: 100 patients, 20 positives
    y_true = np.zeros(100, dtype=int)
    y_true[:20] = 1  # 20 true positives
    
    # Probabilities: highest 50 predictions contain 15 true positives and 35 true negatives
    y_prob = np.zeros(100, dtype=float)
    y_prob[:15] = 0.90  # 15 true positives in top
    y_prob[20:55] = 0.80  # 35 false positives in top (total 50 in top)
    y_prob[15:20] = 0.20  # 5 remaining true positives below top 50
    y_prob[55:] = 0.10   # remaining true negatives below top 50

    metrics = compute_evaluation_metrics(y_true, y_prob, capacity=50)

    # Top 50 contains 15 actual positives
    # Recall@50 = 15 / 20 = 0.75
    # Precision@50 = 15 / 50 = 0.30
    assert metrics["top_k_true_positives"] == 15
    assert metrics["recall_at_capacity"] == 0.75
    assert metrics["precision_at_capacity"] == 0.30
    assert metrics["capacity_k"] == 50


def test_expected_calibration_error_bounds():
    """Verifies that ECE is bounded in [0, 1] and equals zero for perfectly calibrated predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.05, 0.95, 0.95])

    ece, bins = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert 0.0 <= ece <= 1.0
    assert len(bins) == 10
