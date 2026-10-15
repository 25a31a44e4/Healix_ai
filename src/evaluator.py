"""Comprehensive Model Evaluation and Clinical Decision Support Metrics.

Implements:
1. AUC-PR (Average Precision)
2. Recall@50 = (Actual Positives in Top 50) / (Total Actual Positives in Cohort)
3. Precision@50 = (Actual Positives in Top 50) / min(50, Cohort Size)
4. Expected Calibration Error (ECE) across documented probability bins
5. Brier Score
6. Confusion Matrix at documented operating threshold
7. PR Curve and Reliability Diagram data generators
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """Computes Expected Calibration Error (ECE) and bin-level reliability statistics.
    
    Divides probabilities into n_bins equal-width intervals [0, 1/n), [1/n, 2/n), etc.
    ECE = sum_{b=1}^B (|B_b| / N) * | acc(B_b) - conf(B_b) |
    
    Args:
        y_true: 1D binary ground truth array.
        y_prob: 1D array of predicted probabilities.
        n_bins: Number of equal-width bins (default 10).
        
    Returns:
        Tuple of (ece_score, list_of_bin_dictionaries)
    """
    if len(y_true) == 0:
        return 0.0, []

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_details = []
    total_samples = len(y_true)
    weighted_error_sum = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        # Include upper edge on final bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            avg_confidence = float(np.mean(y_prob[in_bin]))
            observed_accuracy = float(np.mean(y_true[in_bin]))
            abs_gap = abs(observed_accuracy - avg_confidence)
            weighted_error_sum += (bin_count / total_samples) * abs_gap
            
            bin_details.append({
                "bin_index": i + 1,
                "bin_range": f"{bin_lower:.1f} - {bin_upper:.1f}",
                "count": bin_count,
                "avg_confidence": round(avg_confidence, 4),
                "observed_accuracy": round(observed_accuracy, 4),
                "calibration_gap": round(abs_gap, 4),
            })
        else:
            bin_details.append({
                "bin_index": i + 1,
                "bin_range": f"{bin_lower:.1f} - {bin_upper:.1f}",
                "count": 0,
                "avg_confidence": round((bin_lower + bin_upper) / 2.0, 4),
                "observed_accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    ece = float(weighted_error_sum)
    return round(ece, 4), bin_details


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.30,
    capacity: int = 50
) -> Dict[str, Any]:
    """Computes all primary evaluation metrics for clinical decision support.
    
    Args:
        y_true: 1D ground truth binary array.
        y_prob: 1D predicted probability array.
        threshold: Classification decision threshold for confusion matrix.
        capacity: Operational capacity for top-K outreach (default 50).
        
    Returns:
        Dict of computed evaluation metrics and curves.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_samples = len(y_true)
    n_positives = int(np.sum(y_true))

    # Guard against empty cohorts or single-class cohorts
    if n_samples == 0:
        return {"error": "Evaluation cohort contains zero records."}

    # 1. AUC-PR (Average Precision)
    if n_positives == 0 or n_positives == n_samples:
        auc_pr = 0.0 if n_positives == 0 else 1.0
        roc_auc = 0.5
    else:
        auc_pr = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))

    # 2. Recall@K and Precision@K
    # Sort indices by predicted probability descending
    k = min(capacity, n_samples)
    sorted_indices = np.argsort(y_prob)[::-1]
    top_k_indices = sorted_indices[:k]
    
    top_k_positives = int(np.sum(y_true[top_k_indices]))
    
    # Recall@K = (positives identified in top K) / (total true positives in cohort)
    recall_at_k = float(top_k_positives / n_positives) if n_positives > 0 else 0.0
    
    # Precision@K = (positives in top K) / K
    precision_at_k = float(top_k_positives / k) if k > 0 else 0.0

    # 3. Calibration: ECE & Brier Score
    ece, bin_details = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    brier = float(brier_score_loss(y_true, y_prob))

    # 4. Confusion Matrix at documented operating threshold
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # 5. Precision-Recall curve points for interactive plotting
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
    
    # Sample down curve points if too dense (for quick plotly rendering)
    step = max(1, len(recalls) // 100)
    pr_curve_data = {
        "recall": [round(float(r), 4) for r in recalls[::step]],
        "precision": [round(float(p), 4) for p in precisions[::step]],
    }

    return {
        "n_samples": n_samples,
        "n_positives": n_positives,
        "prevalence": round(float(n_positives / n_samples), 4),
        "auc_pr": round(auc_pr, 4),
        "roc_auc": round(roc_auc, 4),
        "recall_at_capacity": round(recall_at_k, 4),
        "precision_at_capacity": round(precision_at_k, 4),
        "capacity_k": k,
        "top_k_true_positives": top_k_positives,
        "expected_calibration_error": round(ece, 4),
        "brier_score": round(brier, 4),
        "threshold_used": threshold,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "sensitivity_recall": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "ppv_precision": round(ppv, 4),
            "npv": round(npv, 4),
        },
        "calibration_bins": bin_details,
        "pr_curve_data": pr_curve_data,
    }
