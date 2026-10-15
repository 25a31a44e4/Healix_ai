"""Standalone CLI Model Evaluation Script for CarePulse AI.

Evaluates the calibrated clinical readmission model on the held-out test split.
Outputs formatted metrics table and persists JSON/CSV evaluation reports.

Usage:
    python evaluate.py [--dataset PATH] [--model PATH] [--capacity 50] [--threshold 0.30]
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import DEFAULT_DATA_PATH, MODEL_BUNDLE_PATH, REPORTS_DIR
from src.data_manager import load_dataset, clean_and_deduplicate
from src.preprocessor import patient_grouped_split
from src.models import load_model_bundle, predict_patient_risk, ModelTrainer
from src.evaluator import compute_evaluation_metrics


def main():
    parser = argparse.ArgumentParser(description="CarePulse AI Model Evaluation CLI")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATA_PATH), help="Path to discharge records CSV")
    parser.add_argument("--model", type=str, default=str(MODEL_BUNDLE_PATH), help="Path to model bundle")
    parser.add_argument("--capacity", type=int, default=50, help="Weekly outreach capacity (default 50)")
    parser.add_argument("--threshold", type=float, default=0.30, help="Classification decision threshold (default 0.30)")
    args = parser.parse_args()

    data_path = Path(args.dataset)
    model_path = Path(args.model)

    print("=" * 70)
    print(" CarePulse AI: 30-Day Hospital Readmission Model Evaluation")
    print("=" * 70)

    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Generating 25,000 synthetic records first...")
        from src.data_generator import generate_synthetic_cohort
        generate_synthetic_cohort(num_records=25000, output_path=str(data_path))

    print(f"Loading dataset from: {data_path}")
    raw_df, audit = load_dataset(str(data_path))
    clean_df = clean_and_deduplicate(raw_df)
    print(f"Records loaded: {len(raw_df)} | After deduplication: {len(clean_df)} | Unique patients: {audit.unique_patients}")

    # Split into patient-grouped splits
    print("Performing patient-grouped train/val/test split...")
    train_df, val_df, test_df = patient_grouped_split(clean_df, test_size=0.15, val_size=0.15, random_state=42)
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Check model bundle
    if not model_path.exists():
        print(f"Model bundle not found at {model_path}. Training candidate models now...")
        trainer = ModelTrainer(random_state=42)
        bundle = trainer.train_and_select(train_df, val_df, calibration_method="sigmoid")
        trainer.save_bundle(bundle, path=model_path)
    else:
        print(f"Loading existing model bundle from: {model_path}")
        bundle = load_model_bundle(model_path)

    print(f"Active Model Architecture: {bundle.get('model_name', 'Trained Classifier')}")
    print(f"Probability Calibration: {bundle.get('calibration_method', 'sigmoid').title()} (Platt scaling)")

    # Predict on held-out test set
    print("\nRunning inference on held-out test cohort...")
    test_probs = predict_patient_risk(test_df, bundle)
    y_test = test_df["readmitted_30d"].values

    # Compute evaluation metrics
    metrics = compute_evaluation_metrics(
        y_true=y_test,
        y_prob=test_probs,
        threshold=args.threshold,
        capacity=args.capacity
    )

    # Print Formatted Report Table
    print("\n" + "=" * 70)
    print(" HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 70)
    print(f"Test Cohort Size:            {metrics['n_samples']} patients")
    print(f"Observed Readmission Rate:   {metrics['prevalence']:.2%} ({metrics['n_positives']} positive cases)")
    print("-" * 70)
    print(f"AUC-PR (Average Precision):  {metrics['auc_pr']:.4f}")
    print(f"ROC-AUC:                     {metrics['roc_auc']:.4f}")
    print(f"Expected Calibration Error:  {metrics['expected_calibration_error']:.4f} (10 equal-width bins)")
    print(f"Brier Score:                 {metrics['brier_score']:.4f}")
    print("-" * 70)
    print(f"Weekly Capacity Window (K):  {metrics['capacity_k']} patients")
    print(f"Recall@{metrics['capacity_k']}:                   {metrics['recall_at_capacity']:.2%} "
          f"({metrics['top_k_true_positives']} of {metrics['n_positives']} true readmissions captured)")
    print(f"Precision@{metrics['capacity_k']}:                {metrics['precision_at_capacity']:.2%} "
          f"({metrics['top_k_true_positives']} of {metrics['capacity_k']} contacted patients readmitted)")
    print("-" * 70)
    cm = metrics["confusion_matrix"]
    print(f"Confusion Matrix (Threshold = {args.threshold:.2f}):")
    print(f"  True Negatives (TN):  {cm['tn']:<6} | False Positives (FP): {cm['fp']}")
    print(f"  False Negatives (FN): {cm['fn']:<6} | True Positives (TP):  {cm['tp']}")
    print(f"  Sensitivity (Recall): {cm['sensitivity_recall']:.2%} | Specificity: {cm['specificity']:.2%}")
    print(f"  PPV (Precision):      {cm['ppv_precision']:.2%} | NPV:         {cm['npv']:.2%}")
    print("=" * 70)

    # Save reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "evaluation_summary.json"
    bins_csv_path = REPORTS_DIR / "calibration_bins.csv"
    metrics_csv_path = REPORTS_DIR / "evaluation_metrics.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(metrics["calibration_bins"]).to_csv(bins_csv_path, index=False)
    
    summary_flat = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    pd.DataFrame([summary_flat]).to_csv(metrics_csv_path, index=False)

    print(f"\nSaved evaluation reports:")
    print(f"  - {summary_path}")
    print(f"  - {bins_csv_path}")
    print(f"  - {metrics_csv_path}")
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
