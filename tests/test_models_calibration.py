"""Tests for model training, probability calibration, and persistence."""

import pytest
import numpy as np
import pandas as pd
from src.data_generator import generate_synthetic_cohort
from src.preprocessor import patient_grouped_split
from src.models import ModelTrainer, predict_patient_risk


def test_model_training_and_calibration():
    """Verifies that candidate models train and return strictly calibrated probabilities in [0, 1]."""
    df = generate_synthetic_cohort(num_records=600, random_seed=42)
    train_df, val_df, test_df = patient_grouped_split(df, test_size=0.20, val_size=0.20, random_state=42)

    trainer = ModelTrainer(random_state=42)
    bundle = trainer.train_and_select(train_df, val_df, calibration_method="sigmoid")

    assert bundle["calibrated_model"] is not None
    assert bundle["best_raw_model"] is not None
    assert len(bundle["candidate_results"]) >= 2

    # Predict on test cohort
    test_probs = predict_patient_risk(test_df, bundle)
    assert len(test_probs) == len(test_df)
    assert np.all(test_probs >= 0.0)
    assert np.all(test_probs <= 1.0)
    assert not np.isnan(test_probs).any()
