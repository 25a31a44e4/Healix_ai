"""Tests for preprocessing pipeline and patient-grouped data splitting."""

import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_synthetic_cohort
from src.preprocessor import build_preprocessor, patient_grouped_split, extract_feature_names


def test_patient_grouped_split_zero_leakage():
    """Verifies that patient encounters are strictly segregated into disjoint sets."""
    df = generate_synthetic_cohort(num_records=1000, random_seed=123)
    train_df, val_df, test_df = patient_grouped_split(df, test_size=0.15, val_size=0.15, random_state=42)

    train_pts = set(train_df["patient_id"].unique())
    val_pts = set(val_df["patient_id"].unique())
    test_pts = set(test_df["patient_id"].unique())

    # Assert zero overlap
    assert train_pts.isdisjoint(val_pts), "Train and Val share patient IDs!"
    assert train_pts.isdisjoint(test_pts), "Train and Test share patient IDs!"
    assert val_pts.isdisjoint(test_pts), "Val and Test share patient IDs!"


def test_preprocessor_transformation_and_missing_handling():
    """Verifies that preprocessor handles missing lab values and unseen categories."""
    df = generate_synthetic_cohort(num_records=400, random_seed=42)
    preprocessor, num_cols, cat_cols = build_preprocessor()

    feature_cols = num_cols + cat_cols
    X_train = df[feature_cols].copy()

    # Fit preprocessor
    X_trans = preprocessor.fit_transform(X_train)
    assert not np.isnan(X_trans).any(), "Found NaNs in transformed output!"

    feature_names = extract_feature_names(preprocessor)
    assert len(feature_names) == X_trans.shape[1]

    # Test inference on novel record with missing values and unseen category
    novel_df = pd.DataFrame([{
        "age": 67,
        "length_of_stay": 4,
        "prior_admissions": 1,
        "prior_admissions_6m": 1,
        "medication_count": 8,
        "hemoglobin": np.nan,  # Missing
        "creatinine": np.nan,   # Missing
        "glucose": np.nan,      # Missing
        "sex": "Female",
        "payer_type": "NewUnseenPayer",  # Unseen category
        "primary_diagnosis": "Sepsis",
        "discharge_disposition": "Discharged to Home",
        "followup_scheduled": "Yes",
    }])
    X_novel_trans = preprocessor.transform(novel_df)
    assert not np.isnan(X_novel_trans).any()
