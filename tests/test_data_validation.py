"""Tests for data validation, synthetic data generation, and cleaning policies."""

import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_synthetic_cohort
from src.data_manager import audit_dataset, clean_and_deduplicate
from src.config import DATASET_SCHEMA


def test_synthetic_cohort_generation():
    """Verifies that synthetic data generation conforms to required schema."""
    df = generate_synthetic_cohort(num_records=500, random_seed=42)
    assert len(df) == 500
    for req_col in DATASET_SCHEMA["required_columns"]:
        assert req_col in df.columns, f"Missing required column: {req_col}"
    assert df["readmitted_30d"].isin([0, 1]).all()
    # Check that expired patients cannot be readmitted
    expired_readmissions = df[df["discharge_disposition"] == "Expired"]["readmitted_30d"].sum()
    assert expired_readmissions == 0


def test_duplicate_encounter_policy():
    """Verifies duplicate encounters are detected and deduplicated while retaining unique patients."""
    df = generate_synthetic_cohort(num_records=500, random_seed=42)
    base_audit = audit_dataset(df)
    # Intentionally inject 5 exact duplicate encounters
    dupes = df.head(5).copy()
    tampered_df = pd.concat([df, dupes], ignore_index=True)
    
    audit = audit_dataset(tampered_df)
    assert audit.duplicate_encounters == base_audit.duplicate_encounters + 5
    
    cleaned = clean_and_deduplicate(tampered_df)
    assert cleaned["encounter_id"].nunique() == len(cleaned)


def test_invalid_demographic_filtering():
    """Verifies that records with out-of-range numerical values (e.g. negative age) are handled safely."""
    test_df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "encounter_id": ["E1", "E2", "E3"],
        "age": [-5, 45, 130],
        "length_of_stay": [3, -2, 5],
        "followup_scheduled": [np.nan, "Yes", "No"],
    })
    cleaned = clean_and_deduplicate(test_df)
    # Only P2 has valid age (45), but negative LOS (-2), so P2 is filtered by LOS or age
    # P1 has age -5, P3 has age 130
    assert len(cleaned) == 0 or all((cleaned["age"] >= 0) & (cleaned["age"] <= 120))
