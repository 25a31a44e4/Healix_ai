"""Tests for Weekly Call List prioritization, capacity constraints, and deduplication."""

import pytest
import pandas as pd
import numpy as np
from src.call_list import generate_weekly_call_list
from src.config import NON_ELIGIBLE_DISPOSITIONS


def test_capacity_constraint_and_deduplication():
    """Verifies that the call list strictly obeys capacity and deduplicates encounters by patient."""
    # 200 records, but some patients have multiple encounters
    records = []
    for i in range(100):
        records.append({
            "patient_id": f"P-{i}",
            "encounter_id": f"E-{i}-1",
            "discharge_disposition": "Discharged to Home",
            "age": 65,
            "medication_count": 8,
            "primary_diagnosis": "Pneumonia",
        })
        # Add second encounter for first 30 patients
        if i < 30:
            records.append({
                "patient_id": f"P-{i}",
                "encounter_id": f"E-{i}-2",
                "discharge_disposition": "Discharged to Home",
                "age": 65,
                "medication_count": 8,
                "primary_diagnosis": "Pneumonia",
            })

    df = pd.DataFrame(records)
    risks = np.random.uniform(0.1, 0.9, size=len(df))

    call_list = generate_weekly_call_list(df, risks, capacity=50, include_overrides=False)

    # 1. Strict capacity bound
    assert len(call_list) == 50

    # 2. Strict patient uniqueness (one slot per individual)
    assert call_list["patient_id"].nunique() == 50


def test_eligibility_filter():
    """Verifies non-eligible discharge dispositions (e.g. Expired, Hospice) are excluded from call list."""
    records = [
        {"patient_id": "P1", "encounter_id": "E1", "discharge_disposition": "Expired", "age": 70, "medication_count": 5},
        {"patient_id": "P2", "encounter_id": "E2", "discharge_disposition": "Hospice - Home", "age": 80, "medication_count": 5},
        {"patient_id": "P3", "encounter_id": "E3", "discharge_disposition": "Discharged to Home", "age": 60, "medication_count": 5},
    ]
    df = pd.DataFrame(records)
    risks = np.array([0.95, 0.90, 0.40])

    call_list = generate_weekly_call_list(df, risks, capacity=50, include_overrides=False)
    assert "P1" not in call_list["patient_id"].values
    assert "P2" not in call_list["patient_id"].values
    assert "P3" in call_list["patient_id"].values
