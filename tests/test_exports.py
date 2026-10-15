"""Tests for CSV exports and persistence functions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.call_list import generate_weekly_call_list, export_call_list_csv


def test_export_call_list_csv(tmp_path):
    """Verifies that the weekly call list exports cleanly to CSV format."""
    df = pd.DataFrame([
        {
            "patient_id": f"P-{i}",
            "encounter_id": f"E-{i}",
            "age": 65,
            "sex": "Female",
            "payer_type": "Medicare",
            "primary_diagnosis": "Congestive Heart Failure",
            "discharge_disposition": "Discharged to Home",
            "prior_admissions_6m": 1,
            "medication_count": 10,
            "followup_scheduled": "Yes",
            "length_of_stay": 4,
        }
        for i in range(25)
    ])
    risks = np.linspace(0.8, 0.2, 25)

    call_list = generate_weekly_call_list(df, risks, capacity=20, include_overrides=False)
    out_file = tmp_path / "test_call_list.csv"

    exported_path = export_call_list_csv(call_list, output_path=str(out_file))
    assert Path(exported_path).exists()

    loaded = pd.read_csv(exported_path)
    assert len(loaded) == 20
    assert "priority_rank" in loaded.columns
    assert "patient_id" in loaded.columns
    assert "calibrated_risk" in loaded.columns
    assert "recommended_action" in loaded.columns
