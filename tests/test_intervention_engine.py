"""Tests for interpretable rules-based intervention recommendation engine."""

import pytest
import pandas as pd
import numpy as np
from src.intervention_engine import recommend_interventions


def test_pharmacist_review_triggered_by_polypharmacy():
    """Verifies that high medication count triggers clinical pharmacist review."""
    patient = pd.Series({
        "medication_count": 15,
        "followup_scheduled": "Yes",
        "prior_admissions_6m": 0,
        "primary_diagnosis": "Cellulitis",
        "discharge_disposition": "Discharged to Home",
        "creatinine": 0.9,
        "hemoglobin": 13.5,
        "glucose": 110,
    })
    recs = recommend_interventions(patient)
    keys = [r.intervention_key for r in recs]
    assert "pharmacist_review" in keys
    pharm_rec = next(r for r in recs if r.intervention_key == "pharmacist_review")
    assert pharm_rec.priority == "High"
    assert "polypharmacy" in pharm_rec.supporting_evidence.lower()


def test_appointment_booking_triggered_by_missing_followup():
    """Verifies that missing appointment triggers booking assistance."""
    patient = pd.Series({
        "medication_count": 4,
        "followup_scheduled": "No",
        "prior_admissions_6m": 0,
        "primary_diagnosis": "Hypertension",
        "discharge_disposition": "Discharged to Home",
    })
    recs = recommend_interventions(patient)
    keys = [r.intervention_key for r in recs]
    assert "appointment_assistance" in keys


def test_lab_review_triggered_by_abnormal_creatinine():
    """Verifies abnormal renal labs trigger outpatient clinician review."""
    patient = pd.Series({
        "medication_count": 4,
        "followup_scheduled": "Yes",
        "prior_admissions_6m": 0,
        "primary_diagnosis": "Hypertension",
        "discharge_disposition": "Discharged to Home",
        "creatinine": 2.4,  # High (normal 0.6-1.3)
    })
    recs = recommend_interventions(patient)
    keys = [r.intervention_key for r in recs]
    assert "clinician_lab_review" in keys
