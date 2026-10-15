"""Tests for Fairness and Algorithmic Bias Audit Engine."""

import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_synthetic_cohort
from src.fairness import audit_fairness_subgroups


def test_fairness_subgroup_audit():
    """Verifies subgroup auditing and disparity computation across age, sex, and payer."""
    df = generate_synthetic_cohort(num_records=500, random_seed=42)
    risks = np.random.uniform(0.1, 0.8, size=len(df))

    audit = audit_fairness_subgroups(df, risks)
    assert "subgroup_audits" in audit
    assert "mitigation_report" in audit

    sub_audits = audit["subgroup_audits"]
    assert "Age Group" in sub_audits
    assert "Sex" in sub_audits
    assert "Payer Type" in sub_audits

    # Check structure of subgroup entries
    sex_records = sub_audits["Sex"]
    for entry in sex_records:
        assert "subgroup" in entry
        assert "sample_size" in entry
        assert "status" in entry


def test_small_subgroup_handling():
    """Verifies that subgroups with N < 30 are safely handled without throwing errors."""
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(15)],
        "age": [85] * 15,
        "sex": ["Female"] * 15,
        "payer_type": ["Uninsured"] * 15,
        "readmitted_30d": [1, 0] * 7 + [0],
    })
    risks = np.array([0.5] * 15)

    audit = audit_fairness_subgroups(df, risks)
    payer_recs = audit["subgroup_audits"]["Payer Type"]
    assert any("Insufficient data" in r["status"] for r in payer_recs)
