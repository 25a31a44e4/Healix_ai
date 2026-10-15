"""Interactive Clinical Impact & Capacity Simulator.

Estimates the hypothetical population health and operational impact of targeted
interventions on post-discharge readmission rates under constrained outreach capacity.

Calculates:
- Expected readmissions without intervention
- Estimated readmissions averted across individual risk profiles & intervention types
- Net post-intervention readmissions
- Multi-dimensional sensitivity matrix (Intervention Efficacy x Outreach Adherence)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.config import INTERVENTION_TYPES
from src.intervention_engine import recommend_interventions


def simulate_outreach_impact(
    selected_patients_df: pd.DataFrame,
    calibrated_risks: np.ndarray,
    global_rrr_override: Optional[float] = None,
    adherence_rate: float = 0.80,
    use_intervention_specific_rrr: bool = True
) -> Dict[str, Any]:
    """Simulates expected hospital readmissions averted for a targeted cohort.
    
    Args:
        selected_patients_df: DataFrame of selected patients (e.g. top 50).
        calibrated_risks: Array of calibrated probabilities for selected patients.
        global_rrr_override: Uniform relative risk reduction (e.g. 0.25).
        adherence_rate: Proportion of contacted patients who complete the intervention (0.0 to 1.0).
        use_intervention_specific_rrr: Whether to use tailored RRR for each intervention.
        
    Returns:
        Dict of simulated outcomes, sensitivity tables, and clinical assumptions.
    """
    n_patients = len(selected_patients_df)
    if n_patients == 0:
        return {
            "n_patients": 0,
            "expected_readmissions": 0.0,
            "averted_readmissions": 0.0,
            "net_readmissions": 0.0,
            "relative_reduction_pct": 0.0,
            "sensitivity_matrix": [],
        }

    p_risks = np.asarray(calibrated_risks, dtype=float)
    expected_without = float(np.sum(p_risks))

    # Calculate averted readmissions per patient
    averted_per_patient = []
    applied_rrrs = []
    assigned_interventions = []

    for i, (_, row) in enumerate(selected_patients_df.iterrows()):
        risk_i = p_risks[i]
        
        if global_rrr_override is not None:
            rrr_i = float(global_rrr_override)
            applied_rrrs.append(rrr_i)
            assigned_interventions.append("Uniform Outreach")
        elif use_intervention_specific_rrr:
            recs = recommend_interventions(row)
            if recs:
                rrr_i = recs[0].relative_risk_reduction
                assigned_interventions.append(recs[0].title)
            else:
                rrr_i = 0.20
                assigned_interventions.append("Standard Outreach")
            applied_rrrs.append(rrr_i)
        else:
            rrr_i = 0.25
            applied_rrrs.append(rrr_i)
            assigned_interventions.append("Default Care Coordination")

        # Readmissions averted for individual i
        averted_i = risk_i * rrr_i * adherence_rate
        averted_per_patient.append(averted_i)

    total_averted = float(np.sum(averted_per_patient))
    net_readmissions = max(0.0, expected_without - total_averted)
    relative_reduction_pct = (total_averted / expected_without) * 100 if expected_without > 0 else 0.0

    # Sensitivity Analysis Grid: RRR (10% - 35%) x Adherence (50% - 95%)
    rrr_levels = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    adherence_levels = [0.50, 0.65, 0.80, 0.95]

    sensitivity_rows = []
    for rrr in rrr_levels:
        row_dict = {"assumed_rrr_pct": int(rrr * 100)}
        for adh in adherence_levels:
            col_name = f"adh_{int(adh * 100)}pct"
            # sum(p_i * rrr * adh) = expected_without * rrr * adh
            hypothetical_averted = expected_without * rrr * adh
            row_dict[col_name] = round(hypothetical_averted, 1)
        sensitivity_rows.append(row_dict)

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    return {
        "n_patients": n_patients,
        "average_risk": round(float(np.mean(p_risks)), 4),
        "expected_without_intervention": round(expected_without, 2),
        "estimated_averted_readmissions": round(total_averted, 2),
        "net_readmissions_after_intervention": round(net_readmissions, 2),
        "relative_cohort_reduction_pct": round(relative_reduction_pct, 2),
        "mean_applied_rrr": round(float(np.mean(applied_rrrs)), 3),
        "adherence_rate": adherence_rate,
        "sensitivity_table": sensitivity_df.to_dict(orient="records"),
        "assumptions": [
            "Baseline risk is derived from the calibrated machine learning model for each patient.",
            "Relative Risk Reduction (RRR) estimates are synthetic planning parameters derived from published literature on transition care.",
            "Adherence represents the proportion of contacted patients who actively complete follow-up appointments or pharmacist reviews.",
            "This model provides planning projections for capacity allocation and does not guarantee individual patient outcomes.",
        ]
    }
