"""Interpretable Rules-Based Clinical Intervention Recommendation Engine.

Maps patient-specific clinical evidence to targeted post-discharge care interventions:
- Pharmacist Medication Reconciliation
- Appointment-Booking & Transit Assistance
- Nurse Care Coordination
- Home Health / Mobility Assessment
- Urgent Clinician Lab Follow-up

All recommendations provide transparent supporting evidence, priority rating,
and require clinician review.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.config import INTERVENTION_TYPES, LAB_REFERENCE_RANGES


class InterventionRecommendation:
    """Represents a discrete clinical intervention proposed for a patient."""

    def __init__(
        self,
        intervention_key: str,
        title: str,
        category: str,
        supporting_evidence: str,
        reason: str,
        priority: str,
        confidence: str,
        relative_risk_reduction: float,
        clinician_review_required: bool = True
    ):
        self.intervention_key = intervention_key
        self.title = title
        self.category = category
        self.supporting_evidence = supporting_evidence
        self.reason = reason
        self.priority = priority
        self.confidence = confidence
        self.relative_risk_reduction = relative_risk_reduction
        self.clinician_review_required = clinician_review_required

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_key": self.intervention_key,
            "title": self.title,
            "category": self.category,
            "supporting_evidence": self.supporting_evidence,
            "reason": self.reason,
            "priority": self.priority,
            "confidence": self.confidence,
            "relative_risk_reduction": self.relative_risk_reduction,
            "clinician_review_required": self.clinician_review_required,
        }


def recommend_interventions(patient: pd.Series) -> List[InterventionRecommendation]:
    """Generates tailored intervention pathways based on discharge data.
    
    Args:
        patient: A single patient encounter record (pd.Series or dict-like).
        
    Returns:
        List of InterventionRecommendation objects ordered by priority.
    """
    recommendations: List[InterventionRecommendation] = []
    
    # Extract clinical values
    med_count = patient.get("medication_count", 0)
    followup = str(patient.get("followup_scheduled", "")).strip().title()
    prior_6m = patient.get("prior_admissions_6m", 0)
    prior_12m = patient.get("prior_admissions", 0)
    diag = str(patient.get("primary_diagnosis", ""))
    disp = str(patient.get("discharge_disposition", ""))
    age = patient.get("age", 0)
    los = patient.get("length_of_stay", 0)
    hb = patient.get("hemoglobin", np.nan)
    cr = patient.get("creatinine", np.nan)
    gl = patient.get("glucose", np.nan)

    # 1. Polypharmacy & Medication Reconciliation
    if med_count >= 14:
        recommendations.append(InterventionRecommendation(
            intervention_key="pharmacist_review",
            title=INTERVENTION_TYPES["pharmacist_review"]["title"],
            category=INTERVENTION_TYPES["pharmacist_review"]["tag"],
            supporting_evidence=f"Discharge medication count is {med_count} (severe polypharmacy >= 14).",
            reason="High medication count elevates risks of adverse drug events, duplicate therapies, and adherence obstacles.",
            priority="High",
            confidence="High",
            relative_risk_reduction=INTERVENTION_TYPES["pharmacist_review"]["default_rrr"],
            clinician_review_required=True
        ))
    elif med_count >= 9:
        recommendations.append(InterventionRecommendation(
            intervention_key="pharmacist_review",
            title=INTERVENTION_TYPES["pharmacist_review"]["title"],
            category=INTERVENTION_TYPES["pharmacist_review"]["tag"],
            supporting_evidence=f"Discharge medication count is {med_count} (moderate polypharmacy >= 9).",
            reason="Comprehensive reconciliation can clarify dose adjustments and confirm pharmacy delivery.",
            priority="Moderate",
            confidence="Moderate",
            relative_risk_reduction=INTERVENTION_TYPES["pharmacist_review"]["default_rrr"],
            clinician_review_required=True
        ))

    # 2. Appointment Booking & Transit Assistance
    if followup == "No":
        recommendations.append(InterventionRecommendation(
            intervention_key="appointment_assistance",
            title=INTERVENTION_TYPES["appointment_assistance"]["title"],
            category=INTERVENTION_TYPES["appointment_assistance"]["tag"],
            supporting_evidence="No primary care or specialty follow-up appointment was booked prior to discharge.",
            reason="Post-discharge outpatient contact within 7-14 days significantly lowers 30-day decompensation risk.",
            priority="High",
            confidence="High",
            relative_risk_reduction=INTERVENTION_TYPES["appointment_assistance"]["default_rrr"],
            clinician_review_required=False  # Administrative outreach
        ))
    elif pd.isna(patient.get("followup_scheduled")) or followup in ["Unrecorded", "Nan", "None", ""]:
        recommendations.append(InterventionRecommendation(
            intervention_key="appointment_assistance",
            title=INTERVENTION_TYPES["appointment_assistance"]["title"],
            category=INTERVENTION_TYPES["appointment_assistance"]["tag"],
            supporting_evidence="Discharge record does not document follow-up appointment status.",
            reason="Verify whether an outpatient visit is scheduled with the patient's care provider.",
            priority="Moderate",
            confidence="Moderate",
            relative_risk_reduction=INTERVENTION_TYPES["appointment_assistance"]["default_rrr"],
            clinician_review_required=False
        ))

    # 3. Nurse Care Coordination & Chronic Disease Management
    high_utilizer = (prior_6m >= 2) or (prior_12m >= 3)
    chronic_cardiopulmonary = diag in ["Congestive Heart Failure", "COPD Exacerbation", "Chronic Kidney Disease"]
    if high_utilizer or chronic_cardiopulmonary:
        evidence_parts = []
        if prior_6m >= 2:
            evidence_parts.append(f"{prior_6m} admissions in past 6 months")
        if chronic_cardiopulmonary:
            evidence_parts.append(f"primary diagnosis of {diag}")
        
        evidence_str = "; ".join(evidence_parts) if evidence_parts else f"{prior_12m} prior admissions in 12 months"
        
        recommendations.append(InterventionRecommendation(
            intervention_key="nurse_coordination",
            title=INTERVENTION_TYPES["nurse_coordination"]["title"],
            category=INTERVENTION_TYPES["nurse_coordination"]["tag"],
            supporting_evidence=f"Patient history shows: {evidence_str}.",
            reason="Frequent utilization and complex chronic pathology indicate high risk of physiological decompensation.",
            priority="High",
            confidence="High",
            relative_risk_reduction=INTERVENTION_TYPES["nurse_coordination"]["default_rrr"],
            clinician_review_required=True
        ))

    # 4. Post-Acute Home Health / Community Paramedic Evaluation
    home_health_candidate = (
        disp in ["Home Health Care", "Skilled Nursing Facility", "Rehabilitation Facility"]
        or (age >= 78 and los >= 6)
    )
    if home_health_candidate:
        recommendations.append(InterventionRecommendation(
            intervention_key="home_health_assessment",
            title=INTERVENTION_TYPES["home_health_assessment"]["title"],
            category=INTERVENTION_TYPES["home_health_assessment"]["tag"],
            supporting_evidence=f"Discharge disposition: '{disp}', Age: {age}, Length of Stay: {los} days.",
            reason="Assess home safety, mobility limitations, and caregiver support structure to prevent falls and relapse.",
            priority="Moderate",
            confidence="Moderate",
            relative_risk_reduction=INTERVENTION_TYPES["home_health_assessment"]["default_rrr"],
            clinician_review_required=True
        ))

    # 5. Outpatient Laboratory Review
    lab_anomalies = []
    if pd.notna(cr) and cr > 1.6:
        lab_anomalies.append(f"Elevated Serum Creatinine ({cr} mg/dL, normal 0.6–1.3)")
    if pd.notna(hb) and hb < 10.0:
        lab_anomalies.append(f"Low Hemoglobin ({hb} g/dL, normal 12.0–17.5)")
    if pd.notna(gl) and (gl > 220 or gl < 70):
        lab_anomalies.append(f"Abnormal Blood Glucose ({gl} mg/dL, normal 70–140)")

    if lab_anomalies:
        recommendations.append(InterventionRecommendation(
            intervention_key="clinician_lab_review",
            title=INTERVENTION_TYPES["clinician_lab_review"]["title"],
            category=INTERVENTION_TYPES["clinician_lab_review"]["tag"],
            supporting_evidence="; ".join(lab_anomalies) + ".",
            reason="Documented discharge lab abnormalities require outpatient repeat monitoring and clinical evaluation.",
            priority="High",
            confidence="High",
            relative_risk_reduction=INTERVENTION_TYPES["clinician_lab_review"]["default_rrr"],
            clinician_review_required=True
        ))

    # Priority sorting: High > Moderate > Standard
    priority_order = {"High": 0, "Moderate": 1, "Standard": 2}
    recommendations.sort(key=lambda r: priority_order.get(r.priority, 99))
    
    return recommendations
