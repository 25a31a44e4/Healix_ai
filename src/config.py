"""Configuration module for CarePulse AI.
Contains UI color tokens, risk thresholds, file paths, clinical intervention rules,
and operational constants.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATA_PATH = DATA_DIR / "synthetic_discharge_records.csv"
MODEL_BUNDLE_PATH = MODELS_DIR / "carepulse_model_bundle.joblib"
CALL_LIST_CSV_PATH = REPORTS_DIR / "weekly_call_list_latest.csv"
OVERRIDE_LOG_PATH = REPORTS_DIR / "call_list_overrides.json"
CONTACT_LOG_PATH = REPORTS_DIR / "contact_outcomes.json"

# Visual Design Color Tokens (Exact palette specified by design requirements)
COLORS = {
    "deep_navy": "#142D4E",
    "midnight_blue": "#1B3B5F",
    "primary_teal": "#16A6A1",
    "deep_teal": "#117C78",
    "soft_mint": "#DDF5EC",
    "pale_blue": "#EAF3FC",
    "warm_off_white": "#F7F9FC",
    "pure_white": "#FFFFFF",
    "main_text": "#253449",
    "secondary_text": "#738197",
    "warning_amber": "#F2B84B",
    "critical_red": "#D95362",
    "border": "#E3EAF1",
    # Muted risk badge backgrounds
    "risk_low_bg": "#E8F8F2",
    "risk_low_text": "#0E7058",
    "risk_mod_bg": "#EAF3FC",
    "risk_mod_text": "#1A5276",
    "risk_high_bg": "#FEF7E8",
    "risk_high_text": "#A06800",
    "risk_vhigh_bg": "#FDF0F1",
    "risk_vhigh_text": "#9C2131",
}

# Operational Constraints
DEFAULT_WEEKLY_CAPACITY = 50

# Risk Band Cutoffs (Configurable baseline thresholds)
RISK_THRESHOLDS = {
    "low": 0.15,
    "moderate": 0.30,
    "high": 0.50,
}

def get_risk_category(prob: float) -> str:
    """Categorizes calibrated probability into clinical risk tier."""
    if prob < RISK_THRESHOLDS["low"]:
        return "Low"
    elif prob < RISK_THRESHOLDS["moderate"]:
        return "Moderate"
    elif prob < RISK_THRESHOLDS["high"]:
        return "High"
    else:
        return "Very High"

# Clinical Lab Normal Ranges (Adult reference ranges)
LAB_REFERENCE_RANGES = {
    "hemoglobin": {"low": 12.0, "high": 17.5, "unit": "g/dL", "name": "Hemoglobin"},
    "creatinine": {"low": 0.6, "high": 1.3, "unit": "mg/dL", "name": "Serum Creatinine"},
    "glucose": {"low": 70.0, "high": 140.0, "unit": "mg/dL", "name": "Blood Glucose"},
}

# Expected Dataset Schema & Types
DATASET_SCHEMA = {
    "required_columns": [
        "patient_id",
        "encounter_id",
        "age",
        "sex",
        "payer_type",
        "primary_diagnosis",
        "secondary_diagnoses",
        "procedures",
        "length_of_stay",
        "prior_admissions",
        "prior_admissions_6m",
        "medication_count",
        "discharge_disposition",
        "hemoglobin",
        "creatinine",
        "glucose",
        "followup_scheduled",
        "discharge_date",
        "readmitted_30d",
    ],
    "numeric_features": [
        "age",
        "length_of_stay",
        "prior_admissions",
        "prior_admissions_6m",
        "medication_count",
        "hemoglobin",
        "creatinine",
        "glucose",
    ],
    "categorical_features": [
        "sex",
        "payer_type",
        "primary_diagnosis",
        "discharge_disposition",
        "followup_scheduled",
    ],
    "target_column": "readmitted_30d",
    "group_column": "patient_id",
}

# Discharge Dispositions that disqualify from outpatient care calls
NON_ELIGIBLE_DISPOSITIONS = [
    "Expired",
    "Hospice - Home",
    "Hospice - Facility",
    "Left Against Medical Advice",
    "Discharged to Law Enforcement",
]

# Intervention Definitions
INTERVENTION_TYPES = {
    "pharmacist_review": {
        "title": "Clinical Pharmacist Medication Reconciliation",
        "tag": "Pharmacy",
        "default_rrr": 0.22,  # Relative Risk Reduction from comprehensive med rec
    },
    "appointment_assistance": {
        "title": "Rapid Appointment Scheduling & Transit Support",
        "tag": "Care Access",
        "default_rrr": 0.28,
    },
    "nurse_coordination": {
        "title": "High-Intensity Nurse Care Management",
        "tag": "Care Management",
        "default_rrr": 0.35,
    },
    "home_health_assessment": {
        "title": "Post-Acute Home Health / Community Paramedic Evaluation",
        "tag": "Home Health",
        "default_rrr": 0.25,
    },
    "clinician_lab_review": {
        "title": "Urgent Outpatient Clinician Laboratory Follow-up",
        "tag": "Physician Review",
        "default_rrr": 0.20,
    },
}
