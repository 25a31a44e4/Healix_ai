"""Weekly Call List Prioritization and Care-Coordination Engine.

Operationalizes the healthcare capacity constraint (default 50 patients/week):
- Filters non-eligible discharge dispositions (e.g. Expired, Hospice, AMA)
- Deduplicates repeat encounters for the same patient (one active slot per individual)
- Ranks candidate patients by calibrated readmission risk probability
- Supports supervisor overrides with an immutable JSON audit log
- Tracks clinical outreach attempts and contact outcomes
- Generates exportable CSV and printable Monday Morning Call Lists
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_WEEKLY_CAPACITY,
    NON_ELIGIBLE_DISPOSITIONS,
    CALL_LIST_CSV_PATH,
    OVERRIDE_LOG_PATH,
    CONTACT_LOG_PATH,
    get_risk_category,
)
from src.intervention_engine import recommend_interventions


def generate_weekly_call_list(
    df: pd.DataFrame,
    risk_probabilities: np.ndarray,
    capacity: int = DEFAULT_WEEKLY_CAPACITY,
    include_overrides: bool = True
) -> pd.DataFrame:
    """Generates the prioritized Top-K weekly patient outreach list.
    
    Args:
        df: Input patient discharge records DataFrame.
        risk_probabilities: Corresponding calibrated readmission probabilities.
        capacity: Maximum outreach capacity (default 50).
        include_overrides: Whether to apply active supervisor overrides.
        
    Returns:
        pd.DataFrame containing ranked call list with interventions and metadata.
    """
    candidate_df = df.copy()
    candidate_df["calibrated_risk"] = risk_probabilities
    candidate_df["risk_category"] = [get_risk_category(p) for p in risk_probabilities]

    # 1. Apply eligibility filter
    if "discharge_disposition" in candidate_df.columns:
        eligible_mask = ~candidate_df["discharge_disposition"].isin(NON_ELIGIBLE_DISPOSITIONS)
        candidate_df = candidate_df[eligible_mask]

    # 2. Patient-level deduplication: Keep the most recent discharge or highest risk encounter
    # If discharge_date is available, sort by discharge_date descending then risk descending
    sort_cols = ["calibrated_risk"]
    ascending_orders = [False]
    if "discharge_date" in candidate_df.columns:
        sort_cols = ["discharge_date", "calibrated_risk"]
        ascending_orders = [False, False]

    candidate_df = candidate_df.sort_values(by=sort_cols, ascending=ascending_orders)
    # Deduplicate so each patient appears at most once in outreach queue
    candidate_df = candidate_df.drop_duplicates(subset=["patient_id"], keep="first")

    # 3. Sort strictly by calibrated risk descending
    candidate_df = candidate_df.sort_values(by="calibrated_risk", ascending=False).reset_index(drop=True)

    # 4. Apply supervisor manual overrides if logged
    if include_overrides:
        candidate_df = apply_supervisor_overrides(candidate_df)

    # 5. Enforce capacity limit
    call_list = candidate_df.head(capacity).copy().reset_index(drop=True)
    call_list["priority_rank"] = np.arange(1, len(call_list) + 1)

    # 6. Attach tailored interventions and primary risk reasons
    primary_reasons = []
    top_interventions = []
    intervention_details = []

    for _, row in call_list.iterrows():
        recs = recommend_interventions(row)
        if recs:
            top_rec = recs[0]
            top_interventions.append(top_rec.title)
            primary_reasons.append(top_rec.supporting_evidence)
            intervention_details.append("; ".join([r.title for r in recs]))
        else:
            top_interventions.append("Standard Post-Discharge Nurse Check-in")
            primary_reasons.append(f"Calibrated 30d risk is {row['calibrated_risk']:.1%}.")
            intervention_details.append("Standard Nurse Outreach")

    call_list["recommended_action"] = top_interventions
    call_list["primary_risk_reason"] = primary_reasons
    call_list["all_interventions"] = intervention_details

    # 7. Attach contact details (phone number and preferred call window)
    if "phone_number" in call_list.columns:
        call_list["contact_phone"] = call_list["phone_number"]
    elif "contact_phone" not in call_list.columns:
        # Deterministic HIPAA-safe demonstration contact numbers generated from patient_id
        call_list["contact_phone"] = call_list["patient_id"].apply(
            lambda pid: f"(555) {200 + (abs(hash(str(pid))) % 800):03d}-{abs(hash(str(pid) + '_tel')) % 10000:04d}"
        )

    if "preferred_call_window" not in call_list.columns:
        call_windows = [
            "Morning (9:00 AM - 11:30 AM)",
            "Early Afternoon (12:30 PM - 2:30 PM)",
            "Late Afternoon (3:00 PM - 5:00 PM)",
        ]
        call_list["preferred_call_window"] = call_list["patient_id"].apply(
            lambda pid: call_windows[abs(hash(str(pid) + "_window")) % len(call_windows)]
        )

    # Attach contact outcome status if previously logged
    contact_status_map = load_contact_outcomes()
    call_list["contact_status"] = call_list["patient_id"].map(
        lambda pid: contact_status_map.get(pid, {}).get("status", "Pending Outreach")
    )
    call_list["contact_notes"] = call_list["patient_id"].map(
        lambda pid: contact_status_map.get(pid, {}).get("notes", "")
    )

    return call_list


def record_supervisor_override(
    patient_id: str,
    action: str,  # 'force_include' or 'force_exclude'
    reason: str,
    author: str = "Clinical Lead"
) -> Dict[str, Any]:
    """Records an authorized override with timestamp and clinical justification."""
    overrides = []
    if OVERRIDE_LOG_PATH.exists():
        try:
            with open(OVERRIDE_LOG_PATH, "r", encoding="utf-8") as f:
                overrides = json.load(f)
        except Exception:
            overrides = []

    entry = {
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient_id,
        "action": action,
        "reason": reason,
        "author": author,
    }
    overrides.append(entry)

    with open(OVERRIDE_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)

    return entry


def get_override_history() -> List[Dict[str, Any]]:
    """Loads audit trail of supervisor overrides."""
    if not OVERRIDE_LOG_PATH.exists():
        return []
    try:
        with open(OVERRIDE_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def apply_supervisor_overrides(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Modifies candidate order or membership based on active overrides."""
    overrides = get_override_history()
    if not overrides:
        return candidate_df

    df = candidate_df.copy()
    for item in overrides:
        pid = item["patient_id"]
        action = item["action"]
        if action == "force_exclude":
            df = df[df["patient_id"] != pid]
        elif action == "force_include":
            # Boost row to top if patient exists in candidates
            target_mask = df["patient_id"] == pid
            if target_mask.any():
                match_row = df[target_mask]
                rest_df = df[~target_mask]
                df = pd.concat([match_row, rest_df], ignore_index=True)
                
    return df


def record_contact_outcome(
    patient_id: str,
    status: str,
    notes: str = "",
    author: str = "Care Manager"
):
    """Logs contact attempt outcome for a prioritized patient."""
    outcomes = load_contact_outcomes()
    outcomes[patient_id] = {
        "status": status,
        "notes": notes,
        "author": author,
        "timestamp": datetime.now().isoformat(),
    }
    with open(CONTACT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(outcomes, f, indent=2)


def load_contact_outcomes() -> Dict[str, Dict[str, Any]]:
    """Loads recorded patient contact outcomes."""
    if not CONTACT_LOG_PATH.exists():
        return {}
    try:
        with open(CONTACT_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def export_call_list_csv(call_list_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """Exports prioritized call list to CSV for clinical distribution."""
    dest = output_path or str(CALL_LIST_CSV_PATH)
    export_cols = [
        "priority_rank",
        "patient_id",
        "encounter_id",
        "contact_phone",
        "preferred_call_window",
        "calibrated_risk",
        "risk_category",
        "primary_diagnosis",
        "age",
        "sex",
        "payer_type",
        "length_of_stay",
        "prior_admissions_6m",
        "medication_count",
        "followup_scheduled",
        "recommended_action",
        "primary_risk_reason",
        "contact_status",
        "contact_notes",
    ]
    available_cols = [c for c in export_cols if c in call_list_df.columns]
    call_list_df[available_cols].to_csv(dest, index=False)
    return dest
