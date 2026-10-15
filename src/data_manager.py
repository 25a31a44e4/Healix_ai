"""Data Management and Ingestion module for CarePulse AI.

Provides:
- CSV ingestion with customizable column mappings
- Data validation and clinical range checking
- Explicit duplicate encounter vs. repeat patient admission handling
- Missingness audits and data quality warnings report
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from src.config import DATASET_SCHEMA, LAB_REFERENCE_RANGES


class DataQualityAudit:
    """Encapsulates validation and quality metrics of an ingested healthcare dataset."""
    def __init__(
        self,
        total_records: int,
        unique_patients: int,
        unique_encounters: int,
        duplicate_encounters: int,
        missing_counts: Dict[str, int],
        missing_percentages: Dict[str, float],
        invalid_values: Dict[str, int],
        target_distribution: Dict[str, Any],
        warnings: List[str]
    ):
        self.total_records = total_records
        self.unique_patients = unique_patients
        self.unique_encounters = unique_encounters
        self.duplicate_encounters = duplicate_encounters
        self.missing_counts = missing_counts
        self.missing_percentages = missing_percentages
        self.invalid_values = invalid_values
        self.target_distribution = target_distribution
        self.warnings = warnings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "unique_patients": self.unique_patients,
            "unique_encounters": self.unique_encounters,
            "duplicate_encounters": self.duplicate_encounters,
            "missing_counts": self.missing_counts,
            "missing_percentages": self.missing_percentages,
            "invalid_values": self.invalid_values,
            "target_distribution": self.target_distribution,
            "warnings": self.warnings,
        }


def load_dataset(
    file_path: str,
    column_mapping: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, DataQualityAudit]:
    """Loads a discharge dataset from CSV and conducts a comprehensive quality audit.
    
    Args:
        file_path: Path to CSV dataset.
        column_mapping: Optional dictionary mapping {user_col: standard_col}.
        
    Returns:
        Tuple of (audited DataFrame with standard names, DataQualityAudit).
    """
    df = pd.read_csv(file_path)
    
    # Apply column mappings if supplied
    if column_mapping:
        df = df.rename(columns=column_mapping)
        
    audit = audit_dataset(df)
    return df, audit


def audit_dataset(df: pd.DataFrame) -> DataQualityAudit:
    """Runs data quality validation checks across the dataset."""
    total_records = len(df)
    warnings = []
    
    # Check required schema
    missing_cols = [col for col in DATASET_SCHEMA["required_columns"] if col not in df.columns]
    if missing_cols:
        warnings.append(f"Missing required columns in dataset: {', '.join(missing_cols)}")
        
    unique_patients = int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0
    unique_encounters = int(df["encounter_id"].nunique()) if "encounter_id" in df.columns else 0
    duplicate_encounters = total_records - unique_encounters if "encounter_id" in df.columns else 0
    
    if duplicate_encounters > 0:
        warnings.append(
            f"Detected {duplicate_encounters} duplicate encounter IDs. "
            "Policy: Keep the first occurrence and deduplicate prior to model training."
        )
        
    # Check repeat patient admissions
    if unique_patients < total_records and unique_patients > 0:
        repeat_admissions = total_records - unique_patients - duplicate_encounters
        # This is expected in hospital epidemiology
        pass
        
    # Missingness analysis
    missing_counts = {}
    missing_percentages = {}
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        missing_counts[col] = n_miss
        missing_percentages[col] = round((n_miss / total_records) * 100, 2)
        
    # Check expected 10-20% lab missingness
    for lab in ["hemoglobin", "creatinine", "glucose"]:
        if lab in df.columns:
            pct = missing_percentages[lab]
            if pct > 40.0:
                warnings.append(f"Excessive missingness in lab test '{lab}': {pct}% missing.")
                
    # Invalid numerical values check
    invalid_values = {}
    if "age" in df.columns:
        invalid_age = int(((df["age"] < 0) | (df["age"] > 125)).sum())
        if invalid_age > 0:
            invalid_values["age"] = invalid_age
            warnings.append(f"Found {invalid_age} records with invalid age (<0 or >125).")
            
    if "length_of_stay" in df.columns:
        invalid_los = int((df["length_of_stay"] < 0).sum())
        if invalid_los > 0:
            invalid_values["length_of_stay"] = invalid_los
            warnings.append(f"Found {invalid_los} records with negative length of stay.")
            
    if "hemoglobin" in df.columns:
        invalid_hb = int(((df["hemoglobin"] < 1.0) | (df["hemoglobin"] > 30.0)).dropna().sum())
        if invalid_hb > 0:
            invalid_values["hemoglobin"] = invalid_hb
            warnings.append(f"Found {invalid_hb} records with physiologically implausible hemoglobin.")
            
    if "creatinine" in df.columns:
        invalid_cr = int(((df["creatinine"] < 0.1) | (df["creatinine"] > 25.0)).dropna().sum())
        if invalid_cr > 0:
            invalid_values["creatinine"] = invalid_cr
            warnings.append(f"Found {invalid_cr} records with physiologically implausible creatinine.")
            
    # Target distribution
    target_distribution = {}
    if DATASET_SCHEMA["target_column"] in df.columns:
        tgt_col = DATASET_SCHEMA["target_column"]
        val_counts = df[tgt_col].value_counts().to_dict()
        pos_count = int(val_counts.get(1, 0))
        neg_count = int(val_counts.get(0, 0))
        readmission_rate = float(pos_count / total_records) if total_records > 0 else 0.0
        target_distribution = {
            "readmitted_positive": pos_count,
            "readmitted_negative": neg_count,
            "readmission_rate": round(readmission_rate, 4),
        }
    else:
        warnings.append(f"Target column '{DATASET_SCHEMA['target_column']}' is missing. Cannot train supervised model.")
        
    return DataQualityAudit(
        total_records=total_records,
        unique_patients=unique_patients,
        unique_encounters=unique_encounters,
        duplicate_encounters=duplicate_encounters,
        missing_counts=missing_counts,
        missing_percentages=missing_percentages,
        invalid_values=invalid_values,
        target_distribution=target_distribution,
        warnings=warnings
    )


def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning rules and deduplication policy.
    
    Policy:
    1. Removes exact duplicate encounters (keeps first).
    2. Drops records with physiologically impossible demographic values (age < 0 or > 120).
    3. Retains legitimate repeat patient encounters (different encounter_ids) for longitudinal history.
    """
    clean_df = df.copy()
    
    # 1. Deduplicate by encounter_id
    if "encounter_id" in clean_df.columns:
        clean_df = clean_df.drop_duplicates(subset=["encounter_id"], keep="first")
        
    # 2. Filter invalid ages
    if "age" in clean_df.columns:
        clean_df = clean_df[(clean_df["age"] >= 0) & (clean_df["age"] <= 120)]
        
    # 3. Filter negative length of stay
    if "length_of_stay" in clean_df.columns:
        clean_df = clean_df[clean_df["length_of_stay"] >= 0]
        
    # 4. Fill missing followup_scheduled with 'Unrecorded'
    if "followup_scheduled" in clean_df.columns:
        clean_df["followup_scheduled"] = clean_df["followup_scheduled"].fillna("Unrecorded")
        
    return clean_df.reset_index(drop=True)
