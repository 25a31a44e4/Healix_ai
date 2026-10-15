"""Synthetic Healthcare Data Generator for CarePulse AI.

Generates 25,000 clinically plausible hospital discharge records with:
- Realistic chronic illness clusters (CHF, COPD, Diabetes, AMI, Sepsis)
- Multi-encounter patient histories (legitimate repeat visits)
- Controlled ~15% missingness in lab panels
- Known, documented probabilistic data-generating process
- Repeat duplicate encounters for deduplication testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from src.config import DEFAULT_DATA_PATH


def generate_synthetic_cohort(
    num_records: int = 25000,
    random_seed: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Generates synthetic hospital discharge cohort with clinical dependencies.
    
    Args:
        num_records: Total discharge records to generate (default 25,000).
        random_seed: Reproducibility seed.
        output_path: Path to save CSV file, or None.
        
    Returns:
        pd.DataFrame: Formatted discharge records dataset.
    """
    rng = np.random.default_rng(random_seed)
    
    # Generate unique patient IDs (~80% unique patients to simulate repeat admissions)
    num_unique_patients = int(num_records * 0.80)
    unique_patient_ids = [f"PT-{100000 + i}" for i in range(num_unique_patients)]
    
    # Assign patients to encounters (most 1, some 2-4 encounters)
    patient_assignment = rng.choice(unique_patient_ids, size=num_records, replace=True)
    
    # Demographics generated at encounter level with patient consistency
    # Build a lookup for persistent patient traits (age, sex)
    patient_sex_map = {pid: rng.choice(["Female", "Male"], p=[0.52, 0.48]) for pid in unique_patient_ids}
    patient_base_age_map = {pid: int(np.clip(rng.normal(64, 16), 18, 98)) for pid in unique_patient_ids}
    
    sexes = [patient_sex_map[pid] for pid in patient_assignment]
    ages = [patient_base_age_map[pid] for pid in patient_assignment]
    
    # Payer types heavily correlate with age
    payers = []
    for age in ages:
        if age >= 65:
            payer = rng.choice(["Medicare", "Commercial", "Medicaid", "Uninsured"], p=[0.78, 0.14, 0.06, 0.02])
        else:
            payer = rng.choice(["Commercial", "Medicaid", "Medicare", "Uninsured"], p=[0.55, 0.25, 0.10, 0.10])
        payers.append(payer)
        
    # Primary Diagnoses with base prevalence
    diagnoses = [
        "Congestive Heart Failure",
        "COPD Exacerbation",
        "Type 2 Diabetes with Complications",
        "Acute Myocardial Infarction",
        "Pneumonia",
        "Sepsis",
        "Cardiac Arrhythmia",
        "Total Hip / Knee Replacement",
        "Gastrointestinal Bleeding",
        "Cellulitis / Skin Infection",
        "Hypertensive Crisis",
        "Chronic Kidney Disease",
    ]
    diag_probs = [0.15, 0.14, 0.13, 0.09, 0.11, 0.08, 0.07, 0.06, 0.06, 0.04, 0.04, 0.03]
    primary_diagnoses = rng.choice(diagnoses, size=num_records, p=diag_probs)
    
    # Secondary diagnoses pool
    secondary_pool = [
        "Hypertension", "Dyslipidemia", "Depression", "Obesity", "Tobacco Use Disorder",
        "Peripheral Vascular Disease", "Atrial Fibrillation", "Anemia", "Hypothyroidism", "Osteoarthritis"
    ]
    secondary_diagnoses = []
    for _ in range(num_records):
        k = rng.integers(1, 4)
        secs = rng.choice(secondary_pool, size=k, replace=False)
        secondary_diagnoses.append(", ".join(secs))
        
    # Procedures performed
    proc_pool = [
        "None", "Diagnostic Cardiac Catheterization", "Echocardiogram", "Chest CT Scan",
        "Bronchoscopy", "Endoscopy", "Joint Arthroplasty", "Hemodialysis Session", "Blood Transfusion"
    ]
    procedures = rng.choice(proc_pool, size=num_records, p=[0.35, 0.12, 0.15, 0.12, 0.05, 0.07, 0.06, 0.04, 0.04])
    
    # Length of stay (days) - log-normal distribution shifted by diagnosis severity
    los_base = rng.lognormal(mean=1.2, sigma=0.6, size=num_records)
    length_of_stay = []
    for i, diag in enumerate(primary_diagnoses):
        mult = 1.4 if diag in ["Sepsis", "Congestive Heart Failure", "Acute Myocardial Infarction"] else 1.0
        val = int(np.clip(los_base[i] * mult, 1, 30))
        length_of_stay.append(val)
        
    # Prior hospital admissions in last 12m and last 6m
    prior_admissions = []
    prior_admissions_6m = []
    for _ in range(num_records):
        p12 = int(rng.poisson(lam=1.1))
        # 6-month prior admissions cannot exceed 12-month
        p6 = int(rng.binomial(n=p12, p=0.65))
        prior_admissions.append(p12)
        prior_admissions_6m.append(p6)
        
    # Medication Count (Polypharmacy common in chronic diseases)
    medication_counts = []
    for i, diag in enumerate(primary_diagnoses):
        base_meds = rng.integers(3, 14)
        if diag in ["Congestive Heart Failure", "Type 2 Diabetes with Complications", "Chronic Kidney Disease"]:
            base_meds += rng.integers(2, 8)
        medication_counts.append(int(np.clip(base_meds, 1, 30)))
        
    # Discharge Disposition
    disposition_types = [
        "Discharged to Home",
        "Home Health Care",
        "Skilled Nursing Facility",
        "Rehabilitation Facility",
        "Left Against Medical Advice",
        "Hospice - Home",
        "Expired",
    ]
    disp_probs = [0.68, 0.16, 0.08, 0.04, 0.02, 0.01, 0.01]
    discharge_dispositions = rng.choice(disposition_types, size=num_records, p=disp_probs)
    
    # Laboratory panels with ~15% clinically missing values
    # Hemoglobin (g/dL): Normal 12 - 17.5
    hb_base = rng.normal(13.2, 2.1, size=num_records)
    # Creatinine (mg/dL): Normal 0.7 - 1.3
    cr_base = rng.lognormal(mean=0.1, sigma=0.45, size=num_records)
    # Glucose (mg/dL): Normal 80 - 130
    gl_base = rng.normal(128, 42, size=num_records)
    
    hemoglobin = []
    creatinine = []
    glucose = []
    
    for i in range(num_records):
        # 15% missingness for each lab independently
        hb_val = np.nan if rng.random() < 0.15 else round(float(np.clip(hb_base[i], 6.0, 20.0)), 1)
        cr_val = np.nan if rng.random() < 0.15 else round(float(np.clip(cr_base[i], 0.4, 7.5)), 2)
        gl_val = np.nan if rng.random() < 0.15 else round(float(np.clip(gl_base[i], 45.0, 450.0)), 1)
        
        hemoglobin.append(hb_val)
        creatinine.append(cr_val)
        glucose.append(gl_val)
        
    # Follow-up appointment scheduled at discharge
    # Approximately 72% yes, 23% no, 5% unrecorded/missing
    followup_status = []
    for _ in range(num_records):
        r = rng.random()
        if r < 0.72:
            followup_status.append("Yes")
        elif r < 0.95:
            followup_status.append("No")
        else:
            followup_status.append(np.nan)
            
    # Discharge Dates spread over the past 365 days
    base_date = datetime(2026, 10, 1)
    dates = [
        (base_date - timedelta(days=int(rng.integers(1, 365)))).strftime("%Y-%m-%d")
        for _ in range(num_records)
    ]
    
    # Encounter IDs (mostly unique, with 50 duplicate encounters to test deduplication audit)
    encounter_ids = [f"ENC-{200000 + i}" for i in range(num_records)]
    # Intentionally duplicate 50 encounter IDs
    duplicate_indices = rng.choice(num_records, size=50, replace=False)
    for idx in duplicate_indices:
        target_idx = rng.choice([j for j in range(num_records) if j != idx])
        encounter_ids[idx] = encounter_ids[target_idx]
        
    # Target Generation (readmitted_30d) based on authentic logistic risk function
    # Log-odds calculation:
    # Baseline intercept corresponding to ~20% readmission rate
    logits = np.full(num_records, -1.85)
    
    for i in range(num_records):
        # Age contribution
        if ages[i] > 75:
            logits[i] += 0.35
        elif ages[i] > 65:
            logits[i] += 0.18
            
        # Diagnosis severity
        diag = primary_diagnoses[i]
        if diag in ["Congestive Heart Failure", "COPD Exacerbation"]:
            logits[i] += 0.65
        elif diag in ["Sepsis", "Chronic Kidney Disease"]:
            logits[i] += 0.50
        elif diag == "Acute Myocardial Infarction":
            logits[i] += 0.35
        elif diag == "Total Hip / Knee Replacement":
            logits[i] -= 0.40  # Planned elective procedure typically has lower readmission
            
        # Prior admissions is a massive clinical predictor
        logits[i] += 0.32 * prior_admissions_6m[i]
        logits[i] += 0.15 * min(prior_admissions[i], 5)
        
        # Length of stay
        if length_of_stay[i] > 7:
            logits[i] += 0.30
        elif length_of_stay[i] > 4:
            logits[i] += 0.15
            
        # Medication polypharmacy
        if medication_counts[i] >= 15:
            logits[i] += 0.45
        elif medication_counts[i] >= 10:
            logits[i] += 0.25
            
        # Lack of follow-up appointment
        if followup_status[i] == "No":
            logits[i] += 0.40
            
        # Lab abnormalities if measured
        if not np.isnan(hemoglobin[i]) and hemoglobin[i] < 10.0:
            logits[i] += 0.30
        if not np.isnan(creatinine[i]) and creatinine[i] > 1.8:
            logits[i] += 0.45
        if not np.isnan(glucose[i]) and glucose[i] > 220:
            logits[i] += 0.25
            
        # Payer effect
        if payers[i] in ["Medicaid", "Uninsured"]:
            logits[i] += 0.20
            
        # Disposition effect
        if discharge_dispositions[i] in ["Skilled Nursing Facility", "Home Health Care"]:
            logits[i] += 0.25
            
    # Calculate probabilities via sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    # Bernoulli trial with random seed
    readmitted = (rng.random(size=num_records) < probs).astype(int)
    
    # Clinical constraint: Expired patients cannot be readmitted
    for i in range(num_records):
        if discharge_dispositions[i] == "Expired":
            readmitted[i] = 0
            
    # Assemble DataFrame
    df = pd.DataFrame({
        "patient_id": patient_assignment,
        "encounter_id": encounter_ids,
        "age": ages,
        "sex": sexes,
        "payer_type": payers,
        "primary_diagnosis": primary_diagnoses,
        "secondary_diagnoses": secondary_diagnoses,
        "procedures": procedures,
        "length_of_stay": length_of_stay,
        "prior_admissions": prior_admissions,
        "prior_admissions_6m": prior_admissions_6m,
        "medication_count": medication_counts,
        "discharge_disposition": discharge_dispositions,
        "hemoglobin": hemoglobin,
        "creatinine": creatinine,
        "glucose": glucose,
        "followup_scheduled": followup_status,
        "discharge_date": dates,
        "readmitted_30d": readmitted,
    })
    
    if output_path:
        df.to_csv(output_path, index=False)
        
    return df


if __name__ == "__main__":
    print(f"Generating 25,000 synthetic records to {DEFAULT_DATA_PATH}...")
    df = generate_synthetic_cohort(num_records=25000, output_path=str(DEFAULT_DATA_PATH))
    print(f"Dataset generated successfully! Shape: {df.shape}")
    print(f"Readmission Rate: {df['readmitted_30d'].mean():.2%}")
    print(f"Unique Patients: {df['patient_id'].nunique()}")
