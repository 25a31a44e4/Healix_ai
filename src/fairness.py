"""Fairness & Bias Audit Engine for Clinical Machine Learning Models.

Audits algorithmic parity across:
- Age brackets (<45, 45-64, 65-74, 75+)
- Sex (Female, Male)
- Payer types (Medicare, Medicaid, Commercial, Uninsured)

Evaluates sample size, readmission prevalence, AUC-PR, calibration error (ECE),
Brier score, and representation in the Top-50 weekly call list.
Includes a structured data-quality and bias mitigation report.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.evaluator import compute_expected_calibration_error, compute_evaluation_metrics
from sklearn.metrics import average_precision_score, brier_score_loss


def categorize_age_group(age: float) -> str:
    """Bins patient age into standard epidemiological cohorts."""
    if age < 45:
        return "<45"
    elif age < 65:
        return "45-64"
    elif age < 75:
        return "65-74"
    else:
        return "75+"


def audit_fairness_subgroups(
    df: pd.DataFrame,
    risk_probabilities: np.ndarray,
    top_50_patient_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Runs algorithmic bias audit across age, sex, and payer attributes.
    
    Args:
        df: Input evaluation DataFrame containing target 'readmitted_30d'.
        risk_probabilities: Calibrated model probabilities.
        top_50_patient_ids: Optional list of patient IDs selected for top-50 outreach.
        
    Returns:
        Dictionary containing subgroup metrics, representation rates, and mitigation analysis.
    """
    audit_df = df.copy()
    audit_df["calibrated_prob"] = risk_probabilities
    audit_df["age_group"] = audit_df["age"].apply(categorize_age_group)

    target_col = "readmitted_30d"
    has_target = target_col in audit_df.columns
    total_records = len(audit_df)

    # Set of top-50 patients for representation auditing
    top_50_set = set(top_50_patient_ids) if top_50_patient_ids is not None else set()

    subgroup_categories = {
        "Age Group": "age_group",
        "Sex": "sex",
        "Payer Type": "payer_type",
    }

    results = {}

    for cat_name, col_name in subgroup_categories.items():
        if col_name not in audit_df.columns:
            continue

        cat_results = []
        groups = sorted(audit_df[col_name].dropna().unique())

        for grp in groups:
            sub_mask = audit_df[col_name] == grp
            sub_df = audit_df[sub_mask]
            sub_size = len(sub_df)
            pop_share = (sub_size / total_records) * 100 if total_records > 0 else 0.0

            # Measure top-50 representation
            if top_50_set:
                in_top50_count = sum(pid in top_50_set for pid in sub_df["patient_id"])
                top50_share = (in_top50_count / len(top_50_set)) * 100 if len(top_50_set) > 0 else 0.0
            else:
                in_top50_count = 0
                top50_share = 0.0

            # Missing lab rate in subgroup (data quality parity)
            lab_cols = [c for c in ["hemoglobin", "creatinine", "glucose"] if c in sub_df.columns]
            if lab_cols:
                sub_missing_lab_pct = float(sub_df[lab_cols].isna().mean().mean() * 100)
            else:
                sub_missing_lab_pct = 0.0

            # Insufficient data guard
            if sub_size < 30:
                cat_results.append({
                    "subgroup": str(grp),
                    "sample_size": sub_size,
                    "population_share_pct": round(pop_share, 1),
                    "top50_share_pct": round(top50_share, 1),
                    "prevalence_pct": None,
                    "auc_pr": None,
                    "ece": None,
                    "brier_score": None,
                    "missing_lab_pct": round(sub_missing_lab_pct, 1),
                    "status": "Insufficient data (N < 30)",
                })
                continue

            if not has_target:
                cat_results.append({
                    "subgroup": str(grp),
                    "sample_size": sub_size,
                    "population_share_pct": round(pop_share, 1),
                    "top50_share_pct": round(top50_share, 1),
                    "prevalence_pct": None,
                    "auc_pr": None,
                    "ece": None,
                    "brier_score": None,
                    "missing_lab_pct": round(sub_missing_lab_pct, 1),
                    "status": "Target outcome unavailable",
                })
                continue

            y_sub = sub_df[target_col].values
            p_sub = sub_df["calibrated_prob"].values
            pos_count = int(np.sum(y_sub))

            if pos_count == 0 or pos_count == sub_size:
                cat_results.append({
                    "subgroup": str(grp),
                    "sample_size": sub_size,
                    "population_share_pct": round(pop_share, 1),
                    "top50_share_pct": round(top50_share, 1),
                    "prevalence_pct": round((pos_count / sub_size) * 100, 2),
                    "auc_pr": None,
                    "ece": None,
                    "brier_score": None,
                    "missing_lab_pct": round(sub_missing_lab_pct, 1),
                    "status": "Insufficient class diversity (single class observed)",
                })
                continue

            # Compute group metrics
            sub_auc_pr = float(average_precision_score(y_sub, p_sub))
            sub_ece, _ = compute_expected_calibration_error(y_sub, p_sub, n_bins=10)
            sub_brier = float(brier_score_loss(y_sub, p_sub))
            sub_prev = float(pos_count / sub_size) * 100

            cat_results.append({
                "subgroup": str(grp),
                "sample_size": sub_size,
                "population_share_pct": round(pop_share, 1),
                "top50_share_pct": round(top50_share, 1),
                "prevalence_pct": round(sub_prev, 2),
                "auc_pr": round(sub_auc_pr, 4),
                "ece": round(sub_ece, 4),
                "brier_score": round(sub_brier, 4),
                "missing_lab_pct": round(sub_missing_lab_pct, 1),
                "status": "Evaluated",
            })

        results[cat_name] = cat_results

    # Calculate Disparity Summaries & Mitigation Report
    mitigation_report = generate_mitigation_report(results, audit_df)

    return {
        "subgroup_audits": results,
        "mitigation_report": mitigation_report,
    }


def generate_mitigation_report(subgroup_audits: Dict[str, List[Dict[str, Any]]], df: pd.DataFrame) -> Dict[str, Any]:
    """Generates analytical conclusions, disparities, and recommended bias mitigations."""
    disparities = []
    recommendations = []

    for cat_name, records in subgroup_audits.items():
        evaluated_records = [r for r in records if r["status"] == "Evaluated"]
        if len(evaluated_records) >= 2:
            auc_prs = [r["auc_pr"] for r in evaluated_records if r["auc_pr"] is not None]
            eces = [r["ece"] for r in evaluated_records if r["ece"] is not None]

            if auc_prs:
                max_pr = max(auc_prs)
                min_pr = min(auc_prs)
                pr_gap = round(max_pr - min_pr, 4)
                if pr_gap > 0.10:
                    disparities.append(
                        f"Significant AUC-PR disparity in {cat_name}: {pr_gap:.4f} gap "
                        f"between highest ({max_pr:.4f}) and lowest ({min_pr:.4f}) subgroups."
                    )

            if eces:
                max_ece = max(eces)
                min_ece = min(eces)
                ece_gap = round(max_ece - min_ece, 4)
                if ece_gap > 0.05:
                    disparities.append(
                        f"Calibration error disparity in {cat_name}: {ece_gap:.4f} gap "
                        f"(Max ECE: {max_ece:.4f}, Min ECE: {min_ece:.4f})."
                    )

        # Missingness differences
        missing_rates = [r["missing_lab_pct"] for r in records]
        if missing_rates and (max(missing_rates) - min(missing_rates)) > 8.0:
            disparities.append(
                f"Data quality disparity in {cat_name}: Missing lab rate spans {min(missing_rates):.1f}% to {max(missing_rates):.1f}%."
            )

    # Standard clinical bias guidance
    recommendations.extend([
        "Maintain subgroup-specific calibration curves: Group-conditioned probability calibration reduces differential over-prediction or under-prediction.",
        "Preserve demographic variables for auditing: Removing sensitive attributes blinds the audit without resolving underlying correlated disparities.",
        "Implement missingness-pattern monitoring: Disproportionate missing laboratory records in Medicaid or uninsured cohorts require standardized discharge protocols.",
        "Conduct human-in-the-loop clinical review: Outreach priority scores should be paired with nurse manager assessment to avoid systematic demographic blind spots.",
    ])

    return {
        "disparities_detected": disparities if disparities else ["No critical statistical parity violations detected."],
        "recommended_mitigations": recommendations,
    }
