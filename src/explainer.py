"""Explainable AI (XAI) and Clinical Interpretation Engine.

Calculates patient-specific risk factor contributions using SHAP (TreeExplainer/LinearExplainer)
or model feature attributions with guaranteed fallback.
Translates mathematical importances into plain-language clinical insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.config import DATASET_SCHEMA, LAB_REFERENCE_RANGES

HAS_SHAP = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    pass


class ClinicalExplainer:
    """Provides local patient-level risk factor attributions and clinical explanations."""

    def __init__(self, model_bundle: Dict[str, Any]):
        self.bundle = model_bundle
        self.preprocessor = model_bundle["preprocessor"]
        self.raw_model = model_bundle["raw_model"]
        self.feature_names = model_bundle["feature_names"]
        self.model_name = model_bundle.get("model_name", "")
        self.shap_explainer = None
        self._init_shap()

    def _init_shap(self):
        """Initializes SHAP explainer based on model architecture."""
        if not HAS_SHAP:
            return

        try:
            # Tree-based models (XGBoost, LightGBM, Random Forest)
            if hasattr(self.raw_model, "estimators_") or "XGB" in str(type(self.raw_model)) or "LGBM" in str(type(self.raw_model)):
                self.shap_explainer = shap.TreeExplainer(self.raw_model)
            elif hasattr(self.raw_model, "coef_"):
                # Linear model
                self.shap_explainer = shap.LinearExplainer(self.raw_model, masker=shap.maskers.Independent(data=np.zeros((1, len(self.feature_names)))))
        except Exception as e:
            print(f"SHAP explainer init note (using robust attribution fallback): {e}")
            self.shap_explainer = None

    def explain_patient(
        self,
        patient_row: pd.Series,
        top_k: int = 6
    ) -> Dict[str, Any]:
        """Explains risk factors for an individual patient record.
        
        Args:
            patient_row: Single row pd.Series or 1-row DataFrame.
            top_k: Number of top driving features to return.
            
        Returns:
            Dictionary containing attributions, chart data, and clinical explanations.
        """
        row_df = pd.DataFrame([patient_row]) if isinstance(patient_row, pd.Series) else patient_row.copy()
        
        # Prepare inputs
        feature_cols = [c for c in row_df.columns if c in DATASET_SCHEMA["numeric_features"] or c in DATASET_SCHEMA["categorical_features"]]
        X_input = row_df[feature_cols].copy()
        for col in DATASET_SCHEMA["categorical_features"]:
            if col not in X_input.columns:
                X_input[col] = "Missing"
        for col in DATASET_SCHEMA["numeric_features"]:
            if col not in X_input.columns:
                X_input[col] = np.nan

        X_trans = self.preprocessor.transform(X_input)

        contributions = []
        
        # Compute local attribution
        if self.shap_explainer is not None:
            try:
                shap_vals = self.shap_explainer.shap_values(X_trans)
                # Handle binary classification output formats
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    vals = shap_vals[1][0]
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    vals = shap_vals[0, :, 1]
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
                    vals = shap_vals[0]
                else:
                    vals = np.array(shap_vals).flatten()
                    
                for idx, feat_name in enumerate(self.feature_names):
                    if idx < len(vals):
                        contributions.append({
                            "feature": feat_name,
                            "importance": float(vals[idx]),
                        })
            except Exception:
                contributions = self._compute_fallback_attribution(X_trans)
        else:
            contributions = self._compute_fallback_attribution(X_trans)

        # Sort by absolute impact
        contributions = sorted(contributions, key=lambda x: abs(x["importance"]), reverse=True)
        top_factors = contributions[:top_k]

        # Generate clinical plain-language narratives
        clinical_statements = []
        for factor in top_factors:
            narrative = self._narrate_feature(factor["feature"], factor["importance"], row_df.iloc[0])
            if narrative:
                clinical_statements.append(narrative)

        return {
            "top_factors": top_factors,
            "clinical_statements": clinical_statements,
            "all_attributions": contributions,
        }

    def _compute_fallback_attribution(self, X_trans: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback attribution using linear coefficients or tree feature importances."""
        contributions = []
        x_vals = X_trans[0]
        
        if hasattr(self.raw_model, "coef_"):
            coefs = self.raw_model.coef_[0]
            for idx, feat_name in enumerate(self.feature_names):
                if idx < len(coefs):
                    contributions.append({
                        "feature": feat_name,
                        "importance": float(coefs[idx] * x_vals[idx]),
                    })
        elif hasattr(self.raw_model, "feature_importances_"):
            importances = self.raw_model.feature_importances_
            for idx, feat_name in enumerate(self.feature_names):
                if idx < len(importances):
                    # Direction approximated by whether standardized value is above or below zero
                    sign = 1.0 if x_vals[idx] >= 0 else -1.0
                    contributions.append({
                        "feature": feat_name,
                        "importance": float(importances[idx] * sign * 0.1),
                    })
        else:
            for idx, feat_name in enumerate(self.feature_names):
                contributions.append({
                    "feature": feat_name,
                    "importance": 0.01,
                })
        return contributions

    def _narrate_feature(self, feature_name: str, impact: float, patient_row: pd.Series) -> Optional[Dict[str, Any]]:
        """Converts feature attribution to an honest, non-causal clinical narrative."""
        direction = "increases" if impact > 0 else "decreases"
        badge = "Risk Driver" if impact > 0 else "Protective Factor"
        
        # Prior Admissions
        if "prior_admissions_6m" in feature_name:
            val = patient_row.get("prior_admissions_6m", 0)
            text = f"Patient has {val} documented hospital admission(s) in the preceding 6 months."
            return {"title": "Prior 6-Month Utilization", "text": text, "badge": badge, "impact": impact, "direction": direction}
            
        if "prior_admissions" in feature_name:
            val = patient_row.get("prior_admissions", 0)
            text = f"Patient has {val} total hospital admission(s) documented over the past 12 months."
            return {"title": "Annual Utilization History", "text": text, "badge": badge, "impact": impact, "direction": direction}

        # Medication Count
        if "medication_count" in feature_name:
            val = patient_row.get("medication_count", 0)
            if val >= 10:
                text = f"High medication count ({val} active medications), presenting polypharmacy complexity."
            else:
                text = f"Patient prescribed {val} active medications at discharge."
            return {"title": "Medication Regimen Complexity", "text": text, "badge": badge, "impact": impact, "direction": direction}

        # Follow-up appointment
        if "followup_scheduled" in feature_name:
            status = patient_row.get("followup_scheduled", "")
            if status == "No":
                text = "No post-discharge follow-up appointment was documented at discharge time."
                return {"title": "Follow-up Appointment Missing", "text": text, "badge": badge, "impact": impact, "direction": direction}
            elif status == "Yes":
                text = "A post-discharge outpatient follow-up appointment is confirmed and documented."
                return {"title": "Follow-up Appointment Confirmed", "text": text, "badge": badge, "impact": impact, "direction": direction}
            elif pd.isna(status) or status == "Unrecorded":
                text = "Follow-up appointment status was unrecorded in the discharge summary."
                return {"title": "Follow-up Status Unrecorded", "text": text, "badge": badge, "impact": impact, "direction": direction}

        # Length of stay
        if "length_of_stay" in feature_name:
            val = patient_row.get("length_of_stay", 0)
            text = f"Index hospitalization length of stay was {val} days."
            return {"title": "Length of Hospital Stay", "text": text, "badge": badge, "impact": impact, "direction": direction}

        # Diagnoses
        if "primary_diagnosis" in feature_name:
            diag = patient_row.get("primary_diagnosis", "Documented Condition")
            text = f"Primary index admission diagnosis: {diag}."
            return {"title": f"Primary Diagnosis: {diag}", "text": text, "badge": badge, "impact": impact, "direction": direction}

        # Lab values
        if "creatinine" in feature_name and "missing" not in feature_name:
            val = patient_row.get("creatinine", np.nan)
            if pd.notna(val):
                text = f"Serum creatinine is {val} mg/dL (Reference: 0.6–1.3 mg/dL)."
                return {"title": "Serum Creatinine Level", "text": text, "badge": badge, "impact": impact, "direction": direction}

        if "hemoglobin" in feature_name and "missing" not in feature_name:
            val = patient_row.get("hemoglobin", np.nan)
            if pd.notna(val):
                text = f"Hemoglobin is {val} g/dL (Reference: 12.0–17.5 g/dL)."
                return {"title": "Hemoglobin Level", "text": text, "badge": badge, "impact": impact, "direction": direction}

        if "glucose" in feature_name and "missing" not in feature_name:
            val = patient_row.get("glucose", np.nan)
            if pd.notna(val):
                text = f"Blood glucose is {val} mg/dL (Reference: 70–140 mg/dL)."
                return {"title": "Blood Glucose Level", "text": text, "badge": badge, "impact": impact, "direction": direction}

        # Age
        if "age" in feature_name:
            val = patient_row.get("age", 0)
            text = f"Patient age is {val} years."
            return {"title": "Patient Age Cohort", "text": text, "badge": badge, "impact": impact, "direction": direction}

        return None
