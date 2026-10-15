"""Machine Learning Model Training, Selection, and Probability Calibration.

Trains and compares:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting (XGBoost, LightGBM, or Scikit-learn HistGradientBoosting)

Performs probability calibration (Sigmoid / Platt scaling and Isotonic)
and persists model bundles with metadata.
"""

import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import MODEL_BUNDLE_PATH, DATASET_SCHEMA
from src.preprocessor import build_preprocessor, extract_feature_names

# Check optional gradient boosting libraries
HAS_XGBOOST = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    pass

HAS_LIGHTGBM = False
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    pass


class ModelTrainer:
    """Trains candidate models, calibrates probabilities, and bundles artifacts."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor, self.num_cols, self.cat_cols = build_preprocessor()
        self.fitted_preprocessor = None
        self.feature_names = []
        self.candidate_results = {}
        self.best_model_name = ""
        self.best_raw_model = None
        self.calibrated_model = None

    def _get_candidate_models(self) -> Dict[str, Any]:
        """Returns candidate model instances."""
        candidates = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            ),
        }

        # Choose best available gradient boosting engine
        if HAS_XGBOOST:
            candidates["Gradient Boosting (XGBoost)"] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1
            )
        elif HAS_LIGHTGBM:
            candidates["Gradient Boosting (LightGBM)"] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.08,
                random_state=self.random_state,
                verbose=-1
            )
        else:
            candidates["Gradient Boosting (HistGB)"] = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=6,
                random_state=self.random_state
            )

        return candidates

    def train_and_select(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        calibration_method: str = "sigmoid"
    ) -> Dict[str, Any]:
        """Trains candidate models on training set, selects best on validation set,
        and applies probability calibration.
        
        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            calibration_method: 'sigmoid' (Platt scaling) or 'isotonic'.
            
        Returns:
            Dictionary containing model bundle metadata and candidate metrics.
        """
        target_col = DATASET_SCHEMA["target_column"]
        feature_cols = [c for c in train_df.columns if c in DATASET_SCHEMA["numeric_features"] or c in DATASET_SCHEMA["categorical_features"]]

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values

        # 1. Fit preprocessor strictly on training data
        print("Fitting preprocessing pipeline on training split...")
        X_train_raw = train_df[feature_cols]
        X_val_raw = val_df[feature_cols]

        X_train_trans = self.preprocessor.fit_transform(X_train_raw)
        X_val_trans = self.preprocessor.transform(X_val_raw)
        self.fitted_preprocessor = self.preprocessor
        self.feature_names = extract_feature_names(self.fitted_preprocessor)

        candidates = self._get_candidate_models()
        self.candidate_results = {}

        best_auc_pr = -1.0
        best_name = None
        best_model = None

        print("Evaluating candidate models on validation split...")
        for name, model in candidates.items():
            start_t = time.time()
            model.fit(X_train_trans, y_train)
            fit_time = round(time.time() - start_t, 3)

            val_preds_proba = model.predict_proba(X_val_trans)[:, 1]
            val_auc_pr = average_precision_score(y_val, val_preds_proba)
            val_roc_auc = roc_auc_score(y_val, val_preds_proba)
            val_brier = brier_score_loss(y_val, val_preds_proba)

            self.candidate_results[name] = {
                "val_auc_pr": round(float(val_auc_pr), 4),
                "val_roc_auc": round(float(val_roc_auc), 4),
                "val_brier_score": round(float(val_brier), 4),
                "training_time_seconds": fit_time,
            }

            if val_auc_pr > best_auc_pr:
                best_auc_pr = val_auc_pr
                best_name = name
                best_model = model

        self.best_model_name = best_name
        self.best_raw_model = best_model

        # 2. Fit probability calibration on validation split (prefit)
        print(f"Selected best model: {best_name} (Val AUC-PR: {best_auc_pr:.4f})")
        print(f"Applying '{calibration_method}' probability calibration on validation split...")
        
        try:
            from sklearn.frozen import FrozenEstimator
            calibrator = CalibratedClassifierCV(
                estimator=FrozenEstimator(self.best_raw_model),
                method=calibration_method
            )
            calibrator.fit(X_val_trans, y_val)
        except (ImportError, ValueError, TypeError):
            try:
                calibrator = CalibratedClassifierCV(
                    estimator=self.best_raw_model,
                    method=calibration_method,
                    cv="prefit"
                )
                calibrator.fit(X_val_trans, y_val)
            except Exception:
                calibrator = CalibratedClassifierCV(
                    estimator=self.best_raw_model,
                    method=calibration_method,
                    cv=3
                )
                calibrator.fit(X_train_trans, y_train)
        self.calibrated_model = calibrator

        # Assemble bundle
        bundle = {
            "preprocessor": self.fitted_preprocessor,
            "calibrated_model": self.calibrated_model,
            "raw_model": self.best_raw_model,
            "best_raw_model": self.best_raw_model,
            "model_name": self.best_model_name,
            "calibration_method": calibration_method,
            "candidate_results": self.candidate_results,
            "feature_names": self.feature_names,
            "trained_at": datetime.now().isoformat(),
            "train_sample_count": len(train_df),
            "val_sample_count": len(val_df),
        }

        return bundle

    def save_bundle(self, bundle: Dict[str, Any], path: Optional[str] = None):
        """Saves model bundle to disk using joblib."""
        save_path = path or MODEL_BUNDLE_PATH
        joblib.dump(bundle, save_path)
        print(f"CarePulse model bundle saved successfully to {save_path}")


def load_model_bundle(path: Optional[str] = None) -> Dict[str, Any]:
    """Loads saved model bundle from disk."""
    bundle_path = path or MODEL_BUNDLE_PATH
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {bundle_path}. Please train model first.")
    return joblib.load(bundle_path)


def predict_patient_risk(
    df: pd.DataFrame,
    bundle: Dict[str, Any]
) -> np.ndarray:
    """Returns calibrated readmission probability strictly bounded in [0, 1].
    
    Args:
        df: Input DataFrame with patient features.
        bundle: Loaded model bundle dictionary.
        
    Returns:
        1D np.ndarray of calibrated probabilities.
    """
    preprocessor = bundle["preprocessor"]
    calibrated_model = bundle["calibrated_model"]

    # Select only required feature columns
    feature_cols = [c for c in df.columns if c in DATASET_SCHEMA["numeric_features"] or c in DATASET_SCHEMA["categorical_features"]]
    
    # Fill any missing required categorical columns if not present
    X_input = df[feature_cols].copy()
    for col in DATASET_SCHEMA["categorical_features"]:
        if col not in X_input.columns:
            X_input[col] = "Missing"
    for col in DATASET_SCHEMA["numeric_features"]:
        if col not in X_input.columns:
            X_input[col] = np.nan

    X_trans = preprocessor.transform(X_input)
    probs = calibrated_model.predict_proba(X_trans)[:, 1]
    
    # Guarantee numeric bounds [0, 1]
    return np.clip(probs, 0.0, 1.0)
