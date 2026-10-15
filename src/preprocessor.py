"""Preprocessing and Feature Engineering Pipeline for CarePulse AI.

Enforces:
- Strict patient-grouped train/val/test splits (zero patient leakage)
- Imputation with missingness indicators for laboratory values
- Robust categorical encoding handling unseen levels at inference
- Standard scaling fitted solely on training splits
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import DATASET_SCHEMA


def build_preprocessor() -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Builds scikit-learn ColumnTransformer for clinical tabular data."""
    num_cols = DATASET_SCHEMA["numeric_features"]
    cat_cols = DATASET_SCHEMA["categorical_features"]
    
    # Numerical pipeline: Median imputer with explicit missingness indicator + StandardScaler
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    
    # Categorical pipeline: Most frequent imputer + OneHotEncoder (handle unknown levels safely)
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop"
    )
    
    return preprocessor, num_cols, cat_cols


def patient_grouped_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits data into train, val, and test partitions grouped by patient_id.
    
    Prevents data leakage: All encounters for a single patient exist exclusively
    in one partition.
    
    Args:
        df: Input DataFrame with 'patient_id' column.
        test_size: Proportion of unique patients in test set.
        val_size: Proportion of unique patients in validation set.
        random_state: Reproducibility seed.
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if "patient_id" not in df.columns:
        raise ValueError("Cannot perform grouped split: 'patient_id' column missing.")
        
    groups = df["patient_id"].values
    
    # First split: Hold out Test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))
    
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    # Second split: Train vs Validation
    val_ratio_of_train_val = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_of_train_val, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["patient_id"].values))
    
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    
    # Verify zero patient overlap
    train_pts = set(train_df["patient_id"].unique())
    val_pts = set(val_df["patient_id"].unique())
    test_pts = set(test_df["patient_id"].unique())
    
    assert train_pts.isdisjoint(val_pts), "Data leakage detected: train and val share patient IDs!"
    assert train_pts.isdisjoint(test_pts), "Data leakage detected: train and test share patient IDs!"
    assert val_pts.isdisjoint(test_pts), "Data leakage detected: val and test share patient IDs!"
    
    return train_df, val_df, test_df


def extract_feature_names(fitted_transformer: ColumnTransformer) -> List[str]:
    """Extracts human-readable feature names from a fitted ColumnTransformer."""
    feature_names = []
    for name, pipe, cols in fitted_transformer.transformers_:
        if name == "remainder":
            continue
        if name == "num":
            imputer = pipe.named_steps["imputer"]
            # Base features
            feature_names.extend(cols)
            # Imputer indicators if added
            if hasattr(imputer, "indicator_") and imputer.indicator_ is not None:
                indicator_indices = imputer.indicator_.features_
                for idx in indicator_indices:
                    feature_names.append(f"{cols[idx]}_is_missing")
        elif name == "cat":
            encoder = pipe.named_steps["encoder"]
            cat_names = encoder.get_feature_names_out(cols)
            feature_names.extend(cat_names)
            
    return feature_names
