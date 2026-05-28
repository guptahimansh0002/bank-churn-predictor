"""Feature engineering utilities for bank customer churn prediction."""

from pathlib import Path
from typing import List

import joblib
import pandas as pd

from config import (
    ACTIVE_MEMBER_COLUMN,
    AGE_COLUMN,
    AGE_TENURE_INTERACTION_COLUMN,
    BALANCE_COLUMN,
    BALANCE_SALARY_RATIO_COLUMN,
    ESTIMATED_SALARY_COLUMN,
    FEATURES_PATH,
    NUM_PRODUCTS_COLUMN,
    PRODUCT_ENGAGEMENT_COLUMN,
    TARGET_COLUMN,
    TENURE_COLUMN,
    ZERO_BALANCE_FLAG_COLUMN,
)

# This module contains reusable feature creation and feature-name persistence helpers.


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived churn-prediction features from configured source columns."""
    transformed_df = df.copy()
    transformed_df[BALANCE_SALARY_RATIO_COLUMN] = transformed_df[BALANCE_COLUMN] / (
        transformed_df[ESTIMATED_SALARY_COLUMN] + 1
    )
    transformed_df[AGE_TENURE_INTERACTION_COLUMN] = (
        transformed_df[AGE_COLUMN] * transformed_df[TENURE_COLUMN]
    )
    transformed_df[PRODUCT_ENGAGEMENT_COLUMN] = (
        transformed_df[NUM_PRODUCTS_COLUMN] * transformed_df[ACTIVE_MEMBER_COLUMN]
    )
    transformed_df[ZERO_BALANCE_FLAG_COLUMN] = (
        transformed_df[BALANCE_COLUMN] == 0
    ).astype(int)
    return transformed_df


def get_feature_names(df: pd.DataFrame) -> List[str]:
    """Return and persist all feature columns excluding the configured target."""
    feature_names = [column for column in df.columns if column != TARGET_COLUMN]
    Path(FEATURES_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_names, FEATURES_PATH)
    return feature_names
