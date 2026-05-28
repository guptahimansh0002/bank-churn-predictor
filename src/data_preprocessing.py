"""Data preprocessing utilities for bank customer churn prediction."""

# This module contains reusable data loading, missing-value handling,
# categorical encoding, and numeric scaling helpers for the ML pipeline.

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_COLS,
    DROP_COLUMNS,
    ENCODER_PATH,
    NUMERIC_COLS,
    SCALER_PATH,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def _validate_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """Validate that all required columns are present in a dataframe."""
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a persisted artifact when needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _get_encoder_feature_names(encoder: OneHotEncoder) -> List[str]:
    """Return encoded categorical feature names across supported sklearn versions."""
    if hasattr(encoder, "get_feature_names_out"):
        return encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    return encoder.get_feature_names(CATEGORICAL_COLS).tolist()


def load_data(path: str) -> pd.DataFrame:
    """Load, validate, and clean the raw churn dataset."""
    df = pd.read_csv(path)
    required_columns = DROP_COLUMNS + CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COLUMN]
    _validate_columns(df, required_columns)

    cleaned_df = df.drop(columns=DROP_COLUMNS)
    logger.info("Loaded dataset with shape %s and dropped configured columns.", df.shape)
    return cleaned_df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric nulls with median and categorical nulls with mode."""
    missing_counts = df.isnull().sum()
    logger.info("Missing value counts before filling: %s", missing_counts.to_dict())

    cleaned_df = df.copy()
    numeric_columns = [column for column in NUMERIC_COLS if column in cleaned_df.columns]
    categorical_columns = [
        column for column in CATEGORICAL_COLS if column in cleaned_df.columns
    ]

    for column in numeric_columns:
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

    for column in categorical_columns:
        mode_values = cleaned_df[column].mode(dropna=True)
        if mode_values.empty:
            raise ValueError(f"Cannot fill missing values without a mode for: {column}")
        cleaned_df[column] = cleaned_df[column].fillna(mode_values.iloc[0])

    return cleaned_df


def encode_categorical(
    df: pd.DataFrame, fit: bool = True
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """One-hot encode configured categorical columns and persist or load encoder."""
    _validate_columns(df, CATEGORICAL_COLS)

    transformed_df = df.copy()
    if fit:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        encoded_values = encoder.fit_transform(transformed_df[CATEGORICAL_COLS])
        _ensure_parent_dir(ENCODER_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        logger.info("Fitted and saved categorical encoder to %s.", ENCODER_PATH)
    else:
        encoder = joblib.load(ENCODER_PATH)
        encoded_values = encoder.transform(transformed_df[CATEGORICAL_COLS])
        logger.info("Loaded categorical encoder from %s.", ENCODER_PATH)

    encoded_columns = _get_encoder_feature_names(encoder)
    encoded_df = pd.DataFrame(
        encoded_values,
        columns=encoded_columns,
        index=transformed_df.index,
    )

    transformed_df = transformed_df.drop(columns=CATEGORICAL_COLS)
    transformed_df = pd.concat([transformed_df, encoded_df], axis=1)
    return transformed_df, encoder


def scale_features(
    df: pd.DataFrame, fit: bool = True
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale configured numeric columns and persist or load scaler."""
    _validate_columns(df, NUMERIC_COLS)

    transformed_df = df.copy()
    if fit:
        scaler = StandardScaler()
        transformed_df[NUMERIC_COLS] = scaler.fit_transform(transformed_df[NUMERIC_COLS])
        _ensure_parent_dir(SCALER_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Fitted and saved numeric scaler to %s.", SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        transformed_df[NUMERIC_COLS] = scaler.transform(transformed_df[NUMERIC_COLS])
        logger.info("Loaded numeric scaler from %s.", SCALER_PATH)

    return transformed_df, scaler
