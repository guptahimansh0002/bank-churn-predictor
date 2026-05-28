"""Prediction utilities for bank customer churn inference."""

# This module is the single prediction interface used by the application layer.

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_COLS,
    CHURN_FLAG_KEY,
    CHURN_PROBABILITY_KEY,
    ENCODER_PATH,
    FEATURES_PATH,
    HIGH_RISK_LABEL,
    HIGH_RISK_MIN,
    LOW_RISK_LABEL,
    LOW_RISK_MAX,
    MEDIUM_RISK_LABEL,
    MODEL_PATH,
    MISSING_FEATURE_FILL_VALUE,
    NUMERIC_COLS,
    POSITIVE_CLASS_INDEX,
    RISK_LABEL_KEY,
    SCALER_PATH,
    THRESHOLD,
)
from src.feature_engineering import create_features

logger = logging.getLogger(__name__)


def _require_artifact(path: str) -> None:
    """Raise a clear error when a required prediction artifact is missing."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Required prediction artifact not found: {path}")


def _get_encoder_feature_names(encoder: Any) -> List[str]:
    """Return encoded categorical feature names across supported sklearn versions."""
    if hasattr(encoder, "get_feature_names_out"):
        return encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    return encoder.get_feature_names(CATEGORICAL_COLS).tolist()


def _validate_input(input_dict: Dict[str, Any]) -> None:
    """Validate that raw prediction input contains required feature columns."""
    required_columns = NUMERIC_COLS + CATEGORICAL_COLS
    missing_columns = [
        column for column in required_columns if column not in input_dict
    ]
    if missing_columns:
        raise ValueError(f"Missing required input fields: {missing_columns}")


def _risk_label(churn_probability: float) -> str:
    """Return the configured risk label for a churn probability."""
    if churn_probability < LOW_RISK_MAX:
        return LOW_RISK_LABEL
    if churn_probability > HIGH_RISK_MIN:
        return HIGH_RISK_LABEL
    return MEDIUM_RISK_LABEL


@lru_cache(maxsize=1)
def load_artifacts() -> Tuple[Any, Any, Any, List[str]]:
    """Load trained model, scaler, encoder, and feature names from configured paths."""
    artifact_paths = [MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURES_PATH]
    for artifact_path in artifact_paths:
        _require_artifact(artifact_path)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    logger.info("Loaded prediction artifacts from configured paths.")
    return model, scaler, encoder, feature_names


def preprocess_input(input_dict: Dict[str, Any]) -> np.ndarray:
    """Transform raw input into a model-ready numpy array in saved feature order."""
    _validate_input(input_dict)
    _, scaler, encoder, feature_names = load_artifacts()

    input_df = pd.DataFrame([input_dict])
    input_df = create_features(input_df)

    encoded_values = encoder.transform(input_df[CATEGORICAL_COLS])
    encoded_columns = _get_encoder_feature_names(encoder)
    encoded_df = pd.DataFrame(
        encoded_values,
        columns=encoded_columns,
        index=input_df.index,
    )

    input_df = input_df.drop(columns=CATEGORICAL_COLS)
    input_df = pd.concat([input_df, encoded_df], axis=1)
    input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])
    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=MISSING_FEATURE_FILL_VALUE,
    )

    return input_df.to_numpy()


def predict_churn(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predict churn probability, churn flag, and risk label for one customer."""
    model, _, _, _ = load_artifacts()
    features = preprocess_input(input_dict)

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model must support predict_proba for churn scoring.")

    churn_probability = float(model.predict_proba(features)[0][POSITIVE_CLASS_INDEX])
    churn_flag = int(churn_probability >= THRESHOLD)

    prediction = {
        CHURN_PROBABILITY_KEY: churn_probability,
        CHURN_FLAG_KEY: churn_flag,
        RISK_LABEL_KEY: _risk_label(churn_probability),
    }
    logger.info("Generated churn prediction: %s", prediction)
    return prediction
