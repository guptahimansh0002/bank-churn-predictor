"""Model training pipeline for bank customer churn prediction."""

# This module trains candidate churn models and persists the best production artifact.

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import (
    CATEGORICAL_COLS,
    DATA_PATH,
    DATA_PATH_FALLBACK,
    FEATURES_PATH,
    MAX_ITER,
    MODEL_PATH,
    NUMERIC_COLS,
    RANDOM_STATE,
    SHAP_FEATURE_COLUMN,
    SHAP_IMPORTANCE_COLUMN,
    SHAP_IMPORTANCE_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_preprocessing import (
    encode_categorical,
    handle_missing_values,
    load_data,
    scale_features,
)
from src.feature_engineering import create_features, get_feature_names
from src.model_evaluation import evaluate_all_models

logger = logging.getLogger(__name__)


def _resolve_data_path() -> str:
    """Return the configured data path, falling back to the local dataset casing."""
    if Path(DATA_PATH).is_file():
        return DATA_PATH
    if Path(DATA_PATH_FALLBACK).is_file():
        return DATA_PATH_FALLBACK
    raise FileNotFoundError(f"Training data not found: {DATA_PATH}")


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a persisted training artifact when needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _candidate_models() -> Dict[str, Any]:
    """Create candidate classifiers for model selection."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def _prepare_training_frame() -> pd.DataFrame:
    """Load raw data and apply shared preprocessing before splitting."""
    df = load_data(_resolve_data_path())
    df = df[NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COLUMN]]
    df = handle_missing_values(df)
    return create_features(df)


def _fit_transform_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit preprocessing artifacts on train data and transform train/test sets."""
    X_train_encoded, _ = encode_categorical(X_train, fit=True)
    X_test_encoded, _ = encode_categorical(X_test, fit=False)

    X_train_scaled, _ = scale_features(X_train_encoded, fit=True)
    X_test_scaled, _ = scale_features(X_test_encoded, fit=False)

    feature_names = get_feature_names(X_train_scaled)
    X_train_scaled = X_train_scaled.reindex(columns=feature_names)
    X_test_scaled = X_test_scaled.reindex(columns=feature_names, fill_value=0)
    return X_train_scaled, X_test_scaled


def _save_feature_importance(model: Any, feature_names: List[str]) -> None:
    """Save a feature importance CSV compatible with the dashboard."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).mean(axis=0)
    else:
        importance = np.zeros(len(feature_names))

    importance_df = (
        pd.DataFrame(
            {
                SHAP_FEATURE_COLUMN: feature_names,
                SHAP_IMPORTANCE_COLUMN: importance,
            }
        )
        .sort_values(SHAP_IMPORTANCE_COLUMN, ascending=False)
        .reset_index(drop=True)
    )
    _ensure_parent_dir(SHAP_IMPORTANCE_PATH)
    importance_df.to_csv(SHAP_IMPORTANCE_PATH, index=False)
    logger.info("Saved feature importance to %s.", SHAP_IMPORTANCE_PATH)


def run_training_pipeline() -> Any:
    """Run the end-to-end training pipeline and persist required artifacts."""
    logging.basicConfig(level=logging.INFO)

    df = _prepare_training_frame()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train_processed, X_test_processed = _fit_transform_train_test(X_train, X_test)

    models = _candidate_models()
    X_train_array = X_train_processed.to_numpy()
    X_test_array = X_test_processed.to_numpy()
    for model_name, model in models.items():
        logger.info("Training model: %s", model_name)
        model.fit(X_train_array, y_train)

    comparison_df = evaluate_all_models(models, X_test_array, y_test)
    best_model_name = comparison_df.iloc[0]["model"]
    best_model = models[best_model_name]

    _ensure_parent_dir(MODEL_PATH)
    joblib.dump(best_model, MODEL_PATH)
    _save_feature_importance(best_model, joblib.load(FEATURES_PATH))

    logger.info("Saved best model '%s' to %s.", best_model_name, MODEL_PATH)
    return best_model
