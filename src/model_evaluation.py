"""Model evaluation utilities for bank customer churn prediction."""

# This module contains metric calculation, model comparison, ROC plotting,
# and SHAP-based feature importance reporting helpers.

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import (
    METRIC_ACCURACY,
    METRIC_CONFUSION_MATRIX,
    METRIC_F1,
    METRIC_PRECISION,
    METRIC_RECALL,
    METRIC_ROC_AUC,
    MODEL_NAME_COLUMN,
    ROC_CURVES_PATH,
    SHAP_FEATURE_COLUMN,
    SHAP_IMPORTANCE_COLUMN,
    SHAP_IMPORTANCE_PATH,
    SHAP_SUMMARY_PATH,
    THRESHOLD,
)

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for an evaluation artifact when needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _get_model_scores(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    """Return continuous positive-class scores for ROC and threshold metrics."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    logger.warning("Model has no probability or decision scores; using predictions.")
    return model.predict(X_test)


def _get_model_name(model_name: Any, model: Any) -> str:
    """Return a stable display name for a model comparison row."""
    if isinstance(model_name, str):
        return model_name
    return model.__class__.__name__


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluate a fitted model and return standard classification metrics."""
    y_scores = _get_model_scores(model, X_test)
    y_pred = (y_scores >= THRESHOLD).astype(int)

    metrics = {
        METRIC_ACCURACY: accuracy_score(y_test, y_pred),
        METRIC_PRECISION: precision_score(y_test, y_pred, zero_division=0),
        METRIC_RECALL: recall_score(y_test, y_pred, zero_division=0),
        METRIC_F1: f1_score(y_test, y_pred, zero_division=0),
        METRIC_ROC_AUC: roc_auc_score(y_test, y_scores),
        METRIC_CONFUSION_MATRIX: confusion_matrix(y_test, y_pred).tolist(),
    }
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def evaluate_all_models(
    models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Evaluate multiple fitted models and return a ROC-AUC sorted comparison."""
    comparison_rows = []
    for model_name, model in models.items():
        logger.info("Evaluating model: %s", model_name)
        metrics = evaluate_model(model, X_test, y_test)
        metrics[MODEL_NAME_COLUMN] = _get_model_name(model_name, model)
        comparison_rows.append(metrics)

    comparison_df = pd.DataFrame(comparison_rows)
    return comparison_df.sort_values(METRIC_ROC_AUC, ascending=False).reset_index(
        drop=True
    )


def plot_roc_curves(
    models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """Plot ROC curves for all fitted models and save the figure."""
    _ensure_parent_dir(ROC_CURVES_PATH)
    plt.figure()

    for model_name, model in models.items():
        y_scores = _get_model_scores(model, X_test)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_scores)
        roc_auc = roc_auc_score(y_test, y_scores)
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{_get_model_name(model_name, model)} ({METRIC_ROC_AUC}: {roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_CURVES_PATH)
    plt.close()
    logger.info("Saved ROC curves to %s.", ROC_CURVES_PATH)


def compute_shap_values(
    model: Any, X_test: pd.DataFrame, feature_names: list
) -> None:
    """Compute SHAP values and save a summary plot plus feature importance CSV."""
    import shap

    _ensure_parent_dir(SHAP_SUMMARY_PATH)
    _ensure_parent_dir(SHAP_IMPORTANCE_PATH)

    X_shap = pd.DataFrame(X_test, columns=feature_names)
    explainer = shap.Explainer(model, X_shap)
    shap_values = explainer(X_shap)

    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH)
    plt.close()

    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, -1]

    importance = np.abs(values).mean(axis=0)
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
    importance_df.to_csv(SHAP_IMPORTANCE_PATH, index=False)
    logger.info("Saved SHAP summary to %s.", SHAP_SUMMARY_PATH)
    logger.info("Saved SHAP importance to %s.", SHAP_IMPORTANCE_PATH)
