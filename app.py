"""Streamlit dashboard for the Bank Customer Churn Prediction project."""

# The application layer delegates all inference to predict_churn().

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src.predict import predict_churn


def _binary_value(label: str) -> int:
    """Convert configured binary UI labels to model input values."""
    return int(label == config.YES_LABEL)


def _collect_customer_inputs(prefix: str, container: Any) -> Dict[str, Any]:
    """Collect raw customer inputs from a Streamlit container."""
    customer_input = {
        config.CREDIT_SCORE_COLUMN: container.slider(
            config.CREDIT_SCORE_COLUMN,
            min_value=config.CREDIT_SCORE_RANGE[0],
            max_value=config.CREDIT_SCORE_RANGE[1],
            value=config.CREDIT_SCORE_DEFAULT,
            key=f"{prefix}_{config.CREDIT_SCORE_COLUMN}",
        ),
        config.AGE_COLUMN: container.slider(
            config.AGE_COLUMN,
            min_value=config.AGE_RANGE[0],
            max_value=config.AGE_RANGE[1],
            value=config.AGE_DEFAULT,
            key=f"{prefix}_{config.AGE_COLUMN}",
        ),
        config.TENURE_COLUMN: container.slider(
            config.TENURE_COLUMN,
            min_value=config.TENURE_RANGE[0],
            max_value=config.TENURE_RANGE[1],
            value=config.TENURE_DEFAULT,
            key=f"{prefix}_{config.TENURE_COLUMN}",
        ),
        config.BALANCE_COLUMN: container.slider(
            config.BALANCE_COLUMN,
            min_value=config.BALANCE_RANGE[0],
            max_value=config.BALANCE_RANGE[1],
            value=config.BALANCE_DEFAULT,
            key=f"{prefix}_{config.BALANCE_COLUMN}",
        ),
        config.NUM_PRODUCTS_COLUMN: container.slider(
            config.NUM_PRODUCTS_COLUMN,
            min_value=config.NUM_PRODUCTS_RANGE[0],
            max_value=config.NUM_PRODUCTS_RANGE[1],
            value=config.NUM_PRODUCTS_DEFAULT,
            key=f"{prefix}_{config.NUM_PRODUCTS_COLUMN}",
        ),
        config.HAS_CR_CARD_COLUMN: _binary_value(
            container.radio(
                config.HAS_CR_CARD_COLUMN,
                [config.YES_LABEL, config.NO_LABEL],
                key=f"{prefix}_{config.HAS_CR_CARD_COLUMN}",
                horizontal=True,
            )
        ),
        config.ACTIVE_MEMBER_COLUMN: _binary_value(
            container.radio(
                config.ACTIVE_MEMBER_COLUMN,
                [config.YES_LABEL, config.NO_LABEL],
                key=f"{prefix}_{config.ACTIVE_MEMBER_COLUMN}",
                horizontal=True,
            )
        ),
        config.ESTIMATED_SALARY_COLUMN: container.slider(
            config.ESTIMATED_SALARY_COLUMN,
            min_value=config.ESTIMATED_SALARY_RANGE[0],
            max_value=config.ESTIMATED_SALARY_RANGE[1],
            value=config.ESTIMATED_SALARY_DEFAULT,
            key=f"{prefix}_{config.ESTIMATED_SALARY_COLUMN}",
        ),
        config.GEOGRAPHY_COLUMN: container.selectbox(
            config.GEOGRAPHY_COLUMN,
            config.GEOGRAPHY_OPTIONS,
            key=f"{prefix}_{config.GEOGRAPHY_COLUMN}",
        ),
        config.GENDER_COLUMN: container.selectbox(
            config.GENDER_COLUMN,
            config.GENDER_OPTIONS,
            key=f"{prefix}_{config.GENDER_COLUMN}",
        ),
    }
    return customer_input


def _probability_gauge(probability: float) -> go.Figure:
    """Build a Plotly gauge chart for churn probability."""
    probability_percent = probability * config.PROBABILITY_SCALE
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_percent,
            number={"suffix": "%"},
            gauge={
                "axis": {
                    "range": [config.GAUGE_MIN, config.GAUGE_MAX],
                },
                "bar": {"color": _risk_color(probability)},
            },
        )
    )


def _risk_color(probability: float) -> str:
    """Return the configured display color for a churn probability."""
    if probability < config.LOW_RISK_MAX:
        return config.LOW_RISK_COLOR
    if probability > config.HIGH_RISK_MIN:
        return config.HIGH_RISK_COLOR
    return config.MEDIUM_RISK_COLOR


def _recommendation(risk_label: str) -> str:
    """Return the configured recommendation for a risk label."""
    recommendations = {
        config.LOW_RISK_LABEL: config.LOW_RISK_RECOMMENDATION,
        config.MEDIUM_RISK_LABEL: config.MEDIUM_RISK_RECOMMENDATION,
        config.HIGH_RISK_LABEL: config.HIGH_RISK_RECOMMENDATION,
    }
    return recommendations[risk_label]


def _render_prediction(prediction: Dict[str, Any]) -> None:
    """Render prediction probability, risk label, and recommendation."""
    probability = prediction[config.CHURN_PROBABILITY_KEY]
    risk_label = prediction[config.RISK_LABEL_KEY]
    st.plotly_chart(_probability_gauge(probability), use_container_width=True)
    st.markdown(
        f"<h3 style='color: {_risk_color(probability)}'>{risk_label} Risk</h3>",
        unsafe_allow_html=True,
    )
    st.info(_recommendation(risk_label))


def _render_feature_importance() -> None:
    """Render top SHAP feature importances from the configured CSV artifact."""
    importance_path = Path(config.SHAP_IMPORTANCE_PATH)
    if not importance_path.is_file():
        st.warning(f"Feature importance artifact not found: {config.SHAP_IMPORTANCE_PATH}")
        return

    importance_df = pd.read_csv(importance_path)
    top_features = importance_df.head(config.IMPORTANCE_TOP_N)
    figure = px.bar(
        top_features.sort_values(config.SHAP_IMPORTANCE_COLUMN),
        x=config.SHAP_IMPORTANCE_COLUMN,
        y=config.SHAP_FEATURE_COLUMN,
        orientation="h",
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_what_if(customer_input: Dict[str, Any]) -> None:
    """Render auto-updating what-if probability and session trend."""
    try:
        prediction = predict_churn(customer_input)
    except (FileNotFoundError, ValueError, AttributeError) as error:
        st.error(str(error))
        return

    probability = prediction[config.CHURN_PROBABILITY_KEY]
    input_signature = tuple(sorted(customer_input.items()))
    history = st.session_state.setdefault(config.WHAT_IF_HISTORY_KEY, [])
    last_input = st.session_state.get(config.WHAT_IF_LAST_INPUT_KEY)

    if input_signature != last_input:
        history.append(
            {
                config.TREND_STEP_COLUMN: len(history) + 1,
                config.CHURN_PROBABILITY_KEY: probability,
            }
        )
        st.session_state[config.WHAT_IF_LAST_INPUT_KEY] = input_signature

    _render_prediction(prediction)
    trend_df = pd.DataFrame(history)
    if not trend_df.empty:
        figure = px.line(
            trend_df,
            x=config.TREND_STEP_COLUMN,
            y=config.CHURN_PROBABILITY_KEY,
            markers=True,
        )
        st.plotly_chart(figure, use_container_width=True)


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title=config.APP_TITLE, layout=config.APP_LAYOUT)
    st.title(config.APP_TITLE)

    tab_risk, tab_importance, tab_what_if = st.tabs(
        [
            config.TAB_RISK_CALCULATOR,
            config.TAB_FEATURE_IMPORTANCE,
            config.TAB_WHAT_IF,
        ]
    )

    with st.sidebar:
        st.header(config.SIDEBAR_HEADER)
        risk_input = _collect_customer_inputs(config.TAB_RISK_CALCULATOR, st.sidebar)

    with tab_risk:
        if st.button(config.CALCULATE_RISK_BUTTON):
            try:
                prediction = predict_churn(risk_input)
                _render_prediction(prediction)
            except (FileNotFoundError, ValueError, AttributeError) as error:
                st.error(str(error))

    with tab_importance:
        _render_feature_importance()

    with tab_what_if:
        what_if_input = _collect_customer_inputs(config.TAB_WHAT_IF, st)
        _render_what_if(what_if_input)


if __name__ == "__main__":
    main()
