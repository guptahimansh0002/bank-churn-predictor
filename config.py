"""Project configuration for bank customer churn prediction."""

# Data and artifact paths.
DATA_PATH = "data/european_bank.csv"
DATA_PATH_FALLBACK = "data/european_Bank.csv"
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"
ENCODER_PATH = "artifacts/encoder.pkl"
FEATURES_PATH = "artifacts/feature_names.pkl"
ROC_CURVES_PATH = "artifacts/roc_curves.png"
SHAP_SUMMARY_PATH = "artifacts/shap_summary.png"
SHAP_IMPORTANCE_PATH = "artifacts/shap_importance.csv"

# Dataset schema.
TARGET_COLUMN = "Exited"
DROP_COLUMNS = ["CustomerId", "Surname"]
CATEGORICAL_COLS = ["Geography", "Gender"]
NUMERIC_COLS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
CREDIT_SCORE_COLUMN = "CreditScore"
BALANCE_COLUMN = "Balance"
ESTIMATED_SALARY_COLUMN = "EstimatedSalary"
AGE_COLUMN = "Age"
TENURE_COLUMN = "Tenure"
NUM_PRODUCTS_COLUMN = "NumOfProducts"
HAS_CR_CARD_COLUMN = "HasCrCard"
ACTIVE_MEMBER_COLUMN = "IsActiveMember"
GEOGRAPHY_COLUMN = "Geography"
GENDER_COLUMN = "Gender"
BALANCE_SALARY_RATIO_COLUMN = "balance_salary_ratio"
AGE_TENURE_INTERACTION_COLUMN = "age_tenure_interaction"
PRODUCT_ENGAGEMENT_COLUMN = "product_engagement"
ZERO_BALANCE_FLAG_COLUMN = "zero_balance_flag"

# Training and inference settings.
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.5
MAX_ITER = 1000

# Evaluation output schema.
MODEL_NAME_COLUMN = "model"
METRIC_ACCURACY = "accuracy"
METRIC_PRECISION = "precision"
METRIC_RECALL = "recall"
METRIC_F1 = "f1"
METRIC_ROC_AUC = "roc_auc"
METRIC_CONFUSION_MATRIX = "confusion_matrix"
SHAP_FEATURE_COLUMN = "feature"
SHAP_IMPORTANCE_COLUMN = "importance"

# Prediction output schema and risk bands.
CHURN_PROBABILITY_KEY = "churn_probability"
CHURN_FLAG_KEY = "churn_flag"
RISK_LABEL_KEY = "risk_label"
LOW_RISK_LABEL = "Low"
MEDIUM_RISK_LABEL = "Medium"
HIGH_RISK_LABEL = "High"
LOW_RISK_MAX = 0.3
HIGH_RISK_MIN = 0.6
POSITIVE_CLASS_INDEX = 1
MISSING_FEATURE_FILL_VALUE = 0

# Streamlit app labels and controls.
APP_TITLE = "Bank Churn Intelligence"
APP_LAYOUT = "wide"
TAB_RISK_CALCULATOR = "Churn Risk Calculator"
TAB_FEATURE_IMPORTANCE = "Feature Importance"
TAB_WHAT_IF = "What-If Simulator"
CALCULATE_RISK_BUTTON = "Calculate Risk"
SIDEBAR_HEADER = "Customer Profile"
YES_LABEL = "Yes"
NO_LABEL = "No"
GEOGRAPHY_OPTIONS = ["France", "Spain", "Germany"]
GENDER_OPTIONS = ["Male", "Female"]
CREDIT_SCORE_RANGE = (300, 850)
AGE_RANGE = (18, 100)
TENURE_RANGE = (0, 10)
BALANCE_RANGE = (0.0, 250000.0)
NUM_PRODUCTS_RANGE = (1, 4)
ESTIMATED_SALARY_RANGE = (0.0, 200000.0)
CREDIT_SCORE_DEFAULT = 650
AGE_DEFAULT = 40
TENURE_DEFAULT = 5
BALANCE_DEFAULT = 75000.0
NUM_PRODUCTS_DEFAULT = 2
ESTIMATED_SALARY_DEFAULT = 100000.0
IMPORTANCE_TOP_N = 10
PROBABILITY_SCALE = 100
GAUGE_MIN = 0
GAUGE_MAX = 100
LOW_RISK_COLOR = "#16a34a"
MEDIUM_RISK_COLOR = "#ca8a04"
HIGH_RISK_COLOR = "#dc2626"
LOW_RISK_RECOMMENDATION = "Customer appears stable. Continue standard engagement and periodic satisfaction checks."
MEDIUM_RISK_RECOMMENDATION = "Monitor this customer closely and consider targeted retention offers or service outreach."
HIGH_RISK_RECOMMENDATION = "Prioritize immediate retention action with personalized outreach and account review."
WHAT_IF_HISTORY_KEY = "what_if_history"
WHAT_IF_LAST_INPUT_KEY = "what_if_last_input"
TREND_STEP_COLUMN = "step"
