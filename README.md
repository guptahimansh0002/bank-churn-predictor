# 🏦 Bank Customer Churn Prediction & Risk Scoring System

Predict customer churn risk, explain churn drivers, and simulate retention scenarios with a production-ready machine learning dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Production-green)

---

## 📌 Table of Contents

- [🏦 Bank Customer Churn Prediction & Risk Scoring System](#-bank-customer-churn-prediction--risk-scoring-system)
- [📌 Table of Contents](#-table-of-contents)
- [📖 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [📁 Project Structure](#-project-structure)
- [🧰 Tech Stack](#-tech-stack)
- [🗃️ Dataset Description](#️-dataset-description)
- [🤖 ML Models Used](#-ml-models-used)
- [📊 Model Evaluation Metrics](#-model-evaluation-metrics)
- [🧪 Feature Engineering](#-feature-engineering)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🐳 Docker Deployment](#-docker-deployment)
- [🖥️ Usage Guide](#️-usage-guide)
- [📸 Screenshots](#-screenshots)
- [🏁 Project Results](#-project-results)
- [🚀 Future Improvements](#-future-improvements)
- [👨‍💻 Author](#-author)
- [📄 License](#-license)

---

## 📖 Project Overview

The **Bank Customer Churn Prediction & Risk Scoring System** is an end-to-end machine learning project designed to identify customers who are likely to leave a bank.

It combines data preprocessing, feature engineering, model training, evaluation, explainability, and a Streamlit dashboard into a structured production-style project.

The system predicts whether a customer may churn based on banking, demographic, and account-activity attributes.

It also produces an interpretable churn probability score that can support business decision-making.

The project is built to be useful beyond a notebook workflow.

It includes reusable Python modules, saved artifacts, model persistence, Docker deployment support, and a dashboard interface for business users.

### Business Problem

Customer churn is a major challenge for banks and financial institutions.

Acquiring a new customer is often more expensive than retaining an existing customer.

Banks need early warning systems that help identify customers with high churn risk before they leave.

This project solves that problem by:

- Predicting churn probability for individual customers.
- Classifying customers into risk levels.
- Explaining the main drivers behind model decisions.
- Supporting what-if analysis for retention planning.
- Providing a dashboard for analysts and stakeholders.

### Target Users

This project is built for:

- Banks and financial institutions.
- Customer retention teams.
- Business analysts.
- Data analysts.
- ML engineers.
- Risk analytics teams.
- Final-year students building portfolio ML projects.

---

## ✨ Key Features

### Churn Risk Calculator

The dashboard includes an interactive churn risk calculator.

Users can enter customer details such as credit score, age, geography, balance, tenure, card ownership, account activity, and estimated salary.

The system returns:

- Churn probability.
- Churn flag.
- Risk label.
- Business recommendation.

### SHAP Explainability

The project supports model explainability through SHAP-based feature importance.

The explainability module helps users understand which features contribute most to churn prediction.

This is important for financial decision-making because model transparency improves trust and usability.

### What-If Simulator

The what-if simulator allows users to modify customer attributes and observe changes in churn probability.

This helps business teams explore possible retention actions.

For example, analysts can simulate the impact of:

- Higher engagement.
- Different account balances.
- More products.
- Increased tenure.
- Different customer activity status.

### Docker Deployment

The project includes a production-ready Dockerfile.

The containerized Streamlit app can be built and deployed consistently across environments.

The Docker setup includes:

- Python slim base image.
- Cached dependency installation.
- Non-root application user.
- Streamlit health check.
- Exposed Streamlit port.

### Production MLOps Pipeline

The project follows a modular ML pipeline structure.

It separates:

- Data loading.
- Missing value handling.
- Encoding.
- Scaling.
- Feature engineering.
- Model training.
- Model evaluation.
- Prediction serving.
- Dashboard usage.

This makes the project easier to maintain, test, extend, and deploy.

---

## 📁 Project Structure

```text
bank-churn-predictor/
├── data/
│   └── churn.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── predict.py
├── models/
│   └── best_model.pkl
├── artifacts/
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── feature_names.pkl
│   ├── shap_importance.csv
│   ├── shap_summary.png
│   └── roc_curves.png
├── app.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── config.py
└── README.md
```

### File and Folder Descriptions

| Path | Description |
|---|---|
| `data/` | Stores raw dataset files used by the training pipeline. |
| `data/churn.csv` | Customer churn dataset used for model training and evaluation. |
| `src/` | Source package containing reusable ML pipeline modules. |
| `src/__init__.py` | Marks `src` as a Python package. |
| `src/data_preprocessing.py` | Handles data loading, validation, missing values, encoding, and scaling. |
| `src/feature_engineering.py` | Creates derived churn-related features and saves final feature names. |
| `src/model_training.py` | Runs the full training pipeline and saves the best model. |
| `src/model_evaluation.py` | Computes model metrics, plots ROC curves, and generates explainability outputs. |
| `src/predict.py` | Provides the single prediction interface used by the Streamlit app. |
| `models/` | Stores trained model files. |
| `models/best_model.pkl` | Serialized best-performing trained ML model. |
| `artifacts/` | Stores preprocessing and explainability artifacts. |
| `artifacts/scaler.pkl` | Saved scaler used to standardize numeric input features. |
| `artifacts/encoder.pkl` | Saved encoder used for categorical feature transformation. |
| `artifacts/feature_names.pkl` | Saved final feature order used during model inference. |
| `artifacts/shap_importance.csv` | Feature importance table used by the dashboard. |
| `artifacts/shap_summary.png` | SHAP summary plot generated during model evaluation. |
| `artifacts/roc_curves.png` | ROC curve comparison plot for trained models. |
| `app.py` | Streamlit dashboard for churn risk scoring and analysis. |
| `Dockerfile` | Production container definition for deploying the Streamlit app. |
| `requirements.txt` | Pinned Python dependencies required by the project. |
| `.dockerignore` | Files and folders excluded from the Docker build context. |
| `config.py` | Central configuration file for paths, columns, thresholds, and constants. |
| `README.md` | Project documentation for GitHub and portfolio presentation. |

---

## 🧰 Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| Programming Language | Python 3.12 | Core development language for data processing, ML, and app logic. |
| Data Processing | pandas | Data loading, cleaning, transformation, and tabular manipulation. |
| Numerical Computing | NumPy | Numerical operations and array handling. |
| ML | scikit-learn | Preprocessing, model training, evaluation, and baseline algorithms. |
| ML | XGBoost | Gradient boosting model for high-performance churn prediction. |
| Frontend | Streamlit | Interactive dashboard and user interface. |
| Visualization | Plotly | Interactive gauge, bar, and trend charts. |
| Explainability | SHAP | Model interpretation and feature importance explanation. |
| Persistence | joblib | Saving and loading models, scalers, encoders, and feature names. |
| Evaluation | matplotlib | ROC and SHAP plot generation in backend evaluation workflows. |
| Deployment | Docker | Containerized deployment for consistent runtime behavior. |
| Configuration | Python config module | Centralized constants and paths used across the project. |

---

## 🗃️ Dataset Description

The dataset contains customer-level banking records.

Each row represents one bank customer.

The goal is to predict whether the customer exited the bank.

| Column | Description |
|---|---|
| `RowNumber` | Sequential row identifier in the original dataset. |
| `CustomerId` | Unique customer identifier. |
| `Surname` | Customer surname. |
| `CreditScore` | Customer credit score. |
| `Geography` | Customer country or region, such as France, Spain, or Germany. |
| `Gender` | Customer gender. |
| `Age` | Customer age in years. |
| `Tenure` | Number of years the customer has been with the bank. |
| `Balance` | Account balance maintained by the customer. |
| `NumOfProducts` | Number of bank products used by the customer. |
| `HasCrCard` | Indicates whether the customer has a credit card. |
| `IsActiveMember` | Indicates whether the customer is an active bank member. |
| `EstimatedSalary` | Estimated annual salary of the customer. |
| `Exited` | Target variable indicating whether the customer churned. |

### Target Variable

The target variable is:

```text
Exited
```

Target interpretation:

| Value | Meaning |
|---|---|
| `0` | Customer did not churn. |
| `1` | Customer churned. |

The model predicts the probability that `Exited = 1`.

---

## 🤖 ML Models Used

The training pipeline can evaluate multiple supervised classification models.

| Model | Type | Purpose |
|---|---|---|
| Logistic Regression | Linear classification model | Provides an interpretable baseline model for churn classification. |
| Decision Tree | Tree-based classification model | Captures non-linear decision rules and feature splits. |
| Random Forest | Ensemble tree model | Reduces variance and improves predictive stability. |
| Gradient Boosting | Boosted ensemble model | Builds strong predictive performance through sequential error correction. |
| XGBoost | Optimized gradient boosting model | Provides high-performance churn classification with strong tabular-data results. |

### Model Selection Strategy

Models are compared using evaluation metrics such as ROC-AUC, F1-score, recall, precision, and accuracy.

The best model is saved as:

```text
models/best_model.pkl
```

This saved model is then used by the prediction interface and Streamlit dashboard.

---

## 📊 Model Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures the overall percentage of correct predictions. |
| Precision | Measures how many predicted churn customers actually churned. |
| Recall | Measures how many actual churn customers were correctly identified. |
| F1 | Balances precision and recall into a single metric. |
| ROC-AUC | Measures how well the model separates churners from non-churners across thresholds. |

### Why ROC-AUC Matters

ROC-AUC is especially useful for churn prediction because business teams often care about ranking customers by risk.

A model with higher ROC-AUC is better at assigning higher scores to customers who are more likely to churn.

### Why Recall Matters

Recall is important when the business wants to identify as many churn-risk customers as possible.

Missing a high-risk customer can result in lost revenue and missed retention opportunities.

### Why Precision Matters

Precision is important when retention campaigns are expensive.

Higher precision means fewer resources are spent on customers who are unlikely to churn.

---

## 🧪 Feature Engineering

The project creates derived features to improve model signal and business interpretability.

### Derived Features

1. `balance_salary_ratio`

```text
balance_salary_ratio = Balance / (EstimatedSalary + 1)
```

This feature measures balance size relative to estimated salary.

2. `age_tenure_interaction`

```text
age_tenure_interaction = Age * Tenure
```

This feature captures the combined effect of customer age and relationship duration.

3. `product_engagement`

```text
product_engagement = NumOfProducts * IsActiveMember
```

This feature reflects how product ownership interacts with customer activity.

4. `zero_balance_flag`

```text
zero_balance_flag = (Balance == 0).astype(int)
```

This feature identifies customers with zero account balance.

### Feature Engineering Benefit

These features help the model capture behavioral patterns that may not be obvious from raw columns alone.

They also make churn drivers easier to interpret for business stakeholders.

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/bank-churn-predictor.git
```

```bash
cd bank-churn-predictor
```

### 2. Create a Conda Environment

```bash
conda create -n bank-churn python=3.12 -y
```

### 3. Activate the Environment

```bash
conda activate bank-churn
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the Dataset

Place the churn dataset at:

```text
data/european_Bank.csv
```

The training pipeline expects the configured dataset path to be available before training.

### 6. Run the Training Pipeline

```bash
python -c "from src.model_training import run_training_pipeline; run_training_pipeline()"
```

This command trains the models and creates required artifacts.

Expected generated files:

```text
models/best_model.pkl
artifacts/scaler.pkl
artifacts/encoder.pkl
artifacts/feature_names.pkl
artifacts/shap_importance.csv
```

### 7. Launch the Streamlit App

```bash
streamlit run app.py
```

### 8. Open the App in Browser

```text
http://localhost:8501
```

---

## 🐳 Docker Deployment

The project includes a production-ready Dockerfile for deploying the Streamlit dashboard.

### Build the Docker Image

```bash
docker build -t bank-churn-predictor .
```

### Run the Docker Container

```bash
docker run -p 8501:8501 bank-churn-predictor
```

### Access the Application

Open the following URL:

```text
http://localhost:8501
```

### Docker Health Check

The Dockerfile includes a health check for the Streamlit endpoint:

```text
/healthz
```

The health check validates that the app is running and reachable inside the container.

---

## 🖥️ Usage Guide

The Streamlit dashboard contains three main tabs.

### Tab 1: Churn Risk Calculator

Use this tab to calculate churn risk for an individual customer.

Steps:

1. Enter customer details in the sidebar.
2. Click the **Calculate Risk** button.
3. View the churn probability gauge.
4. Review the risk label.
5. Read the recommended retention action.

Inputs include:

- Credit score.
- Age.
- Tenure.
- Balance.
- Number of products.
- Credit card ownership.
- Active member status.
- Estimated salary.
- Geography.
- Gender.

Risk levels:

| Risk Label | Probability Range | Meaning |
|---|---|---|
| Low | Less than 0.30 | Customer is unlikely to churn. |
| Medium | 0.30 to 0.60 | Customer has moderate churn risk. |
| High | Greater than 0.60 | Customer has high churn risk. |

### Tab 2: Feature Importance

Use this tab to understand which features have the strongest influence on churn predictions.

The dashboard loads:

```text
artifacts/shap_importance.csv
```

It displays the top 10 most important features in a horizontal Plotly bar chart.

This helps analysts identify the most important churn drivers.

### Tab 3: What-If Simulator

Use this tab to simulate how changes in customer attributes affect churn probability.

The probability updates automatically when input values change.

No submit button is required.

The simulator is useful for:

- Testing retention scenarios.
- Understanding sensitivity to customer attributes.
- Exploring risk movement over time.
- Comparing possible customer engagement strategies.

---

## 📸 Screenshots

Add screenshots after running the Streamlit app locally.

### Risk Calculator

```text
[Risk Calculator]
```

Placeholder for the churn probability gauge, risk label, and recommendation panel.

### Feature Importance

```text
[Feature Importance]
```

Placeholder for the top feature importance bar chart.

### What-If Simulator

```text
[What-If Simulator]
```

Placeholder for the interactive what-if churn probability trend chart.

---

## 🏁 Project Results

The final results table can be updated after running the training pipeline.

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| Gradient Boosting | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |

### Result Interpretation

The best model should be selected based on a balance of ROC-AUC, F1-score, recall, and operational business needs.

For churn prediction, ROC-AUC and recall are often especially important.

ROC-AUC helps rank customers by churn risk.

Recall helps identify as many true churners as possible.

Precision helps control the cost of retention campaigns.

---

## 🚀 Future Improvements

- Add MLflow tracking for experiment management and model versioning.
- Add automated unit tests and CI/CD checks for pipeline reliability.
- Add a batch prediction workflow for scoring large customer portfolios.
- Add drift monitoring to detect changes in customer behavior and model performance.
- Deploy the app to a cloud platform such as AWS, Azure, GCP, Render, or Streamlit Community Cloud.

---

## 👨‍💻 Author

| Field | Details |
|---|---|
| Name | Himanshu Gupta |
| Role | ML Engineer \| Final Year B.Tech CSE |
| LinkedIn | https://www.linkedin.com/in/himanshu-gupta-ml/ |
| GitHub | https://github.com/guptahimansh0002 |

---

## 📄 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project under the terms of the MIT License.

---

## ✅ Portfolio Notes

This project is designed as a professional GitHub portfolio project.

It demonstrates:

- End-to-end machine learning development.
- Modular Python project structure.
- Streamlit dashboard development.
- Model explainability.
- Docker-based deployment readiness.
- Business-focused risk scoring.
- Practical churn analytics workflow.

---

## 🔎 Quick Reference

### Train Model

```bash
python -c "from src.model_training import run_training_pipeline; run_training_pipeline()"
```

### Run App

```bash
streamlit run app.py
```

### Build Docker Image

```bash
docker build -t bank-churn-predictor .
```

### Run Docker Container

```bash
docker run -p 8501:8501 bank-churn-predictor
```

### App URL

```text
http://localhost:8501
```
