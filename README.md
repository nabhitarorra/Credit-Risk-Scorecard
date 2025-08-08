# Credit Risk Scorecard

A Streamlit dashboard for visualizing credit risk using XGBoost and SHAP. Built using a proxy dataset (Titanic) for demonstration.

## Features

- Train/test split using basic borrower features
- ROC AUC scoring and classification report
- SHAP value visualizations for feature transparency
- Live credit scoring form for user input

## Technologies Used

- Python, Streamlit
- XGBoost for binary classification
- SHAP for model explainability
- Matplotlib for charts

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
