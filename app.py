import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# Set up the Streamlit UI
st.set_page_config(page_title="Credit Risk Scorecard", layout="wide")
st.title("📊 Credit Risk Scoring Dashboard")

# Load Dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].dropna()
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    return df

# Train Model
@st.cache_resource
def train_model(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, (y_pred > 0.5).astype(int), output_dict=True)

    return model, X_test, y_test, y_pred, auc, report

# Load data and model
df = load_data()
model, X_test, y_test, y_pred, auc, report = train_model(df)

# Main Dashboard
st.subheader("Model Performance")
st.metric(label="ROC AUC Score", value=f"{auc:.3f}")
st.json({
    "Precision (1)": f"{report['1']['precision']:.2f}",
    "Recall (1)": f"{report['1']['recall']:.2f}",
    "F1 Score (1)": f"{report['1']['f1-score']:.2f}"
})

# SHAP Visualization
st.subheader("Feature Importance (SHAP Values)")
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

fig, ax = plt.subplots(figsize=(10, 5))
shap.plots.beeswarm(shap_values, max_display=5, show=False)
st.pyplot(fig)

# Predict on Custom Input
st.subheader("🔢 Credit Risk Prediction Tool")
with st.form("prediction_form"):
    col1, col2, col3, col4 = st.columns(4)
    Pclass = col1.selectbox("Passenger Class (Proxy for Income Level)", [1, 2, 3])
    Sex = col2.selectbox("Sex", ["Male", "Female"])
    Age = col3.slider("Age", 18, 80, 35)
    Fare = col4.slider("Fare Amount", 0.0, 500.0, 32.0)
    submitted = st.form_submit_button("Predict Risk")

    if submitted:
        sample = pd.DataFrame.from_dict({
            "Pclass": [Pclass],
            "Sex": [1 if Sex == "Male" else 0],
            "Age": [Age],
            "Fare": [Fare]
        })
        prob = model.predict_proba(sample)[0][1]
        st.success(f"Predicted Probability of Default: {prob:.2%}")

# Footer
st.markdown("---")
st.markdown("Created by Nabhit Arorra | Demo dataset: Titanic | Model: XGBoost + SHAP + Streamlit")
