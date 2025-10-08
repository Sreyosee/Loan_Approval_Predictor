# task3_loan_approval_prediction.py
# Final working Streamlit app

import json
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ---------------------------------------------------
# MUST be first Streamlit command
# ---------------------------------------------------
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💳", layout="centered")

# ---------------------------------------------------
# Load trained model and schema
# ---------------------------------------------------
@st.cache_resource
def load_assets():
    model = load("loan_approval_pipeline.joblib")
    with open("schema.json", "r") as f:
        schema = json.load(f)
    return model, schema

try:
    model, schema = load_assets()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Make sure 'loan_approval_pipeline.joblib' and 'schema.json' are in the same folder as this app.")
    st.stop()

# ---------------------------------------------------
# Setup columns and defaults
# ---------------------------------------------------
num_cols = schema.get("num_cols", [])
cat_cols = schema.get("cat_cols", [])
defaults = schema.get("defaults", {})

st.title("💳 Loan Approval Predictor")
st.caption("Preprocessing + SMOTE are built into this trained pipeline.")

# ---------------------------------------------------
# SINGLE form with a unique key
# ---------------------------------------------------
FORM_KEY = "loan_form_unique"  # ✅ unique key (fixed)

with st.form(key=FORM_KEY):
    st.subheader("Applicant Details")
    colA, colB = st.columns(2)
    values = {}

    # Numeric fields
    for i, c in enumerate(num_cols):
        with (colA if i % 2 == 0 else colB):
            values[c] = st.number_input(c, value=float(defaults.get(c, 0.0)))

    # Categorical fields
    for i, c in enumerate(cat_cols):
        with (colA if i % 2 == 0 else colB):
            values[c] = st.text_input(c, value=str(defaults.get(c, "")))

    submitted = st.form_submit_button("Predict")

# ---------------------------------------------------
# Predict after submission
# ---------------------------------------------------
if submitted:
    X_infer = pd.DataFrame([values])
    pred = model.predict(X_infer)[0]
    prob = model.predict_proba(X_infer)[0][1] if hasattr(model, "predict_proba") else None

    st.markdown("---")
    st.subheader("Result")
    label = "✅ Approved" if int(pred) == 1 else "❌ Not Approved"
    st.markdown(f"**Prediction:** {label}")
    if prob is not None:
        st.markdown(f"**Approval probability:** `{prob:.3f}`")

    st.success("Prediction complete!")
