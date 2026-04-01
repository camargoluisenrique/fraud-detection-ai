import streamlit as st
import pandas as pd
from model import train_model, predict

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Fraud Detection System", layout="centered")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

section.main > div {
    padding-top: 3rem;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    font-weight: 600;
    letter-spacing: 0.3px;
}

.stButton>button {
    border-radius: 8px;
    background-color: #1f77b4;
    color: white;
    font-weight: 500;
    padding: 0.6rem 1.2rem;
}

.stButton>button:hover {
    background-color: #155a8a;
}

[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 600;
}

body {
    overflow-x: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model, X_test, scaler = train_model()
    return model, X_test, scaler

model, X_test, scaler = load_model()

# =========================
# HEADER
# =========================
st.markdown("## Fraud Detection System")
st.caption("Real-time transaction monitoring using machine learning")

st.divider()

# =========================
# INPUT
# =========================
st.markdown("### Transaction Input")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount ($)", 
        min_value=0.0, 
        max_value=50000.0, 
        value=100.0,
        key="amount_input"
    )

with col2:
    time = st.slider(
        "Time (seconds)", 
        min_value=0, 
        max_value=172800, 
        value=10000,
        key="time_slider"
    )

# =========================
# SETTINGS
# =========================
st.markdown("### Detection Settings")

threshold = st.slider(
    "Fraud Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5,
    key="threshold_slider"
)

# =========================
# BASE SAMPLE
# =========================
sample = X_test.iloc[0].copy()
sample["Amount"] = amount
sample["Time"] = time

# =========================
# PREDICTION
# =========================
if st.button("Evaluate Transaction"):

    input_data = sample.to_dict()
    result = predict(model, input_data)
    prob = result["probability"]

    st.divider()

    # =========================
    # RESULT
    # =========================
    st.markdown("### Risk Assessment")

    st.metric("Fraud Probability", f"{prob:.2%}")
    st.progress(prob)

    if prob > threshold:
        st.markdown(
            "<div style='padding:10px; border-radius:8px; background-color:#3a0f0f; color:#ff4b4b;'>"
            "<b>High Risk Transaction</b>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='padding:10px; border-radius:8px; background-color:#0f2e1f; color:#00c853;'>"
            "<b>Transaction Approved</b>"
            "</div>",
            unsafe_allow_html=True
        )

    # =========================
    # ANALYSIS
    # =========================
    st.markdown("### Analysis")

    if prob > threshold:
        st.write("The transaction shows patterns consistent with fraudulent behavior.")
    else:
        st.write("The transaction is consistent with normal behavior patterns.")

    # =========================
    # MODEL INFO
    # =========================
    st.markdown("### Model Information")

    st.write("""
This model is trained on highly imbalanced financial data.

It prioritizes recall to ensure fraudulent transactions are detected,
even at the cost of some false positives.
""")