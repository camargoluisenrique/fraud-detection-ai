import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.model import load_model, load_sample, predict, get_metrics

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Fraud Detection System", layout="centered")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model_cached():
    return load_model()

@st.cache_data
def load_data():
    return load_sample()

model = load_model_cached()
X_sample = load_data()

#  DATA REAL PARA MÉTRICAS 
df_full = pd.read_csv("data/fraud_sample.csv")
y_eval = df_full["Class"]
X_eval = df_full.drop("Class", axis=1)

# =========================
# HEADER
# =========================
st.title("Fraud Detection System")
st.caption("Real-time transaction monitoring using machine learning")

st.divider()

# =========================
# INPUT
# =========================
st.subheader("Transaction Input")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount ($)", 0.0, 50000.0, 100.0
    )

with col2:
    time = st.slider(
        "Time (seconds)", 0, 172800, 10000
    )

# =========================
# THRESHOLD
# =========================
st.subheader("Detection Settings")

threshold = st.slider(
    "Fraud Threshold", 0.0, 1.0, 0.5
)

# =========================
# BASE SAMPLE
# =========================
sample = X_sample.iloc[0].copy()
sample["Amount"] = amount
sample["Time"] = time

# =========================
# PREDICTION
# =========================
if st.button("Evaluate Transaction"):

    result = predict(model, sample.to_dict())
    prob = result["probability"]

    st.divider()
    st.subheader("Risk Assessment")

    st.metric("Fraud Probability", f"{prob:.2%}")
    st.progress(prob)

    if prob > threshold:
        st.error("High Risk Transaction")
    else:
        st.success("Transaction Approved")

    st.subheader("Analysis")

    if prob > threshold:
        st.write("Patterns indicate potential fraudulent behavior.")
    else:
        st.write("Transaction appears normal.")

    st.subheader("Model Info")

    st.write("""
Model optimized for imbalanced classification.

Focus:
- High recall
- Fraud detection priority
- Real-world risk scenarios
""")

# =========================
#  MODEL EVALUATION (NIVEL PRO)
# =========================
st.divider()
st.subheader("Model Evaluation")

fpr, tpr, cm = get_metrics(model, X_eval, y_eval)

# ROC Curve
fig, ax = plt.subplots(facecolor="#0e1117")

ax.plot(fpr, tpr, color="#1f77b4", linewidth=2)
ax.plot([0,1], [0,1], linestyle="--", color="gray")

ax.set_facecolor("#0e1117")
ax.set_title("ROC Curve", color="white")
ax.set_xlabel("False Positive Rate", color="white")
ax.set_ylabel("True Positive Rate", color="white")

ax.tick_params(colors="white")

st.pyplot(fig)

# Confusion Matrix
fig2, ax2 = plt.subplots(facecolor="#0e1117")

im = ax2.imshow(cm, cmap="Blues")

ax2.set_title("Confusion Matrix", color="white")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax2.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > cm.max()/2 else "black"
        )

ax2.tick_params(colors="white")

st.pyplot(fig2)