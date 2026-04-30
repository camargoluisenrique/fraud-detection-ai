# Fraud Detection System

End-to-end machine learning application for detecting fraudulent financial transactions in real time.

This project focuses on a practical and production-oriented approach to fraud detection, where model performance is evaluated not only by accuracy, but by its ability to correctly identify rare and high-risk events.

---

## Live Demo

https://fraud-detection-ai-s2xftbqgz8mffnelg5xgsr.streamlit.app/

---

## Problem Context

Fraud detection is a highly imbalanced classification problem. In real-world scenarios, fraudulent transactions represent a very small percentage of the data, but missing them has a disproportionate impact.

Because of this, the objective is not simply to maximize accuracy, but to design a system that:

- Identifies as many fraudulent cases as possible  
- Maintains an acceptable level of false positives  
- Supports real-time decision making  

---

## Approach

The project implements a complete machine learning workflow:

- Data preprocessing and feature handling  
- Model training using Random Forest  
- Class imbalance handling via class weighting  
- Evaluation using ROC AUC and confusion matrix  
- Deployment as an interactive application with Streamlit  

The system allows dynamic threshold adjustment, making it possible to tune the model behavior depending on business requirements.

---

## Model

- Algorithm: Random Forest Classifier  
- Imbalance handling: class_weight="balanced"  
- Output: probability-based prediction  
- Decision rule: configurable threshold  

---

## Performance

The model achieves strong performance on an imbalanced dataset:

- ROC AUC: ~0.98  
- High recall on fraudulent class  
- Low false positive rate relative to class distribution  

Confusion matrix (example):

- True Negatives: 2458  
- False Positives: 2  
- False Negatives: 10  
- True Positives: 482  

These results reflect a model that is effective at identifying fraudulent behavior while keeping unnecessary alerts under control.

---

## Application

The system is deployed as an interactive interface that allows:

- Real-time transaction evaluation  
- Manual adjustment of fraud detection threshold  
- Visualization of model performance (ROC curve and confusion matrix)  
- Immediate risk assessment based on input features  

This simulates a simplified version of a production monitoring tool.

---

## Project Structure

fraud-detection-ai/
│
├── app.py
├── README.md
├── requirements.txt
│
├── src/
│ ├── model.py
│ ├── prepare_sample.py
│
├── data/
│ └── fraud_sample.csv
│
├── outputs/
│ └── models/
│
└── notebooks/


---

## Key Considerations

- Prioritized recall over precision due to business impact  
- Avoided unnecessary model complexity in favor of interpretability  
- Ensured consistent input handling for real-time predictions  
- Designed the system to be easily extensible (API integration, batch scoring)

---

## Potential Extensions

- Integration with REST API (FastAPI)  
- Real-time streaming data pipeline  
- Advanced models (Gradient Boosting, XGBoost)  
- Threshold optimization based on cost-sensitive metrics  

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

---

Author

Luis Enrique Camargo Rangel
Data Scientist | Applied Machine Learning