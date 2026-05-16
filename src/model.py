import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "fraud_sample.csv")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "fraud_model.pkl")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# =========================
# GLOBAL
# =========================
model_input_columns = None

# =========================
# TRAIN MODEL
# =========================
def train_and_save_model():
    global model_input_columns

    df = pd.read_csv(DATA_PATH)

    y = df["Class"]
    X = df.drop("Class", axis=1)

    model_input_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # métricas
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC: {roc:.4f}")

    # guardar modelo
    joblib.dump((model, model_input_columns), MODEL_PATH)

    return model, X_test, y_test

# =========================
# LOAD MODEL
# =========================
def load_model():
    global model_input_columns

    if os.path.exists(MODEL_PATH):
        model, model_input_columns = joblib.load(MODEL_PATH)
        return model

    model, _, _ = train_and_save_model()
    return model

# =========================
# LOAD SAMPLE DATA
# =========================
def load_sample():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Class", axis=1)
    return X

# =========================
# PREDICT
# =========================
def predict(model, input_data):
    global model_input_columns

    df = pd.DataFrame([input_data])

    for col in model_input_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_input_columns]

    prob = model.predict_proba(df)[0][1]

    return {
        "probability": prob,
        "prediction": int(prob > 0.5)
    }

# =========================
# RUN
# =========================
if __name__ == "__main__":
    train_and_save_model()



def get_metrics(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    fpr, tpr, _ = roc_curve(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    return fpr, tpr, cm