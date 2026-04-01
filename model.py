import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

#  GLOBAL (clave para producción)
model_input_columns = None


# =========================
# TRAIN MODEL
# =========================
def train_model():
    global model_input_columns

    #  cargar dataset (sample para producción)
    df = pd.read_csv("data/fraud_sample.csv")

    #  target
    y = df["Class"]
    X = df.drop("Class", axis=1)

    #  guardar columnas (CLAVE)
    model_input_columns = X.columns.tolist()

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # modelo
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # =========================
    # METRICS (para consola)
    # =========================
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test, y_prob)
    print(f"\n ROC AUC: {roc:.4f}")

    return model, X_test, y_test


# =========================
# PREDICT
# =========================
def predict(model, input_data):
    global model_input_columns

    # convertir input a DataFrame
    df = pd.DataFrame([input_data])

    #  asegurar columnas correctas
    for col in model_input_columns:
        if col not in df.columns:
            df[col] = 0

    #  ordenar columnas
    df = df[model_input_columns]

    # predicción
    prob = model.predict_proba(df)[0][1]

    return {
        "probability": prob,
        "prediction": int(prob > 0.5)
    }