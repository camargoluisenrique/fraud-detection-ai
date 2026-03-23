import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv("data/fraud.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y

# =========================
# TRAIN MODEL
# =========================
def train_model():
    X, y = load_data()

    print("Class distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"\n🔥 ROC AUC: {auc:.4f}")

    return model, X_test, y_test

# =========================
# PREDICT
# =========================
def predict(model, input_data):
    df = pd.DataFrame([input_data])

    prob = model.predict_proba(df)[0][1]

    return {
        "probability": float(prob)
    }