import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import json

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

# Scale features for LogisticRegression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "logistic_regression": (LogisticRegression(max_iter=1000, n_jobs=-1), True),
    "decision_tree": (DecisionTreeClassifier(max_depth=15, random_state=42), False),
    "random_forest": (RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1), False),
}

results = {}
best_model_name = None
best_f1 = 0
best_model = None

for name, (model, needs_scaling) in models.items():
    print(f"\nTraining {name}...")
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    f1_macro = f1_score(y_test, preds, average="macro")
    f1_weighted = f1_score(y_test, preds, average="weighted")

    print(classification_report(y_test, preds))
    print(f"{name} -> F1 macro: {f1_macro:.4f}, F1 weighted: {f1_weighted:.4f}")

    results[name] = {"f1_macro": f1_macro, "f1_weighted": f1_weighted}

    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_model_name = name
        best_model = model

print(f"\nBest model: {best_model_name} (F1 macro: {best_f1:.4f})")

# Save best model
joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")  # save scaler in case best model needs it
print(f"Saved best model to {MODELS_DIR / 'best_model.pkl'}")
feature_info = {
    "columns": X_train.columns.tolist(),
    "best_model": best_model_name,
    "metrics": results,
}

with open(MODELS_DIR / "feature_info.json", "w") as f:
    json.dump(feature_info, f, indent=2)

print(f"Saved feature info to {MODELS_DIR / 'feature_info.json'}")