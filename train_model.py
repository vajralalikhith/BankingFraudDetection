from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"

TARGET = "Class"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put creditcard.csv inside the data folder."
        )

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")

    if df.isnull().any().any():
        print("Warning: missing values detected. Filling numeric missing values with medians.")
        df = df.fillna(df.median(numeric_only=True))

    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("The 'Class' column must use 0 for legitimate and 1 for fraud.")

    features = [column for column in df.columns if column != TARGET]
    X = df[features]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(features)}")
    print(f"Legitimate transactions: {(y == 0).sum():,}")
    print(f"Fraudulent transactions: {(y == 1).sum():,}")
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=[0, 1],
                                target_names=["Legitimate", "Fraud"], zero_division=0))
    print("Confusion matrix [rows=actual, columns=predicted]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    try:
        print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")
    except ValueError:
        print("ROC-AUC: unavailable for this test split.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "target": TARGET,
            "metrics": {
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
            },
            "test_size": 0.20,
            "random_state": 42,
        },
        MODEL_PATH,
    )
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
