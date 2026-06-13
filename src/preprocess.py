import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def load_train_data():
    values = pd.read_csv(RAW_DIR / "train_values.csv")
    labels = pd.read_csv(RAW_DIR / "train_labels.csv")
    df = values.merge(labels, on="building_id")
    return df

def preprocess(df):
    df = df.copy()

    # Cap the age outlier (995 -> treat as 100, a sensible "old building" cap)
    df["age"] = df["age"].replace(995, 100)

    # Drop high-cardinality geo columns and building_id for now
    df = df.drop(columns=["building_id", "geo_level_2_id", "geo_level_3_id"])

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    geo_cols = ["geo_level_1_id"]

    df = pd.get_dummies(df, columns=cat_cols + geo_cols, drop_first=True)

    return df

if __name__ == "__main__":
    df = load_train_data()
    df = preprocess(df)

    X = df.drop(columns=["damage_grade"])
    y = df["damage_grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    PROCESSED_DIR.mkdir(exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Columns:", X_train.columns.tolist())