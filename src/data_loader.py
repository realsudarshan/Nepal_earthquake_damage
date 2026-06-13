import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def load_train_data():
    values = pd.read_csv(DATA_DIR / "train_values.csv")
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    df = values.merge(labels, on="building_id")
    return df

if __name__ == "__main__":
    df = load_train_data()
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nHead:\n", df.head())
    print("\nTarget distribution:\n", df["damage_grade"].value_counts())
    print("\nDtypes:\n", df.dtypes.value_counts())
    print("\nMissing values:\n", df.isnull().sum().sum())