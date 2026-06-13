import pandas as pd
from src.data_loader import load_train_data

df = load_train_data()

print("=== Categorical columns - unique values ===")
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique -> {df[col].unique()[:10]}")

print("\n=== Geo level cardinality ===")
for col in ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n=== Numeric summary ===")
print(df[["count_floors_pre_eq", "age", "area_percentage", "height_percentage", "count_families"]].describe())

print("\n=== Damage grade by foundation_type ===")
print(pd.crosstab(df["foundation_type"], df["damage_grade"], normalize="index"))