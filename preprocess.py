"""
Step 1: Data cleaning, feature engineering and train/validation/test split.
Run:  python preprocess.py
"""
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir(Path(__file__).resolve().parent)   # paths below are relative to the project root
# Kaggle "Cardiovascular Disease dataset" (sulianova), see README for source and download_data.py
RAW = "data/raw/cardiovascular_disease_kaggle_sulianova_cardio_train.csv"
RANDOM_STATE = 42

# ---------- 1. Load (the file uses ';' as separator) ----------
df = pd.read_csv(RAW, sep=";")
print("Raw shape:", df.shape)
report = {"raw_rows": len(df)}

# ---------- 2. Basic checks ----------
print("Missing values:", int(df.isna().sum().sum()))
dups = df.duplicated(subset=[c for c in df.columns if c != "id"]).sum()
print("Duplicate rows (ignoring id):", int(dups))
df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"])
report["after_duplicates"] = len(df)

# ---------- 3. Fix units / create useful features ----------
df["age_years"] = (df["age"] / 365.25).round().astype(int)   # age is stored in days
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
df = df.drop(columns=["id", "age"])

# ---------- 4. Remove impossible / extreme values (outliers) ----------
# Blood pressure: systolic must be higher than diastolic and be in a human range
before = len(df)
df = df[(df["ap_hi"].between(70, 250)) & (df["ap_lo"].between(40, 200))]
df = df[df["ap_hi"] > df["ap_lo"]]
report["removed_bp"] = before - len(df)

# Height / weight / BMI: remove physiologically impossible entries
before = len(df)
df = df[df["height"].between(130, 210) & df["weight"].between(35, 200)]
df = df[df["bmi"].between(12, 60)]
report["removed_hw"] = before - len(df)

report["clean_rows"] = len(df)
print(report)

# ---------- 5. Final column order ----------
cols = ["age_years", "gender", "height", "weight", "bmi", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
df = df[cols].reset_index(drop=True)
df.to_csv("data/cardio_clean.csv", index=False)

# ---------- 6. Split: 70% train / 15% validation / 15% test (stratified) ----------
X, y = df.drop(columns="cardio"), df["cardio"]
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)

for name, X_, y_ in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    out = X_.copy()
    out["cardio"] = y_
    out.to_csv(f"data/{name}.csv", index=False)
    print(f"{name}: {out.shape}, positive rate = {y_.mean():.3f}")

pd.Series(report).to_csv("outputs/cleaning_report.csv", header=["rows"])
print("Done. Saved cleaned data and splits in data/")
