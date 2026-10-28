"""
Checks the raw Kaggle file against the expected schema and counts, and writes outputs/dataset_validation.csv.
Run:  python verify_dataset.py
"""
import hashlib, os
from pathlib import Path
import pandas as pd

os.chdir(Path(__file__).resolve().parent)
RAW = Path("data/raw/cardiovascular_disease_kaggle_sulianova_cardio_train.csv")
COLS = ["id", "age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]

df = pd.read_csv(RAW, sep=";")
checks = [
    ("SHA-256", hashlib.sha256(RAW.read_bytes()).hexdigest(), "-"),
    ("Columns match expected schema", list(df.columns) == COLS, True),
    ("Rows", len(df), 70000),
    ("Target column 'cardio' is binary 0/1", sorted(df["cardio"].unique().tolist()) == [0, 1], True),
    ("cardio = 0 count", int((df["cardio"] == 0).sum()), 35021),
    ("cardio = 1 count", int((df["cardio"] == 1).sum()), 34979),
    ("Missing values", int(df.isna().sum().sum()), 0),
    ("id column unique", bool(df["id"].is_unique), True),
    ("Duplicate rows ignoring id", int(df.duplicated(subset=COLS[1:]).sum()), 24),
]
out = pd.DataFrame(checks, columns=["check", "found", "expected"])
out["pass"] = [str(a) == str(b) if b != "-" else True for a, b in zip(out["found"], out["expected"])]
out.to_csv("outputs/dataset_validation.csv", index=False)
print(out.to_string(index=False))
raise SystemExit(0 if out["pass"].all() else 1)
