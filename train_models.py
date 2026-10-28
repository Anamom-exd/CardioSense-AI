"""
Step 2: Train all classifiers on the SAME train/val/test split, compute metrics from real
predictions, pick the final model by VALIDATION ROC-AUC, and save every result file the UI
and the report read.
Run:  python train_models.py      (after python preprocess.py)
"""
import json, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve)

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent)   # paths below are relative to the project root
RS = 42

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")
X_train, y_train = train.drop(columns="cardio"), train["cardio"]
X_val, y_val = val.drop(columns="cardio"), val["cardio"]
X_test, y_test = test.drop(columns="cardio"), test["cardio"]

# ---------- Leakage / feature-order checks ----------
assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns), "feature order differs between splits"
assert "cardio" not in X_train.columns
feat_key = lambda d: pd.util.hash_pandas_object(d.round(6), index=False)
tr_keys = set(feat_key(X_train))
leak = pd.DataFrame([
    {"check": "Target column absent from features", "result": "pass"},
    {"check": "Same feature order in train/val/test", "result": "pass"},
    {"check": "Scalers fitted inside Pipelines on training data only", "result": "pass"},
    {"check": "Val rows whose feature vector also occurs in train", "result": int(feat_key(X_val).isin(tr_keys).sum())},
    {"check": "Test rows whose feature vector also occurs in train", "result": int(feat_key(X_test).isin(tr_keys).sum())},
])
leak.to_csv("outputs/leakage_check.csv", index=False)
print(leak.to_string(index=False))

models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=RS))]),
    "KNN": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=25))]),
    # criterion="entropy" -> information-gain splits (the ID3 idea). scikit-learn implements an optimised CART
    # (binary splits, numeric thresholds), so this is entropy-based but NOT an exact ID3 implementation.
    "Decision Tree (entropy)": DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=RS),
    # probability=False keeps training fast; ROC-AUC uses decision_function scores (same ranking idea).
    # Trained on the full training split, like every other model.
    "SVM (RBF)": Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", C=1.0, gamma="scale"))]),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=RS),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=RS),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=RS),
    "Neural Network (MLP)": Pipeline([("sc", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                                                                      early_stopping=True, random_state=RS))]),
}


def score_vec(model, X):
    """Probability of class 1, or the SVM decision_function margin (ranking score for ROC-AUC)."""
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def scores(model, X, y):
    pred = model.predict(X)
    proba = score_vec(model, X)
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred),
                Recall=recall_score(y, pred), F1=f1_score(y, pred), ROC_AUC=roc_auc_score(y, proba))


rows, fitted = [], {}
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    secs = time.time() - t0
    v, t = scores(model, X_val, y_val), scores(model, X_test, y_test)
    row = {"Model": name, "Train_Accuracy": accuracy_score(y_train, model.predict(X_train))}
    row.update({f"Val_{k}": x for k, x in v.items()})
    row.update({f"Test_{k}": x for k, x in t.items()})
    row["Train_Time_s"] = secs
    rows.append(row)
    fitted[name] = model
    print(f"{name:24s} val AUC={v['ROC_AUC']:.4f}  test acc={t['Accuracy']:.4f}  test AUC={t['ROC_AUC']:.4f}  ({secs:.1f}s)", flush=True)

results = pd.DataFrame(rows).sort_values("Val_ROC_AUC", ascending=False).reset_index(drop=True)
results.round(4).to_csv("outputs/model_comparison.csv", index=False)
print("\n", results.round(4).to_string(index=False))

# Final model: highest VALIDATION ROC-AUC among models that output probabilities (the app needs them).
# The SVM is compared in the table but is not deployable without predict_proba.
# The test set is never used for selection.
eligible = results[results["Model"].map(lambda n: hasattr(fitted[n], "predict_proba"))].reset_index(drop=True)
best_name = eligible.loc[0, "Model"]
best = fitted[best_name]
joblib.dump({"model": best, "features": list(X_train.columns), "name": best_name,
             "n_train": len(train), "n_total": len(train) + len(val) + len(test)}, "models/best_model.joblib")
print("\nBest model (by validation ROC-AUC):", best_name)

# Predictions of the final model, used by the UI threshold analysis
pd.concat([pd.DataFrame({"split": "validation", "y_true": y_val.values, "proba": best.predict_proba(X_val)[:, 1]}),
           pd.DataFrame({"split": "test", "y_true": y_test.values, "proba": best.predict_proba(X_test)[:, 1]})]
          ).to_csv("outputs/best_model_predictions.csv", index=False)

top = results[results["Model"] == best_name].iloc[0]
json.dump({"final_model": best_name, "selection_criterion": "highest validation ROC-AUC",
           "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
           "features": list(X_train.columns),
           "validation": {k: round(float(top[f"Val_{k}"]), 4) for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]},
           "test": {k: round(float(top[f"Test_{k}"]), 4) for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]}},
          open("outputs/metrics.json", "w"), indent=2)

# ---------- Extra tables for the report (final model, test set) ----------
pb_test = best.predict_proba(X_test)[:, 1]
cut_rows = []
for c in (0.5, 0.4, 0.3):
    pr_ = pb_test >= c
    cut_rows.append({"Cutoff": c, "Accuracy": accuracy_score(y_test, pr_), "Precision": precision_score(y_test, pr_),
                     "Recall": recall_score(y_test, pr_), "Missed": int(((~pr_) & (y_test == 1)).sum()),
                     "False_alarms": int((pr_ & (y_test == 0)).sum())})
pd.DataFrame(cut_rows).round(4).to_csv("outputs/threshold_table.csv", index=False)

bp_bins = pd.cut(X_test["ap_hi"], [0, 119, 129, 139, 159, 1000], labels=["<120", "120-129", "130-139", "140-159", ">=160"])
pred_test = best.predict(X_test)
bp_rows = []
for g in bp_bins.cat.categories:
    mk = (bp_bins == g).to_numpy()
    pos = y_test.to_numpy()[mk] == 1
    bp_rows.append({"Systolic_BP": g, "Patients": int(mk.sum()), "Disease_rate": y_test.to_numpy()[mk].mean(),
                    "Accuracy": (pred_test[mk] == y_test.to_numpy()[mk]).mean(),
                    "Recall": (pred_test[mk][pos] == 1).mean() if pos.any() else np.nan})
pd.DataFrame(bp_rows).round(4).to_csv("outputs/bp_group_performance.csv", index=False)

cases = {"A": (50, 2, 165, 70, 120, 80, 1, 1, 0, 0, 1), "B": (55, 1, 160, 70, 130, 85, 1, 1, 0, 0, 1),
         "C": (60, 2, 170, 90, 160, 100, 3, 2, 1, 0, 0)}
ex = []
for k, (a, g, h, w, hi, lo, ch, gl, sm, al, ac) in cases.items():
    r = pd.DataFrame([{"age_years": a, "gender": g, "height": h, "weight": w, "bmi": w / (h / 100) ** 2, "ap_hi": hi,
                       "ap_lo": lo, "cholesterol": ch, "gluc": gl, "smoke": sm, "alco": al, "active": ac}])[list(X_train.columns)]
    ex.append({"Case": k, "BMI": round(r["bmi"][0], 1), "Probability": round(float(best.predict_proba(r)[0, 1]), 4)})
pd.DataFrame(ex).to_csv("outputs/example_predictions.csv", index=False)

# ---------- Plots ----------
plot_df = results.set_index("Model")[["Test_Accuracy", "Test_F1", "Test_ROC_AUC"]]
ax = plot_df.plot(kind="bar", figsize=(11, 5), ylim=(0.6, 0.85))
ax.set_title("Model comparison on the test set"); ax.set_ylabel("Score")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150); plt.close()

plt.figure(figsize=(7, 6))
for name, m in fitted.items():
    pr = score_vec(m, X_test)
    fpr, tpr, _ = roc_curve(y_test, pr)
    plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, pr):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves (test set)"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=150); plt.close()

ConfusionMatrixDisplay(confusion_matrix(y_test, best.predict(X_test)),
                       display_labels=["No disease", "Disease"]).plot(cmap="Blues")
plt.title(f"Confusion matrix (test) - {best_name}"); plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150); plt.close()

fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
for ax, (name, m) in zip(axes.ravel(), fitted.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, m.predict(X_test)),
                           display_labels=["No", "Yes"]).plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(name, fontsize=10)
plt.suptitle("Confusion matrices - all models (test set)"); plt.tight_layout()
plt.savefig("outputs/confusion_matrices_all.png", dpi=130); plt.close()

from sklearn.inspection import permutation_importance
pi = permutation_importance(best, X_val, y_val, n_repeats=5, random_state=RS, scoring="roc_auc", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X_val.columns).sort_values()
imp.plot(kind="barh", figsize=(7, 5), title=f"Permutation importance - {best_name}")
plt.xlabel("Drop in ROC-AUC when feature is shuffled"); plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150); plt.close()
imp.sort_values(ascending=False).round(4).to_csv("outputs/feature_importance.csv", header=["importance"])

clean = pd.read_csv("data/cardio_clean.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
clean.groupby("cardio")["age_years"].plot(kind="hist", alpha=0.5, bins=25, ax=axes[0], legend=True)
axes[0].set_title("Age by disease (0/1)")
clean.boxplot(column="ap_hi", by="cardio", ax=axes[1]); axes[1].set_title("Systolic BP by disease")
clean.boxplot(column="bmi", by="cardio", ax=axes[2]); axes[2].set_title("BMI by disease")
plt.suptitle(""); plt.tight_layout(); plt.savefig("outputs/eda.png", dpi=150); plt.close()

import seaborn as sns
plt.figure(figsize=(9, 7))
sns.heatmap(clean.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation matrix"); plt.tight_layout()
plt.savefig("outputs/correlation.png", dpi=150); plt.close()
print("Plots and result files saved in outputs/")
