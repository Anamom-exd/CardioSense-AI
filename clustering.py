"""
Optional exploratory analysis (separate from the prediction pipeline): K-Means clustering.
The label `cardio` is NOT used to form clusters; it is only used afterwards to describe them.
Fitted on the training split only. Run:  python clustering.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.chdir(Path(__file__).resolve().parent)
RS = 42
train = pd.read_csv("data/train.csv")
X, y = train.drop(columns="cardio"), train["cardio"]
Z = StandardScaler().fit_transform(X)

# choose k by silhouette on a fixed random sample (silhouette is O(n^2))
idx = np.random.RandomState(RS).choice(len(Z), 5000, replace=False)
sil = {k: silhouette_score(Z[idx], KMeans(k, n_init=10, random_state=RS).fit_predict(Z[idx])) for k in range(2, 7)}
k = max(sil, key=sil.get)
print("silhouette by k:", {a: round(b, 4) for a, b in sil.items()}, "-> k =", k)

km = KMeans(k, n_init=10, random_state=RS).fit(Z)
prof = X.assign(cluster=km.labels_, cardio=y.values).groupby("cluster").agg(
    size=("cardio", "size"), age_years=("age_years", "mean"), bmi=("bmi", "mean"), ap_hi=("ap_hi", "mean"),
    cholesterol=("cholesterol", "mean"), smoke=("smoke", "mean"), active=("active", "mean"), disease_rate=("cardio", "mean"))
prof.round(3).to_csv("outputs/kmeans_profile.csv")
pd.Series(sil, name="silhouette").rename_axis("k").round(4).to_csv("outputs/kmeans_silhouette.csv")
print(prof.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(sil), list(sil.values()), "o-"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Silhouette score")
axes[0].set_title("Choosing k (5,000-row sample)")
P = PCA(2, random_state=RS).fit_transform(Z)
s = axes[1].scatter(P[::6, 0], P[::6, 1], c=km.labels_[::6], cmap="tab10", s=6, alpha=0.6)
axes[1].set_title(f"K-Means clusters (k={k}) in PCA space"); axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
plt.tight_layout(); plt.savefig("outputs/kmeans_clusters.png", dpi=150); plt.close()
