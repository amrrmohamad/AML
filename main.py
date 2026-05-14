"""
End-to-End Anti-Money Laundering (AML) Detection Pipeline
Covers all 5 phases from the project requirements:
  Phase 1 - Data Preprocessing & Feature Engineering
  Phase 2 - Dimensionality Reduction (PCA) with justification
  Phase 3 - Behavioral Customer Segmentation (K-Means) with justification
  Phase 4 - Predictive Risk Scoring (XGBoost) with tuning & class imbalance handling
  Phase 5 - Continuous Learning (true partial/batch training)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report,
    silhouette_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

# ============================================================
# PHASE 1: Data Preprocessing & Feature Engineering
# ============================================================
print("=" * 60)
print("PHASE 1: Data Preprocessing & Feature Engineering")
print("=" * 60)

df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# Keep useful columns
df = df[
    [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]
]
print("\n[1.1] Missing values per column:")
print(df.isnull().sum())

df.dropna(subset=["isFraud"], inplace=True)

numeric_cols = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
for col in numeric_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"  Filled nulls in '{col}' with median = {median_val:.2f}")

# --- Encoding ---
encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])
print("\n[1.2] Encoded 'type' classes:", list(encoder.classes_))

# --- Feature engineering: derived balance-change features ---
# These capture the "delta" pattern characteristic of layering/smurfing
df["orig_balance_change"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
df["dest_balance_change"] = df["newbalanceDest"] - df["oldbalanceDest"]

print("\n[1.3] Dataset shape after preprocessing:", df.shape)
print(df.head(3))

# ============================================================
# PHASE 2: Dimensionality Reduction (PCA) with justification
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: PCA with Explained Variance Justification")
print("=" * 60)

feature_cols = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_change",
    "dest_balance_change",
]

X_raw = df[feature_cols].copy()
y = df["isFraud"].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca_full = PCA()
pca_full.fit(X_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained) + 1), explained, color="steelblue")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o", color="darkorange")
plt.axhline(y=0.90, color="red", linestyle="--", label="90% threshold")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig("pca_explained_variance.png", dpi=120)
plt.close()
print("  [Saved] pca_explained_variance.png")

# Choose n_components that explain ≥ 90% variance
n_components = int(np.argmax(cumulative >= 0.90)) + 1
print(f"\n[2.1] Components needed for ≥90% variance: {n_components}")
for i, (ev, cv) in enumerate(zip(explained, cumulative), 1):
    marker = " <-- selected" if i == n_components else ""
    print(f"  PC{i}: {ev:.4f} individual | {cv:.4f} cumulative{marker}")

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print(f"\n[2.2] PCA output shape: {X_pca.shape}")

# ============================================================
# PHASE 3: Behavioral Segmentation (K-Means) with justification
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: K-Means Clustering with Elbow & Silhouette")
print("=" * 60)

# Use a sample for speed (silhouette is O(n²) in memory)
sample_size = min(20_000, len(X_pca))
idx = np.random.choice(len(X_pca), sample_size, replace=False)
X_sample = X_pca[idx]

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sample)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_sample, labels, sample_size=5000)
    silhouette_scores.append(sil)
    print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.4f}")

# --- Elbow + Silhouette plots ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(list(k_range), inertias, marker="o", color="steelblue")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.subplot(1, 2, 2)
plt.plot(list(k_range), silhouette_scores, marker="o", color="darkorange")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.tight_layout()
plt.savefig("kmeans_selection.png", dpi=120)
plt.close()
print("  [Saved] kmeans_selection.png")

# Select k with highest silhouette score
optimal_k = list(k_range)[int(np.argmax(silhouette_scores))]
print(f"\n[3.1] Optimal k selected by silhouette score: {optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["behavioral_segment"] = kmeans_final.fit_predict(X_pca)
print(
    f"[3.2] Cluster distribution:\n{df['behavioral_segment'].value_counts().sort_index()}"
)

# ============================================================
# PHASE 4: Predictive Risk Scoring (XGBoost)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: XGBoost with Hyperparameter Tuning")
print("=" * 60)

# Rebuild feature matrix including the new behavioral_segment
X = df[feature_cols + ["behavioral_segment"]].copy()
y = df["isFraud"].copy()

X_scaled_full = scaler.fit_transform(X)  # re-scale with new feature

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_full, y, test_size=0.2, random_state=42, stratify=y
)

# Class imbalance ratio — used for scale_pos_weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos
print(f"\n[4.1] Class imbalance ratio (neg/pos): {spw:.1f}  → used as scale_pos_weight")

# Hyperparameter search space
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
}

base_model = XGBClassifier(
    scale_pos_weight=spw,  # handles class imbalance
    eval_metric="aucpr",  # area under PR curve — better for imbalanced data
    random_state=42,
    use_label_encoder=False,
)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring="f1",  # optimise for F1 (balances precision & recall)
    cv=3,
    random_state=42,
    verbose=1,
    n_jobs=-1,
)

print("\n[4.2] Running RandomizedSearchCV (20 iterations, 3-fold CV)...")
search.fit(X_train, y_train)

best_params = search.best_params_
print(f"\n[4.3] Best hyperparameters:\n{best_params}")

model = search.best_estimator_

# --- Evaluation ---
y_pred = model.predict(X_test)
print("\n[4.4] Classification Report (focus on Recall & F1):")
print(classification_report(y_test, y_pred, digits=4))

# Note: Accuracy is intentionally omitted as the primary metric.
# With ~99.9% legitimate transactions, a model predicting all-legitimate
# achieves ~99.9% accuracy yet catches zero fraud. Recall and F1-score
# are the meaningful metrics for AML detection.

# ============================================================
# PHASE 5: Continuous Learning (true partial/batch training)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: Continuous Learning — Batch Update (No Full Retrain)")
print("=" * 60)

# Simulate a new batch of transactions arriving the following month.
# STRICT CONSTRAINT: update model weights using ONLY new data.
# We do NOT combine old + new data and retrain from scratch.

# Split training data into historical and new-batch portions
historical_X = X_train[:5000]
historical_y = y_train[:5000]

new_batch_X = X_train[5000:10000]
new_batch_y = y_train[5000:10000]

print(f"\n[5.1] Historical batch size : {len(historical_X)}")
print(f"[5.2] New (incoming) batch size: {len(new_batch_X)}")

# Train initial model on historical data using the best hyperparams
initial_model = XGBClassifier(
    **best_params,
    scale_pos_weight=spw,
    eval_metric="aucpr",
    random_state=42,
    use_label_encoder=False,
)
initial_model.fit(historical_X, historical_y)
print("\n[5.3] Initial model trained on historical batch.")

# Evaluate before update
y_pred_before = initial_model.predict(X_test)
print("\n[5.4] Performance BEFORE batch update:")
print(classification_report(y_test, y_pred_before, digits=4))

updated_model = XGBClassifier(
    **best_params,
    scale_pos_weight=spw,
    eval_metric="aucpr",
    random_state=42,
    use_label_encoder=False,
)
updated_model.fit(new_batch_X, new_batch_y, xgb_model=initial_model.get_booster())

print("\n[5.5] Model updated with new batch data (no retrain from scratch).")

# Evaluate after update
y_pred_after = updated_model.predict(X_test)
print("\n[5.6] Performance AFTER batch update:")
print(classification_report(y_test, y_pred_after, digits=4))

print("\n" + "=" * 60)
print("Pipeline complete. Outputs saved:")
print("  - pca_explained_variance.png")
print("  - kmeans_selection.png")
print("=" * 60)
