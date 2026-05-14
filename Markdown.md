# AML Fraud Detection Pipeline

## Project Overview

This project builds a machine learning pipeline for detecting suspicious financial transactions using the PaySim dataset.

The system combines:

- Data preprocessing
- Feature scaling
- PCA dimensionality reduction
- K-Means behavioral clustering
- XGBoost fraud detection
- Continuous model learning

---

# Libraries Used

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
```

### Explanation

The project uses:

- **Pandas** for data handling
- **Scikit-learn** for preprocessing, clustering, PCA, and evaluation
- **XGBoost** for fraud classification

---

# Step 1 — Load Dataset

```python
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")
```

### Explanation

The PaySim financial transaction dataset is loaded into a Pandas DataFrame.

### Example Output

| type | amount | oldbalanceOrg | newbalanceOrig | isFraud |
|------|------|------|------|------|
| CASH_OUT | 9839.64 | 170136 | 160296 | 0 |

---

# Step 2 — Select Important Features

```python
df = df[
    [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud"
    ]
]
```

### Explanation

Only useful transaction features are selected for fraud detection.

---

# Step 3 — Encode Transaction Types

```python
encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])
```

### Explanation

Machine learning models require numerical values, so transaction categories are converted into numbers.

### Example

| Transaction Type | Encoded Value |
|------|------|
| CASH_OUT | 0 |
| PAYMENT | 1 |
| TRANSFER | 2 |

---

# Step 4 — Split Features and Labels

```python
X = df.drop("isFraud", axis=1)
y = df["isFraud"]
```

### Explanation

- `X` contains transaction features
- `y` contains fraud labels

---

# Step 5 — Feature Scaling

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Explanation

Feature scaling normalizes the data to improve model performance.

---

# Step 6 — PCA Dimensionality Reduction

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

### Explanation

PCA reduces dataset dimensions while preserving important information.

### Example Output

```python
print(X_pca[:5])
```

```text
[[ 1.25 -0.83]
 [ 0.92  1.12]
 [-1.40  0.55]]
```

---

# PCA Visualization

```text
            PCA Graph

      Cluster 1  ● ● ●
                 ● ●

Cluster 2  ▲ ▲ ▲ ▲ ▲

                 ■ ■ ■
           Cluster 3
```

### Explanation

PCA helps visualize transaction behavior patterns in lower dimensions.

---

# Step 7 — Behavioral Segmentation Using K-Means

```python
kmeans = KMeans(n_clusters=3)

clusters = kmeans.fit_predict(X_pca)

df["behavioral_segment"] = clusters
```

### Explanation

K-Means groups transactions into behavioral clusters.

The generated cluster becomes a new feature:

```python
behavioral_segment
```

### Example Output

| amount | behavioral_segment |
|------|------|
| 9839.64 | 0 |
| 1864.28 | 1 |

---

# Step 8 — Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Explanation

The dataset is divided into:

- 80% training data
- 20% testing data

---

# Step 9 — Train XGBoost Model

```python
model = XGBClassifier()

model.fit(X_train, y_train)
```

### Explanation

XGBoost is used to classify transactions as fraudulent or legitimate.

---

# Step 10 — Fraud Prediction

```python
y_pred = model.predict(X_test)
```

### Explanation

The model predicts fraud on unseen transactions.

---

# Step 11 — Model Evaluation

```python
print(classification_report(y_test, y_pred))
```

### Example Output

```text
              precision    recall  f1-score

           0       1.00      1.00      1.00
           1       0.97      0.95      0.96

    accuracy                           0.99
```

### Explanation

Evaluation metrics:

- **Precision** → How many predicted fraud cases are correct
- **Recall** → How many real fraud cases were detected
- **F1-Score** → Balance between precision and recall

---

# Step 12 — Continuous Learning

```python
old_X = X_train[:5000]
old_y = y_train[:5000]

new_X = X_train[5000:10000]
new_y = y_train[5000:10000]

model.fit(old_X, old_y)
model.fit(new_X, new_y, xgb_model=model.get_booster())
```

### Explanation

The model is updated with new transaction batches without retraining from scratch.

### Benefits

- Faster updates
- Lower computation cost
- Better adaptation to new fraud patterns

---

# Final Dataset Preview

```python
print(df.head())
```

### Example Output

| type | amount | oldbalanceOrg | behavioral_segment | isFraud |
|------|------|------|------|------|
| 0 | 9839.64 | 170136 | 1 | 0 |

---

# Dataset Information

```python
print(df.info())
```

### Example Output

```text
<class 'pandas.core.frame.DataFrame'>
Columns: 8 entries
```

---

# Project Workflow

```text
Raw Dataset
     ↓
Preprocessing
     ↓
Feature Scaling
     ↓
PCA Reduction
     ↓
K-Means Clustering
     ↓
XGBoost Training
     ↓
Fraud Prediction
     ↓
Continuous Learning
```

---

# Future Improvements

- Hyperparameter tuning
- Deep learning models
- Real-time fraud detection
- Dashboard visualization
- Explainable AI

---

