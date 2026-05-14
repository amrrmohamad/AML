import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

#=============== Keep useful columns only===================
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

encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])

# ============== split feature and target ===========

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# ============= scale the data ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========== Apply PCA =====================
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print(X_pca[:5])

# ============= apply K-means ===============
kmeans = KMeans(n_clusters=3)

clusters = kmeans.fit_predict(X_pca)
df["behavioral_segment"] = clusters


# ============ test split ================
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===========train XGBoost =============
model = XGBClassifier()

model.fit(X_train, y_train)

# ========= make perdictions =========
y_pred = model.predict(X_test)

# ========= evaluate the model =======
print(classification_report(y_test, y_pred))

# ========= continouous learning ===
old_X = X_train[:5000]
old_y = y_train[:5000]

new_X = X_train[5000:10000]
new_y = y_train[5000:10000]

model.fit(old_X, old_y)
model.fit(new_X, new_y, xgb_model=model.get_booster())

print("Model updated with new batch data")
print(df.head())


print(df.info())
