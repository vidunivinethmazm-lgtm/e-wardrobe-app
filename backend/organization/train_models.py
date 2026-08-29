"""
Run once (or whenever the dataset changes) to train and persist all ML models.

Usage (from the repository root):
    python -m backend.organization.train_models
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sri_lanka_smart_wardrobe_dataset.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from dataset.\n")

# ── Shared encoders ──────────────────────────────────────────────────────────

occasion_enc = LabelEncoder()
df["occasion_encoded"] = occasion_enc.fit_transform(df["occasion"])

fabric_enc = LabelEncoder()
df["fabric_encoded"] = fabric_enc.fit_transform(df["fabric"])

joblib.dump(occasion_enc, os.path.join(MODELS_DIR, "occasion_encoder.pkl"))
joblib.dump(fabric_enc,   os.path.join(MODELS_DIR, "fabric_encoder.pkl"))

# ── Model 2: Smart Storage Layout — KMeans Clustering ───────────────────────
# 5 clusters match the 5 distinct occasion groups.
# StandardScaler ensures both features contribute equally to distance.

X_raw = df[["total_wear_count", "occasion_encoded"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
joblib.dump(scaler, os.path.join(MODELS_DIR, "kmeans_scaler.pkl"))

kmeans = KMeans(n_clusters=5, n_init=20, random_state=42)
kmeans.fit(X_scaled)
joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_storage.pkl"))

sil = silhouette_score(X_scaled, kmeans.labels_)
print(f"[KMeans Clustering]   Silhouette Score : {sil:.4f}  ({sil*100:.2f}%)")

# ── Model 3: Wear Count / Lifespan — Linear Regression ─────────────────────

X_reg = df[["total_wear_count", "fabric_encoded", "max_wears_before_wash"]].values
y_reg = df["sustainability_score"].values

reg = LinearRegression()
reg.fit(X_reg, y_reg)
joblib.dump(reg, os.path.join(MODELS_DIR, "sustainability_regressor.pkl"))

r2 = reg.score(X_reg, y_reg)
print(f"[Linear Regression]   R² Score         : {r2:.4f}  ({r2*100:.2f}%)")

# ── Model 4: Alerts & Insights — Isolation Forest ───────────────────────────
# Accuracy is measured using z-score ground truth:
#   items beyond ±2 std of their occasion group = true anomalies.

X_iso = df[["total_wear_count", "occasion_encoded"]].values
iso   = IsolationForest(contamination=0.1, random_state=42)
iso.fit(X_iso)
joblib.dump(iso, os.path.join(MODELS_DIR, "isolation_forest.pkl"))

# Build per-occasion z-scores to define ground truth anomalies
df["z_score"]     = df.groupby("occasion")["total_wear_count"].transform(
    lambda x: (x - x.mean()) / x.std()
)
df["true_anomaly"] = (df["z_score"].abs() > 2).astype(int)
df["pred_anomaly"] = (iso.predict(X_iso) == -1).astype(int)

iso_acc = (df["true_anomaly"] == df["pred_anomaly"]).mean()
print(f"[Isolation Forest]    Accuracy         : {iso_acc:.4f}  ({iso_acc*100:.2f}%)")

print(f"\nAll models saved to: {MODELS_DIR}")
