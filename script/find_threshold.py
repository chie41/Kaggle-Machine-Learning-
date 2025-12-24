import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score

# Load model
with open("best_optimized_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load train features
train_features = pd.read_csv("train_features.csv")   # nếu bạn đã lưu trước đó
y = train_features["target"]
X = train_features.drop(["object_id", "target"], axis=1).fillna(0)

# Predict probability
y_proba = model.predict_proba(X)[:, 1]

best_f1 = 0
best_thresh = 0

print("Scanning thresholds...")

for t in np.arange(0.01, 0.50, 0.005):
    y_pred = (y_proba >= t).astype(int)
    f1 = f1_score(y, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("\n======================")
print(f"Best threshold = {best_thresh:.4f}")
print(f"Best F1 = {best_f1:.4f}")
print("======================")
