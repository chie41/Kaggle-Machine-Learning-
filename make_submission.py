import os
import pandas as pd
import numpy as np
import pickle
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# import toàn bộ hàm feature engineering từ file train
from tde_advanced_optimization  import *

BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LOAD FINAL MODEL + THRESHOLD + FEATURE HEADER
# ============================================================

print("Loading trained model...")
with open(os.path.join(BASE, "best_optimized_model.pkl"), "rb") as f:
    model = pickle.load(f)

print("Loading best threshold...")
with open(os.path.join(BASE, "best_threshold.txt"), "r") as f:
    best_thresh = float(f.read().strip())

print("Loading feature header...")
feature_header = pd.read_csv(
    os.path.join(BASE, "train_features_header.csv")
).columns.tolist()


# ============================================================
# LOAD TEST LIGHTCURVES
# ============================================================

def load_all_test_lightcurves(base_path='./'):
    all_lightcurves = []

    for i in range(1, 21):
        split_name = f"split_{i:02d}"
        file_path = os.path.join(base_path, split_name, "test_full_lightcurves.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_lightcurves.append(df)
        else:
            print(f"Warning: {file_path} not found")

    if len(all_lightcurves) == 0:
        raise ValueError("No test files found!")

    return pd.concat(all_lightcurves, ignore_index=True)


print("Loading test lightcurves...")
test_lightcurves = load_all_test_lightcurves(base_path=BASE)

# Merge extinction info
test_log = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))[["object_id"]]
test_lightcurves["EBV"] = 0   # test không có EBV, đặt 0 (như baseline)


# Apply extinction correction
test_lightcurves["Flux_corrected"] = test_lightcurves.apply(
    lambda row: apply_extinction_correction(row["Flux"], 0, row["Filter"]),
    axis=1
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("Extracting features from test data...")
test_features = create_advanced_features(test_lightcurves)

# Align with training features
test_features = test_features.set_index("object_id")
test_features = test_features.reindex(columns=feature_header)
test_features = test_features.fillna(0)
test_features = test_features.reset_index()


# ============================================================
# PREDICT
# ============================================================

print("Predicting probabilities...")
y_proba = model.predict_proba(test_features.drop("object_id", axis=1))[:, 1]

print("Applying threshold...")
y_pred = (y_proba >= best_thresh).astype(int)


# ============================================================
# SAVE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "object_id": test_features["object_id"],
    "target": y_pred
})

submission_path = os.path.join(BASE, "submission.csv")
submission.to_csv(submission_path, index=False)

print("\n================================================")
print("Submission file created!")
print(f"Saved to: {submission_path}")
print("================================================")
