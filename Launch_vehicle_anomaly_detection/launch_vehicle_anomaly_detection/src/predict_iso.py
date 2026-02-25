"""
predict_iso.py  —  Isolation Forest Inference
==============================================
Author : Devika
Day    : 9 (Phase 3 — Detection Logic)

Loads the pre-trained Isolation Forest model (models/iso_forest.pkl),
runs prediction on test_anomalies.csv, converts output from (-1/1)
to (1/0) format, evaluates against ground-truth labels, and saves
the full results to data/iso_forest_results.csv.
"""

import os
import joblib
import numpy as np
import pandas as pd

# ─── paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TEST_CSV = os.path.join(DATA_DIR, "test_anomalies.csv")
MODEL_PKL = os.path.join(MODEL_DIR, "iso_forest.pkl")
OUTPUT_CSV = os.path.join(DATA_DIR, "iso_forest_results.csv")

# Feature columns used for prediction (drop 'time' and label)
FEATURE_COLS = ["altitude", "velocity", "engine_temp", "fuel_pressure", "vibration"]


# ─── evaluation helper ───────────────────────────────────────────────
def evaluate(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """Compute classification metrics."""
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }


# ─── main ─────────────────────────────────────────────────────────────
def main():
    # 1. Validate files exist
    if not os.path.exists(MODEL_PKL):
        print(f"[ERROR] Model not found: {MODEL_PKL}")
        return
    if not os.path.exists(TEST_CSV):
        print(f"[ERROR] Test data not found: {TEST_CSV}")
        return

    # 2. Load the trained Isolation Forest model
    print(f"Loading model from {MODEL_PKL} ...")
    model = joblib.load(MODEL_PKL)
    print(f"  Model type: {type(model).__name__}")

    # 3. Load test data
    df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(df)} rows from test_anomalies.csv")
    print(f"  Columns: {list(df.columns)}")

    # 4. Prepare features (drop time and label)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X_test = df[available_features].values
    labels = df["is_anomaly"].values.astype(int)
    print(f"  Using features: {available_features}\n")

    # 5. Run model.predict()
    #    Isolation Forest returns: +1 for inliers (normal), -1 for outliers (anomaly)
    raw_predictions = model.predict(X_test)

    # 6. Convert from (-1/+1) → (1/0)
    #    -1 (outlier/anomaly) → 1
    #    +1 (inlier/normal)   → 0
    predictions = np.where(raw_predictions == -1, 1, 0)

    # 7. Evaluate against ground-truth
    metrics = evaluate(predictions, labels)

    print("=" * 60)
    print("  Isolation Forest — Evaluation Report")
    print("=" * 60)
    print(f"  True Positives  (correct detections) : {metrics['TP']}")
    print(f"  True Negatives  (correct normals)    : {metrics['TN']}")
    print(f"  False Positives (false alarms)        : {metrics['FP']}")
    print(f"  False Negatives (missed anomalies)    : {metrics['FN']}")
    print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  F1-Score  : {metrics['F1-Score']:.4f}")

    # 8. Summary
    n_detected = int(predictions.sum())
    n_actual = int(labels.sum())
    correct = int(np.sum((predictions == 1) & (labels == 1)))

    print(f"\n  Total data points        : {len(df)}")
    print(f"  Actual anomalies         : {n_actual}")
    print(f"  Detected anomalies       : {n_detected}")
    print(f"  Correctly detected       : {correct}")
    print(f"  Detection rate           : {correct / max(n_actual, 1):.2%}")
    print("=" * 60)

    # 9. Save results to CSV
    df["iso_prediction"] = predictions
    df["iso_raw_score"] = raw_predictions
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
