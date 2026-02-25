"""
eval_zscore.py  —  Z-Score Anomaly Evaluator
=============================================
Author : Devika
Day    : 7 (Phase 3 — Detection Logic)

Loads test_anomalies.csv, applies a Z-Score check on the 'velocity'
column, and compares the predictions against the ground-truth
'is_anomaly' label to count correct detections.
"""

import os
import numpy as np
import pandas as pd

# ─── paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_CSV = os.path.join(DATA_DIR, "test_anomalies.csv")


# ─── Z-Score detector ────────────────────────────────────────────────
def detect_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Compute element-wise Z-Scores and flag anomalies.

    Parameters
    ----------
    data      : 1-D array of numeric values (e.g. velocity readings).
    threshold : absolute Z-Score above which a point is flagged (default 3).

    Returns
    -------
    Boolean array — True where |z| > threshold.
    """
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros(len(data), dtype=bool)
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold


# ─── evaluation helpers ──────────────────────────────────────────────
def evaluate(predictions: np.ndarray, labels: np.ndarray):
    """Return a dict of classification metrics."""
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = (2 * precision * recall) / max(precision + recall, 1e-9)

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }


# ─── main ─────────────────────────────────────────────────────────────
def main():
    # 1. Load the test dataset
    if not os.path.exists(TEST_CSV):
        print(f"[ERROR] File not found: {TEST_CSV}")
        return

    df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(df)} rows from test_anomalies.csv")
    print(f"Columns: {list(df.columns)}\n")

    # 2. Extract velocity signal and ground-truth labels
    velocity = df["velocity"].values
    labels   = df["is_anomaly"].values.astype(int)

    # 3. Run Z-Score detection at several thresholds for comparison
    thresholds = [2.0, 2.5, 3.0, 3.5]

    print("=" * 60)
    print("  Z-Score Anomaly Detection — Evaluation Report")
    print("=" * 60)

    for thresh in thresholds:
        predicted = detect_zscore(velocity, threshold=thresh).astype(int)
        metrics   = evaluate(predicted, labels)

        print(f"\n--- Threshold = {thresh} ---")
        print(f"  True Positives  (correct detections) : {metrics['TP']}")
        print(f"  True Negatives  (correct normals)    : {metrics['TN']}")
        print(f"  False Positives (false alarms)        : {metrics['FP']}")
        print(f"  False Negatives (missed anomalies)    : {metrics['FN']}")
        print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"  Precision : {metrics['Precision']:.4f}")
        print(f"  Recall    : {metrics['Recall']:.4f}")
        print(f"  F1-Score  : {metrics['F1-Score']:.4f}")

    # 4. Detailed summary for default threshold (3.0)
    default_pred = detect_zscore(velocity, threshold=3.0).astype(int)
    n_detected   = int(default_pred.sum())
    n_actual     = int(labels.sum())
    correct      = int(np.sum((default_pred == 1) & (labels == 1)))

    print("\n" + "=" * 60)
    print("  Summary (threshold = 3.0)")
    print("=" * 60)
    print(f"  Total data points        : {len(df)}")
    print(f"  Actual anomalies         : {n_actual}")
    print(f"  Detected anomalies       : {n_detected}")
    print(f"  Correctly detected       : {correct}")
    print(f"  Detection rate           : {correct / max(n_actual, 1):.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
