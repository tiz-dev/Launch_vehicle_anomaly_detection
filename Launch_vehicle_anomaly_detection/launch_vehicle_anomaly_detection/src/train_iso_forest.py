"""
train_iso_forest.py
-------------------
Day 5 – Isolation Forest Training

Fits sklearn's IsolationForest on clean, normal-flight telemetry
(train_normal.csv), then saves the trained model to models/iso_forest.pkl
for later use in anomaly scoring / evaluation.

Usage
-----
    python src/train_iso_forest.py

Output
------
    models/iso_forest.pkl   – serialised IsolationForest model
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "train_normal.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "iso_forest.pkl")

# ---------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------
N_ESTIMATORS    = 100     # Number of isolation trees
CONTAMINATION   = "auto"  # Expected fraction of outliers in training data
                           # "auto" → decision threshold = -0.5 (scikit-learn default)
MAX_SAMPLES     = "auto"  # Samples used per tree ("auto" = min(256, n_samples))
RANDOM_STATE    = 42      # Reproducibility seed


def load_features(path: str) -> pd.DataFrame:
    """
    Load the CSV and drop the time column.

    Parameters
    ----------
    path : str  – Absolute path to the CSV file.

    Returns
    -------
    X : pd.DataFrame  – Feature matrix (all columns except 'time').
    """
    df = pd.read_csv(path)
    print(f"[load_features] Loaded {len(df):,} rows from '{path}'.")
    print(f"[load_features] Columns found : {list(df.columns)}")

    if "time" not in df.columns:
        raise ValueError("Expected a 'time' column in the CSV; not found.")

    X = df.drop(columns=["time"])
    print(f"[load_features] Feature columns: {list(X.columns)}")
    return X


def train(X: pd.DataFrame) -> IsolationForest:
    """
    Instantiate and fit an IsolationForest on feature matrix X.

    Model overview
    --------------
    IsolationForest builds an ensemble of random isolation trees.
    Each tree recursively partitions the feature space by randomly
    choosing a feature and a random split value.  Anomalous points
    are isolated near the root (short path length); normal points
    require many splits (long path length).

    The anomaly score for sample x is:

        score(x) = -2^( -avg_path_length(x) / c(n) )

    where c(n) is the average path length for a random Binary Search Tree
    built on n samples (normalisation constant).

    Returns –1 for anomalies, +1 for normal samples when calling
    `predict()`.

    Parameters
    ----------
    X : pd.DataFrame  – Training feature matrix (normal data only).

    Returns
    -------
    model : IsolationForest  – Fitted model.
    """
    print(
        f"\n[train] Fitting IsolationForest  "
        f"(n_estimators={N_ESTIMATORS}, contamination={CONTAMINATION}, "
        f"random_state={RANDOM_STATE}) …"
    )

    model = IsolationForest(
        n_estimators  = N_ESTIMATORS,
        contamination = CONTAMINATION,
        max_samples   = MAX_SAMPLES,
        random_state  = RANDOM_STATE,
        n_jobs        = -1,   # use all available CPU cores
    )
    model.fit(X)

    print(f"[train] Training complete. Trees built: {len(model.estimators_)}")
    return model


def save_model(model: IsolationForest, path: str) -> None:
    """
    Serialise the trained model to disk using joblib.

    Parameters
    ----------
    model : IsolationForest  – Fitted model to save.
    path  : str              – Destination file path (.pkl).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"[save_model] Model saved -> '{path}'  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load training data (drop time column)
    X_train = load_features(DATA_PATH)

    # 2. Fit the model
    iso_forest = train(X_train)

    # 3. Quick sanity-check on training data
    preds  = iso_forest.predict(X_train)
    scores = iso_forest.decision_function(X_train)
    n_flagged = (preds == -1).sum()

    print(
        f"\n[sanity] Predictions on training set:\n"
        f"  Normal samples   (+1): {(preds == 1).sum():,}\n"
        f"  Anomaly samples  (-1): {n_flagged:,}  "
        f"({100 * n_flagged / len(X_train):.2f}% of training data flagged)\n"
        f"  Decision score   min : {scores.min():.4f}\n"
        f"  Decision score   max : {scores.max():.4f}\n"
        f"  Decision score   mean: {scores.mean():.4f}"
    )

    # 4. Save to disk
    save_model(iso_forest, MODEL_PATH)

    print("\n[OK] Done. Run evaluation script to score test_anomalies.csv.")
