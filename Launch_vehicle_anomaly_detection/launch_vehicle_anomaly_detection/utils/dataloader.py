"""
dataloader.py  —  Cached Streamlit Data Loader
================================================
Author : Devika
Day    : 11 (Phase 4 — UI)

Provides @st.cache_data-decorated functions that read the project's
CSV files and return Pandas DataFrames.  Caching ensures the files
are read from disk only once per session, keeping the app fast.

Usage (inside any Streamlit page):
    from utils.dataloader import load_test_anomalies, load_train_normal
    df = load_test_anomalies()
"""

import os
import pandas as pd
import streamlit as st

# ─── paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ─── cached loaders ──────────────────────────────────────────────────
@st.cache_data
def load_train_normal() -> pd.DataFrame:
    """Load the clean training telemetry (data/train_normal.csv)."""
    path = os.path.join(DATA_DIR, "train_normal.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_test_anomalies() -> pd.DataFrame:
    """Load the anomaly-injected test data (data/test_anomalies.csv)."""
    path = os.path.join(DATA_DIR, "test_anomalies.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_normal_telemetry() -> pd.DataFrame:
    """Load the base normal telemetry (data/normal_telemetry.csv)."""
    path = os.path.join(DATA_DIR, "normal_telemetry.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_iso_forest_results() -> pd.DataFrame:
    """Load Isolation Forest prediction results (data/iso_forest_results.csv)."""
    path = os.path.join(DATA_DIR, "iso_forest_results.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    """
    Generic loader — pass any CSV filename inside data/.

    Parameters
    ----------
    filename : str
        Name of the file (e.g. 'test_anomalies.csv').

    Returns
    -------
    pd.DataFrame
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df
