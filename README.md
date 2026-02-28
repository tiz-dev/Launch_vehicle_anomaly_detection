# 🚀 Launch Vehicle Telemetry Anomaly Detector

> A beginner-friendly Python project that simulates rocket telemetry data, injects realistic fault patterns, and detects anomalies using statistical methods and machine learning — presented on an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](#-license)

---

## 📌 Project Overview

This project builds a complete **telemetry anomaly detection pipeline** for a simulated launch vehicle.  
It covers every stage — from generating synthetic sensor readings, injecting realistic faults, detecting them with two complementary algorithms, and finally presenting all results on a dark-themed Streamlit dashboard.

| Detail | Value |
|--------|-------|
| **Authors** | Jisto Prakash · Devika P Dinesh |
| **Level** | Beginner |
| **Duration** | 14 days |
| **Stack** | Python · NumPy · Pandas · Matplotlib · Scikit-Learn · Plotly · Streamlit · Joblib |

---

## 📁 Project Structure

```
Launch_vehicle_anomaly_detection/
│
├── launch_vehicle_anomaly_detection/
│   ├── src/
│   │   ├── day1_generator.py       # Synthetic telemetry data generation
│   │   ├── day2_physics.py         # Fuel pressure & vibration simulation
│   │   ├── anomalies.py            # Spike & drift injection + Z-score detection
│   │   ├── assemble_dataset.py     # Combines base signal + physics channels
│   │   ├── make_test.py            # Builds labelled test set with injected faults
│   │   ├── train_iso_forest.py     # Trains Isolation Forest on normal data
│   │   ├── predict_iso.py          # Runs model on test set, writes results CSV
│   │   ├── eval_zscore.py          # Evaluates Z-score detector (precision/recall/F1)
│   │   ├── visualize_flight.py     # Multi-channel flight profile plots
│   │   └── visualize_health.py     # Anomaly health / distribution plots
│   │
│   ├── data/
│   │   ├── normal_telemetry.csv    # Raw simulated telemetry (Day 1 output)
│   │   ├── train_normal.csv        # Full clean training dataset (5 channels)
│   │   ├── test_anomalies.csv      # Labelled test dataset with injected faults
│   │   └── iso_forest_results.csv  # Isolation Forest predictions (app input)
│   │
│   ├── models/
│   │   └── iso_forest.pkl          # Serialised trained Isolation Forest model
│   │
│   ├── utils/
│   │   └── dataloader.py           # Shared data loading helpers
│   │
│   ├── .streamlit/
│   │   └── config.toml             # Streamlit theme configuration
│   │
│   ├── app.py                      # Full Streamlit dashboard (4 pages)
│   └── requirements.txt            # Python dependencies
│
├── project_plan.md                 # 14-day development roadmap
├── modular_work_distribution.md    # Team task breakdown (Jisto / Devika)
└── README.md                       # You are here
```

---

## 🗓️ Development Roadmap

### Phase 1 — Data Simulation & Understanding (Days 1–3)
| Day | File | Status | Description |
|-----|------|--------|-------------|
| Day 1 | `day1_generator.py` | ✅ Done | Generates altitude, velocity & engine temperature using basic physics models. Saves to `normal_telemetry.csv`. |
| Day 2 | `day2_physics.py` | ✅ Done | Simulates fuel tank pressure (exponential decay) and vibration (velocity-linked noise). |
| Day 3 | `anomalies.py` | ✅ Done | Injects **spike** (random, short-lived excursions) and **drift** (cumulative bias) anomalies into any signal. |

### Phase 2 — Rule-Based Anomaly Detection (Days 4–6)
| Day | File | Status | Description |
|-----|------|--------|-------------|
| Day 4 | `assemble_dataset.py` | ✅ Done | Assembles the full 5-channel training CSV from component scripts. |
| Day 5 | `anomalies.py` → `detect_zscore()` | ✅ Done | Z-Score anomaly flagging per channel (threshold: ±3σ). |
| Day 6 | `eval_zscore.py` | ✅ Done | Computes Precision, Recall, F1, and confusion matrix for Z-Score detector. |

### Phase 3 — Machine Learning: Isolation Forest (Days 7–10)
| Day | File | Status | Description |
|-----|------|--------|-------------|
| Day 7 | `make_test.py` | ✅ Done | Builds the labelled test set with spike + drift faults injected. |
| Day 8 | `train_iso_forest.py` | ✅ Done | Trains IsolationForest (100 trees, contamination=auto) on normal data; serialises to `iso_forest.pkl`. |
| Day 9 | `predict_iso.py` | ✅ Done | Loads trained model, predicts on test set, saves `iso_forest_results.csv`. |
| Day 10 | `eval_zscore.py` / `app.py` | ✅ Done | Side-by-side Accuracy, Precision, Recall, F1 comparison in the dashboard. |

### Phase 4 — Dashboard & Final Report (Days 11–14)
| Day | File | Status | Description |
|-----|------|--------|-------------|
| Day 11 | `app.py` — Home + Dashboard | ✅ Done | Dark-themed Streamlit app with hero banner, metric cards, and telemetry signal viewer. |
| Day 12 | `app.py` — Interactive charts | ✅ Done | Interactive multi-channel Plotly charts with anomaly overlays and sample-size slider. |
| Day 13 | `app.py` — Anomaly Explorer | ✅ Done | Filterable anomaly table + Z-score scatter plot; supports ISO Forest / Z-Score / Ground Truth filters. |
| Day 14 | `app.py` — Model Comparison | ✅ Done | Radar chart, confusion matrix heatmaps, and side-by-side metric cards for both detectors. |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Captdumbledore/Launch_vehicle_anomaly_detection.git
cd Launch_vehicle_anomaly_detection
```

### 2. Install dependencies
```bash
pip install -r launch_vehicle_anomaly_detection/requirements.txt
```

### 3. Launch the dashboard
The app auto-generates all data and trains the model on first run if the result files are missing.

```bash
cd launch_vehicle_anomaly_detection
streamlit run app.py
```

> **Tip:** Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔁 Running Individual Pipeline Steps (optional)

If you prefer to run each step manually:

```bash
cd launch_vehicle_anomaly_detection

# Step 1 – Generate base telemetry
python src/day1_generator.py

# Step 2 – Simulate fuel pressure & vibration
python src/day2_physics.py

# Step 3 – Assemble full training dataset
python src/assemble_dataset.py

# Step 4 – Build labelled test set (with injected faults)
python src/make_test.py

# Step 5 – Train Isolation Forest
python src/train_iso_forest.py

# Step 6 – Run model on test set
python src/predict_iso.py

# Step 7 – Evaluate Z-Score detector
python src/eval_zscore.py

# Step 8 – Launch dashboard
streamlit run app.py
```

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **🏠 Home** | Project overview and system pipeline diagram |
| **📊 Dashboard** | Live metric cards, multi-channel telemetry chart with anomaly overlay, anomaly distribution bar chart |
| **🔍 Anomaly Explorer** | Filterable event table, Z-score scatter plot — filter by detection method and ground truth label |
| **📈 Model Comparison** | Side-by-side Accuracy / Precision / Recall / F1, radar chart, and confusion matrix heatmaps for both algorithms |

---

## 🧪 Physics Models Used

### Telemetry Channels (Day 1)
| Sensor | Model | Formula |
|--------|-------|---------|
| Altitude | Quadratic (constant acceleration) | `h = 0.25 × t²` |
| Velocity | Linear + Gaussian noise | `v = 0.5t + N(0, 1)` |
| Engine Temp | Linear drift + noise | `T = 300 + 0.1t + N(0, 5)` |

### Fuel Tank Pressure (Day 2)
| Parameter | Value |
|-----------|-------|
| Initial Pressure (P₀) | 5000 units |
| Model | Exponential decay: `P(t) = P₀ × e^(−k×t)` |
| Decay Rate (k) | 0.008 |
| Noise | Gaussian `N(0, 50)` |

### Vibration (Day 2)
| Parameter | Value |
|-----------|-------|
| Model | Velocity-linked: `V(t) = 0.05 × v(t) + N(0, 0.1)` |

---

## 🔍 Anomaly Detection Methods

### Z-Score (Statistical)
Flags any sample whose absolute Z-score exceeds a threshold:
```
z_i = (x_i − μ) / σ       →  flag if |z_i| > 3
```
- Fast, interpretable, no training required.
- Sensitive to sensor-wide distribution; may miss gradual drift.

### Isolation Forest (Machine Learning)
Builds an ensemble of random isolation trees trained on **normal data only**:
```
score(x) = −2^(−avg_path_length(x) / c(n))
```
Anomalies are isolated near the root (short path) because they are rare and distinctive.

| Hyperparameter | Value |
|----------------|-------|
| `n_estimators` | 100 |
| `contamination` | `auto` |
| `max_samples` | `auto` |
| `random_state` | 42 |

---

## 🧯 Anomaly Types Simulated

| Type | Function | Description |
|------|----------|-------------|
| **Spike** | `inject_spike()` | Random, large, short-lived excursions (uniform ±magnitude at ~1% of samples) |
| **Drift** | `inject_drift()` | Cumulative linear bias starting at a random onset index |

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
scikit-learn
streamlit
plotly
joblib
```

Install all at once:
```bash
pip install -r launch_vehicle_anomaly_detection/requirements.txt
```

---

## 🚧 Constraints

- ✅ No advanced fluid dynamics or aerodynamics  
- ✅ No complex sensor correlations  
- ✅ No real-time streaming — static CSV files only  
- ✅ Beginner-friendly Python only  

---

## 📄 License

This project is for **academic / educational purposes** only.

