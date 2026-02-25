# 🚀 Launch Vehicle Telemetry Anomaly Detector

> A beginner-friendly Python project that simulates rocket telemetry data, injects realistic faults, and detects anomalies using statistical and machine learning methods — with a fully interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit) ![scikit-learn](https://img.shields.io/badge/scikit--learn-IsolationForest-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project builds a **telemetry anomaly detection system** for a simulated launch vehicle.  
It covers the full ML pipeline — from generating synthetic sensor data and injecting anomalies, to detecting them with Z-Score and Isolation Forest, and presenting interactive results on a Streamlit dashboard.

| | |
|---|---|
| **Authors** | Jisto Prakash · Devika P Dinesh |
| **Level** | Beginner |
| **Duration** | 14 Days |
| **Stack** | Python · NumPy · Pandas · Matplotlib · Scikit-Learn · Plotly · Streamlit |

---

## 📁 Project Structure

```
Launch_vehicle_anomaly_detection/
│
├── launch_vehicle_anomaly_detection/
│   ├── src/
│   │   ├── day1_generator.py       # Synthetic telemetry data generation
│   │   ├── day2_physics.py         # Fuel tank pressure simulation (exponential decay)
│   │   ├── anomalies.py            # Anomaly injection (spikes + drift)
│   │   ├── assemble_dataset.py     # Assembles train/test CSV datasets
│   │   ├── train_iso_forest.py     # Trains Isolation Forest → models/iso_forest.pkl
│   │   ├── eval_zscore.py          # Z-Score detection evaluation
│   │   ├── predict_iso.py          # Isolation Forest inference + results CSV
│   │   ├── make_test.py            # Test dataset builder
│   │   ├── visualize_flight.py     # Static flight telemetry plots
│   │   └── visualize_health.py     # Health & anomaly dashboard plots
│   │
│   ├── data/
│   │   ├── normal_telemetry.csv    # Raw generated telemetry
│   │   ├── train_normal.csv        # Clean training set
│   │   ├── test_anomalies.csv      # Test set with injected anomalies
│   │   └── iso_forest_results.csv  # Model predictions output
│   │
│   ├── models/
│   │   └── iso_forest.pkl          # Trained Isolation Forest model
│   │
│   ├── plots/                      # Saved static PNG charts
│   ├── app.py                      # Streamlit interactive dashboard
│   └── requirements.txt            # Python dependencies
│
├── project_plan.md                 # Full 14-day development roadmap
├── modular_work_distribution.md    # Task split between Jisto & Devika
└── README.md                       # You are here
```

---

## ✅ Development Roadmap

### Phase 1 — Data Simulation (Days 1–3)
| Day | File | Status | Description |
|-----|------|--------|-------------|
| Day 1 | `day1_generator.py` | ✅ Done | Generates altitude, velocity & engine temperature using basic physics models |
| Day 2 | `day2_physics.py` | ✅ Done | Simulates fuel tank pressure via exponential decay with Gaussian noise |
| Day 3 | `anomalies.py` | ✅ Done | Injects spike and drift anomalies into telemetry channels |

### Phase 2 — Rule-Based Detection (Days 4–6)
| Day | File | Status | Description |
|-----|------|--------|-------------|
| Day 4–5 | `assemble_dataset.py` | ✅ Done | Assembles clean train set and anomaly-injected test set |
| Day 6 | `eval_zscore.py` | ✅ Done | Z-Score detection with evaluation across multiple thresholds |

### Phase 3 — Machine Learning: Isolation Forest (Days 7–10)
| Day | File | Status | Description |
|-----|------|--------|-------------|
| Day 7–8 | `train_iso_forest.py` | ✅ Done | Trains Isolation Forest on normal data, saves model |
| Day 9–10 | `predict_iso.py` | ✅ Done | Runs inference, evaluates TP/TN/FP/FN, saves results CSV |

### Phase 4 — Dashboard & Final Report (Days 11–14)
| Day | File | Status | Description |
|-----|------|--------|-------------|
| Day 11–14 | `app.py` | ✅ Done | Full interactive Streamlit dashboard with 4 pages |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/tiz-dev/Launch_vehicle_anomaly_detection.git
cd Launch_vehicle_anomaly_detection/launch_vehicle_anomaly_detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install plotly
```

### 3. Run the pipeline (in order)
```bash
python src/day1_generator.py       # Generate base telemetry
python src/day2_physics.py         # Add physics channels
python src/anomalies.py            # Inject anomalies
python src/assemble_dataset.py     # Build train/test CSVs
python src/train_iso_forest.py     # Train Isolation Forest
python src/eval_zscore.py          # Evaluate Z-Score detection
python src/predict_iso.py          # Run model predictions
python src/visualize_flight.py     # (Optional) static plots
python src/visualize_health.py     # (Optional) health plots
```

### 4. Launch the dashboard
```bash
python -m streamlit run app.py
```
Opens at **http://localhost:8501**

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview, pipeline diagram, anomaly types |
| 📊 **Dashboard** | Live metric cards, multi-channel telemetry viewer with anomaly overlays, channel distribution chart |
| 🔍 **Anomaly Explorer** | Filterable anomaly event table, Z-score scatter plot |
| 📈 **Model Comparison** | Side-by-side metrics, performance radar chart, confusion matrices |

---

## 🧪 Physics Models

### Telemetry Channels
| Channel | Model |
|---------|-------|
| Altitude | `h = 0.25 × t²` (quadratic, constant acceleration) |
| Velocity | `v = 0.5t + N(0, 1)` (linear + noise) |
| Engine Temp | `T = 300 + 0.1t + N(0, 5)` (linear drift + noise) |
| Fuel Pressure | `P(t) = 5000 × e^(−0.008t) + N(0, 50)` (exponential decay) |
| Vibration | Random Gaussian process |

### Anomaly Types
| Type | Description |
|------|-------------|
| **Spike** | Sudden, short-lived excursion (simulates sensor bit-flip or surge) |
| **Drift** | Slow, cumulative bias (simulates calibration loss) |

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
scikit-learn
streamlit
plotly
```

---

## 👥 Work Distribution

| Module | Owner |
|--------|-------|
| Telemetry simulation, anomaly injection, dataset assembly | Jisto Prakash |
| Isolation Forest training, Z-Score evaluation, prediction | Devika P Dinesh |
| Streamlit dashboard, visualizations | Jisto Prakash, Devika P Dinesh |

---

## 📄 License

This project is for academic/educational purposes.

