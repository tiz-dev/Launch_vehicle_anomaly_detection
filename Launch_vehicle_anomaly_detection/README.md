# 🚀 Launch Vehicle Telemetry Anomaly Detector

> A beginner-friendly Python project that simulates rocket telemetry data and detects anomalies using statistical methods and machine learning.

---

## 📌 Project Overview

This project builds a **telemetry anomaly detection system** for a simulated launch vehicle over a 14-day development plan.  
It covers the full pipeline — from generating synthetic sensor data, injecting anomalies, detecting them with Z-Score and Isolation Forest, and finally presenting results on a Streamlit dashboard.

**Author:** Jisto Prakash  
**Level:** Beginner  
**Duration:** 14 Days  
**Stack:** Python · NumPy · Pandas · Matplotlib · Scikit-Learn · Streamlit

---

## 📁 Project Structure

```
Launch_vehicle_anomaly_detection/
│
├── launch_vehicle_anomaly_detection/
│   ├── src/
│   │   ├── day1_generator.py       # Synthetic telemetry data generation
│   │   └── day2_physics.py         # Fuel tank pressure simulation (exponential decay)
│   │
│   ├── data/
│   │   └── normal_telemetry.csv    # Generated telemetry data (auto-created)
│   │
│   └── requirements.txt            # Python dependencies
│
├── project_plan.md                 # Full 14-day development roadmap
└── README.md                       # You are here
```

---

## 🗓️ Development Roadmap

### Phase 1 — Data Simulation & Understanding (Days 1–3)
| Day | File | Description |
|-----|------|-------------|
| ✅ Day 1 | `day1_generator.py` | Generates altitude, velocity & engine temperature using basic physics models. Saves to `normal_telemetry.csv`. |
| ✅ Day 2 | `day2_physics.py` | Simulates fuel tank pressure using an **exponential decay model** with Gaussian noise. |
| ⬜ Day 3 | `day3_anomaly_injector.py` | Injects point & drift anomalies into telemetry data. |

### Phase 2 — Rule-Based Anomaly Detection (Days 4–6)
| Day | File | Description |
|-----|------|-------------|
| ⬜ Day 4 | — | Statistical analysis: Mean & Standard Deviation |
| ⬜ Day 5 | `day5_zscore.py` | Z-Score based anomaly flagging (threshold: ±3σ) |
| ⬜ Day 6 | — | Visualize flagged anomalies on charts |

### Phase 3 — Machine Learning: Isolation Forest (Days 7–10)
| Day | File | Description |
|-----|------|-------------|
| ⬜ Day 7 | — | Concept: Isolation Forest |
| ⬜ Day 8 | `day8_isolation_forest.py` | Train model on normal data, predict on anomaly data |
| ⬜ Day 9 | — | Evaluate: Accuracy, Precision, Recall |
| ⬜ Day 10 | — | Tune `contamination` parameter |

### Phase 4 — Dashboard & Final Report (Days 11–14)
| Day | File | Description |
|-----|------|-------------|
| ⬜ Day 11 | `app.py` | Setup Streamlit, display raw dataframe |
| ⬜ Day 12 | `app.py` | Add interactive line charts |
| ⬜ Day 13 | `app.py` | Integrate anomaly detection into dashboard |
| ⬜ Day 14 | — | Final polish, demo, and submission |

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

### 3. Run Day 1 — Generate Telemetry Data
```bash
python launch_vehicle_anomaly_detection/src/day1_generator.py
```
Outputs: `launch_vehicle_anomaly_detection/data/normal_telemetry.csv`

### 4. Run Day 2 — Simulate Fuel Tank Pressure
```bash
python launch_vehicle_anomaly_detection/src/day2_physics.py
```

---

## 🧪 Physics Models Used

### Day 1 — Altitude & Velocity
| Sensor | Model | Formula |
|--------|-------|---------|
| Altitude | Quadratic (constant acceleration) | `h = 0.25 × t²` |
| Velocity | Linear + noise | `v = 0.5t + N(0, 1)` |
| Engine Temp | Linear drift + noise | `T = 300 + 0.1t + N(0, 5)` |

### Day 2 — Fuel Tank Pressure
| Parameter | Value |
|-----------|-------|
| Initial Pressure (P₀) | 5000 units |
| Model | Exponential decay: `P(t) = P₀ × e^(−k×t)` |
| Decay Rate (k) | 0.008 |
| Noise | Gaussian `N(0, 50)` |

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
scikit-learn
streamlit
```

---

## 🚧 Constraints

- ✅ No advanced fluid dynamics or aerodynamics
- ✅ No complex sensor correlations  
- ✅ No real-time streaming — static CSV files only  
- ✅ Beginner-friendly Python only

---

## 📄 License

This project is for academic/educational purposes.
