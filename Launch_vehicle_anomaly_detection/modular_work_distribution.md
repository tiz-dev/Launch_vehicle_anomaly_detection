# Modular Work Distribution: Launch Vehicle Anomaly Detection

**Objective**: Split the 14-day project equally between **Jisto** and **Devika** with strict modularity.
**Guiding Principle**: "Shared Nothing" Architecture where possible. clear Inputs/Outputs (I/O) for every task.

---

## Phase 1: Data Simulation (Days 1-4)

### Day 1: Basic Telemetry (Completed)
**Status**: Done.
**Code reference** (`src/day1_generator.py`):
```python
import numpy as np
import pandas as pd
import os

def generate_telemetry():
    DURATION = 600
    RATE = 10
    total_points = DURATION * RATE
    time = np.linspace(0, DURATION, total_points)
    altitude = (time ** 2) * 0.25 
    velocity = time * 0.5 + np.random.normal(0, 1.0, total_points)
    return pd.DataFrame({'time': time, 'altitude': altitude, 'velocity': velocity})
```

---

### Day 2: Advanced Physics (Correlation)
**Goal**: Add realistic sensor columns to the DataFrame.
**Independence Strategy**: Each person writes a *pure function* that takes a numpy array and returns a numpy array. They can be developed in separate files (`src/physics_jisto.py`, `src/physics_devika.py`).

#### Jisto: Hydraulics (Fuel Pressure)
*   **Task**: Simulate fuel pressure dropping as fuel is burned.
*   **Prompt for Model**:
    > "Write a Python function `simulate_pressure(time_vector)` that simulates rocket fuel tank pressure. It should start at 5000 units and decrease exponentially as time increases, adding some random Gaussian noise. Input is a numpy array of time. Output is a numpy array of pressure values."
*   **Interface Code**:
    ```python
    def simulate_pressure(time_vector: np.ndarray) -> np.ndarray:
        # returns pressure array matching length of time_vector
        pass
    ```

#### Devika: Vibration (Aerodynamics)
*   **Task**: Simulate airframe vibration peaking at Max-Q (maximum dynamic pressure).
*   **Prompt for Model**:
    > "Write a Python function `simulate_vibration(velocity_vector)` that simulates rocket vibration. Vibration should be proportional to velocity but peak specifically when velocity is around 300 m/s (Max-Q), then decrease. Add random noise. Input is velocity array. Output is vibration array."
*   **Interface Code**:
    ```python
    def simulate_vibration(velocity_vector: np.ndarray) -> np.ndarray:
        # returns vibration array matching length of velocity_vector
        pass
    ```

---

### Day 3: Anomaly Injection (The Chaos)
**Goal**: Create functions to corrupt data.
**Independence Strategy**: Separate "Injector" classes or modules.

#### Jisto: Sudden Failures (Spikes/Flatlines)
*   **Task**: functions for sudden, sharp anomalies.
*   **Prompt for Model**:
    > "Write a Python function `inject_spike(signal, magnitude=100, probability=0.01)` that takes a data array and adds sudden large spikes to random points."
*   **Interface Code**:
    ```python
    def inject_spike(data: np.ndarray) -> np.ndarray:
        # Returns copy of data with random spikes
        pass
    ```

#### Devika: Gradual Failures (Drift/Noise)
*   **Task**: functions for subtle, creeping anomalies.
*   **Prompt for Model**:
    > "Write a Python function `inject_drift(signal, drift_factor=0.1)` that takes a data array and adds a cumulative drift (increasing bias) starting from a random point in time."
*   **Interface Code**:
    ```python
    def inject_drift(data: np.ndarray) -> np.ndarray:
        # Returns copy of data with added drift
        pass
    ```

---

### Day 4: Dataset Assembly
**Goal**: Combine work into final CSVs.
**Independence Strategy**: Checkpoint files. Jisto makes `train.csv`, Devika makes `test.csv`.

#### Jisto: Training Data (Reference)
*   **Task**: Import functions from Day 1 & 2. Generate clean data.
*   **Prompt for Model**:
    > "create a script that imports `generate_telemetry` (Day 1), `simulate_pressure` (Jisto Day 2), and `simulate_vibration` (Devika Day 2). Generate 600s of data and save it as `data/train_normal.csv`."

#### Devika: Testing Data (Anomalies)
*   **Task**: Take clean data and apply Day 3 injectors.
*   **Prompt for Model**:
    > "Create a script that generates clean data (like Jisto's task) but then imports `inject_spike` (Jisto Day 3) and `inject_drift` (Devika Day 3) to corrupt random segments. Save as `data/test_anomalies.csv` with a label column 'is_anomaly' (1 for True, 0 for False)."

---

## Phase 2: Analysis (Days 5-6)

### Day 5: Visualization
**Goal**: Plot flight data.
**Independence Strategy**: Work on different plot files.

#### Jisto: Flight Metrics
*   **Task**: `visualize_flight.py`. Plot Altitude/Velocity.
*   **Code Spec**: Use Matplotlib `subplots`. Focus on X=Time, Y=Altitude/Velocity.

#### Devika: Health Metrics
*   **Task**: `visualize_health.py`. Plot Pressure/Vibration/Temp.
*   **Code Spec**: Use Matplotlib. Focus on correlating Vibration peaks with Velocity.

---

## Phase 3: Detection Logic (Days 7-10)

### Day 7: Z-Score (The Rule-Based Engine)
**Goal**: Statistical detection.

#### Jisto: The Algorithm
*   **Task**: Implement the detector.
*   **Prompt for Model**:
    > "Write a function `detect_zscore(data, threshold=3)` that calculates (data - mean) / std. Returns valid indices where z > threshold."
*   **Interface Code**:
    ```python
    def detect_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        # returns boolean encoding or indices of anomalies
        pass
    ```

#### Devika: The Evaluator
*   **Task**: Run the detector on test data.
*   **Prompt for Model**:
    > "Write a script that loads `test_anomalies.csv`, applies a Z-Score check on the 'velocity' column, and compares the results against the 'is_anomaly' label to count correct detections."

### Day 9: Isolation Forest (The ML Engine)

#### Jisto: Training
*   **Task**: Train model on `train_normal.csv`. Save `model.pkl`.
*   **Prompt for Model**:
    > "Write a script using `sklearn.ensemble.IsolationForest`. Fit it on `train_normal.csv` (drop time column). Save the trained model using `joblib` or `pickle` to `models/iso_forest.pkl`."

#### Devika: Inference
*   **Task**: Load `model.pkl`. Predict on `test_anomalies.csv`.
*   **Prompt for Model**:
    > "Write a script that loads `models/iso_forest.pkl`. Load `test_anomalies.csv`. Run `model.predict()`. Convert output (-1/1) to (1/0) format and save results."

---

## Phase 4: UI (Days 11-14)

### Day 11: Streamlit Foundation

#### Jisto: App Skeleton
*   **Task**: `app.py`. Layout, Sidebar, Title.
*   **Prompt**: "Create a Streamlit app with a sidebar for navigation ('Home', 'Dashboard')."

#### Devika: Data Loader
*   **Task**: `utils/dataloader.py`.
*   **Prompt**: "Write a cached Streamlit function `@st.cache_data` that reads the CSV files and returns a Pandas DataFrame."

---

## Summary of Deliverables

| Day | Jisto (File/Function) | Devika (File/Function) |
| :--- | :--- | :--- |
| **1** | `generate_telemetry()` (Done) | `generate_telemetry()` (Done) |
| **2** | `src/physics.py`: `simulate_pressure` | `src/physics.py`: `simulate_vibration` |
| **3** | `src/anomalies.py`: `inject_spike` | `src/anomalies.py`: `inject_drift` |
| **4** | `scripts/make_train.py` | `scripts/make_test.py` |
| **5** | `scripts/plot_flight.py` | `scripts/plot_health.py` |
| **7** | `src/algorithms.py`: `detect_zscore` | `scripts/eval_zscore.py` |
| **9** | `scripts/train_iso.py` | `scripts/predict_iso.py` |
| **11**| `app.py` (Layout) | `utils/data.py` (Loader) |
