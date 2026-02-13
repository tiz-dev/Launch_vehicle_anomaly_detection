# Launch Vehicle Telemetry Anomaly Detection - Project Plan (14 Days)

**Goal:** Build a telemetry anomaly detector using Python and Machine Learning.
**Level:** Beginner (No complex physics, no real-time streaming).
**Tools:** Python, Pandas, Scikit-Learn, Streamlit.

## Phase 1: Data Simulation & Understanding (Days 1-3)
**Objective:** Generate synthetic data using simple math.

- **Day 1: Basic Data Generation**
  - Create `day1_generator.py`.
  - Simulate **Time** (0 to 600s).
  - Simulate **Altitude** using a quadratic function ($y = ax^2$).
  - Simulate **Velocity** using a linear function ($v = ax + b$).
  - Save data to `normal_telemetry.csv`.

- **Day 2: Visualizing the Flight Path**
  - Create `day2_visualizer.py`.
  - Load `normal_telemetry.csv`.
  - Plot Altitude vs. Time and Velocity vs. Time using Matplotlib.
  - *Goal:* Verify the data looks like a rocket launch (smooth curves).

- **Day 3: Introducing Anomalies**
  - Create `day3_anomaly_injector.py`.
  - Load `normal_telemetry.csv`.
  - Inject **Point Anomalies** (Spikes in sensor data).
  - Inject **Drift Anomalies** (Gradual shift in values).
  - Save as `anomaly_telemetry.csv`.

## Phase 2: Anomaly Detection - Rule-Based (Days 4-6)
**Objective:** Detect anomalies using simple statistics.

- **Day 4: Statistical Analysis**
  - Calculate Mean and Standard Deviation for Altitude and Velocity.
  - Understand "Normal Distribution".

- **Day 5: Z-Score Implementation**
  - Create `day5_zscore.py`.
  - Calculate Z-Score for every data point.
  - Flag data points with Z-Score > 3 as anomalies.

- **Day 6: Visualization of Anomalies**
  - Plot the data again, highlighting the anomalies in Ref/Red.

## Phase 3: Machine Learning - Isolation Forest (Days 7-10)
**Objective:** Use Scikit-Learn to detect anomalies automatically.

- **Day 7: Intro to Isolation Forest**
  - Understand the concept: "Easier to isolate outliers".
  - Install `scikit-learn`.

- **Day 8: Training the Model**
  - Create `day8_isolation_forest.py`.
  - Train `IsolationForest` on `normal_telemetry.csv`.
  - Predict anomalies on `anomaly_telemetry.csv`.

- **Day 9: Evaluating Performance**
  - Compare Model predictions vs. True anomalies (injected in Day 3).
  - Calculate Accuracy, Precision, and Recall (Simple version).

- **Day 10: Tuning the Model**
  - Experiment with `contamination` parameter.
  - See how it affects detection.

## Phase 4: Dashboard & Final Report (Days 11-14)
**Objective:** Build a simple GUI to show results.

- **Day 11: Setup Streamlit**
  - Install `streamlit`.
  - Create `app.py`.
  - Display the raw dataframe.

- **Day 12: Static Dashboard - Charts**
  - Add Line Charts for Altitude, Velocity.
  - Add a "Load Data" button.

- **Day 13: Integrating the Model**
  - Add a "Run Anomaly Detection" button.
  - Highlight anomalies on the charts in the Dashboard.

- **Day 14: Final Polish & Demo**
  - Add Title, Description, and Student Name.
  - Record a short demo video / Take screenshots.
  - **Submit Project!**

---
**Constraints Checklist:**
- [x] No advanced physics (Fluid dynamics, Aerodynamics).
- [x] No complex correlations.
- [x] No real-time simulation (Static CSV files only).
