"""
make_test.py  –  Day 4, Devika
-------------------------------
Generates a test dataset with labelled anomalies by:

    1. Producing clean telemetry exactly like assemble_dataset.py
       (Day 1 base + Day 2 physics columns).

    2. Importing inject_spike (Jisto – Day 3) and inject_drift
       (Devika – Day 3) and independently corrupting random segments
       of the sensor columns.

    3. Tracking which rows were changed and adding an `is_anomaly`
       label column  (1 = anomalous, 0 = normal).

    4. Saving the result to  data/test_anomalies.csv.

Interface contract
------------------
    Input  : none  (all parameters are configured below)
    Output : data/test_anomalies.csv
             Columns: time, altitude, velocity, engine_temp,
                      fuel_pressure, vibration, is_anomaly
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure sibling imports work regardless of working directory
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from day1_generator import generate_telemetry
from day2_physics   import simulate_pressure, simulate_vibration
from anomalies      import inject_spike, inject_drift

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
RANDOM_SEED = 42          # For reproducibility (comment out for true randomness)

# inject_spike settings (applied to fuel_pressure)
SPIKE_MAGNITUDE   = 300   # ±  units added at spike points
SPIKE_PROBABILITY = 0.02  # 2 % of samples will be spiked

# inject_drift settings (applied to engine_temp)
DRIFT_FACTOR = 0.08       # Bias per sample added after a random onset

# ---------------------------------------------------------------------------
# Helper: compare two arrays and return a boolean mask of changed rows
# ---------------------------------------------------------------------------
def _changed_mask(original: np.ndarray, corrupted: np.ndarray) -> np.ndarray:
    """Return a boolean array that is True wherever the values differ."""
    return ~np.isclose(original, corrupted, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# 1. Generate clean base telemetry  (Day 1)
# ---------------------------------------------------------------------------
print("[1/5] Generating base telemetry (Day 1)…")
np.random.seed(RANDOM_SEED)
df = generate_telemetry()

# ---------------------------------------------------------------------------
# 2. Add physics sensor columns  (Day 2)
# ---------------------------------------------------------------------------
print("[2/5] Adding fuel pressure – simulate_pressure() (Jisto – Day 2)…")
df["fuel_pressure"] = simulate_pressure(df["time"].to_numpy())

print("[3/5] Adding airframe vibration – simulate_vibration() (Devika – Day 2)…")
df["vibration"] = simulate_vibration(df["velocity"].to_numpy())

# Keep a clean copy for label comparison
df_clean = df.copy()

# ---------------------------------------------------------------------------
# 3. Inject spike anomalies into fuel_pressure  (Jisto – Day 3)
# ---------------------------------------------------------------------------
print("[4/5] Injecting spike anomalies into fuel_pressure (Jisto – Day 3)…")
fp_original  = df["fuel_pressure"].to_numpy().copy()
fp_corrupted = inject_spike(
    fp_original,
    magnitude=SPIKE_MAGNITUDE,
    probability=SPIKE_PROBABILITY,
)
df["fuel_pressure"] = fp_corrupted
spike_mask = _changed_mask(fp_original, fp_corrupted)

# ---------------------------------------------------------------------------
# 4. Inject drift anomaly into engine_temp  (Devika – Day 3)
# ---------------------------------------------------------------------------
print("[5/5] Injecting drift anomaly into engine_temp (Devika – Day 3)…")
et_original  = df["engine_temp"].to_numpy().copy()
et_corrupted = inject_drift(et_original, drift_factor=DRIFT_FACTOR)
df["engine_temp"] = et_corrupted
drift_mask = _changed_mask(et_original, et_corrupted)

# ---------------------------------------------------------------------------
# 5. Build the is_anomaly label column
#    A row is anomalous if ANY of its sensor columns was corrupted.
# ---------------------------------------------------------------------------
anomaly_mask = spike_mask | drift_mask
df["is_anomaly"] = anomaly_mask.astype(int)

# ---------------------------------------------------------------------------
# 6. Save to data/test_anomalies.csv
# ---------------------------------------------------------------------------
output_dir  = os.path.join(SRC_DIR, "..", "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_anomalies.csv")

df.to_csv(output_path, index=False)

# ---------------------------------------------------------------------------
# 7. Summary report
# ---------------------------------------------------------------------------
n_total    = len(df)
n_anomaly  = int(df["is_anomaly"].sum())
n_normal   = n_total - n_anomaly
n_spike    = int(spike_mask.sum())
n_drift    = int(drift_mask.sum())

print(f"\n{'='*55}")
print(f"  ✅  Saved → {os.path.normpath(output_path)}")
print(f"{'='*55}")
print(f"  Total rows      : {n_total:,}")
print(f"  Normal rows     : {n_normal:,}  ({100*n_normal/n_total:.1f} %)")
print(f"  Anomalous rows  : {n_anomaly:,}  ({100*n_anomaly/n_total:.1f} %)")
print(f"    ↳ spike rows  : {n_spike:,}  (fuel_pressure – inject_spike)")
print(f"    ↳ drift rows  : {n_drift:,}  (engine_temp   – inject_drift)")
print(f"  Columns         : {list(df.columns)}")
print(f"\n  Label distribution:")
print(df["is_anomaly"].value_counts().rename({0: "Normal (0)", 1: "Anomaly (1)"}).to_string())
print(f"\n  Quick stats (corrupted signal columns):")
print(df[["fuel_pressure", "engine_temp", "is_anomaly"]].describe().round(2).to_string())
