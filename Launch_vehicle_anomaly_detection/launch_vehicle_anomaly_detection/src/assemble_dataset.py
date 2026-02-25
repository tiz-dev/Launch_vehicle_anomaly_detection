"""
assemble_dataset.py
-------------------
Assembles the full training dataset by combining:
  - Day 1  : generate_telemetry()  → time, altitude, velocity, engine_temp
  - Day 2a : simulate_pressure()   → fuel_pressure   (Jisto)
  - Day 2b : simulate_vibration()  → vibration       (Devika)

Output: data/train_normal.csv  (600 s × 10 Hz = 6 000 rows, no anomalies)
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure imports work regardless of working directory
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from day1_generator import generate_telemetry
from day2_physics   import simulate_pressure, simulate_vibration

# ---------------------------------------------------------------------------
# 1. Base telemetry (Day 1) — returns a DataFrame with time/altitude/velocity/engine_temp
# ---------------------------------------------------------------------------
print("[1/3] Generating base telemetry (Day 1)...")
df = generate_telemetry()

# ---------------------------------------------------------------------------
# 2. Fuel pressure — Jisto Day 2 (takes the time vector)
# ---------------------------------------------------------------------------
print("[2/3] Simulating fuel tank pressure (Jisto – Day 2)...")
df["fuel_pressure"] = simulate_pressure(df["time"].to_numpy())

# ---------------------------------------------------------------------------
# 3. Airframe vibration — Devika Day 2 (takes the velocity vector)
# ---------------------------------------------------------------------------
print("[3/3] Simulating airframe vibration (Devika – Day 2)...")
df["vibration"] = simulate_vibration(df["velocity"].to_numpy())

# ---------------------------------------------------------------------------
# 4. Save to data/train_normal.csv
# ---------------------------------------------------------------------------
output_dir  = os.path.join(SRC_DIR, "..", "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "train_normal.csv")

df.to_csv(output_path, index=False)

# ---------------------------------------------------------------------------
# 5. Quick summary
# ---------------------------------------------------------------------------
print(f"\n✅  Saved → {os.path.normpath(output_path)}")
print(f"   Rows    : {len(df):,}")
print(f"   Columns : {list(df.columns)}")
print(f"\n   Quick stats:")
print(df.describe().round(2).to_string())
