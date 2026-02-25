"""
visualize_health.py  –  Day 5, Devika
--------------------------------------
Plots sensor health metrics from the test dataset:
    • Panel 1 : Fuel Pressure over time  (with anomaly highlights)
    • Panel 2 : Engine Temperature over time  (with drift region highlighted)
    • Panel 3 : Vibration over time  (with anomaly highlights)
    • Panel 4 : Vibration vs Velocity scatter  (Max-Q correlation)

Input  : data/test_anomalies.csv
Output : plots/health_metrics.png  +  interactive Matplotlib window
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SRC_DIR, "..")
DATA_PATH  = os.path.join(BASE_DIR, "data",  "test_anomalies.csv")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
OUTPUT_PNG = os.path.join(PLOTS_DIR, "health_metrics.png")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("[1/3] Loading data from:", os.path.normpath(DATA_PATH))
df = pd.read_csv(DATA_PATH)

required = {"time", "fuel_pressure", "engine_temp", "vibration", "velocity", "is_anomaly"}
missing  = required - set(df.columns)
if missing:
    print(f"ERROR: Missing columns in CSV: {missing}")
    sys.exit(1)

is_anomaly = df["is_anomaly"].astype(bool)
is_normal  = ~is_anomaly
n_total    = len(df)
n_anomaly  = is_anomaly.sum()

print(f"       Rows: {n_total:,}  |  Anomalous: {n_anomaly:,} "
      f"({100*n_anomaly/n_total:.1f}%)")

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
print("[2/3] Building plots…")

DARK_BG   = "#0f1117"
CLR_NORM  = "#4fc3f7"   # light blue  – normal signal
CLR_ANOM  = "#ff5252"   # red         – anomalous points
CLR_DRIFT = "#ce93d8"   # purple      – drift overlay
CLR_VEL   = "#81c784"   # green       – velocity reference
CLR_TEMP  = "#ffb74d"   # amber       – engine temp
CLR_VIBR  = "#f06292"   # pink        – vibration

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4a",
    "axes.labelcolor":   "#d0d0d0",
    "xtick.color":       "#a0a0a0",
    "ytick.color":       "#a0a0a0",
    "text.color":        "#e0e0e0",
    "grid.color":        "#2e3140",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.framealpha": 0.3,
    "legend.facecolor":  "#1a1d27",
    "legend.edgecolor":  "#3a3d4a",
    "font.family":       "DejaVu Sans",
})

fig = plt.figure(figsize=(16, 14), facecolor=DARK_BG)
fig.suptitle(
    "Launch Vehicle — Sensor Health Metrics Dashboard  (Day 5 · Devika)",
    fontsize=15, fontweight="bold", color="#ffffff", y=0.98
)

gs = gridspec.GridSpec(
    4, 2,
    figure=fig,
    hspace=0.55,
    wspace=0.35,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.06,
)

ax1 = fig.add_subplot(gs[0, :])   # Fuel Pressure – full width
ax2 = fig.add_subplot(gs[1, :])   # Engine Temp   – full width
ax3 = fig.add_subplot(gs[2, :])   # Vibration     – full width
ax4 = fig.add_subplot(gs[3, 0])   # Vibration vs Velocity scatter
ax5 = fig.add_subplot(gs[3, 1])   # Anomaly count bar chart

time = df["time"]

# ── Helper ──────────────────────────────────────────────────────────────────
def _plot_signal_with_anomalies(ax, x, y_all, mask_anom, color_normal, color_anom,
                                 ylabel, title, unit=""):
    ax.plot(x[is_normal], y_all[is_normal],
            color=color_normal, linewidth=0.7, alpha=0.85, label="Normal")
    ax.scatter(x[mask_anom], y_all[mask_anom],
               color=color_anom, s=6, zorder=5, label=f"Anomaly ({mask_anom.sum():,} pts)")
    ax.set_ylabel(f"{ylabel}\n{unit}", fontsize=9)
    ax.set_title(title, fontsize=10, pad=4, color="#e0e0e0")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True)
    ax.tick_params(labelsize=8)


# ── Panel 1: Fuel Pressure ───────────────────────────────────────────────────
_plot_signal_with_anomalies(
    ax1,
    time, df["fuel_pressure"], is_anomaly,
    CLR_NORM, CLR_ANOM,
    "Fuel Pressure", "① Fuel Pressure  —  inject_spike anomalies highlighted",
    unit="(units)"
)
ax1.set_xlabel("Time (s)", fontsize=8)

# ── Panel 2: Engine Temperature with drift region ────────────────────────────
ax2.plot(time[is_normal], df["engine_temp"][is_normal],
         color=CLR_TEMP, linewidth=0.7, alpha=0.85, label="Normal")
ax2.scatter(time[is_anomaly], df["engine_temp"][is_anomaly],
            color=CLR_DRIFT, s=6, zorder=5,
            label=f"Drift region ({is_anomaly.sum():,} pts)")

# Shade the drift region
drift_start_idx = df.index[is_anomaly].min() if is_anomaly.any() else None
if drift_start_idx is not None:
    drift_t = df["time"].iloc[drift_start_idx]
    ax2.axvspan(drift_t, time.iloc[-1],
                color=CLR_DRIFT, alpha=0.08, label=f"Drift onset ≈ {drift_t:.0f} s")
    ax2.axvline(drift_t, color=CLR_DRIFT, linewidth=1.2,
                linestyle="--", alpha=0.7)

ax2.set_ylabel("Engine Temp\n(K)", fontsize=9)
ax2.set_xlabel("Time (s)", fontsize=8)
ax2.set_title("② Engine Temperature  —  inject_drift region shaded",
              fontsize=10, pad=4, color="#e0e0e0")
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True)
ax2.tick_params(labelsize=8)

# ── Panel 3: Vibration ───────────────────────────────────────────────────────
_plot_signal_with_anomalies(
    ax3,
    time, df["vibration"], is_anomaly,
    CLR_VIBR, CLR_ANOM,
    "Vibration", "③ Airframe Vibration  —  Max-Q peak visible near centre of flight",
    unit="(g)"
)
# Mark Max-Q velocity reference on vibration plot
maxq_idx = df["vibration"].idxmax()
ax3.annotate(
    f"Max-Q peak\n(v ≈ {df['velocity'].iloc[maxq_idx]:.0f} m/s)",
    xy=(time.iloc[maxq_idx], df["vibration"].iloc[maxq_idx]),
    xytext=(time.iloc[maxq_idx] + 30, df["vibration"].iloc[maxq_idx] * 0.8),
    arrowprops=dict(arrowstyle="->", color="#ffeb3b", lw=1.2),
    fontsize=8, color="#ffeb3b",
)
ax3.set_xlabel("Time (s)", fontsize=8)

# ── Panel 4: Vibration vs Velocity scatter ───────────────────────────────────
sc = ax4.scatter(
    df.loc[is_normal,  "velocity"], df.loc[is_normal,  "vibration"],
    c=CLR_VIBR, s=2, alpha=0.4, label="Normal"
)
ax4.scatter(
    df.loc[is_anomaly, "velocity"], df.loc[is_anomaly, "vibration"],
    c=CLR_ANOM, s=5, alpha=0.7, zorder=5, label="Anomaly"
)
# Max-Q reference line
ax4.axvline(300, color="#ffeb3b", linewidth=1.2, linestyle="--",
            alpha=0.8, label="Max-Q (v=300 m/s)")
ax4.set_xlabel("Velocity (m/s)", fontsize=9)
ax4.set_ylabel("Vibration (g)",  fontsize=9)
ax4.set_title("④ Vibration vs Velocity\n(Max-Q Correlation)", fontsize=10, pad=4)
ax4.legend(fontsize=8, markerscale=2)
ax4.grid(True)
ax4.tick_params(labelsize=8)

# ── Panel 5: Anomaly composition bar ────────────────────────────────────────
categories  = ["Normal\nrows", "Anomalous\nrows"]
counts      = [int(is_normal.sum()), int(is_anomaly.sum())]
colors_bar  = [CLR_NORM, CLR_ANOM]
bars = ax5.bar(categories, counts, color=colors_bar, edgecolor="#3a3d4a", width=0.5)

for bar, count in zip(bars, counts):
    ax5.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 20,
             f"{count:,}\n({100*count/n_total:.1f}%)",
             ha="center", va="bottom", fontsize=9, color="#e0e0e0")

ax5.set_ylabel("Row count", fontsize=9)
ax5.set_title("⑤ Dataset Label Distribution", fontsize=10, pad=4)
ax5.set_ylim(0, max(counts) * 1.18)
ax5.grid(True, axis="y")
ax5.tick_params(labelsize=9)

# ---------------------------------------------------------------------------
# Save + show
# ---------------------------------------------------------------------------
os.makedirs(PLOTS_DIR, exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print(f"[3/3] Saved → {os.path.normpath(OUTPUT_PNG)}")
plt.tight_layout()
plt.show()
