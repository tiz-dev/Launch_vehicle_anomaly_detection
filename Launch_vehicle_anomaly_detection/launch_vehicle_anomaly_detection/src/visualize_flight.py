"""
visualize_flight.py  –  Day 5, Jisto
--------------------------------------
Plots flight metrics from train_normal.csv:
    • Altitude  vs Time
    • Velocity  vs Time

Independence Strategy: works on its own plot file, reads from data/train_normal.csv.
Output: Matplotlib figure (shown on screen + optionally saved as plots/flight_metrics.png)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Path setup ────────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "..", "data")
PLOT_DIR = os.path.join(SRC_DIR, "..", "plots")
CSV_PATH = os.path.join(DATA_DIR, "train_normal.csv")

# ── Load data ─────────────────────────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    print(f"[ERROR] Could not find {CSV_PATH}")
    print("        Run assemble_dataset.py first to generate train_normal.csv.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"[visualize_flight] Loaded {len(df):,} rows from train_normal.csv")

# ── Plot setup ────────────────────────────────────────────────────────────────
plt.style.use("dark_background")

fig = plt.figure(figsize=(13, 8))
fig.suptitle(
    "🚀  Launch Vehicle — Flight Metrics (Day 5, Jisto)",
    fontsize=15, fontweight="bold", color="white", y=0.98
)

gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.42)

# ── Panel 1: Altitude vs Time ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])

ax1.plot(
    df["time"], df["altitude"],
    color="#17A2B8", linewidth=1.2, alpha=0.95, label="Altitude (m)"
)
ax1.fill_between(df["time"], df["altitude"], alpha=0.12, color="#17A2B8")

# Annotate peak
peak_idx = df["altitude"].idxmax()
ax1.annotate(
    f"  Peak: {df['altitude'].iloc[peak_idx]:,.0f} m",
    xy=(df["time"].iloc[peak_idx], df["altitude"].iloc[peak_idx]),
    xytext=(df["time"].iloc[peak_idx] - 80, df["altitude"].iloc[peak_idx] * 0.82),
    arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.4),
    color="#FFD700", fontsize=9, fontweight="bold"
)

ax1.set_xlabel("Time (s)", fontsize=9, color="#AAAAAA")
ax1.set_ylabel("Altitude (m)", fontsize=10, color="#17A2B8")
ax1.set_title("Altitude vs Time", fontsize=11, color="white", pad=6)
ax1.tick_params(colors="#AAAAAA")
ax1.spines["bottom"].set_color("#444")
ax1.spines["left"].set_color("#444")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(loc="upper left", fontsize=9, framealpha=0.3)
ax1.grid(alpha=0.15)

# Stats box – Altitude
alt_stats = (
    f"Start : {df['altitude'].iloc[0]:.0f} m\n"
    f"End   : {df['altitude'].iloc[-1]:,.0f} m\n"
    f"Rate  : +{df['altitude'].iloc[-1]/600:.0f} m/s avg climb"
)
ax1.text(
    0.02, 0.97, alt_stats,
    transform=ax1.transAxes, fontsize=7.5,
    verticalalignment="top", color="#CCCCCC",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", alpha=0.7, edgecolor="#17A2B8")
)

# ── Panel 2: Velocity vs Time ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])

ax2.plot(
    df["time"], df["velocity"],
    color="#28B463", linewidth=1.0, alpha=0.9, label="Velocity (m/s)"
)
ax2.fill_between(df["time"], df["velocity"], alpha=0.10, color="#28B463")

# Mark Max-Q velocity (~300 m/s)
maxq_vel   = 300.0
maxq_rows  = df[df["velocity"].between(maxq_vel - 5, maxq_vel + 5)]
if not maxq_rows.empty:
    maxq_t = maxq_rows["time"].iloc[0]
    ax2.axvline(maxq_t, color="#E74C3C", linestyle="--", linewidth=1.2,
                alpha=0.8, label=f"Max-Q ≈ t={maxq_t:.0f}s")
    ax2.annotate(
        "  Max-Q\n  ~300 m/s",
        xy=(maxq_t, maxq_vel),
        color="#E74C3C", fontsize=8.5, fontweight="bold",
        xytext=(maxq_t + 15, maxq_vel - 30)
    )

ax2.set_xlabel("Time (s)", fontsize=9, color="#AAAAAA")
ax2.set_ylabel("Velocity (m/s)", fontsize=10, color="#28B463")
ax2.set_title("Velocity vs Time", fontsize=11, color="white", pad=6)
ax2.tick_params(colors="#AAAAAA")
ax2.spines["bottom"].set_color("#444")
ax2.spines["left"].set_color("#444")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(loc="upper left", fontsize=9, framealpha=0.3)
ax2.grid(alpha=0.15)

# Stats box – Velocity
vel_stats = (
    f"Start : {df['velocity'].iloc[0]:.1f} m/s\n"
    f"End   : {df['velocity'].iloc[-1]:.1f} m/s\n"
    f"Max   : {df['velocity'].max():.1f} m/s"
)
ax2.text(
    0.02, 0.97, vel_stats,
    transform=ax2.transAxes, fontsize=7.5,
    verticalalignment="top", color="#CCCCCC",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", alpha=0.7, edgecolor="#28B463")
)

# ── Save + Show ───────────────────────────────────────────────────────────────
os.makedirs(PLOT_DIR, exist_ok=True)
save_path = os.path.join(PLOT_DIR, "flight_metrics.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
print(f"[visualize_flight] Plot saved → {os.path.normpath(save_path)}")

plt.show()
