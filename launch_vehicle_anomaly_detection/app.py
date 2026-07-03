"""
app.py  –  Launch Vehicle Telemetry Anomaly Detector
=====================================================
Full-featured Streamlit dashboard with:
  • Live metric cards (total samples, anomaly counts, F1, accuracy)
  • Interactive multi-channel telemetry charts with anomaly overlays
  • Isolation Forest vs Z-score comparison
  • Detected anomaly table with filtering
  • Channel distribution plots
  • Premium dark-space UI with custom CSS
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Add src/ to path for local imports ────────────────────────────────────
BASE_DIR_INIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR_INIT, "src"))

# ── Page config (must be first) ────────────────────────────────────────────
st.set_page_config(
    page_title="LV Anomaly Detector",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Main background */
  .stApp { background: #0a0e1a; }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0d1b2a 0%, #0a0e1a 100%);
      border-right: 1px solid #1e3a5f;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
      background: linear-gradient(135deg, #0d1b2a 0%, #112240 100%);
      border: 1px solid #1e3a5f;
      border-radius: 12px;
      padding: 16px 20px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.4);
      transition: transform 0.2s, box-shadow 0.2s;
  }
  [data-testid="metric-container"]:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(0, 120, 255, 0.15);
  }
  [data-testid="stMetricLabel"] { color: #8892b0 !important; font-size: 0.78rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
  [data-testid="stMetricValue"] { color: #e6f1ff !important; font-size: 2rem !important; font-weight: 700 !important; }
  [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

  /* Section headers */
  .section-header {
      color: #64ffda;
      font-size: 1.1rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      padding: 0 0 8px 0;
      border-bottom: 1px solid #1e3a5f;
      margin-bottom: 16px;
  }

  /* Hero banner */
  .hero-banner {
      background: linear-gradient(135deg, #0d1b2a 0%, #112240 50%, #0a192f 100%);
      border: 1px solid #1e3a5f;
      border-radius: 16px;
      padding: 28px 36px;
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
  }
  .hero-banner::before {
      content: '';
      position: absolute;
      top: -50%;
      right: -10%;
      width: 300px;
      height: 300px;
      background: radial-gradient(circle, rgba(100,255,218,0.06) 0%, transparent 70%);
      border-radius: 50%;
  }
  .hero-title {
      font-size: 2rem;
      font-weight: 700;
      color: #e6f1ff;
      margin: 0 0 6px 0;
  }
  .hero-sub {
      color: #8892b0;
      font-size: 1rem;
      margin: 0;
  }
  .hero-badge {
      display: inline-block;
      background: rgba(100,255,218,0.1);
      border: 1px solid #64ffda;
      color: #64ffda;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 20px;
      margin-right: 6px;
      letter-spacing: 0.06em;
  }

  /* Table */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* Divider */
  hr { border-color: #1e3a5f !important; margin: 24px 0 !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

  /* Info / alert boxes */
  .stAlert { border-radius: 10px; }

  /* Sidebar radio */
  [data-testid="stSidebar"] .stRadio > label { color: #8892b0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

ISO_RESULTS_CSV  = os.path.join(DATA_DIR, "iso_forest_results.csv")
TEST_CSV         = os.path.join(DATA_DIR, "test_anomalies.csv")
TRAIN_CSV        = os.path.join(DATA_DIR, "train_normal.csv")

FEATURE_COLS = ["altitude", "velocity", "engine_temp", "fuel_pressure", "vibration"]
CHANNEL_LABELS = {
    "altitude":     "Altitude (m)",
    "velocity":     "Velocity (m/s)",
    "engine_temp":  "Engine Temp (°C)",
    "fuel_pressure":"Fuel Pressure (kPa)",
    "vibration":    "Vibration (g)",
}

PLOTLY_THEME = dict(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0d1b2a",
    font_color="#8892b0",
    font_family="Inter",
)

# ── Auto-generate pipeline (runs on Streamlit Cloud if data missing) ──────
@st.cache_resource
def auto_generate_data():
    """Run the full pipeline to create data and model if they don't exist."""
    iso_results = os.path.join(BASE_DIR_INIT, "data", "iso_forest_results.csv")
    if os.path.exists(iso_results):
        return True  # Already have results, skip generation

    try:
        import joblib
        from sklearn.ensemble import IsolationForest
        from day1_generator import generate_telemetry
        from day2_physics   import simulate_pressure, simulate_vibration
        from anomalies      import inject_spike, inject_drift

        data_dir  = os.path.join(BASE_DIR_INIT, "data")
        model_dir = os.path.join(BASE_DIR_INIT, "models")
        os.makedirs(data_dir,  exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        np.random.seed(42)

        # 1. Generate base telemetry
        df_base = generate_telemetry()
        df_base["fuel_pressure"] = simulate_pressure(df_base["time"].to_numpy())
        df_base["vibration"]     = simulate_vibration(df_base["velocity"].to_numpy())
        df_base.to_csv(os.path.join(data_dir, "train_normal.csv"), index=False)

        # 2. Build test set with anomalies
        df_test = df_base.copy()
        fp_orig = df_test["fuel_pressure"].to_numpy().copy()
        fp_corr = inject_spike(fp_orig, magnitude=300, probability=0.02)
        df_test["fuel_pressure"] = fp_corr
        spike_mask = ~np.isclose(fp_orig, fp_corr, rtol=0, atol=1e-10)

        et_orig = df_test["engine_temp"].to_numpy().copy()
        et_corr = inject_drift(et_orig, drift_factor=0.08)
        df_test["engine_temp"] = et_corr
        drift_mask = ~np.isclose(et_orig, et_corr, rtol=0, atol=1e-10)

        df_test["is_anomaly"] = (spike_mask | drift_mask).astype(int)
        df_test.to_csv(os.path.join(data_dir, "test_anomalies.csv"), index=False)

        # 3. Train Isolation Forest on normal data
        FEATURE_COLS_GEN = ["altitude", "velocity", "engine_temp", "fuel_pressure", "vibration"]
        X_train = df_base[FEATURE_COLS_GEN].values
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(X_train)
        joblib.dump(model, os.path.join(model_dir, "iso_forest.pkl"))

        # 4. Predict on test set
        X_test       = df_test[FEATURE_COLS_GEN].values
        raw_pred     = model.predict(X_test)
        df_test["iso_prediction"] = np.where(raw_pred == -1, 1, 0)
        df_test["iso_raw_score"]  = raw_pred
        df_test.to_csv(iso_results, index=False)

        return True
    except Exception as e:
        st.error(f"⚠️ Auto-generation failed: {e}")
        return False

auto_generate_data()

# ── Data loader ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(ISO_RESULTS_CSV):
        return None, None
    df = pd.read_csv(ISO_RESULTS_CSV)
    # Ensure columns exist
    if "iso_prediction" not in df.columns:
        df["iso_prediction"] = 0
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = 0

    # Z-score detection (per-channel, threshold=3)
    z_flags = pd.Series(False, index=df.index)
    z_scores_max = pd.Series(0.0, index=df.index)
    for col in FEATURE_COLS:
        if col in df.columns:
            z = (df[col] - df[col].mean()) / df[col].std()
            z_flags = z_flags | (z.abs() > 3)
            z_scores_max = z_scores_max.combine(z.abs(), max)
    df["z_flag"] = z_flags.astype(int)
    df["z_score_max"] = z_scores_max

    # Isolation Forest anomaly score (re-derive if missing)
    if "iso_anomaly_score" not in df.columns:
        try:
            import joblib
            model_path = os.path.join(MODEL_DIR, "iso_forest.pkl")
            if os.path.exists(model_path):
                _iso_model = joblib.load(model_path)
                X = df[FEATURE_COLS].values
                df["iso_anomaly_score"] = -_iso_model.score_samples(X)  # higher = more anomalous
            else:
                df["iso_anomaly_score"] = np.nan
        except Exception:
            df["iso_anomaly_score"] = np.nan

    metrics = {}
    gt = df["is_anomaly"].values
    iso_pred = df["iso_prediction"].values
    z_pred   = df["z_flag"].values

    def calc(pred, gt):
        tp = int(np.sum((pred == 1) & (gt == 1)))
        tn = int(np.sum((pred == 0) & (gt == 0)))
        fp = int(np.sum((pred == 1) & (gt == 0)))
        fn = int(np.sum((pred == 0) & (gt == 1)))
        acc  = (tp + tn) / max(len(gt), 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        return dict(TP=tp, TN=tn, FP=fp, FN=fn,
                    Accuracy=acc, Precision=prec, Recall=rec, F1=f1)

    metrics["iso"] = calc(iso_pred, gt)
    metrics["z"]   = calc(z_pred, gt)
    return df, metrics


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 20px 0;">
      <div style="font-size:2.5rem;">🚀</div>
      <div style="color:#64ffda; font-weight:700; font-size:1.05rem; letter-spacing:0.04em;">LV ANOMALY DETECTOR</div>
      <div style="color:#8892b0; font-size:0.75rem; margin-top:4px;">Telemetry Intelligence System</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        options=["🏠  Home", "📊  Dashboard", "🔍  Anomaly Explorer", "📈  Model Comparison"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("""
    <div style="padding: 12px; background: rgba(100,255,218,0.05); border: 1px solid rgba(100,255,218,0.15); border-radius: 10px;">
      <div style="color:#64ffda; font-size:0.72rem; font-weight:600; letter-spacing:0.08em; margin-bottom:8px;">TELEMETRY CHANNELS</div>
      <div style="color:#8892b0; font-size:0.78rem; line-height:1.8;">
        📡 Altitude (m)<br>
        💨 Velocity (m/s)<br>
        🔥 Engine Temp (°C)<br>
        💧 Fuel Pressure (kPa)<br>
        📳 Vibration (g)
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Version 1.0  ·  Launch Vehicle Anomaly Detector")


# ── Load data ──────────────────────────────────────────────────────────────
df, metrics = load_data()
data_ok = df is not None


# ══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════
def show_home():
    st.markdown("""
    <div class="hero-banner">
      <p class="hero-title">🚀 Launch Vehicle Telemetry<br>Anomaly Detector</p>
      <p class="hero-sub">Real-time anomaly detection on synthetic rocket flight data using statistical &amp; ML methods.</p>
      <div style="margin-top:16px;">
        <span class="hero-badge">Z-SCORE</span>
        <span class="hero-badge">ISOLATION FOREST</span>
        <span class="hero-badge">6,000 SAMPLES</span>
        <span class="hero-badge">5 CHANNELS</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-header">About This System</div>', unsafe_allow_html=True)
        st.markdown("""
        This prototype ingests synthetic launch vehicle telemetry, injects realistic fault patterns,
        and detects anomalies using two complementary algorithms.

        **Detection Methods:**
        | Method | Strategy |
        |---|---|
        | **Z-Score** | Statistical — flags samples beyond ±3σ from mean |
        | **Isolation Forest** | ML-based — anomalies isolated faster in random trees |

        **Anomaly Types Simulated:**
        - 🔺 **Spikes** — sudden, short-lived sensor excursions
        - 📉 **Drift** — slow, cumulative calibration loss
        """)

    with col2:
        st.markdown('<div class="section-header">System Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        ```
        ┌─────────────────────────────┐
        │   1. Telemetry Simulation   │
        │      day1_generator.py      │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │   2. Physics Augmentation   │
        │      day2_physics.py        │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │   3. Anomaly Injection      │
        │      anomalies.py           │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │   4. Model Training         │
        │      train_iso_forest.py    │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────────┐
        │   5. Detection & Evaluation │
        │      eval_zscore.py         │
        │      predict_iso.py         │
        └─────────────────────────────┘
        ```
        """)

    st.info("👈  Use the sidebar to navigate to the **Dashboard** or **Anomaly Explorer**.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
def show_dashboard():
    st.markdown("""
    <div style="margin-bottom:20px;">
      <span style="font-size:1.6rem;font-weight:700;color:#e6f1ff;">📊 Telemetry Dashboard</span><br>
      <span style="color:#8892b0;font-size:0.9rem;">Live anomaly detection results across all telemetry channels</span>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.error("⚠️  Results file not found. Please run `python src/predict_iso.py` first.", icon="🚨")
        return

    # ── Metric cards ──────────────────────────────────────────────────────
    total = len(df)
    actual_anom   = int(df["is_anomaly"].sum())
    iso_detected  = int(df["iso_prediction"].sum())
    z_detected    = int(df["z_flag"].sum())
    iso_f1        = metrics["iso"]["F1"]
    iso_acc       = metrics["iso"]["Accuracy"]
    iso_prec      = metrics["iso"]["Precision"]
    iso_rec       = metrics["iso"]["Recall"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Samples",       f"{total:,}")
    c2.metric("Actual Anomalies",    f"{actual_anom:,}",   f"{actual_anom/total*100:.1f}% of data")
    c3.metric("ISO Forest Detected", f"{iso_detected:,}",  f"Recall {iso_rec*100:.1f}%")
    c4.metric("Z-Score Detected",    f"{z_detected:,}",    f"Precision {metrics['z']['Precision']*100:.1f}%")
    c5.metric("ISO Forest F1",       f"{iso_f1:.3f}",      f"Acc {iso_acc*100:.1f}%")

    st.markdown("---")

    # ── Channel selector ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Telemetry Signal Viewer</div>', unsafe_allow_html=True)

    selected_channels = st.multiselect(
        "Select channels to visualize:",
        options=FEATURE_COLS,
        default=["altitude", "velocity", "engine_temp"],
        format_func=lambda x: CHANNEL_LABELS[x],
    )

    overlay = st.radio(
        "Anomaly overlay:",
        ["Isolation Forest", "Z-Score", "Ground Truth", "None"],
        horizontal=True,
        index=0,
    )

    sample_size = st.slider("Samples to display:", 500, len(df), min(2000, len(df)), step=100)
    df_plot = df.iloc[:sample_size]

    if selected_channels:
        n = len(selected_channels)
        fig = make_subplots(
            rows=n, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=[CHANNEL_LABELS[c] for c in selected_channels],
        )

        colors_normal = ["#64ffda", "#7eb3ff", "#ff9f7e", "#b58cf7", "#ffd166"]
        colors_anom   = "#ff4b4b"

        overlay_col = {
            "Isolation Forest": "iso_prediction",
            "Z-Score":          "z_flag",
            "Ground Truth":     "is_anomaly",
            "None":             None,
        }[overlay]

        for i, ch in enumerate(selected_channels, start=1):
            color = colors_normal[i % len(colors_normal)]
            t = df_plot["time"].values if "time" in df_plot.columns else np.arange(sample_size)

            # Normal trace
            fig.add_trace(go.Scatter(
                x=t, y=df_plot[ch],
                mode="lines",
                name=CHANNEL_LABELS[ch],
                line=dict(color=color, width=1.2),
                opacity=0.9,
                showlegend=(i == 1),
            ), row=i, col=1)

            # Anomaly scatter overlay
            if overlay_col:
                mask = df_plot[overlay_col] == 1
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=t[mask.values],
                        y=df_plot[ch].values[mask.values],
                        mode="markers",
                        name="Anomaly" if i == 1 else None,
                        marker=dict(color=colors_anom, size=5, symbol="x"),
                        showlegend=(i == 1),
                    ), row=i, col=1)

            fig.update_yaxes(
                row=i, col=1,
                gridcolor="#1e3a5f",
                title_font=dict(color="#8892b0", size=10),
                tickfont=dict(color="#8892b0", size=9),
                title_text=CHANNEL_LABELS[ch].split("(")[0].strip(),
            )

        fig.update_xaxes(
            gridcolor="#1e3a5f",
            tickfont=dict(color="#8892b0", size=9),
            title_text="Time (s)",
            title_font=dict(color="#8892b0"),
            row=n, col=1,
        )
        fig.update_layout(
            height=220 * n,
            **PLOTLY_THEME,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                        font=dict(color="#8892b0"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one channel above.", icon="📡")

    st.markdown("---")

    # ── Anomaly distribution bar chart ────────────────────────────────────
    st.markdown('<div class="section-header">Anomaly Distribution Per Channel</div>', unsafe_allow_html=True)

    anom_df = df[df["is_anomaly"] == 1]
    channel_counts = {}
    for ch in FEATURE_COLS:
        if ch in df.columns:
            mu, sig = df[ch].mean(), df[ch].std()
            z = (anom_df[ch] - mu).abs() / sig
            channel_counts[CHANNEL_LABELS[ch]] = int((z > 2.5).sum())

    bar_fig = go.Figure(go.Bar(
        x=list(channel_counts.keys()),
        y=list(channel_counts.values()),
        marker=dict(
            color=list(channel_counts.values()),
            colorscale="Teal",
            showscale=False,
        ),
        text=list(channel_counts.values()),
        textposition="outside",
        textfont=dict(color="#64ffda"),
    ))
    bar_fig.update_layout(
        height=320,
        **PLOTLY_THEME,
        xaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#8892b0")),
        yaxis=dict(gridcolor="#1e3a5f", tickfont=dict(color="#8892b0"), title="Anomalous Samples"),
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(bar_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY EXPLORER
# ══════════════════════════════════════════════════════════════════════════
def show_anomaly_explorer():
    st.markdown("""
    <div style="margin-bottom:20px;">
      <span style="font-size:1.6rem;font-weight:700;color:#e6f1ff;">🔍 Anomaly Explorer</span><br>
      <span style="color:#8892b0;font-size:0.9rem;">Browse and filter detected anomaly events</span>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.error("⚠️  Run `python src/predict_iso.py` first.", icon="🚨")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        method_filter = st.selectbox("Detection Method", ["All", "Isolation Forest", "Z-Score", "Both"])
    with col2:
        gt_filter = st.selectbox("Ground Truth", ["All", "True Anomalies Only", "False Positives Only"])
    with col3:
        max_rows = st.slider("Max rows shown", 50, 500, 100)

    # Build filtered view
    adf = df.copy()
    if method_filter == "Isolation Forest":
        adf = adf[adf["iso_prediction"] == 1]
    elif method_filter == "Z-Score":
        adf = adf[adf["z_flag"] == 1]
    elif method_filter == "Both":
        adf = adf[(adf["iso_prediction"] == 1) & (adf["z_flag"] == 1)]

    if gt_filter == "True Anomalies Only":
        adf = adf[adf["is_anomaly"] == 1]
    elif gt_filter == "False Positives Only":
        adf = adf[adf["is_anomaly"] == 0]

    # Label columns nicely
    display_cols = {
        "time":           "Time (s)",
        "altitude":       "Altitude (m)",
        "velocity":       "Velocity (m/s)",
        "engine_temp":    "Engine Temp",
        "fuel_pressure":  "Fuel Pressure",
        "vibration":      "Vibration (g)",
        "is_anomaly":     "Ground Truth",
        "iso_prediction": "ISO Detected",
        "z_flag":         "Z-Score Flag",
    }
    available = [c for c in display_cols if c in adf.columns]
    disp = adf[available].rename(columns=display_cols).head(max_rows)

    # Summary row
    total_shown = len(adf)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Matching Events", f"{total_shown:,}")
    col_b.metric("True Positives",  f"{int(adf['is_anomaly'].sum()):,}" if 'is_anomaly' in adf.columns else "—")
    col_c.metric("False Positives", f"{int((adf['is_anomaly'] == 0).sum()):,}" if 'is_anomaly' in adf.columns else "—")

    st.dataframe(
        disp.style
            .highlight_between(subset=["Ground Truth"] if "Ground Truth" in disp.columns else [], left=1, right=1, color="#3d1a1a")
            .highlight_between(subset=["ISO Detected"] if "ISO Detected" in disp.columns else [], left=1, right=1, color="#1a3d2b")
            .format(precision=3),
        use_container_width=True,
        height=420,
    )

    st.markdown("---")

    # ── Scatter: z_score_max vs channel value coloured by prediction ──────
    st.markdown('<div class="section-header">Z-Score Distribution</div>', unsafe_allow_html=True)
    ch_scatter = st.selectbox("Channel for scatter:", FEATURE_COLS, format_func=lambda x: CHANNEL_LABELS[x])

    if "z_score_max" in df.columns:
        sdf = df.sample(min(2000, len(df)), random_state=42)
        scatter_color = sdf["iso_prediction"].map({0: "#64ffda", 1: "#ff4b4b"})
        sc_fig = go.Figure(go.Scatter(
            x=sdf[ch_scatter],
            y=sdf["z_score_max"],
            mode="markers",
            marker=dict(color=scatter_color, size=4, opacity=0.6),
            text=sdf["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
        ))
        sc_fig.add_hline(y=3, line_dash="dash", line_color="#ffd166",
                         annotation_text="Z=3 threshold", annotation_font_color="#ffd166")
        sc_fig.update_layout(
            height=340,
            **PLOTLY_THEME,
            xaxis=dict(title=CHANNEL_LABELS[ch_scatter], gridcolor="#1e3a5f", tickfont=dict(color="#8892b0")),
            yaxis=dict(title="Max Z-Score", gridcolor="#1e3a5f", tickfont=dict(color="#8892b0")),
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(sc_fig, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # ML MODEL — ISOLATION FOREST VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
      <div style="width:4px;height:32px;background:linear-gradient(180deg,#64ffda,#7eb3ff);border-radius:2px;"></div>
      <div>
        <div style="color:#e6f1ff;font-size:1.1rem;font-weight:700;letter-spacing:0.03em;">🤖 Isolation Forest — ML Model Insights</div>
        <div style="color:#8892b0;font-size:0.8rem;">Anomaly scores and detections from the trained Isolation Forest model</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    has_iso_score = "iso_anomaly_score" in df.columns and df["iso_anomaly_score"].notna().any()

    if not has_iso_score:
        st.warning("⚠️ Isolation Forest anomaly scores are not available. Ensure the model file exists at `models/iso_forest.pkl`.", icon="🤖")
    else:
        sdf2 = df.sample(min(3000, len(df)), random_state=7)

        # ── Row 1: Score Distribution Histogram + Score vs Channel Scatter ──
        col_hist, col_ch_sc = st.columns(2)

        with col_hist:
            st.markdown('<div class="section-header">Anomaly Score Distribution</div>', unsafe_allow_html=True)

            normal_scores = df[df["iso_prediction"] == 0]["iso_anomaly_score"].dropna()
            anom_scores   = df[df["iso_prediction"] == 1]["iso_anomaly_score"].dropna()

            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(
                x=normal_scores,
                name="Normal (ISO)",
                marker_color="#64ffda",
                opacity=0.65,
                nbinsx=60,
                hovertemplate="Score: %{x:.4f}<br>Count: %{y}<extra>Normal</extra>",
            ))
            hist_fig.add_trace(go.Histogram(
                x=anom_scores,
                name="Anomaly (ISO)",
                marker_color="#ff4b4b",
                opacity=0.75,
                nbinsx=60,
                hovertemplate="Score: %{x:.4f}<br>Count: %{y}<extra>Anomaly</extra>",
            ))

            # Decision boundary — typical split near the median of anomaly scores
            if len(anom_scores) > 0 and len(normal_scores) > 0:
                boundary = (normal_scores.max() + anom_scores.min()) / 2
                hist_fig.add_vline(
                    x=boundary, line_dash="dash", line_color="#ffd166", line_width=1.5,
                    annotation_text="Decision boundary",
                    annotation_font_color="#ffd166",
                    annotation_position="top right",
                )

            hist_fig.update_layout(
                barmode="overlay",
                height=320,
                **PLOTLY_THEME,
                xaxis=dict(title="Anomaly Score (higher = more anomalous)",
                           gridcolor="#1e3a5f", tickfont=dict(color="#8892b0")),
                yaxis=dict(title="Count", gridcolor="#1e3a5f", tickfont=dict(color="#8892b0")),
                legend=dict(font=dict(color="#8892b0"), bgcolor="rgba(0,0,0,0)",
                            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=10),
            )
            st.plotly_chart(hist_fig, use_container_width=True)

        with col_ch_sc:
            st.markdown('<div class="section-header">Score vs Channel (ISO Prediction)</div>', unsafe_allow_html=True)
            ch_iso = st.selectbox(
                "Channel:",
                FEATURE_COLS,
                format_func=lambda x: CHANNEL_LABELS[x],
                key="iso_ch_select",
            )

            # Four groups: TP / TN / FP / FN for richer insight
            group_colors = {
                "True Positive":  "#ff4b4b",
                "True Negative":  "#64ffda",
                "False Positive": "#ffd166",
                "False Negative": "#b58cf7",
            }

            def get_group(row):
                if row["iso_prediction"] == 1 and row["is_anomaly"] == 1:
                    return "True Positive"
                elif row["iso_prediction"] == 0 and row["is_anomaly"] == 0:
                    return "True Negative"
                elif row["iso_prediction"] == 1 and row["is_anomaly"] == 0:
                    return "False Positive"
                else:
                    return "False Negative"

            sdf2["pred_group"] = sdf2.apply(get_group, axis=1)

            iso_sc_fig = go.Figure()
            for grp, color in group_colors.items():
                mask = sdf2["pred_group"] == grp
                if mask.any():
                    iso_sc_fig.add_trace(go.Scatter(
                        x=sdf2.loc[mask, ch_iso],
                        y=sdf2.loc[mask, "iso_anomaly_score"],
                        mode="markers",
                        name=grp,
                        marker=dict(color=color, size=4, opacity=0.65),
                        hovertemplate=f"{grp}<br>{CHANNEL_LABELS[ch_iso]}: %{{x:.2f}}<br>Score: %{{y:.4f}}<extra></extra>",
                    ))

            iso_sc_fig.update_layout(
                height=320,
                **PLOTLY_THEME,
                xaxis=dict(title=CHANNEL_LABELS[ch_iso], gridcolor="#1e3a5f", tickfont=dict(color="#8892b0")),
                yaxis=dict(title="Anomaly Score", gridcolor="#1e3a5f", tickfont=dict(color="#8892b0")),
                legend=dict(font=dict(color="#8892b0"), bgcolor="rgba(0,0,0,0)",
                            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=10),
            )
            st.plotly_chart(iso_sc_fig, use_container_width=True)

        # ── Row 2: Anomaly Score Timeline ──────────────────────────────────
        st.markdown('<div class="section-header">Anomaly Score Timeline</div>', unsafe_allow_html=True)

        t_col = "time" if "time" in df.columns else None
        t_vals = df[t_col].values if t_col else np.arange(len(df))
        t_label = "Time (s)" if t_col else "Sample Index"

        # Subsample for performance
        step = max(1, len(df) // 3000)
        t_sub   = t_vals[::step]
        score_sub = df["iso_anomaly_score"].values[::step]
        pred_sub  = df["iso_prediction"].values[::step]
        gt_sub    = df["is_anomaly"].values[::step]

        tl_fig = go.Figure()

        # Score line — normal regions
        tl_fig.add_trace(go.Scatter(
            x=t_sub, y=score_sub,
            mode="lines",
            name="Anomaly Score",
            line=dict(color="#7eb3ff", width=1.2),
            opacity=0.8,
            hovertemplate="Score: %{y:.4f}<extra>Anomaly Score</extra>",
        ))

        # Overlay ISO-flagged points
        iso_mask = pred_sub == 1
        if iso_mask.any():
            tl_fig.add_trace(go.Scatter(
                x=t_sub[iso_mask],
                y=score_sub[iso_mask],
                mode="markers",
                name="ISO Detected",
                marker=dict(color="#ff4b4b", size=5, symbol="circle"),
                hovertemplate="Score: %{y:.4f}<extra>ISO Flagged</extra>",
            ))

        # Overlay ground-truth anomaly windows as a shaded band
        gt_mask = gt_sub == 1
        if gt_mask.any():
            tl_fig.add_trace(go.Scatter(
                x=t_sub[gt_mask],
                y=score_sub[gt_mask],
                mode="markers",
                name="Ground Truth Anomaly",
                marker=dict(color="#ffd166", size=4, symbol="diamond", opacity=0.7),
                hovertemplate="Score: %{y:.4f}<extra>True Anomaly</extra>",
            ))

        tl_fig.update_layout(
            height=300,
            **PLOTLY_THEME,
            xaxis=dict(title=t_label, gridcolor="#1e3a5f", tickfont=dict(color="#8892b0")),
            yaxis=dict(title="Anomaly Score", gridcolor="#1e3a5f", tickfont=dict(color="#8892b0")),
            legend=dict(font=dict(color="#8892b0"), bgcolor="rgba(0,0,0,0)",
                        orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(tl_fig, use_container_width=True)

        # ── ISO summary stats ───────────────────────────────────────────────
        st.markdown('<div class="section-header">Model Decision Summary</div>', unsafe_allow_html=True)
        iso_m = metrics["iso"]
        s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
        s_col1.metric("Avg Score (Normal)",  f"{df[df['iso_prediction']==0]['iso_anomaly_score'].mean():.4f}")
        s_col2.metric("Avg Score (Anomaly)", f"{df[df['iso_prediction']==1]['iso_anomaly_score'].mean():.4f}")
        s_col3.metric("True Positives",  f"{iso_m['TP']:,}")
        s_col4.metric("False Positives", f"{iso_m['FP']:,}")
        s_col5.metric("False Negatives", f"{iso_m['FN']:,}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
def show_model_comparison():
    st.markdown("""
    <div style="margin-bottom:20px;">
      <span style="font-size:1.6rem;font-weight:700;color:#e6f1ff;">📈 Model Comparison</span><br>
      <span style="color:#8892b0;font-size:0.9rem;">Isolation Forest vs Z-Score — side-by-side performance metrics</span>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.error("⚠️  Run `python src/predict_iso.py` first.", icon="🚨")
        return

    col1, col2 = st.columns(2)

    def metric_card(col, name, m, color):
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg,#0d1b2a,#112240);
                border: 1px solid {color}33; border-radius:14px; padding:20px 24px; margin-bottom:12px;">
              <div style="color:{color};font-size:0.75rem;font-weight:700;letter-spacing:0.1em;margin-bottom:12px;">
                ▶ {name}
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div><div style="color:#8892b0;font-size:0.7rem;">ACCURACY</div>
                     <div style="color:#e6f1ff;font-size:1.4rem;font-weight:700;">{m['Accuracy']:.3f}</div></div>
                <div><div style="color:#8892b0;font-size:0.7rem;">F1-SCORE</div>
                     <div style="color:#e6f1ff;font-size:1.4rem;font-weight:700;">{m['F1']:.3f}</div></div>
                <div><div style="color:#8892b0;font-size:0.7rem;">PRECISION</div>
                     <div style="color:#e6f1ff;font-size:1.4rem;font-weight:700;">{m['Precision']:.3f}</div></div>
                <div><div style="color:#8892b0;font-size:0.7rem;">RECALL</div>
                     <div style="color:#e6f1ff;font-size:1.4rem;font-weight:700;">{m['Recall']:.3f}</div></div>
              </div>
              <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:8px;color:#8892b0;font-size:0.75rem;">
                <div>TP: <span style="color:#64ffda">{m['TP']:,}</span></div>
                <div>TN: <span style="color:#64ffda">{m['TN']:,}</span></div>
                <div>FP: <span style="color:#ff4b4b">{m['FP']:,}</span></div>
                <div>FN: <span style="color:#ff4b4b">{m['FN']:,}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    metric_card(col1, "ISOLATION FOREST", metrics["iso"], "#64ffda")
    metric_card(col2, "Z-SCORE (σ = 3)", metrics["z"],   "#7eb3ff")

    st.markdown("---")

    # ── Radar chart ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Performance Radar</div>', unsafe_allow_html=True)
    cats = ["Accuracy", "Precision", "Recall", "F1"]
    iso_vals = [metrics["iso"][k] for k in cats]
    z_vals   = [metrics["z"][k]   for k in cats]

    radar_fig = go.Figure()
    fill_colors = {"#64ffda": "rgba(100,255,218,0.15)", "#7eb3ff": "rgba(126,179,255,0.15)"}
    for vals, name, color in [(iso_vals, "Isolation Forest", "#64ffda"), (z_vals, "Z-Score", "#7eb3ff")]:
        radar_fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name=name,
            line=dict(color=color, width=2),
            fillcolor=fill_colors[color],
        ))
        radar_fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="none",
            showlegend=False,
            line=dict(color=color, width=2),
        ))

    radar_fig.update_layout(
        polar=dict(
            bgcolor="#0d1b2a",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e3a5f",
                            tickfont=dict(color="#8892b0", size=9)),
            angularaxis=dict(tickfont=dict(color="#8892b0", size=11), gridcolor="#1e3a5f"),
        ),
        **PLOTLY_THEME,
        height=380,
        legend=dict(font=dict(color="#8892b0"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=10, b=10),
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("---")

    # ── Confusion matrix heatmaps ─────────────────────────────────────────
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    mid_colors = {"#64ffda": "rgba(100,255,218,0.33)", "#7eb3ff": "rgba(126,179,255,0.33)"}

    def cm_fig(m, title, color):
        z_data = [[m["TN"], m["FP"]], [m["FN"], m["TP"]]]
        labels = [["TN", "FP"], ["FN", "TP"]]
        fig = go.Figure(go.Heatmap(
            z=z_data,
            x=["Predicted Normal", "Predicted Anomaly"],
            y=["Actual Normal", "Actual Anomaly"],
            colorscale=[[0, "#0a0e1a"], [0.5, mid_colors[color]], [1, color]],
            showscale=False,
            text=labels,
            texttemplate="%{text}<br>%{z:,}",
            textfont=dict(color="#e6f1ff", size=13),
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(color="#8892b0", size=13)),
            **PLOTLY_THEME,
            height=280,
            xaxis=dict(tickfont=dict(color="#8892b0")),
            yaxis=dict(tickfont=dict(color="#8892b0")),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    c1.plotly_chart(cm_fig(metrics["iso"], "Isolation Forest", "#64ffda"), use_container_width=True)
    c2.plotly_chart(cm_fig(metrics["z"],   "Z-Score",          "#7eb3ff"), use_container_width=True)


# ── Router ─────────────────────────────────────────────────────────────────
if page == "🏠  Home":
    show_home()
elif page == "📊  Dashboard":
    show_dashboard()
elif page == "🔍  Anomaly Explorer":
    show_anomaly_explorer()
elif page == "📈  Model Comparison":
    show_model_comparison()
