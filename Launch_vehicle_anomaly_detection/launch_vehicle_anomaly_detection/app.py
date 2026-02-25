"""
app.py  –  Day 11: Streamlit Foundation
----------------------------------------
Launch Vehicle Telemetry Anomaly Detector
Streamlit app skeleton with sidebar navigation (Home | Dashboard).

Run with:
    streamlit run app.py
"""

import streamlit as st

# ---------------------------------------------------------------
# PAGE CONFIG  (must be the very first Streamlit call)
# ---------------------------------------------------------------
st.set_page_config(
    page_title   = "Launch Vehicle Anomaly Detector",
    page_icon    = "🚀",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------
# SIDEBAR  –  Navigation
# ---------------------------------------------------------------
st.sidebar.title("🚀 LV Anomaly Detector")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    label   = "Navigate",
    options = ["Home", "Dashboard"],
    index   = 0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Day 11 · Streamlit Foundation")


# ---------------------------------------------------------------
# PAGE: HOME
# ---------------------------------------------------------------
def show_home() -> None:
    st.title("🚀 Launch Vehicle Telemetry Anomaly Detector")
    st.markdown(
        """
        Welcome to the **Launch Vehicle Telemetry Anomaly Detector** — 
        a lightweight prototype that ingests synthetic flight telemetry, 
        injects realistic faults, and flags anomalies using two complementary 
        algorithms:

        | Method | Approach |
        |---|---|
        | **Z-Score** | Statistical — flags samples beyond ±3σ from the mean |
        | **Isolation Forest** | ML-based — anomalies are isolated faster in random trees |

        ---
        ### Telemetry Channels Monitored
        - **Altitude** (m)
        - **Velocity** (m/s)
        - **Engine Temperature** (°C)
        - **Fuel Pressure** (kPa)
        - **Vibration** (g)

        ---
        ### Anomaly Types Simulated
        - **Spikes** — sudden, short-lived excursions (sensor bit-flip / surge)
        - **Drift**  — slow, cumulative sensor bias (calibration loss)

        ---
        > Use the **Dashboard** page to explore the telemetry and anomaly detection results.
        """
    )
    st.info("Select **Dashboard** in the sidebar to get started.", icon="👈")


# ---------------------------------------------------------------
# PAGE: DASHBOARD
# ---------------------------------------------------------------
def show_dashboard() -> None:
    st.title("📊 Telemetry Dashboard")
    st.markdown("Real-time anomaly detection results will appear here.")

    st.markdown("---")

    # ── Placeholder metric cards ──────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Total Samples",    value="6,000")
    col2.metric(label="Z-Score Flags",    value="—",    delta=None)
    col3.metric(label="Isolation Forest", value="—",    delta=None)
    col4.metric(label="Model Status",     value="Loaded ✅")

    st.markdown("---")

    # ── Placeholder chart area ────────────────────────────────
    st.subheader("Telemetry Signal Viewer")
    st.info(
        "⚙️  Chart integration coming next — will plot altitude, velocity, "
        "engine_temp, fuel_pressure, and vibration with anomaly overlays.",
        icon="📈",
    )

    st.markdown("---")

    # ── Placeholder anomaly table ─────────────────────────────
    st.subheader("Detected Anomalies")
    st.info(
        "⚙️  Anomaly table coming next — will list timestamps, channel, "
        "detection method, and anomaly score.",
        icon="🔍",
    )


# ---------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------
if page == "Home":
    show_home()
elif page == "Dashboard":
    show_dashboard()
