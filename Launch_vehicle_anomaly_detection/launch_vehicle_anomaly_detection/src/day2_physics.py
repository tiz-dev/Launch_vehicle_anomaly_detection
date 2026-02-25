import numpy as np
import matplotlib.pyplot as plt

# --- PRESSURE CONFIGURATION ---
INITIAL_PRESSURE = 5000   # Starting tank pressure (arbitrary units)
DECAY_RATE       = 0.008  # Controls how fast pressure drops (higher = faster decay)
NOISE_STD        = 50     # Standard deviation of Gaussian noise (units)

# --- VIBRATION CONFIGURATION ---
MAX_Q_VELOCITY   = 300.0  # Velocity (m/s) at which Max-Q / peak vibration occurs
MAX_Q_WIDTH      = 80.0   # Gaussian sigma – controls how sharply vibration peaks (m/s)
VIBR_AMPLITUDE   = 10.0   # Peak vibration amplitude (arbitrary g-units)
VIBR_NOISE_STD   = 0.5    # Gaussian sensor noise on vibration signal


def simulate_pressure(time_vector: np.ndarray) -> np.ndarray:
    """
    Simulates rocket fuel tank pressure over a flight.

    Physics model
    -------------
    Fuel is consumed continuously, so tank pressure decays exponentially:

        P(t) = P0 * exp(-k * t)  +  noise

    Where:
        P0    – initial pressure (5000 units)
        k     – decay rate constant (controls consumption speed)
        noise – Gaussian random noise ~ N(0, NOISE_STD)

    Parameters
    ----------
    time_vector : np.ndarray
        1-D array of time values (in seconds).

    Returns
    -------
    pressure : np.ndarray
        1-D array of pressure values (same shape as time_vector).
    """
    # Deterministic exponential decay
    clean_signal = INITIAL_PRESSURE * np.exp(-DECAY_RATE * time_vector)

    # Gaussian noise – models sensor jitter / micro-turbulence
    noise = np.random.normal(loc=0.0, scale=NOISE_STD, size=time_vector.shape)

    pressure = clean_signal + noise
    return pressure


def simulate_vibration(velocity_vector: np.ndarray) -> np.ndarray:
    """
    Simulates rocket airframe vibration throughout a flight.

    Physics model
    -------------
    Aerodynamic load (dynamic pressure Q) peaks when the rocket reaches
    roughly 300 m/s in the lower atmosphere.  Vibration is modelled as
    a Gaussian bell-curve centred on that Max-Q velocity:

        V_base(v) = A * exp( -( (v - v_maxq)^2 / (2 * sigma^2) ) )

    A velocity-proportional factor (v / v_maxq) ensures that vibration
    also scales with overall speed, not just proximity to Max-Q.
    Gaussian noise is then added to mimic sensor jitter.

    Parameters
    ----------
    velocity_vector : np.ndarray
        1-D array of velocity values (in m/s).

    Returns
    -------
    vibration : np.ndarray
        1-D array of vibration amplitude values (same shape as velocity_vector).
    """
    # Bell-curve centred on Max-Q velocity
    gaussian_peak = np.exp(
        -((velocity_vector - MAX_Q_VELOCITY) ** 2) / (2 * MAX_Q_WIDTH ** 2)
    )

    # Scale amplitude: peak vibration at Max-Q, proportional to speed elsewhere
    velocity_scale = velocity_vector / MAX_Q_VELOCITY  # 0 → 0, at Max-Q → 1
    clean_signal = VIBR_AMPLITUDE * gaussian_peak * velocity_scale

    # Gaussian noise – models sensor jitter / structural micro-oscillations
    noise = np.random.normal(loc=0.0, scale=VIBR_NOISE_STD, size=velocity_vector.shape)

    vibration = clean_signal + noise
    return vibration


# ------------------------------------------------------------------
# Quick self-test / visual check when run directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Use the same time vector convention as day1_generator.py
    DURATION_SECONDS = 600
    SAMPLING_RATE_HZ = 10
    TOTAL_POINTS     = DURATION_SECONDS * SAMPLING_RATE_HZ

    time     = np.linspace(0, DURATION_SECONDS, TOTAL_POINTS)
    # Velocity: linear ramp (matches day1_generator.py model: v = t * 0.5)
    velocity = time * 0.5
    pressure = simulate_pressure(time)
    vibration = simulate_vibration(velocity)

    # ── Pressure stats ──────────────────────────────────────────────
    print("=== simulate_pressure() Quick Stats ===")
    print(f"  Time points   : {len(time)}")
    print(f"  Pressure start: {pressure[0]:.2f} units")
    print(f"  Pressure end  : {pressure[-1]:.2f} units")
    print(f"  Min / Max     : {pressure.min():.2f} / {pressure.max():.2f}")

    # ── Vibration stats ─────────────────────────────────────────────
    peak_idx = np.argmax(vibration)
    print("\n=== simulate_vibration() Quick Stats ===")
    print(f"  Velocity points   : {len(velocity)}")
    print(f"  Vibration min     : {vibration.min():.4f} g")
    print(f"  Vibration max     : {vibration.max():.4f} g  (at v={velocity[peak_idx]:.1f} m/s, t={time[peak_idx]:.1f}s)")
    print(f"  Expected peak ~   : {MAX_Q_VELOCITY} m/s  (Max-Q)")

    # ── Plots ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(time, velocity, color='seagreen', linewidth=0.8, label='Velocity')
    axes[0].axvline(time[peak_idx], color='red', linestyle='--', alpha=0.6, label=f'Peak vibration (t={time[peak_idx]:.0f}s)')
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Day 2 – Velocity Profile")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(time, vibration, color='darkorange', linewidth=0.8, label='Vibration')
    axes[1].axvline(time[peak_idx], color='red', linestyle='--', alpha=0.6)
    axes[1].set_ylabel("Vibration (g)")
    axes[1].set_title("Day 2 – Simulated Airframe Vibration (Max-Q Peak + Noise)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(time, pressure, color='royalblue', linewidth=0.8, label='Tank Pressure')
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Pressure (units)")
    axes[2].set_title("Day 2 – Simulated Fuel Tank Pressure (Exponential Decay + Noise)")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
