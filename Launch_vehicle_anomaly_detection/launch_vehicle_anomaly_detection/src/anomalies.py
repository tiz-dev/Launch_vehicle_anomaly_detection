import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# CONFIGURATION DEFAULTS
# ---------------------------------------------------------------
DEFAULT_MAGNITUDE    = 100   # How big a spike is (±magnitude)
DEFAULT_PROBABILITY  = 0.01  # Probability that any given sample gets a spike
DEFAULT_DRIFT_FACTOR = 0.1   # Drift step per sample (added cumulatively after onset)


def inject_spike(
    signal: np.ndarray,
    magnitude: float = DEFAULT_MAGNITUDE,
    probability: float = DEFAULT_PROBABILITY,
) -> np.ndarray:
    """
    Inject sudden, large spikes into a signal at random positions.

    Model
    -----
    A spike at index i means the sensor momentarily reports an
    extreme value (e.g., a transient voltage surge, bit-flip, or
    sudden pressure drop).  Each sample is independently spiked
    with probability `probability`:

        corrupted[i] = signal[i] + spike_i

    where spike_i ~ Uniform(-magnitude, +magnitude) if spiked,
    else 0.

    Parameters
    ----------
    signal      : np.ndarray  – 1-D original sensor readings.
    magnitude   : float       – Maximum absolute spike amplitude
                                (added on TOP of the real value).
    probability : float       – Probability [0, 1] that any single
                                sample is spiked. Default 0.01 = 1%.

    Returns
    -------
    corrupted   : np.ndarray  – Copy of `signal` with spikes added.
                                Shape is identical to input.

    Notes
    -----
    * Returns a **copy** — the original array is never mutated.
    * Spike direction is random (can be positive or negative).
    * Use np.random.seed() before calling for reproducible results.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if magnitude < 0:
        raise ValueError(f"magnitude must be >= 0, got {magnitude}")

    corrupted = signal.copy()
    n = len(signal)

    # Boolean mask: True where a spike will be injected
    spike_mask = np.random.random(n) < probability

    # Random spike amplitudes: uniform in [-magnitude, +magnitude]
    spike_amplitudes = np.random.uniform(-magnitude, magnitude, size=n)

    # Apply only where mask is True
    corrupted[spike_mask] += spike_amplitudes[spike_mask]

    spike_indices = np.where(spike_mask)[0]
    print(f"[inject_spike] {len(spike_indices)} spike(s) injected "
          f"out of {n} samples "
          f"({100 * len(spike_indices) / n:.2f}% hit rate).")

    return corrupted


def inject_drift(
    signal: np.ndarray,
    drift_factor: float = DEFAULT_DRIFT_FACTOR,
) -> np.ndarray:
    """
    Inject a cumulative drift (increasing bias) into a signal.

    Model
    -----
    A realistic "creeping" sensor failure: from a random onset index
    onwards, an ever-growing bias is added to each sample:

        onset  = randint(0, n-1)     – random start point
        bias_i = drift_factor * (i - onset)   for i >= onset
                 0                            for i <  onset

    So by the end of the array the accumulated offset is:
        drift_factor * (n - 1 - onset)

    Parameters
    ----------
    signal       : np.ndarray  – 1-D original sensor readings.
    drift_factor : float       – Bias added *per sample* after onset.
                                 Larger values = faster drift.
                                 Default 0.1.

    Returns
    -------
    corrupted    : np.ndarray  – Copy of `signal` with drift added.
                                 Shape is identical to input.

    Notes
    -----
    * Returns a **copy** — the original array is never mutated.
    * Drift is always positive (upward bias). Negate drift_factor for
      a downward drift.
    * Use np.random.seed() before calling for reproducible results.
    """
    if drift_factor < 0:
        raise ValueError(f"drift_factor must be >= 0, got {drift_factor}. "
                         "Use a negative value intentionally by negating outside.")

    n = len(signal)
    corrupted = signal.copy()

    # Random onset: drift can start anywhere except the very last sample
    onset = np.random.randint(0, n - 1)

    # Cumulative bias: 0 before onset, linearly growing after
    bias = np.zeros(n)
    post_onset_indices = np.arange(n - onset)        # [0, 1, 2, ...]
    bias[onset:] = drift_factor * post_onset_indices

    corrupted += bias

    total_drift = bias[-1]
    print(f"[inject_drift] Drift onset at index {onset} "
          f"(t ≈ {onset} samples). "
          f"Total accumulated bias at end: +{total_drift:.4f} units.")

    return corrupted


def detect_zscore(
    data: np.ndarray,
    threshold: float = 3.0,
) -> np.ndarray:
    """
    Detect anomalies in a 1-D signal using the Z-score method.

    Model
    -----
    For each sample x_i, the Z-score measures how many standard
    deviations it lies from the global mean:

        z_i = (x_i - μ) / σ

    where  μ = mean(data)  and  σ = std(data).

    A sample is flagged as an anomaly when |z_i| > threshold.
    The conventional threshold is 3 (i.e., outside ±3 σ), which
    covers ~99.7 % of a normal distribution.

    Parameters
    ----------
    data      : np.ndarray  – 1-D array of sensor readings.
    threshold : float       – Minimum |z| to be called an anomaly.
                              Default 3.0.

    Returns
    -------
    anomaly_indices : np.ndarray  – Integer indices where |z| > threshold.
                                    Empty array if none found.

    Raises
    ------
    ValueError  – If `data` has zero length or zero standard deviation
                  (a flat signal has no variation to score against).

    Notes
    -----
    * The function uses **population** std (ddof=0) to match the
      classic z-score formula.
    * Both positive AND negative extremes are flagged (|z| > threshold).
    * Use np.random.seed() before injecting anomalies for reproducible
      demonstrations.

    Examples
    --------
    >>> clean = np.zeros(100)
    >>> clean[42] = 999          # obvious spike
    >>> detect_zscore(clean)
    array([42])
    """
    if len(data) == 0:
        raise ValueError("data must be non-empty.")

    mean = np.mean(data)
    std  = np.std(data, ddof=0)

    if std == 0.0:
        raise ValueError(
            "Standard deviation is zero — cannot compute Z-scores on a "
            "constant (flat) signal."
        )

    z_scores = (data - mean) / std
    anomaly_indices = np.where(np.abs(z_scores) > threshold)[0]

    print(
        f"[detect_zscore] μ={mean:.4f}  σ={std:.4f}  threshold=±{threshold}  "
        f"→  {len(anomaly_indices)} anomaly sample(s) detected "
        f"out of {len(data)} total."
    )

    return anomaly_indices


# ---------------------------------------------------------------
# Quick self-test / visual check when run directly
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Use the Day 2 pressure signal as test input
    DURATION_SECONDS = 600
    SAMPLING_RATE_HZ = 10
    TOTAL_POINTS     = DURATION_SECONDS * SAMPLING_RATE_HZ

    np.random.seed(42)   # reproducible demo

    time  = np.linspace(0, DURATION_SECONDS, TOTAL_POINTS)
    clean = 5000 * np.exp(-0.008 * time)   # exponential decay (Day 2)

    # ── inject_spike ─────────────────────────────────────────────────
    spiked = inject_spike(clean, magnitude=500, probability=0.02)

    print("\n=== inject_spike() Quick Stats ===")
    print(f"  Samples total  : {TOTAL_POINTS}")
    print(f"  Clean   max    : {clean.max():.2f}  |  min: {clean.min():.2f}")
    print(f"  Spiked  max    : {spiked.max():.2f}  |  min: {spiked.min():.2f}")

    # ── inject_drift ─────────────────────────────────────────────────
    drifted = inject_drift(clean, drift_factor=0.5)

    print("\n=== inject_drift() Quick Stats ===")
    print(f"  Samples total  : {TOTAL_POINTS}")
    print(f"  Clean   end    : {clean[-1]:.2f}")
    print(f"  Drifted end    : {drifted[-1]:.2f}  (+{drifted[-1]-clean[-1]:.2f} drift)")

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # — Spike panel —
    axes[0].plot(time, clean,  color="steelblue", linewidth=0.8, label="Clean Signal",  alpha=0.7)
    axes[0].plot(time, spiked, color="tomato",    linewidth=0.8, label="Spiked Signal", alpha=0.9)
    diff = spiked - clean
    axes[0].scatter(time[diff != 0], spiked[diff != 0],
                    color="red", s=20, zorder=5, label="Spike Points")
    axes[0].set_ylabel("Pressure (units)")
    axes[0].set_title("Day 3 – inject_spike(): Sudden Anomaly Injection")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # — Drift panel —
    axes[1].plot(time, clean,   color="steelblue",  linewidth=0.8, label="Clean Signal",   alpha=0.7)
    axes[1].plot(time, drifted, color="darkorchid", linewidth=0.8, label="Drifted Signal", alpha=0.9)
    # Mark the drift onset
    drift_start = np.where(drifted != clean)[0]
    if len(drift_start):
        axes[1].axvline(time[drift_start[0]], color="purple", linestyle="--",
                        alpha=0.7, label=f"Drift onset (t={time[drift_start[0]]:.0f}s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Pressure (units)")
    axes[1].set_title("Day 3 – inject_drift(): Gradual Anomaly Injection")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
