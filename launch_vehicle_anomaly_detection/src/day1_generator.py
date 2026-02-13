import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
DURATION_SECONDS = 600  # Total flight time
SAMPLING_RATE_HZ = 10   # Data points per second
TOTAL_POINTS = DURATION_SECONDS * SAMPLING_RATE_HZ

def generate_telemetry():
    """Generates synthetic launch vehicle telemetry data."""
    
    # 1. Time Vector
    time = np.linspace(0, DURATION_SECONDS, TOTAL_POINTS)
    
    # 2. Altitude (Curve: goes up)
    # y = x^2 (basic physics approximation for constant acceleration)
    # We normalize it to reach a peak of ~100km (100,000 meters)
    altitude = (time ** 2) * 0.25 
    
    # 3. Velocity (Linear increase, derivative of Altitude)
    # v = u + at (velocity increases linearly with constant acceleration)
    velocity = time * 0.5 + np.random.normal(0, 1.0, TOTAL_POINTS) # Add small noise
    
    # 4. Engine Temperature (Fluctuates around a mean)
    # Random walk: starts at 300K, drifts slightly
    base_temp = 300 # Kelvin
    temp_noise = np.random.normal(0, 5, TOTAL_POINTS)
    temperature = base_temp + (time * 0.1) + temp_noise

    # 5. Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'altitude': altitude,
        'velocity': velocity,
        'engine_temp': temperature
    })
    
    return df

if __name__ == "__main__":
    print("Generating telemetry data...")
    df = generate_telemetry()
    
    # Save to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "../data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "normal_telemetry.csv")
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    
    # Simple Plot to Verify
    print("Plotting validation chart...")
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['altitude'], label='Altitude')
    plt.plot(df['time'], df['velocity'], label='Velocity')
    plt.legend()
    plt.title("Day 1: Basic Telemetry Check")
    plt.show()
