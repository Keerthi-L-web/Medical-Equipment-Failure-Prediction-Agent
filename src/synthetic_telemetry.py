"""
Synthetic Telemetry Generator Module for Medical Equipment Failure Prediction Agent
Generates realistic physics-based sensor degradation trajectories for monitored medical devices.
Reproducible using random_state = 42.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .config import RANDOM_SEED, TELEMETRY_TIMESTEPS_PER_DEVICE, SIMULATED_DEGRADATION_RATIO, PROCESSED_DATA_DIR

def generate_synthetic_telemetry(devices_df: pd.DataFrame, num_devices: int = 1000) -> pd.DataFrame:
    """
    Generate realistic multi-signal telemetry time-series for a subset of real device IDs.
    """
    np.random.seed(RANDOM_SEED)

    # Sample representative device IDs
    sample_df = devices_df.head(num_devices).copy()
    device_ids = sample_df['id'].tolist()
    device_types = dict(zip(sample_df['id'], sample_df['device_type']))

    records = []
    end_time = datetime(2018, 1, 1, 12, 0, 0)
    time_delta = timedelta(hours=6)

    print(f"[SyntheticTelemetry] Generating time-series telemetry for {len(device_ids):,} devices ({TELEMETRY_TIMESTEPS_PER_DEVICE} timesteps each)...")

    # Designate a subset of devices to undergo progressive degradation
    num_degrading = int(len(device_ids) * SIMULATED_DEGRADATION_RATIO)
    degrading_device_ids = set(np.random.choice(device_ids, size=num_degrading, replace=False))

    for dev_id in device_ids:
        dev_type = device_types.get(dev_id, 'Other')
        is_degrading = dev_id in degrading_device_ids
        
        # Start onset of degradation randomly between timestep 10 and 30
        degradation_onset = np.random.randint(10, 30) if is_degrading else TELEMETRY_TIMESTEPS_PER_DEVICE + 1

        # Baseline specifications per device type
        base_temp = 36.5 + np.random.normal(0, 0.5)
        base_vib = 0.15 + np.random.uniform(0.01, 0.05)
        base_power = 120.0 + np.random.normal(0, 5.0)
        base_voltage = 220.0 + np.random.normal(0, 1.0)
        base_error = 0.1 + np.random.exponential(0.1)
        base_battery = 98.0 - np.random.uniform(0, 5.0)
        accumulated_hours = np.random.randint(100, 3000)

        start_time = end_time - (time_delta * TELEMETRY_TIMESTEPS_PER_DEVICE)

        for t in range(TELEMETRY_TIMESTEPS_PER_DEVICE):
            curr_time = start_time + (time_delta * t)
            accumulated_hours += 6

            # Compute degradation factor (0.0 healthy -> 1.0 critical failure)
            if is_degrading and t >= degradation_onset:
                deg_progress = (t - degradation_onset) / (TELEMETRY_TIMESTEPS_PER_DEVICE - degradation_onset)
                deg_factor = np.clip(deg_progress ** 1.5, 0.0, 1.0)
            else:
                deg_factor = 0.0

            # Signals with physics-based coupled drift + Gaussian noise
            noise_temp = np.random.normal(0, 0.2)
            temp = base_temp + (deg_factor * 12.5) + noise_temp

            noise_vib = np.random.normal(0, 0.03)
            vib = base_vib + (deg_factor * 1.8) + noise_vib

            noise_power = np.random.normal(0, 2.0)
            power = base_power + (deg_factor * 45.0) + (deg_factor * np.random.normal(0, 10.0)) + noise_power

            voltage_noise = np.random.normal(0, 0.5 + deg_factor * 4.0)
            voltage = base_voltage + voltage_noise

            error_rate = np.clip(base_error + (deg_factor * 12.0) + np.random.exponential(0.2), 0.0, 25.0)
            
            battery_health = np.clip(base_battery - (t * 0.1) - (deg_factor * 35.0) + np.random.normal(0, 0.3), 5.0, 100.0)

            # Combined continuous health score (100 = healthy, 0 = failed)
            raw_health = 100.0 - (
                (temp - 36.5) * 2.5 + 
                (vib - 0.15) * 20.0 + 
                (error_rate * 3.0) + 
                (100.0 - battery_health) * 0.3
            )
            health_score = float(np.clip(raw_health, 0.0, 100.0))

            # Failure label triggers when composite degradation crosses critical threshold
            telemetry_failure_label = 1 if (health_score < 35.0 and deg_factor > 0.6) else 0

            records.append({
                'device_id': dev_id,
                'timestamp': curr_time.strftime('%Y-%m-%d %H:%M:%S'),
                'device_type': dev_type,
                'temperature': round(temp, 2),
                'vibration': round(vib, 3),
                'power_consumption': round(power, 2),
                'voltage': round(voltage, 2),
                'operating_hours': accumulated_hours,
                'error_rate': round(error_rate, 2),
                'battery_health': round(battery_health, 1),
                'health_score': round(health_score, 1),
                'telemetry_failure_label': telemetry_failure_label
            })

    telemetry_df = pd.DataFrame(records)

    # Save synthetic telemetry CSV
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_DIR, "synthetic_telemetry.csv")
    telemetry_df.to_csv(out_path, index=False)
    print(f"[SyntheticTelemetry] Saved synthetic telemetry dataset with {len(telemetry_df):,} rows to {out_path}")

    return telemetry_df

def engineer_telemetry_features(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal rolling and rate-of-change telemetry features per device.
    Summarizes recent telemetry state into a single row per device.
    """
    df = telemetry_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['device_id', 'timestamp'])

    # Derive rolling averages and rates of change per device
    grouped = df.groupby('device_id')

    df['rolling_temp_mean_5'] = grouped['temperature'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['rolling_vib_mean_5'] = grouped['vibration'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['rolling_power_mean_5'] = grouped['power_consumption'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    df['temperature_change_rate'] = grouped['temperature'].diff().fillna(0.0)
    df['vibration_change_rate'] = grouped['vibration'].diff().fillna(0.0)
    df['power_change_rate'] = grouped['power_consumption'].diff().fillna(0.0)
    df['voltage_variability'] = grouped['voltage'].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0.0)

    # Take the latest timestep snapshot for each device as current telemetry state
    latest_telemetry = grouped.last().reset_index()

    telemetry_feature_cols = [
        'device_id', 'temperature', 'vibration', 'power_consumption', 'voltage',
        'operating_hours', 'error_rate', 'battery_health', 'health_score',
        'rolling_temp_mean_5', 'rolling_vib_mean_5', 'rolling_power_mean_5',
        'temperature_change_rate', 'vibration_change_rate', 'power_change_rate',
        'voltage_variability', 'telemetry_failure_label'
    ]

    return latest_telemetry[telemetry_feature_cols]
