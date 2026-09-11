"""
Data Loader Module for Medical Equipment Failure Prediction Agent
Loads raw datasets from D:\ or data/raw directory.
"""
import os
import pandas as pd
from .config import BASE_DIR, RAW_DATA_DIR

def get_raw_filepath(filename: str) -> str:
    """Find dataset file in RAW_DATA_DIR or D:\\ drive."""
    candidates = [
        os.path.join(RAW_DATA_DIR, filename),
        os.path.join("D:\\", filename),
        os.path.join("d:\\", filename),
    ]
    # Check prefix variations (e.g. devices-1681209661.csv)
    prefix = filename.split('.')[0]
    for c in candidates:
        if os.path.exists(c):
            return c
            
    if os.path.exists("D:\\"):
        for f in os.listdir("D:\\"):
            if f.startswith(prefix) and f.endswith(".csv"):
                return os.path.join("D:\\", f)
                
    raise FileNotFoundError(f"Source CSV for {filename} not found in {RAW_DATA_DIR} or D:\\ drive.")

def load_raw_data():
    """Load devices, events, and manufacturers CSV files into Pandas DataFrames."""
    dev_path = get_raw_filepath("devices.csv")
    evt_path = get_raw_filepath("events.csv")
    mfr_path = get_raw_filepath("manufacturers.csv")

    print(f"[DataLoader] Loading devices from {dev_path}...")
    devices_df = pd.read_csv(dev_path, low_memory=False)

    print(f"[DataLoader] Loading events from {evt_path}...")
    events_df = pd.read_csv(evt_path, low_memory=False)

    print(f"[DataLoader] Loading manufacturers from {mfr_path}...")
    manufacturers_df = pd.read_csv(mfr_path, low_memory=False)

    return devices_df, events_df, manufacturers_df
