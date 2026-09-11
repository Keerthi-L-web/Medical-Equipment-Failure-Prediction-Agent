"""
EDA & Data Inspection Script for Medical Equipment Failure Prediction Agent
Analyzes devices.csv, events.csv, manufacturers.csv and generates detailed statistical report.
"""
import os
import sys
import pandas as pd
import numpy as np

def find_file(filename):
    """Search for dataset file in expected paths."""
    candidates = [
        os.path.join("data", "raw", filename),
        os.path.join("..", "data", "raw", filename),
        os.path.join("D:\\", filename),
        os.path.join("d:\\", filename),
    ]
    # Check variations like devices-1681209661.csv
    prefix = filename.split('.')[0]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Check D:\ for matching prefixes
    d_files = os.listdir("D:\\") if os.path.exists("D:\\") else []
    for f in d_files:
        if f.startswith(prefix) and f.endswith(".csv"):
            return os.path.join("D:\\", f)
    raise FileNotFoundError(f"Could not find dataset file for {filename}")

def inspect_data():
    print("=" * 60)
    print("STARTING PHASE 1: DATA INSPECTION AND EDA")
    print("=" * 60)

    devices_path = find_file("devices.csv")
    events_path = find_file("events.csv")
    manufacturers_path = find_file("manufacturers.csv")

    print(f"Loading devices from: {devices_path}")
    print(f"Loading events from: {events_path}")
    print(f"Loading manufacturers from: {manufacturers_path}")

    df_dev = pd.read_csv(devices_path, low_memory=False)
    df_evt = pd.read_csv(events_path, low_memory=False)
    df_mfr = pd.read_csv(manufacturers_path, low_memory=False)

    print("\n--- 1. DEVICES DATASET SUMMARY ---")
    print(f"Total Device Rows: {len(df_dev):,}")
    print(f"Missing Values per Column:\n{df_dev.isnull().sum()}")
    print(f"\nUnique Risk Classes: {df_dev['risk_class'].value_counts(dropna=False).to_dict()}")
    print(f"Unique Classifications Count: {df_dev['classification'].nunique()}")
    print(f"Unique Device Codes Count: {df_dev['code'].nunique()}")
    print(f"Implanted Distribution:\n{df_dev['implanted'].value_counts(dropna=False).to_dict()}")
    print(f"Country Distribution Top 10:\n{df_dev['country'].value_counts(dropna=False).head(10).to_dict()}")
    print(f"Unique Manufacturers in Devices: {df_dev['manufacturer_id'].nunique()}")

    # Quantity distribution
    if 'quantity_in_commerce' in df_dev.columns:
        print(f"Quantity in Commerce Stats:\n{df_dev['quantity_in_commerce'].describe()}")

    # Date range for devices
    if 'created_at' in df_dev.columns:
        dev_created = pd.to_datetime(df_dev['created_at'], errors='coerce')
        print(f"Devices Created At Date Range: {dev_created.min()} to {dev_created.max()}")

    print("\n--- 2. EVENTS DATASET SUMMARY ---")
    print(f"Total Event Rows: {len(df_evt):,}")
    print(f"Missing Values per Column:\n{df_evt.isnull().sum()}")
    print(f"\nUnique Actions Top 10:\n{df_evt['action'].value_counts(dropna=False).head(10).to_dict()}")
    print(f"Unique Action Summaries (Recall/etc):\n{df_evt['action_summary'].value_counts(dropna=False).head(10).to_dict()}")
    print(f"Action Level Distribution:\n{df_evt['action_level'].value_counts(dropna=False).to_dict()}")
    print(f"Top Causes:\n{df_evt['determined_cause'].value_counts(dropna=False).head(10).to_dict()}")
    print(f"Top Reasons:\n{df_evt['reason'].value_counts(dropna=False).head(10).to_dict()}")
    print(f"Event Types:\n{df_evt['type'].value_counts(dropna=False).to_dict()}")
    print(f"Status Distribution:\n{df_evt['status'].value_counts(dropna=False).to_dict()}")

    # Date parse for events
    evt_dates = pd.to_datetime(df_evt['date'], errors='coerce')
    print(f"\nEvent Date Range: {evt_dates.min()} to {evt_dates.max()}")
    print(f"Events with valid dates: {evt_dates.notnull().sum():,} / {len(df_evt):,}")

    devices_with_events = df_evt['device_id'].nunique()
    print(f"Unique Devices with Events: {devices_with_events:,} / {df_dev['id'].nunique():,} ({devices_with_events / df_dev['id'].nunique():.2%})")

    print("\n--- 3. MANUFACTURERS DATASET SUMMARY ---")
    print(f"Total Manufacturer Rows: {len(df_mfr):,}")
    print(f"Missing Values per Column:\n{df_mfr.isnull().sum()}")
    print(f"Unique Manufacturers: {df_mfr['id'].nunique()}")
    print(f"Top Parent Companies:\n{df_mfr['parent_company'].value_counts(dropna=False).head(10).to_dict()}")

    print("\n--- 4. TEMPORAL ANALYSIS FOR TARGET DEFINITION ---")
    # Date distribution by year
    years = evt_dates.dt.year.value_counts().sort_index()
    print(f"Event Count by Year:\n{years}")

    # Determine cutoff date
    # E.g. pick a cutoff date where historical events exist before and future events exist after
    valid_evt_dates = evt_dates.dropna()
    q75 = valid_evt_dates.quantile(0.75)
    print(f"\n75th Percentile Event Date: {q75}")

    # Let's inspect sample device names to categorize device types
    print("\n--- 5. DEVICE NAME KEYWORD INSPECTION ---")
    sample_names = df_dev['name'].dropna().head(30).tolist()
    print("Sample Device Names:")
    for n in sample_names:
        print(f"  - {n}")

    # Output detailed report to reports/eda_report.md
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "eda_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Exploratory Data Analysis & Schema Audit Report\n\n")
        f.write("## 1. Dataset Overview\n\n")
        f.write(f"- **Devices Total**: `{len(df_dev):,}` rows\n")
        f.write(f"- **Events Total**: `{len(df_evt):,}` rows\n")
        f.write(f"- **Manufacturers Total**: `{len(df_mfr):,}` rows\n")
        f.write(f"- **Devices with Recorded Events**: `{devices_with_events:,}` ({devices_with_events / df_dev['id'].nunique():.2%})\n\n")

        f.write("## 2. Key Column Distributions\n\n")
        f.write("### Risk Class (Devices)\n")
        for k, v in df_dev['risk_class'].value_counts(dropna=False).items():
            f.write(f"- `{k}`: {v:,}\n")
        
        f.write("\n### Event Types & Action Summaries\n")
        for k, v in df_evt['action_summary'].value_counts(dropna=False).head(10).items():
            f.write(f"- `{k}`: {v:,}\n")

        f.write("\n## 3. Date Ranges & Temporal Cutoff Assessment\n\n")
        f.write(f"- **Earliest Event Date**: `{evt_dates.min()}`\n")
        f.write(f"- **Latest Event Date**: `{evt_dates.max()}`\n")
        f.write(f"- **Recommended Temporal Cutoff**: `2016-01-01` (or 75th percentile date)\n")
        f.write(f"- **Prediction Horizon**: 12 Months post-cutoff\n\n")

        f.write("## 4. Key Takeaways & ML Pipeline Recommendations\n\n")
        f.write("1. **Relational Linkage**: Devices link to Manufacturers via `manufacturer_id` and Events link to Devices via `device_id`.\n")
        f.write("2. **Device Categorization**: Map raw `name` & `classification` into major clinical types (Monitoring, Respiratory, Pumps, Imaging, Diagnostic, Surgical, Implantable, Other).\n")
        f.write("3. **Target Imbalance**: Out of all devices, a fraction experience events in the future prediction window. XGBoost `scale_pos_weight` will handle class imbalance.\n")
        f.write("4. **Synthetic Telemetry Requirement**: Synthetic telemetry will be generated for monitored devices with realistic health degradation curves.\n")

    print(f"\nEDA Report written to {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    inspect_data()
