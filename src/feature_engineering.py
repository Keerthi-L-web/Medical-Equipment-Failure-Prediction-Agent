"""
Feature Engineering Module for Medical Equipment Failure Prediction Agent
Constructs strictly temporal historical features pre-cutoff date to avoid data leakage,
and generates the ground truth target `future_event`.
"""
import pandas as pd
import numpy as np
from .config import TEMPORAL_CUTOFF_DATE, PREDICTION_HORIZON_MONTHS

def create_historical_and_target_features(
    devices_df: pd.DataFrame, 
    events_df: pd.DataFrame, 
    cutoff_date_str: str = TEMPORAL_CUTOFF_DATE,
    horizon_months: int = PREDICTION_HORIZON_MONTHS
) -> pd.DataFrame:
    """
    Construct temporal pre-cutoff features and target variable.
    """
    cutoff_date = pd.to_datetime(cutoff_date_str)
    horizon_end_date = cutoff_date + pd.DateOffset(months=horizon_months)

    print(f"[FeatureEngineering] Temporal Cutoff Date: {cutoff_date_str}")
    print(f"[FeatureEngineering] Future Target Horizon: {cutoff_date_str} to {horizon_end_date.strftime('%Y-%m-%d')}")

    evt = events_df.copy()
    evt['event_dt'] = pd.to_datetime(evt['date'], errors='coerce', utc=True).dt.tz_localize(None)

    # Split events into PRE-CUTOFF (history) and POST-CUTOFF (target evaluation)
    pre_cutoff_mask = evt['event_dt'].notnull() & (evt['event_dt'] <= cutoff_date)
    post_cutoff_mask = evt['event_dt'].notnull() & (evt['event_dt'] > cutoff_date) & (evt['event_dt'] <= horizon_end_date)

    hist_evt = evt[pre_cutoff_mask].copy()
    future_evt = evt[post_cutoff_mask].copy()

    print(f"[FeatureEngineering] Historical Pre-Cutoff Events: {len(hist_evt):,}")
    print(f"[FeatureEngineering] Future Post-Cutoff Events: {len(future_evt):,}")

    # 1. Historical Event Aggregations per Device
    hist_evt['is_recall'] = hist_evt['action_summary'].fillna('').astype(str).str.contains('Recall', case=False).astype(int)
    hist_evt['is_high_severity'] = (
        hist_evt['action_level'].fillna('').astype(str).str.contains('High|Class I|Class 1|Level 1', case=False) |
        hist_evt['action_summary'].fillna('').astype(str).str.contains('Recall', case=False)
    ).astype(int)

    # Date deltas relative to cutoff
    hist_evt['days_before_cutoff'] = (cutoff_date - hist_evt['event_dt']).dt.days

    # Pre-cutoff time windows
    hist_evt['in_last_1_yr'] = (hist_evt['days_before_cutoff'] <= 365).astype(int)
    hist_evt['in_last_3_yr'] = (hist_evt['days_before_cutoff'] <= 365 * 3).astype(int)
    hist_evt['in_last_5_yr'] = (hist_evt['days_before_cutoff'] <= 365 * 5).astype(int)

    # Group by device_id
    dev_hist_grp = hist_evt.groupby('device_id')

    hist_features = pd.DataFrame({
        'previous_event_count': dev_hist_grp.size(),
        'events_last_1_year': dev_hist_grp['in_last_1_yr'].sum(),
        'events_last_3_years': dev_hist_grp['in_last_3_yr'].sum(),
        'events_last_5_years': dev_hist_grp['in_last_5_yr'].sum(),
        'previous_recall_count': dev_hist_grp['is_recall'].sum(),
        'previous_high_severity_event_count': dev_hist_grp['is_high_severity'].sum(),
        'min_days_before_cutoff': dev_hist_grp['days_before_cutoff'].min(),
    }).reset_index()

    hist_features['days_since_last_event'] = hist_features['min_days_before_cutoff'].fillna(3650)
    hist_features.drop(columns=['min_days_before_cutoff'], inplace=True)

    hist_features['recent_event_frequency'] = np.where(
        hist_features['previous_event_count'] > 0,
        hist_features['events_last_1_year'] / hist_features['previous_event_count'],
        0.0
    )

    # Merge historical features into main devices dataframe
    df = devices_df.merge(hist_features, left_on='id', right_on='device_id', how='left')
    df.drop(columns=['device_id'], inplace=True, errors='ignore')

    # Fill missing historical event metrics with 0 (meaning no pre-cutoff events recorded)
    hist_cols = [
        'previous_event_count', 'events_last_1_year', 'events_last_3_years',
        'events_last_5_years', 'previous_recall_count', 'previous_high_severity_event_count',
        'recent_event_frequency'
    ]
    df[hist_cols] = df[hist_cols].fillna(0)
    df['days_since_last_event'] = df['days_since_last_event'].fillna(3650)

    # 2. Manufacturer Pre-Cutoff Aggregations
    mfr_hist_grp = df.groupby('manufacturer_id').agg(
        manufacturer_event_count=('previous_event_count', 'sum'),
        manufacturer_recall_count=('previous_recall_count', 'sum'),
    ).reset_index()

    df = df.merge(mfr_hist_grp, on='manufacturer_id', how='left')
    df['manufacturer_event_rate'] = (
        df['manufacturer_event_count'] / np.maximum(1, df['manufacturer_device_count'])
    ).round(3)

    # 3. Ground Truth Target Creation (`future_event`)
    future_device_ids = set(future_evt['device_id'].unique())
    df['future_event'] = df['id'].isin(future_device_ids).astype(int)

    pos_count = df['future_event'].sum()
    total_count = len(df)
    print(f"[FeatureEngineering] Master Target Distribution: {pos_count:,} positive future events out of {total_count:,} devices ({pos_count / total_count:.2%})")

    return df
