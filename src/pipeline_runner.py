"""
End-to-End Master Pipeline Runner for Medical Equipment Failure Prediction Agent
Executes: Data Loading -> Preprocessing -> Temporal Feature Engineering -> Target Creation
-> Synthetic Telemetry Generation -> Anomaly Detection -> XGBoost Training -> Artifact Saving.
"""
import os
import pandas as pd
from .config import PROCESSED_DATA_DIR, SAMPLE_SIZE
from .data_loader import load_raw_data
from .preprocessing import clean_and_derive_device_features
from .feature_engineering import create_historical_and_target_features
from .synthetic_telemetry import generate_synthetic_telemetry, engineer_telemetry_features
from .anomaly_detection import TelemetryAnomalyDetector
from .train_model import train_and_evaluate_xgboost

def run_pipeline(sample_size: int = SAMPLE_SIZE):
    print("=" * 70)
    print("RUNNING END-TO-END MEDICAL EQUIPMENT FAILURE PREDICTION PIPELINE")
    print("=" * 70)

    # Step 1: Load Raw Data
    dev_df, evt_df, mfr_df = load_raw_data()

    # Subsample for hackathon performance if needed
    if sample_size and sample_size < len(dev_df):
        print(f"[Pipeline] Subsampling {sample_size:,} representative devices out of {len(dev_df):,} total devices...")
        dev_df = dev_df.head(sample_size).copy()

    # Step 2: Clean & Preprocess Device & Manufacturer Features
    print("[Pipeline] Step 2: Cleaning devices & merging manufacturer records...")
    dev_df = clean_and_derive_device_features(dev_df, mfr_df)

    # Step 3: Create Pre-Cutoff Historical Features & Target `future_event`
    print("[Pipeline] Step 3: Computing pre-cutoff temporal features & target variable...")
    master_df = create_historical_and_target_features(dev_df, evt_df)

    # Step 4: Generate Synthetic Telemetry Time Series
    print("[Pipeline] Step 4: Generating physics-based synthetic telemetry...")
    telemetry_df = generate_synthetic_telemetry(master_df, num_devices=len(master_df))

    # Step 5: Engineer Telemetry Features & Fit Anomaly Detector
    print("[Pipeline] Step 5: Engineering telemetry features & fitting Isolation Forest...")
    telemetry_features_df = engineer_telemetry_features(telemetry_df)
    
    detector = TelemetryAnomalyDetector()
    telemetry_features_df['telemetry_anomaly_score'] = detector.fit_predict(telemetry_features_df)
    detector.save()

    # Step 6: Merge Telemetry Features with Master Analytical Dataset
    print("[Pipeline] Step 6: Constructing Master Analytical Dataset...")
    master_df = master_df.merge(telemetry_features_df, left_on='id', right_on='device_id', how='left')

    # Step 7: Train & Evaluate XGBoost Classifier
    print("[Pipeline] Step 7: Training & evaluating XGBoost classifier...")
    model, feat_names, metrics = train_and_evaluate_xgboost(master_df, target_col='future_event')

    # Step 8: Precompute Predictions & Composite Scores
    print("[Pipeline] Step 8: Precomputing predictions, risk levels & maintenance priorities...")
    from .train_model import prepare_feature_matrix
    from .risk_engine import calculate_risk_and_priority

    X_full = prepare_feature_matrix(master_df, expected_columns=feat_names)
    probs = model.predict_proba(X_full)[:, 1]

    records = []
    for idx, row in master_df.iterrows():
        p = probs[idx]
        anom = row.get('telemetry_anomaly_score', 0.0)
        rc = row.get('risk_class_encoded', 1)
        prev_evt = row.get('previous_event_count', 0)
        qty_log = row.get('quantity_in_commerce_log', 0.0)

        calc = calculate_risk_and_priority(p, anom, rc, prev_evt, qty_log)
        records.append(calc)

    res_df = pd.DataFrame(records)
    for col in res_df.columns:
        master_df[col] = res_df[col].values

    # Save Master Analytical Dataset
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    master_out_path = os.path.join(PROCESSED_DATA_DIR, "master_dataset.csv")
    master_df.to_csv(master_out_path, index=False)
    print(f"[Pipeline] Master analytical dataset saved to {master_out_path} ({len(master_df):,} rows)")

    print("=" * 70)
    print("PIPELINE EXECUTION COMPLETE SUCCESSFULLY!")
    print(f"Master Dataset: {master_out_path}")
    print(f"XGBoost ROC-AUC: {metrics['roc_auc']:.4f} | Recall: {metrics['recall']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
