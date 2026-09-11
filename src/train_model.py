"""
ML Model Training Module for Medical Equipment Failure Prediction Agent
Trains XGBoost Classifier with scale_pos_weight for imbalanced failure prediction,
evaluates performance metrics, and saves model artifacts.
"""
import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, accuracy_score
)
from .config import RANDOM_SEED, MODELS_DIR, REPORTS_DIR

FEATURE_COLUMNS = [
    # Device features
    'risk_class_encoded', 'device_age_years', 'is_implanted', 'quantity_in_commerce_log',
    # Historical event features
    'previous_event_count', 'events_last_1_year', 'events_last_3_years', 'events_last_5_years',
    'previous_recall_count', 'previous_high_severity_event_count', 'days_since_last_event', 'recent_event_frequency',
    # Manufacturer features
    'manufacturer_device_count', 'manufacturer_event_count', 'manufacturer_recall_count', 'manufacturer_event_rate',
    # Telemetry & Anomaly features
    'temperature', 'vibration', 'power_consumption', 'voltage', 'operating_hours',
    'error_rate', 'battery_health', 'temperature_change_rate', 'vibration_change_rate',
    'power_change_rate', 'voltage_variability', 'telemetry_anomaly_score'
]

def prepare_feature_matrix(df: pd.DataFrame, expected_columns=None):
    """Clean and encode features for model consumption."""
    df_clean = df.copy()

    # One-hot encode device_type if present
    if 'device_type' in df_clean.columns:
        dev_type_dummies = pd.get_dummies(df_clean['device_type'], prefix='devtype', dtype=int)
        df_clean = pd.concat([df_clean, dev_type_dummies], axis=1)

    if expected_columns is not None:
        X = df_clean.reindex(columns=expected_columns, fill_value=0).fillna(0)
        return X

    # Available columns
    available_cols = [c for c in FEATURE_COLUMNS if c in df_clean.columns]
    # Add dummy columns for device types
    type_cols = [c for c in df_clean.columns if c.startswith('devtype_')]
    final_cols = available_cols + type_cols

    X = df_clean[final_cols].fillna(0)
    return X, final_cols

def train_and_evaluate_xgboost(master_df: pd.DataFrame, target_col: str = 'future_event'):
    """
    Train XGBoost model, evaluate metrics, and save model artifacts.
    """
    print("=" * 60)
    print(f"[ModelTraining] Training XGBoost Classifier on Target: {target_col}")
    print("=" * 60)

    X, feature_names = prepare_feature_matrix(master_df)
    y = master_df[target_col].astype(int)

    # Class balance check
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    pos_weight = (num_neg / max(1, num_pos))
    print(f"[ModelTraining] Dataset Size: {len(X):,} samples | Positive: {num_pos:,} | Negative: {num_neg:,} | Scale Pos Weight: {pos_weight:.2f}")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    # XGBoost Classifier
    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_SEED,
        eval_metric='logloss',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions & Probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    pr_auc = average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- MODEL PERFORMANCE EVALUATION ---")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall (Sensitivity): {rec:.4f}  <-- Priority Metric")
    print(f"F1-Score:        {f1:.4f}")
    print(f"ROC-AUC:         {roc_auc:.4f}")
    print(f"PR-AUC:          {pr_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Feature Importance Top 10
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print("\nTop 10 Feature Importances:")
    print(feat_imp.head(10).to_string(index=False))

    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    cols_path = os.path.join(MODELS_DIR, "feature_columns.pkl")

    joblib.dump(model, model_path)
    joblib.dump(feature_names, cols_path)
    print(f"\n[ModelTraining] Model saved to {model_path}")
    print(f"[ModelTraining] Feature names saved to {cols_path}")

    # Generate Evaluation Markdown Report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "model_evaluation.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# XGBoost Predictive Maintenance Model Evaluation\n\n")
        f.write("## 1. Class Distribution\n\n")
        f.write(f"- Total Training Samples: `{len(X_train):,}`\n")
        f.write(f"- Total Testing Samples: `{len(X_test):,}`\n")
        f.write(f"- Positive Class Count (Test): `{num_pos:,}`\n")
        f.write(f"- Negative Class Count (Test): `{num_neg:,}`\n")
        f.write(f"- Imbalance Weight (`scale_pos_weight`): `{pos_weight:.2f}`\n\n")

        f.write("## 2. Evaluation Metrics\n\n")
        f.write(f"| Metric | Value |\n| --- | --- |\n")
        f.write(f"| **Accuracy** | {acc:.4f} |\n")
        f.write(f"| **Precision** | {prec:.4f} |\n")
        f.write(f"| **Recall** (Priority) | **{rec:.4f}** |\n")
        f.write(f"| **F1-Score** | {f1:.4f} |\n")
        f.write(f"| **ROC-AUC** | {roc_auc:.4f} |\n")
        f.write(f"| **PR-AUC** | {pr_auc:.4f} |\n\n")

        f.write("## 3. Confusion Matrix\n\n")
        f.write("```\n")
        f.write(f"[[ True Negatives: {cm[0][0]:<5}  False Positives: {cm[0][1]:<5} ]\n")
        f.write(f" [ False Negatives: {cm[1][0]:<5} True Positives:  {cm[1][1]:<5} ]]\n")
        f.write("```\n\n")

        f.write("## 4. Top Feature Importances\n\n")
        for _, row in feat_imp.head(10).iterrows():
            f.write(f"- `{row['feature']}`: {row['importance']:.4f}\n")

    print(f"[ModelTraining] Evaluation report written to {report_path}")
    return model, feature_names, {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'cm': cm
    }
