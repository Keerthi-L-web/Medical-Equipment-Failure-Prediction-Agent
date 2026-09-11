"""
SHAP Explainability Module for Medical Equipment Failure Prediction Agent
Calculates tree-based SHAP values for XGBoost predictions and formats them into human-readable risk drivers.
"""
import shap
import numpy as np
import pandas as pd

FEATURE_LABEL_MAP = {
    'temperature': 'Elevated Telemetry Temperature',
    'vibration': 'Abnormal Mechanical Vibration',
    'power_consumption': 'Unstable Power Consumption',
    'voltage': 'Voltage Fluctuation',
    'operating_hours': 'Accumulated Operating Hours',
    'error_rate': 'Elevated Telemetry Error Rate',
    'battery_health': 'Low Battery Health / Capacity',
    'temperature_change_rate': 'Rapid Temperature Change Rate',
    'vibration_change_rate': 'Elevated Vibration Change Rate',
    'power_change_rate': 'Power Consumption Change Rate',
    'voltage_variability': 'High Voltage Variability',
    'telemetry_anomaly_score': 'Continuous Telemetry Anomaly Score',
    'previous_event_count': 'Historical Safety Event Count',
    'events_last_1_year': 'Recent Safety Events (Past 1 Year)',
    'events_last_3_years': 'Safety Events (Past 3 Years)',
    'events_last_5_years': 'Safety Events (Past 5 Years)',
    'previous_recall_count': 'Prior Recall Notices',
    'previous_high_severity_event_count': 'Prior High-Severity Safety Alerts',
    'days_since_last_event': 'Recency of Historical Safety Event',
    'recent_event_frequency': 'Recent Safety Event Frequency',
    'device_age_years': 'Advanced Device Operating Age',
    'is_implanted': 'Implantable Device Classification',
    'risk_class_encoded': 'FDA Device Risk Classification',
    'manufacturer_device_count': 'Manufacturer Commercial Fleet Size',
    'manufacturer_event_count': 'Manufacturer Historical Event Volume',
    'manufacturer_recall_count': 'Manufacturer Recall History',
    'manufacturer_event_rate': 'Manufacturer Historical Event Rate',
    'quantity_in_commerce_log': 'Commercial Distribution Scale'
}

def get_feature_label(feat: str) -> str:
    if feat in FEATURE_LABEL_MAP:
        return FEATURE_LABEL_MAP[feat]
    if feat.startswith('devtype_'):
        return f"Device Category: {feat.replace('devtype_', '')}"
    return feat.replace('_', ' ').title()

from .train_model import prepare_feature_matrix

class FailureRiskExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain_sample(self, sample_df: pd.DataFrame, top_n: int = 5) -> dict:
        """
        Compute SHAP values for a single device row or DataFrame slice.
        Returns top positive risk contributors and negative risk mitigators with non-causal rationale.
        """
        X_sample = prepare_feature_matrix(sample_df, expected_columns=self.feature_names)
        shap_vals = self.explainer.shap_values(X_sample)

        if isinstance(shap_vals, list):
            vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif len(shap_vals.shape) == 2:
            vals = shap_vals[0]
        else:
            vals = shap_vals

        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'label': [get_feature_label(f) for f in self.feature_names],
            'shap_value': vals,
            'abs_shap': np.abs(vals),
            'feature_value': X_sample.iloc[0].values
        }).sort_values('abs_shap', ascending=False)

        top_positive = contributions[contributions['shap_value'] > 0].sort_values('shap_value', ascending=False).head(top_n)
        top_negative = contributions[contributions['shap_value'] < 0].sort_values('shap_value', ascending=True).head(top_n)

        # Generate human-readable text bullets (non-causal wording)
        pos_reasons = []
        for _, row in top_positive.iterrows():
            pos_reasons.append(f"{row['label']} (contributed +{row['shap_value']:.2f} to predicted risk)")

        neg_reasons = []
        for _, row in top_negative.iterrows():
            neg_reasons.append(f"{row['label']} (contributed {row['shap_value']:.2f} toward risk reduction)")

        summary_text = (
            f"Top risk contributors: {', '.join(top_positive['label'].head(3))}."
            if not top_positive.empty else "All telemetry parameters and historical records within normal operating baselines."
        )

        return {
            'shap_values': dict(zip(contributions['feature'], contributions['shap_value'])),
            'top_positive': top_positive.to_dict(orient='records'),
            'top_negative': top_negative.to_dict(orient='records'),
            'text_reasons': pos_reasons,
            'text_reasons_pos': pos_reasons,
            'text_reasons_neg': neg_reasons,
            'summary_text': summary_text
        }
