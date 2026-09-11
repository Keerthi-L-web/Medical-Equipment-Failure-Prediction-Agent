"""
Anomaly Detection Module for Medical Equipment Failure Prediction Agent
Uses Isolation Forest on continuous telemetry signals to calculate a normalized anomaly score (0-100).
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from .config import RANDOM_SEED, MODELS_DIR

ANOMALY_FEATURE_COLS = [
    'temperature', 'vibration', 'power_consumption', 
    'voltage', 'error_rate', 'battery_health'
]

class TelemetryAnomalyDetector:
    def __init__(self, contamination: float = 0.15):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        self.scaler = MinMaxScaler(feature_range=(0, 100))
        self.feature_cols = ANOMALY_FEATURE_COLS

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """Fit Isolation Forest on telemetry features and return normalized 0-100 anomaly scores."""
        X = df[self.feature_cols].copy().fillna(0)
        
        # Isolation Forest decision_function returns negative values for anomalies
        self.model.fit(X)
        raw_scores = self.model.decision_function(X)
        
        # Invert score so higher = more anomalous
        inverted_scores = -raw_scores
        
        # Scale to 0-100 range
        scaled_scores = self.scaler.fit_transform(inverted_scores.reshape(-1, 1)).flatten()
        return pd.Series(np.clip(scaled_scores, 0.0, 100.0).round(1), index=df.index)

    def predict_score(self, df: pd.DataFrame) -> pd.Series:
        """Compute anomaly score (0-100) for new telemetry samples."""
        X = df[self.feature_cols].copy().fillna(0)
        raw_scores = self.model.decision_function(X)
        inverted_scores = -raw_scores
        scaled_scores = self.scaler.transform(inverted_scores.reshape(-1, 1)).flatten()
        return pd.Series(np.clip(scaled_scores, 0.0, 100.0).round(1), index=df.index)

    def save(self, filepath: str = None):
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, "anomaly_model.pkl")
        joblib.dump(self, filepath)
        print(f"[AnomalyDetector] Model saved to {filepath}")

    @staticmethod
    def load(filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "anomaly_model.pkl")
        detector = joblib.load(filepath)
        print(f"[AnomalyDetector] Model loaded from {filepath}")
        return detector
