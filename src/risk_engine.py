"""
Risk Engine Module for Medical Equipment Failure Prediction Agent
Calculates failure risk score (0-100), risk levels, and maintenance priority score (0-100).
"""
import pandas as pd
import numpy as np
from .config import RISK_LEVEL_THRESHOLDS, PRIORITY_WEIGHTS, RISK_CLASS_MAP

def get_risk_level(score: float) -> str:
    """Classify 0-100 risk score into LOW, MODERATE, HIGH, CRITICAL."""
    if score >= 80.0:
        return "CRITICAL"
    elif score >= 60.0:
        return "HIGH"
    elif score >= 30.0:
        return "MODERATE"
    else:
        return "LOW"

def get_priority_level(score: float) -> str:
    """Classify 0-100 priority score into LOW, MODERATE, HIGH, CRITICAL."""
    if score >= 80.0:
        return "CRITICAL"
    elif score >= 60.0:
        return "HIGH"
    elif score >= 30.0:
        return "MODERATE"
    else:
        return "LOW"

def calculate_risk_and_priority(
    failure_probability: float,
    anomaly_score: float,
    risk_class_encoded: int,
    previous_event_count: int,
    quantity_in_commerce_log: float
) -> dict:
    """
    Calculate composite failure risk score and maintenance priority score.
    """
    # 1. Failure Risk Score (0-100)
    failure_risk_score = round(float(np.clip(failure_probability * 100.0, 0.0, 100.0)), 1)
    risk_level = get_risk_level(failure_risk_score)

    # 2. Risk Class Score component
    rc_score = float(np.clip(risk_class_encoded * 25.0, 25.0, 100.0))

    # 3. Historical Safety Score component
    safety_score = float(np.clip(previous_event_count * 20.0, 0.0, 100.0))

    # 4. Quantity score component
    qty_score = float(np.clip(quantity_in_commerce_log * 10.0, 0.0, 100.0))

    # 5. Weighted Maintenance Priority Formula using PRIORITY_WEIGHTS
    w = PRIORITY_WEIGHTS
    raw_priority = (
        (failure_risk_score * w.get('failure_risk', 0.40)) +
        (anomaly_score * w.get('anomaly_score', 0.25)) +
        (rc_score * w.get('risk_class', 0.15)) +
        (safety_score * w.get('historical_safety', 0.10)) +
        (qty_score * w.get('quantity_commerce', 0.10))
    )

    priority_score = round(float(np.clip(raw_priority, 0.0, 100.0)), 1)
    priority_level = get_priority_level(priority_score)

    return {
        'failure_risk_score': failure_risk_score,
        'risk_level': risk_level,
        'anomaly_score': round(float(anomaly_score), 1),
        'maintenance_priority_score': priority_score,
        'priority_level': priority_level
    }
