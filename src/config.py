"""
Central Configuration for Medical Equipment Failure Prediction Agent
"""
import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Reproducibility
RANDOM_SEED = 42

# Temporal Split Configuration
TEMPORAL_CUTOFF_DATE = "2016-01-01"
PREDICTION_HORIZON_MONTHS = 24  # Evaluate future events from 2016-01-01 to 2018-01-01

# Subsampling for Hackathon Performance (representative subset)
SAMPLE_SIZE = 10000  # Number of devices to include in analytical dataset

# Synthetic Telemetry Settings
TELEMETRY_TIMESTEPS_PER_DEVICE = 50
SIMULATED_DEGRADATION_RATIO = 0.25  # ~25% of devices exhibit degradation trajectories

# Risk Thresholds (0-100)
RISK_LEVEL_THRESHOLDS = {
    "LOW": (0, 29),
    "MODERATE": (30, 59),
    "HIGH": (60, 79),
    "CRITICAL": (80, 100)
}

# Maintenance Priority Score Weights
PRIORITY_WEIGHTS = {
    "failure_risk": 0.40,
    "anomaly_score": 0.25,
    "risk_class": 0.15,
    "historical_safety": 0.10,
    "quantity_commerce": 0.10
}

# Risk Class Weights (for priority score calculation)
RISK_CLASS_MAP = {
    "Class III": 4,
    "IVD Class III": 4,
    "Class IIB": 3,
    "IVD Class II": 3,
    "Class IIA": 2,
    "IVD Other": 2,
    "Class I": 1,
    "IVD Class I": 1,
    "Unknown": 1
}

# Device Categories Mapping Keywords
DEVICE_CATEGORY_KEYWORDS = {
    "Respiratory": ["ventilator", "respiratory", "oxygen", "airway", "cpap", "breathing", "lung"],
    "Infusion": ["pump", "infusion", "catheter", "syringe", "iv ", "flush", "injection"],
    "Monitoring": ["monitor", "ecg", "pulse", "oximeter", "blood pressure", "telemetry", "cardiac"],
    "Imaging": ["ultrasound", "x-ray", "mri", "ct ", "scanner", "imaging", "fluoroscopy"],
    "Diagnostic": ["glucose", "blood", "test", "analyzer", "assay", "reagent", "strip", "probe"],
    "Surgical": ["surgical", "stent", "suture", "scalpel", "endoscope", "laser", "blade", "forceps"],
    "Implantable": ["implant", "intraocular", "pacemaker", "prosthesis", "graft", "valve", "lens"]
}
