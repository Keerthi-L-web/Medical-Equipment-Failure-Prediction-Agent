"""
Preprocessing Module for Medical Equipment Failure Prediction Agent
Handles data cleaning, normalization, device type categorization, and feature derivation.
"""
import pandas as pd
import numpy as np
from .config import DEVICE_CATEGORY_KEYWORDS, RISK_CLASS_MAP

def infer_device_type(row) -> str:
    """Categorize device into major clinical types based on name, classification, and code."""
    text_corpus = f"{str(row.get('name', ''))} {str(row.get('classification', ''))} {str(row.get('code', ''))}".lower()
    
    for category, keywords in DEVICE_CATEGORY_KEYWORDS.items():
        if any(kw in text_corpus for kw in keywords):
            return category
            
    # Check implanted flag fallback
    if pd.notnull(row.get('implanted')) and str(row.get('implanted')).lower() in ['true', '1', 'yes']:
        return "Implantable"
        
    return "Other"

def clean_and_derive_device_features(devices_df: pd.DataFrame, manufacturers_df: pd.DataFrame) -> pd.DataFrame:
    """Clean devices dataset and compute derived features."""
    df = devices_df.copy()

    # 1. Device Type Categorization
    df['device_type'] = df.apply(infer_device_type, axis=1)

    # 2. Device Age in Years
    if 'created_at' in df.columns:
        created_dt = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
        # Reference snapshot year 2018 (end of events period)
        ref_date = pd.to_datetime('2018-01-01')
        df['device_age_years'] = (ref_date - created_dt).dt.days / 365.25
        df['device_age_years'] = df['device_age_years'].clip(lower=0).fillna(df['device_age_years'].median()).round(1)
    else:
        df['device_age_years'] = 5.0

    # 3. Implanted status
    df['is_implanted'] = df['implanted'].apply(
        lambda x: 1 if pd.notnull(x) and str(x).strip().lower() in ['true', '1', 'yes', 'y'] else 0
    )

    # 4. Quantity in commerce log scale
    df['quantity_in_commerce'] = pd.to_numeric(df['quantity_in_commerce'], errors='coerce').fillna(0)
    df['quantity_in_commerce_log'] = np.log1p(np.maximum(0, df['quantity_in_commerce'])).round(2)

    # 5. Risk Class Encoding
    def extract_risk_class(row) -> str:
        text_corpus = f"{str(row.get('risk_class', ''))} {str(row.get('description', ''))} {str(row.get('classification', ''))}".strip()
        text_lower = text_corpus.lower()

        if 'class iii' in text_lower or 'class 3' in text_lower:
            return 'Class III'
        elif 'class iib' in text_lower or 'class 2b' in text_lower:
            return 'Class IIB'
        elif 'class iia' in text_lower or 'class 2a' in text_lower:
            return 'Class IIA'
        elif 'class ii' in text_lower or 'class 2' in text_lower:
            return 'Class IIA'
        elif 'ivd' in text_lower:
            return 'IVD Other'
        elif 'class i' in text_lower or 'class 1' in text_lower:
            return 'Class I'
        else:
            name_lower = str(row.get('name', '')).lower()
            if any(w in name_lower for w in ['stent', 'implant', 'valve', 'pacemaker', 'catheter', 'intraocular']):
                return 'Class III'
            elif any(w in name_lower for w in ['monitor', 'pump', 'ventilator', 'laser', 'suction']):
                return 'Class IIB'
            else:
                return 'Class IIA'

    df['risk_class_clean'] = df.apply(extract_risk_class, axis=1)
    df['risk_class_encoded'] = df['risk_class_clean'].map(RISK_CLASS_MAP).fillna(2).astype(int)

    # 6. Manufacturer Relational Integration
    mfr_counts = df['manufacturer_id'].value_counts().to_dict()
    df['manufacturer_device_count'] = df['manufacturer_id'].map(mfr_counts).fillna(1).astype(int)

    # Merge manufacturer parent company if available
    mfr_clean = manufacturers_df[['id', 'name', 'parent_company']].copy()
    mfr_clean.rename(columns={'id': 'manufacturer_id', 'name': 'manufacturer_name'}, inplace=True)
    df = df.merge(mfr_clean, on='manufacturer_id', how='left')

    return df
