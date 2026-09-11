"""
Streamlit Dashboard for Medical Equipment Failure Prediction Agent
Interactive multi-view dashboard with real-time telemetry simulation demo.
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.anomaly_detection import TelemetryAnomalyDetector
from src.explainability import FailureRiskExplainer
from src.risk_engine import calculate_risk_and_priority, get_risk_level, get_priority_level
from src.maintenance_agent import AIMaintenanceAgent
from src.train_model import prepare_feature_matrix

import joblib

# Page Configuration
st.set_page_config(
    page_title="Medical Equipment Failure Prediction Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card-critical {
        border-left-color: #EF4444;
        background-color: #FEF2F2;
    }
    .metric-card-warning {
        border-left-color: #F59E0B;
        background-color: #FFFBEB;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-moderate {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .safeguard-banner {
        background-color: #FFF7ED;
        border: 2px solid #F97316;
        border-radius: 8px;
        padding: 1rem;
        color: #7C2D12;
        font-weight: 600;
        margin-top: 1rem;
    }
    .synthetic-disclaimer {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #1E40AF;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_datasets():
    master_path = os.path.join(PROCESSED_DATA_DIR, "master_dataset.csv")
    telemetry_path = os.path.join(PROCESSED_DATA_DIR, "synthetic_telemetry.csv")

    if not os.path.exists(master_path):
        st.error(f"Master dataset not found at {master_path}. Please run `python -m src.pipeline_runner` first.")
        st.stop()

    master_df = pd.read_csv(master_path)
    telemetry_df = pd.read_csv(telemetry_path) if os.path.exists(telemetry_path) else None
    return master_df, telemetry_df

@st.cache_resource
def load_models():
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    cols_path = os.path.join(MODELS_DIR, "feature_columns.pkl")

    if not os.path.exists(model_path):
        st.error("Trained XGBoost model not found. Please run model training pipeline.")
        st.stop()

    model = joblib.load(model_path)
    feature_names = joblib.load(cols_path)
    return model, feature_names

# Load Data & Models
master_df, telemetry_df = load_datasets()
xgb_model, feature_names = load_models()
explainer = FailureRiskExplainer(xgb_model, feature_names)
maintenance_agent = AIMaintenanceAgent()

# Precompute Predictions & Scores for Master Fleet Table
@st.cache_data
def compute_fleet_scores(_df, _model_cols):
    req_cols = ['failure_risk_score', 'risk_level', 'anomaly_score', 'maintenance_priority_score', 'priority_level']
    if all(col in _df.columns for col in req_cols):
        return _df.copy()

    df_calc = _df.copy()
    X = prepare_feature_matrix(df_calc, expected_columns=_model_cols)
    probs = xgb_model.predict_proba(X)[:, 1]
    
    records = []
    for idx, row in df_calc.iterrows():
        p = probs[idx]
        anom = row.get('telemetry_anomaly_score', 0.0)
        rc = row.get('risk_class_encoded', 1)
        prev_evt = row.get('previous_event_count', 0)
        qty_log = row.get('quantity_in_commerce_log', 0.0)

        calc = calculate_risk_and_priority(p, anom, rc, prev_evt, qty_log)
        records.append(calc)

    res_df = pd.DataFrame(records)
    for col in res_df.columns:
        df_calc[col] = res_df[col].values
    return df_calc

fleet_df = compute_fleet_scores(master_df, feature_names)

# Global Session State for Device Selection across Pages
device_options = fleet_df['id'].tolist()
if 'selected_device_id' not in st.session_state or st.session_state['selected_device_id'] not in device_options:
    st.session_state['selected_device_id'] = device_options[0]

# Navigation Sidebar
st.sidebar.image("https://img.icons8.com/color/96/medical-heart.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select System View",
    ["Fleet Risk Overview", "Device Deep Dive & SHAP", "AI Maintenance Agent", "Live Telemetry Simulator Demo"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.success("XGBoost Engine: Active")
st.sidebar.success("Isolation Forest: Active")
st.sidebar.info("Data Pipeline: Synced")

# HEADER & DISCLAIMER BANNER
st.markdown('<div class="main-header">🏥 Medical Equipment Failure Prediction Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predictive Maintenance & Safety Decision-Support System</div>', unsafe_allow_html=True)

st.markdown("""
<div class="synthetic-disclaimer">
    ℹ️ <strong>Transparency Notice</strong>: Historical device, manufacturer, and recall event data are loaded from official regulatory records. 
    To demonstrate real-time continuous sensor monitoring requested by the specification, multi-signal sensor telemetry is simulated using controlled physics degradation trajectories.
</div>
""", unsafe_allow_html=True)

# VIEW 1: FLEET RISK OVERVIEW
if page == "Fleet Risk Overview":
    st.subheader("📊 Fleet Monitoring & Failure Risk Summary")

    total_devices = len(fleet_df)
    critical_cnt = int((fleet_df['risk_level'] == 'CRITICAL').sum())
    high_cnt = int((fleet_df['risk_level'] == 'HIGH').sum())
    mod_cnt = int((fleet_df['risk_level'] == 'MODERATE').sum())
    low_cnt = int((fleet_df['risk_level'] == 'LOW').sum())

    anom_cnt = int((fleet_df['anomaly_score'] >= 60.0).sum())
    priority_alerts = int((fleet_df['maintenance_priority_score'] >= 60.0).sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Fleet Devices", f"{total_devices:,}")
    col2.metric("Critical Risk (>=80%)", f"{critical_cnt:,}", delta_color="inverse")
    col3.metric("High Risk (60-79%)", f"{high_cnt:,}", delta_color="inverse")
    col4.metric("Active Anomaly Alerts", f"{anom_cnt:,}")
    col5.metric("Maintenance Alerts", f"{priority_alerts:,}")

    st.markdown("---")

    # Filters
    st.subheader("📋 Monitored Equipment Fleet Table")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    selected_risk = f_col1.multiselect(
        "Filter by Risk Level",
        ["CRITICAL", "HIGH", "MODERATE", "LOW"],
        default=["CRITICAL", "HIGH", "MODERATE", "LOW"]
    )
    all_types = sorted(fleet_df['device_type'].unique().tolist())
    selected_type = f_col2.multiselect(
        "Filter by Device Category",
        all_types,
        default=all_types
    )
    search_query = f_col3.text_input("Search Device Name / Code / ID", "")

    # STRICT DATA FILTERING (NO DATA MUTATION OR REASSIGNMENT)
    # The risk_level and failure_risk_score are calculated once during pipeline execution.
    # Selecting filter options strictly filters rows matching selected_risk without altering any stored device values.
    filtered_df = fleet_df.copy()
    if selected_risk:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(selected_risk)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    if selected_type:
        filtered_df = filtered_df[filtered_df['device_type'].isin(selected_type)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df['name'].astype(str).str.lower().str.contains(q) |
            filtered_df['id'].astype(str).str.lower().str.contains(q)
        ]

    st.caption(f"Showing **{len(filtered_df):,}** matching devices out of **{total_devices:,}** total fleet devices.")

    if filtered_df.empty:
        st.info("ℹ️ No equipment records match the active filter criteria. Select at least one Risk Level / Category or clear search query.")
    else:
        disp_cols = [
            'id', 'name', 'device_type', 'risk_class_clean', 'failure_risk_score',
            'risk_level', 'anomaly_score', 'maintenance_priority_score', 'priority_level'
        ]
        
        display_df = filtered_df[disp_cols].copy()
        display_df.rename(columns={
            'id': 'Device ID',
            'name': 'Device Name',
            'device_type': 'Category',
            'risk_class_clean': 'FDA Class',
            'failure_risk_score': 'Failure Risk (%)',
            'risk_level': 'Risk Level',
            'anomaly_score': 'Anomaly Score',
            'maintenance_priority_score': 'Priority Score',
            'priority_level': 'Priority Level'
        }, inplace=True)
        display_df.sort_values(by='Priority Score', ascending=False, inplace=True)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

# VIEW 2: DEVICE DEEP DIVE & SHAP
elif page == "Device Deep Dive & SHAP":
    st.subheader("🔍 Single Device Deep Dive & SHAP Risk Drivers")

    selected_dev_id = st.selectbox(
        "Select Device ID to Inspect",
        device_options,
        index=device_options.index(st.session_state['selected_device_id']),
        format_func=lambda x: f"ID: {x} | {fleet_df[fleet_df['id']==x]['name'].values[0]} ({fleet_df[fleet_df['id']==x]['risk_level'].values[0]})"
    )
    st.session_state['selected_device_id'] = selected_dev_id

    dev_row = fleet_df[fleet_df['id'] == selected_dev_id].iloc[0]

    # Device Specs & Metrics
    d_col1, d_col2 = st.columns([1, 2])

    with d_col1:
        st.markdown(f"### **{dev_row['name']}**")
        st.write(f"**Device ID**: `{dev_row['id']}`")
        st.write(f"**Category**: `{dev_row['device_type']}`")
        st.write(f"**FDA Risk Class**: `{dev_row['risk_class_clean']}`")
        st.write(f"**Manufacturer**: `{dev_row.get('manufacturer_name', 'Unknown')}`")
        st.write(f"**Device Age**: `{dev_row['device_age_years']} years`")
        st.write(f"**Implanted**: `{'Yes' if dev_row['is_implanted'] == 1 else 'No'}`")

        st.markdown("---")
        st.markdown("#### **Historical Safety Record**")
        st.write(f"- Total Pre-cutoff Events: `{int(dev_row['previous_event_count'])}`")
        st.write(f"- Events (Past 1 Year): `{int(dev_row['events_last_1_year'])}`")
        st.write(f"- Events (Past 3 Years): `{int(dev_row['events_last_3_years'])}`")
        st.write(f"- Prior Recalls: `{int(dev_row['previous_recall_count'])}`")
        st.write(f"- Prior High-Severity Alerts: `{int(dev_row['previous_high_severity_event_count'])}`")
        st.write(f"- Manufacturer Total Events: `{int(dev_row.get('manufacturer_event_count', 0))}`")

    with d_col2:
        st.markdown("### **Calculated Safety & Risk Metrics**")
        r_col1, r_col2, r_col3 = st.columns(3)

        risk_val = dev_row['failure_risk_score']
        risk_lvl = dev_row['risk_level']
        r_col1.metric("Predicted Failure Risk", f"{risk_val}%", delta=risk_lvl)

        anom_val = dev_row['anomaly_score']
        r_col2.metric("Telemetry Anomaly Score", f"{anom_val}/100")

        prio_val = dev_row['maintenance_priority_score']
        prio_lvl = dev_row['priority_level']
        r_col3.metric("Maintenance Priority", f"{prio_val}/100", delta=prio_lvl)

        st.markdown("---")
        st.markdown("### **🎯 SHAP Feature Importance & Risk Drivers**")

        sample_df = pd.DataFrame([dev_row])
        explanation = explainer.explain_sample(sample_df)

        pos_df = pd.DataFrame(explanation['top_positive'])
        neg_df = pd.DataFrame(explanation['top_negative'])

        t1, t2 = st.tabs(["🔴 Risk Elevating Factors", "🟢 Risk Mitigating Factors"])

        with t1:
            if not pos_df.empty:
                y_pos = pos_df['label'] if 'label' in pos_df.columns else pos_df['feature']
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.barh(y_pos, pos_df['shap_value'], color='#EF4444')
                ax.set_xlabel("SHAP Risk Contribution (+ Increase in Risk)")
                ax.set_title("Top Factors Elevating Failure Risk")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("**Detailed Risk Contributions:**")
                for reason in explanation.get('text_reasons_pos', explanation.get('text_reasons', [])):
                    st.write(f"• {reason}")
            else:
                st.success("No positive risk drivers detected. Equipment telemetry and history are optimal.")

        with t2:
            if not neg_df.empty:
                y_neg = neg_df['label'] if 'label' in neg_df.columns else neg_df['feature']
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.barh(y_neg, neg_df['shap_value'].abs(), color='#10B981')
                ax.set_xlabel("SHAP Risk Mitigation (Magnitude of Risk Reduction)")
                ax.set_title("Top Factors Reducing Failure Risk")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("**Detailed Risk Mitigations:**")
                for reason in explanation.get('text_reasons_neg', []):
                    st.write(f"• {reason}")
            else:
                st.info("No mitigating risk factors recorded.")

    # Telemetry Time-Series Chart
    if telemetry_df is not None:
        st.markdown("---")
        st.markdown("### 📈 Sensor Telemetry Trends")
        st.caption("ℹ️ Simulated telemetry for demonstration purposes.")
        dev_telemetry = telemetry_df[telemetry_df['device_id'] == selected_dev_id].sort_values('timestamp')

        if not dev_telemetry.empty:
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.line_chart(dev_telemetry.set_index('timestamp')[['temperature', 'vibration']])
            with t_col2:
                st.line_chart(dev_telemetry.set_index('timestamp')[['power_consumption', 'voltage']])

# VIEW 3: AI MAINTENANCE AGENT
elif page == "AI Maintenance Agent":
    st.subheader("🤖 AI Maintenance Agent Decision Panel")

    selected_dev_id = st.selectbox(
        "Select Target Equipment for Maintenance Decision Support",
        device_options,
        index=device_options.index(st.session_state['selected_device_id']),
        format_func=lambda x: f"ID: {x} | {fleet_df[fleet_df['id']==x]['name'].values[0]} ({fleet_df[fleet_df['id']==x]['risk_level'].values[0]})"
    )
    st.session_state['selected_device_id'] = selected_dev_id

    dev_row = fleet_df[fleet_df['id'] == selected_dev_id].iloc[0]
    sample_df = pd.DataFrame([dev_row])
    explanation = explainer.explain_sample(sample_df)

    risk_summary = {
        'risk_level': dev_row['risk_level'],
        'priority_level': dev_row['priority_level'],
        'failure_risk_score': dev_row['failure_risk_score'],
        'anomaly_score': dev_row['anomaly_score']
    }

    rec = maintenance_agent.generate_recommendation(dev_row.to_dict(), risk_summary, explanation)

    # Render Maintenance Recommendation Card
    st.markdown(f"## **Recommendation for {rec['device_name']}**")
    st.write(f"**Device Category**: `{rec['device_type']}` | **Device ID**: `{rec['device_id']}` | **FDA Class**: `{dev_row['risk_class_clean']}`")

    a_col1, a_col2, a_col3 = st.columns(3)
    a_col1.markdown(f"#### **Failure Risk**: `{rec['failure_risk_score']}%` ({rec['risk_level']})")
    a_col2.markdown(f"#### **Priority Level**: `{rec['priority_level']}` ({rec['anomaly_score']}/100 Anomaly)")
    a_col3.markdown(f"#### **Maintenance Urgency**: `{rec['urgency']}`")

    st.markdown("---")
    st.markdown("### **Primary Risk Drivers Detected**")
    for d in rec['top_risk_drivers']:
        st.write(f"⚠️ {d}")

    st.markdown("### **Recommended Actionable Servicing Steps**")
    for idx, act in enumerate(rec['recommended_actions'], 1):
        st.write(f"**{idx}.** {act}")

    st.markdown(f"""
    <div class="safeguard-banner">
        🛑 <strong>HUMAN TECHNICIAN REVIEW REQUIRED</strong><br>
        {rec['safeguard_notice']}
    </div>
    """, unsafe_allow_html=True)

# VIEW 4: LIVE TELEMETRY SIMULATOR DEMO
elif page == "Live Telemetry Simulator Demo":
    st.subheader("⚡ Live Telemetry Degradation Simulation Demo")
    st.write("Demonstrates real-time telemetry deterioration, anomaly detection escalation, failure risk calculation, and maintenance agent alert triggering.")

    if telemetry_df is None:
        st.warning("Telemetry dataset not loaded.")
        st.stop()

    selected_dev_id = st.selectbox(
        "Select Device for Live Simulation",
        device_options,
        index=device_options.index(st.session_state['selected_device_id']),
        format_func=lambda x: f"ID: {x} | {fleet_df[fleet_df['id']==x]['name'].values[0]} ({fleet_df[fleet_df['id']==x]['risk_level'].values[0]})"
    )
    st.session_state['selected_device_id'] = selected_dev_id

    dev_telemetry = telemetry_df[telemetry_df['device_id'] == selected_dev_id].sort_values('timestamp').reset_index(drop=True)

    if dev_telemetry.empty:
        st.error("No telemetry records found for selected device.")
        st.stop()

    # Step Progression Slider
    max_step = len(dev_telemetry) - 1
    step_idx = st.slider("Simulated Telemetry Timestep Progression", 0, max_step, 0)

    curr_telemetry = dev_telemetry.iloc[step_idx]

    st.markdown(f"### **Timestep {step_idx + 1} / {max_step + 1} — Timestamp: `{curr_telemetry['timestamp']}`**")

    # Render Current Telemetry Metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Temperature", f"{curr_telemetry['temperature']} °C")
    m2.metric("Vibration", f"{curr_telemetry['vibration']} g")
    m3.metric("Power Cons.", f"{curr_telemetry['power_consumption']} W")
    m4.metric("Voltage", f"{curr_telemetry['voltage']} V")
    m5.metric("Error Rate", f"{curr_telemetry['error_rate']}%")
    m6.metric("Health Score", f"{curr_telemetry['health_score']}/100")

    # Re-calculate dynamic risk for current timestep state
    base_dev_row = fleet_df[fleet_df['id'] == selected_dev_id].iloc[0].to_dict()
    base_dev_row['temperature'] = curr_telemetry['temperature']
    base_dev_row['vibration'] = curr_telemetry['vibration']
    base_dev_row['power_consumption'] = curr_telemetry['power_consumption']
    base_dev_row['voltage'] = curr_telemetry['voltage']
    base_dev_row['error_rate'] = curr_telemetry['error_rate']
    base_dev_row['battery_health'] = curr_telemetry['battery_health']

    # Compute Anomaly Score & XGBoost Probability dynamically
    sim_df = pd.DataFrame([base_dev_row])
    detector = TelemetryAnomalyDetector()
    try:
        detector = TelemetryAnomalyDetector.load()
        sim_anom = float(detector.predict_score(sim_df).iloc[0])
    except Exception:
        sim_anom = float(100.0 - curr_telemetry['health_score'])

    X_sim = prepare_feature_matrix(sim_df, expected_columns=feature_names)
    sim_prob = float(xgb_model.predict_proba(X_sim)[0, 1])

    sim_calc = calculate_risk_and_priority(
        sim_prob, sim_anom, base_dev_row.get('risk_class_encoded', 1),
        base_dev_row.get('previous_event_count', 0), base_dev_row.get('quantity_in_commerce_log', 0.0)
    )

    st.markdown("---")
    st.markdown("### **Real-Time Dynamic Risk & Priority Progression**")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Predicted Failure Risk", f"{sim_calc['failure_risk_score']}%", delta=sim_calc['risk_level'])
    p2.metric("Anomaly Score", f"{sim_calc['anomaly_score']}/100")
    p3.metric("Maintenance Priority", f"{sim_calc['maintenance_priority_score']}/100")
    p4.metric("Priority Status", sim_calc['priority_level'])

    # Trend Chart up to current timestep
    st.markdown("---")
    st.markdown("### **Telemetry Degradation Trajectory (Timestep 1 to Selected)**")
    st.caption("ℹ️ Simulated telemetry for demonstration purposes.")
    history_so_far = dev_telemetry.iloc[:step_idx + 1]
    st.line_chart(history_so_far.set_index('timestamp')[['temperature', 'vibration', 'power_consumption', 'error_rate']])

    if sim_calc['risk_level'] in ['HIGH', 'CRITICAL']:
        st.error(f"🚨 ALERT: {sim_calc['risk_level']} FAILURE RISK DETECTED AT TIMESTEP {step_idx + 1}! Human Technician Inspection Required.")
