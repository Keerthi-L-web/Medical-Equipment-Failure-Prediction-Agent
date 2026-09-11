"""
AI Maintenance Agent Module for Medical Equipment Failure Prediction Agent
Generates actionable, rule-assisted maintenance recommendations with mandatory human review safeguards.
"""
import pandas as pd

class AIMaintenanceAgent:
    def generate_recommendation(
        self,
        device_info: dict,
        risk_summary: dict,
        shap_explanation: dict
    ) -> dict:
        """
        Synthesize device state, SHAP drivers, anomaly score, and priority level
        into a structured maintenance decision-support recommendation.
        """
        risk_level = risk_summary.get('risk_level', 'LOW')
        priority_level = risk_summary.get('priority_level', 'LOW')
        failure_risk = risk_summary.get('failure_risk_score', 0.0)
        anomaly_score = risk_summary.get('anomaly_score', 0.0)
        
        dev_name = device_info.get('name', 'Medical Device')
        dev_type = device_info.get('device_type', 'General')
        is_implanted = bool(device_info.get('is_implanted', 0)) or (dev_type == "Implantable")
        
        top_reasons = shap_explanation.get('text_reasons', shap_explanation.get('text_reasons_pos', []))
        top_drivers_list = [d.split(" (contributed")[0] for d in top_reasons]

        # Determine Maintenance Urgency
        if risk_level == "CRITICAL" or priority_level == "CRITICAL":
            urgency = "IMMEDIATE (Within 24 Hours — Biomedical Engineering Review Required)"
        elif risk_level == "HIGH" or priority_level == "HIGH":
            urgency = "URGENT (Within 48–72 Hours — Inspect Equipment)"
        elif risk_level == "MODERATE" or priority_level == "MODERATE":
            urgency = "SCHEDULED (Within 7–14 Days — Preventive Maintenance)"
        else:
            urgency = "ROUTINE (Next Scheduled Preventive Maintenance Cycle)"

        recommended_actions = []
        shap_str = " ".join(top_reasons).lower()

        # 1. IMPLANTABLE DEVICES — CLINICAL / MONITORING GUIDELINES ONLY
        if is_implanted or dev_type == "Implantable":
            recommended_actions.append("Flag for qualified clinical and biomedical engineering review (implantable medical device protocol).")
            recommended_actions.append("Review patient telemetry transmission logs, battery longevity estimates, and signal telemetry integrity.")
            recommended_actions.append("Check manufacturer field safety notices and regulatory recall bulletins.")
            recommended_actions.append("Do NOT attempt physical disassembly or servicing; coordinate with attending clinical specialist.")

        # 2. CATEGORY-SPECIFIC INSPECTION PROTOCOLS
        elif dev_type == "Respiratory":
            recommended_actions.append("Inspect pressure sensors, flow valves, oxygen mixing valves, and patient circuit compliance.")
            recommended_actions.append("Inspect turbine / blower motor assembly and clear air intake filtration screens.")
            recommended_actions.append("Verify internal backup battery switchover and pressure relief alarm limits.")
            if "temperature" in shap_str:
                recommended_actions.append("Check internal heating unit and thermal sensor calibration.")
            if "vibration" in shap_str:
                recommended_actions.append("Inspect blower motor mounting dampers and check for mechanical noise.")

        elif dev_type == "Infusion":
            recommended_actions.append("Inspect flow accuracy, peristaltic rotor mechanism, and occlusion pressure sensor calibration.")
            recommended_actions.append("Check battery condition, charging circuit, and drop-sensor signal reliability.")
            recommended_actions.append("Verify air-in-line detector function and door latch safety interlock.")
            if "power" in shap_str or "voltage" in shap_str:
                recommended_actions.append("Inspect power supply module, power cord, and battery backup cells.")

        elif dev_type == "Monitoring":
            recommended_actions.append("Inspect ECG / physiological lead connections and signal amplifier integrity.")
            recommended_actions.append("Perform non-invasive blood pressure (NIBP) pneumatic leak test and module calibration.")
            recommended_actions.append("Verify battery condition, display panel brightness, and audible alarm speaker output.")

        elif dev_type == "Imaging":
            recommended_actions.append("Inspect optical / transducer cooling system, thermal exchanger, and coolant levels.")
            recommended_actions.append("Review image quality calibration logs, detector alignment, and gain settings.")
            recommended_actions.append("Check power supply unit (PSU) high-voltage stability and electrical grounding integrity.")

        elif dev_type == "Diagnostic":
            recommended_actions.append("Inspect fluidic sampling channels, reagent delivery lines, and optical measurement sensors.")
            recommended_actions.append("Run reference control calibration checks and inspect cuvette / probe cleanliness.")
            recommended_actions.append("Check reagent storage temperature control and waste discharge lines.")

        elif dev_type == "Surgical":
            recommended_actions.append("Inspect mechanical articulation joints, handpiece cabling, and casing seals.")
            recommended_actions.append("Test electrical safety, insulation resistance, and ground continuity (IEC 60601 compliance).")
            recommended_actions.append("Verify console foot pedal switches and energy output power calibration.")

        else: # Other / Unknown
            recommended_actions.append("Perform standard biomedical equipment physical inspection, checking casing, switches, and cabling.")
            recommended_actions.append("Verify electrical safety test compliance (IEC 60601) and calibration certificates.")
            recommended_actions.append("Run manufacturer built-in self-diagnostic suite (BIT).")

        # Telemetry triggers for non-implantable devices if extra actions needed
        if not is_implanted and dev_type != "Implantable":
            if "battery" in shap_str and not any("battery" in a.lower() for a in recommended_actions):
                recommended_actions.append("Perform battery impedance test and replace internal battery pack if degraded.")
            if "error rate" in shap_str and not any("diagnostic" in a.lower() for a in recommended_actions):
                recommended_actions.append("Perform digital diagnostic check, verify firmware version, and inspect telemetry bus connection.")

        # Historical Safety Record check
        prev_evts = device_info.get('previous_event_count', 0)
        if prev_evts > 0:
            recommended_actions.append(f"Review manufacturer field safety notices and historical logs for {int(prev_evts)} prior safety event(s).")

        safeguard_notice = (
            "HUMAN TECHNICIAN REVIEW REQUIRED: These recommendations are generated by an AI maintenance "
            "decision-support system. A qualified biomedical technician must inspect the equipment "
            "and approve any maintenance action before the device is returned to clinical service."
        )

        return {
            'device_id': device_info.get('id'),
            'device_name': dev_name,
            'device_type': dev_type,
            'risk_level': risk_level,
            'priority_level': priority_level,
            'failure_risk_score': failure_risk,
            'anomaly_score': anomaly_score,
            'urgency': urgency,
            'explanation_summary': shap_explanation.get('summary_text', ''),
            'top_risk_drivers': top_drivers_list if top_drivers_list else top_reasons,
            'recommended_actions': recommended_actions,
            'human_review_required': True,
            'safeguard_notice': safeguard_notice
        }
