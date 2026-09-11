#!/bin/bash
echo "========================================================================="
echo " Medical Equipment Failure Prediction Agent"
echo "========================================================================="
echo "Step 1: Running Data Pipeline and Model Training..."
python -m src.pipeline_runner
if [ $? -ne 0 ]; then
    echo "[ERROR] Pipeline failed!"
    exit 1
fi
echo ""
echo "Step 2: Launching Streamlit Interactive Dashboard..."
streamlit run dashboard/app.py
