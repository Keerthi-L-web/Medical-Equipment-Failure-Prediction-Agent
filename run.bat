@echo off
echo =========================================================================
echo  Medical Equipment Failure Prediction Agent
echo =========================================================================
echo Step 1: Running Data Pipeline and Model Training...
python -m src.pipeline_runner
if %errorlevel% neq 0 (
    echo [ERROR] Pipeline failed!
    exit /b %errorlevel%
)
echo.
echo Step 2: Launching Streamlit Interactive Dashboard...
streamlit run dashboard/app.py
