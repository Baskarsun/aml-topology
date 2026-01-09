@echo off
echo ============================================================
echo Starting AML Dashboard System
echo ============================================================
echo.

cd /d "%~dp0"

echo Starting Flask API on port 5000...
start "AML API" cmd /k "python src\inference_api.py"

timeout /t 5 /nobreak > nul

echo Starting Transaction Simulator...
start "AML Simulator" cmd /k "python transaction_simulator.py --rate 2.0"

timeout /t 2 /nobreak > nul

echo Starting Streamlit Dashboard on port 8501...
start "AML Dashboard" cmd /k "streamlit run dashboard.py"

echo.
echo ============================================================
echo All components started!
echo ============================================================
echo.
echo API: http://localhost:5000
echo Dashboard: http://localhost:8501
echo.
echo Close all cmd windows to stop the system.
echo ============================================================
