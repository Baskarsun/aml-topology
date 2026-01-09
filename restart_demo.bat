@echo off
echo ========================================
echo  AML Detection System - Fresh Restart
echo ========================================
echo.

REM Stop all Python processes
echo [1/4] Stopping existing processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 >nul

REM Clear database
echo [2/4] Clearing metrics database...
del /f /q metrics.db 2>nul
echo Database cleared

REM Start inference API in background
echo [3/4] Starting Inference API...
start "AML Inference API" cmd /c "python src\inference_api.py"
timeout /t 5 >nul

REM Start transaction simulator
echo [4/4] Starting Transaction Simulator...
start "Transaction Simulator" cmd /c "python transaction_simulator.py --rate 3.0"
timeout /t 3 >nul

echo.
echo ========================================
echo  System Ready!
echo ========================================
echo.
echo Next steps:
echo 1. Open new terminal and run: streamlit run dashboard.py
echo 2. Dashboard will open at http://localhost:8501
echo.
echo Press any key to continue...
pause >nul
