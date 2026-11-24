@echo off
title SmartCell - AI-Assisted Blood Cell Screening Research Platform
echo ==============================================================================
echo SMARTCELL AI BLOOD CELL SCREENING RESEARCH PLATFORM
echo ==============================================================================
echo Mode 1: Malaria Screening (Thin Blood Smear Multi-Cell Field Analysis)
echo Mode 2: ALL Leukemia Classification (Single White Blood Cell Screening)
echo Web Interface URL:  http://127.0.0.1:8501
echo ==============================================================================
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501 " ^| findstr "LISTENING"') do (
    echo Stopping previous SmartCell server process on port 8501 (PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
)
start "" http://127.0.0.1:8501
"D:\smartcell\venv\Scripts\python.exe" "D:\smartcell\scripts\smartcell_demo_app.py" --host 127.0.0.1 --port 8501
pause

