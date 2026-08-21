@echo off
title FLX-Gaze

cd /d "%~dp0"

python --version >/dev/null 2>&1
if %errorlevel% neq 0 (
    echo Python not found!
    pause
    exit /b 1
)

python -c "import tensorflow; import cv2; import numpy; import mediapipe" >/dev/null 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

cd /d "%~dp0gaze_correction_system"
python -c "from config import get_config; c,_ = get_config(); exit(0 if c.f != 650 else 1)" >/dev/null 2>&1
if %errorlevel% neq 0 (
    echo Running first-time calibration...
    python auto_calibrate.py --auto
)

echo Starting FLX-Gaze... Press r to toggle, q to quit
python regz_socket_MP_FD.py --auto
pause
