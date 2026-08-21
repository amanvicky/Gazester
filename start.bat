@echo off
title FLX-Gaze

cd /d "%~dp0"

py --version >nul 2>&1
if %errorlevel% neq 0 (
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Python not found!
        echo Please install Python 3.8+ from https://python.org
        pause
        exit /b 1
    )
    set PYCMD=python
) else (
    set PYCMD=py
)

%PYCMD% -c "import tensorflow; import cv2; import numpy; import mediapipe" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

cd /d "%~dp0gaze_correction_system"
%PYCMD% -c "from config import get_config; c,_ = get_config(); exit(0 if c.f != 650 else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo Running first-time calibration...
    %PYCMD% auto_calibrate.py --auto
)

echo Starting FLX-Gaze... Press r to toggle, q to quit
%PYCMD% regz_socket_MP_FD.py --auto
pause
