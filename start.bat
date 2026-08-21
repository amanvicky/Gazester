@echo off
title FLX-Gaze - Eye Gaze Correction System
color 0A

echo.
echo  ============================================
echo     FLX-Gaze - Eye Gaze Correction System
echo  ============================================
echo.

cd gaze_correction_system 2>nul || (
    echo ERROR: Cannot find gaze_correction_system folder
    pause
    exit /b 1
)

:: Step 1: Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
echo       Python OK

:: Step 2: Install packages if needed
echo [2/4] Checking packages...
python -c "import tensorflow; import cv2; import numpy; import dlib" >nul 2>&1
if %errorlevel% neq 0 (
    echo       Installing dependencies (this may take a few minutes)...
    pip install -r requirements.txt >nul 2>&1
    pip install cmake >nul 2>&1
    pip install dlib >nul 2>&1
)
echo       Packages OK

:: Step 3: Calibrate if needed
echo [3/4] Checking calibration...
python -c "from config import get_config; c,_ = get_config(); exit(0 if c.f != 650 else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo       First-time setup: Running calibration...
    echo       Please sit 50cm from camera and wait 5 seconds
    echo.
    python auto_calibrate.py --auto
    echo.
)

:: Step 4: Launch
echo [4/4] Starting FLX-Gaze...
echo.
echo  ============================================
echo     Controls:
echo       'r' - Toggle gaze correction
echo       'q' - Quit
echo  ============================================
echo.

python regz_socket_MP_FD.py --auto

pause
