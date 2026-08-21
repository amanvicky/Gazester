@echo off
echo ============================================
echo FLX-Gaze One-Click Installer
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Installing Python packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)

echo.
echo [2/3] Verifying installation...
python -c "import tensorflow, cv2, numpy, mediapipe; print(chr(39) + " + "All packages OK" + chr(39))"

echo.
echo ============================================
echo Installation complete!
echo Next steps: cd gaze_correction_system then python run.py
echo ============================================
pause
