@echo off
echo ============================================
echo FLX-Gaze One-Click Installer
echo ============================================
echo.

set PROJECT_DIR=%~dp0

echo [1/3] Installing Python packages...
pip install -r "%PROJECT_DIR%requirements.txt"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dlib (may take a few minutes)...
pip install cmake
pip install dlib
if %errorlevel% neq 0 (
    echo WARNING: dlib installation failed
    echo You may need Visual C++ Build Tools
    echo Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
)

echo.
echo [3/3] Verifying installation...
python -c "import tensorflow, cv2, numpy; print('Core packages OK')"
python -c "import dlib; print('dlib OK')" 2>nul || echo WARNING: dlib not available

echo.
echo ============================================
echo Installation complete!
echo.
echo Next steps:
echo   1. cd gaze_correction_system
echo   2. python auto_calibrate.py
echo   3. python run.py
echo ============================================
pause
