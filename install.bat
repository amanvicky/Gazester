@echo off
echo ============================================
echo FLX-Gaze One-Click Installer
echo ============================================
echo.

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

echo [1/3] Installing Python packages...
%PYCMD% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)

echo.
echo [2/3] Verifying installation...
%PYCMD% -c "import tensorflow, cv2, numpy, mediapipe; print('All packages OK')"

echo.
echo ============================================
echo Installation complete!
echo Next steps: Run start.bat
echo ============================================
pause
