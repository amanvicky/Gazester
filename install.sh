#!/bin/bash

echo "============================================"
echo "FLX-Gaze One-Click Installer"
echo "============================================"
echo

cd gaze_correction_system || exit 1

echo "[1/3] Installing Python packages..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install packages"
    exit 1
fi

echo
echo "[2/3] Installing dlib (may take a few minutes)..."
pip3 install cmake
pip3 install dlib
if [ $? -ne 0 ]; then
    echo "WARNING: dlib installation failed"
    echo "You may need to install build tools:"
    echo "  Ubuntu/Debian: sudo apt-get install build-essential cmake"
    echo "  macOS: xcode-select --install"
fi

echo
echo "[3/3] Verifying installation..."
python3 -c "import tensorflow, cv2, numpy, dlib; print('All packages OK')"

echo
echo "============================================"
echo "Installation complete!"
echo
echo "Next steps:"
echo "  1. Calibrate camera: python3 auto_calibrate.py"
echo "  2. Run system: python3 run.py"
echo "============================================"
