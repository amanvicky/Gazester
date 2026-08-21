#!/bin/bash
echo ""
echo "============================================"
echo "  FLX-Gaze - Eye Gaze Correction System"
echo "============================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Step 1: Check Python
echo "[1/4] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found!"
    echo "Install with: sudo apt install python3 python3-pip"
    exit 1
fi
echo "      Python OK"

# Step 2: Install packages if needed
echo "[2/4] Checking packages..."
python3 -c "import tensorflow; import cv2; import numpy; import mediapipe" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "      Installing dependencies (this may take a few minutes)..."
    pip3 install -r "$SCRIPT_DIR/requirements.txt"
fi
echo "      Packages OK"

# Step 3: Calibrate if needed
echo "[3/4] Checking calibration..."
cd "$SCRIPT_DIR/gaze_correction_system"
python3 -c "from config import get_config; c,_ = get_config(); exit(0 if c.f != 650 else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "      First-time setup: Running calibration..."
    echo "      Please sit 50cm from camera and wait 5 seconds"
    echo ""
    python3 auto_calibrate.py --auto
    echo ""
fi

# Step 4: Launch
echo "[4/4] Starting FLX-Gaze..."
echo ""
echo "============================================"
echo "  Controls:"
echo "    'r' - Toggle gaze correction"
echo "    'q' - Quit"
echo "============================================"
echo ""

python3 regz_socket_MP_FD.py --auto
