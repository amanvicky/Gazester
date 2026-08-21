# FLX-Gaze: Eye Gaze Correction System

Real-time eye gaze correction using warping-based convolutional neural network.

Uses **MediaPipe** for face detection (cross-platform, no build tools needed).

## Quick Start (3 steps)

### Step 1: Install

**Windows:**
```bash
install.bat
```

**Linux/macOS:**
```bash
chmod +x install.sh
./install.sh
```

**Or manually:**
```bash
cd gaze_correction_system
pip install -r requirements.txt
```

### Step 2: Calibrate

```bash
cd gaze_correction_system
python auto_calibrate.py
```

- Sit ~50cm from your camera
- Wait for 30 samples (or press 'q' to finish early)
- Focal length is automatically saved to config

### Step 3: Run

```bash
python run.py
```

**Controls:**
- Press **'r'** to toggle gaze correction ON/OFF
- Press **'q'** to quit

---

## Manual Configuration

If you need to adjust settings, edit `config.py`:

```python
--f 527              # Focal length (from calibration)
--P_c_x 0            # Camera X position (cm)
--P_c_y -21          # Camera Y position (cm)
--P_c_z -1           # Camera Z position (cm)
--S_W 62             # Screen width (cm)
--S_H 35             # Screen height (cm)
--tar_ip 127.0.0.1   # Target IP (use 127.0.0.1 for self-demo)
--sender_port 5005   # Port for sending video
--recver_port 5005   # Port for receiving video
```

---

## Two-Person Setup

**Person A:**
```bash
python run.py --tar_ip <PERSON_B_IP> --sender_port 5005 --recver_port 5006
```

**Person B:**
```bash
python run.py --tar_ip <PERSON_A_IP> --sender_port 5006 --recver_port 5005
```

---

## Available Commands

```bash
python run.py              # Launch system
python run.py calibrate    # Run calibration
python run.py check        # Check setup
python auto_calibrate.py   # Calibrate camera
python setup.py            # Install/verify dependencies
```

---

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- OpenCV 4.8+
- NumPy 1.24+
- MediaPipe 0.10+
- Camera (webcam)
- Display (monitor)

---

## Troubleshooting

**Camera not working:**
- Check if camera is connected
- Try changing camera index in config (`--camera 1`)

**No face detected:**
- Ensure good lighting
- Face the camera directly
- Sit 30-100cm from camera

**Low performance:**
- Close other applications
- Use a GPU if available
- Reduce video resolution in config

---

## Project Structure

```
├── install.bat              # Windows installer
├── install.sh               # Linux/macOS installer
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── gaze_correction_system/
    ├── run.py              # Launcher
    ├── auto_calibrate.py   # Camera calibration
    ├── setup.py            # Dependency installer
    ├── config.py           # Configuration
    ├── regz_socket_MP_FD.py # Main application
    ├── flx.py              # Neural network model
    ├── transformation.py   # Spatial transformer
    └── weights/            # Pre-trained weights
```

---

## License

Research use only. Please cite the paper if you use this code.
