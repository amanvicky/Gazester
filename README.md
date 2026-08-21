# FLX-Gaze: Eye Gaze Correction System

Real-time eye gaze correction using warping-based convolutional neural network.

## Paper

@article{Hsu:2019:LMC:3339884.3311784,
author = {Hsu, Chih-Fan and Wang, Yu-Shuen and Lei, Chin-Laung and Chen, Kuan-Ta},
title = {Look at Me! Correcting Eye Gaze in Live Video Communication},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
year = {2019},
doi = {10.1145/3311784}
}

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- OpenCV 4.8+
- NumPy 1.24+
- dlib 19.24+
- scipy 1.10+

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Calibrate Camera

```bash
cd gaze_correction_system
python focal_length_calibration.py
```

Press 'q' to exit. Note the focal length value displayed.

### 2. Configure System

Edit `config.py` and set:
- `--f`: Your camera's focal length (from calibration)
- `--P_c_x`, `--P_c_y`, `--P_c_z`: Camera position relative to screen center (cm)
- `--S_W`, `--S_H`: Screen size (cm)
- `--tar_ip`: Target IP address (use 127.0.0.1 for self-demo)
- `--sender_port`, `--recver_port`: Port numbers

### 3. Run System

```bash
cd gaze_correction_system
python regz_socket_MP_FD.py
```

**Controls:**
- Press 'r' to toggle gaze correction
- Press 'q' to quit

## Project Structure

```
├── gaze_correction_system/          # Inference system
│   ├── config.py                    # Configuration
│   ├── regz_socket_MP_FD.py        # Main application
│   ├── flx.py                      # Neural network model
│   ├── tf_utils.py                 # Model utilities
│   ├── transformation.py           # Spatial transformer
│   ├── focal_length_calibration.py # Camera calibration
│   ├── lm_feat/                    # Face landmark model
│   └── weights/                    # Pre-trained weights
└── training/                       # Training code
    └── code_tf/model_train/        # Training scripts
```

## Technical Details

- **Model**: FLX-Gaze (249K parameters)
- **Input**: 48×64 eye image + 12-channel anchor maps + 2D gaze angle
- **Output**: Gaze-corrected eye image
- **Architecture**: Encoder → Coarse warping → Fine warping → LCM blending

## License

Research use only. Please cite the paper if you use this code.
