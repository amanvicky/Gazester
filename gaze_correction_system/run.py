#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLX-Gaze Automated Launcher
Detects environment, verifies setup, and launches the gaze correction system
"""

import os
import sys
import subprocess
import platform


def print_header():
    print("=" * 60)
    print("FLX-Gaze: Eye Gaze Correction System")
    print("=" * 60)


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERROR: Python 3.8+ required (found {version.major}.{version.minor})")
        return False
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_packages():
    """Check required packages."""
    required = {
        'tensorflow': 'tensorflow',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'dlib': 'dlib',
    }

    missing = []
    for import_name, pkg_name in required.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', '?')
            print(f"  {pkg_name} {version}")
        except ImportError:
            print(f"  {pkg_name} MISSING")
            missing.append(pkg_name)

    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    return True


def check_model_files():
    """Check if model files exist."""
    required_files = [
        ("weights/warping_model/flx/12/L/checkpoint", "Left eye model"),
        ("weights/warping_model/flx/12/R/checkpoint", "Right eye model"),
        ("lm_feat/shape_predictor_68_face_landmarks.dat", "Face landmarks"),
    ]

    all_ok = True
    for path, desc in required_files:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {desc}: {size:,} bytes")
        else:
            print(f"  {desc}: MISSING ({path})")
            all_ok = False

    return all_ok


def check_config():
    """Check if config is set up."""
    try:
        from config import get_config
        conf, _ = get_config()

        # Check if focal length has been calibrated
        if conf.f == 650:  # Default value
            print("  WARNING: Focal length not calibrated (using default 650)")
            print("  Run: python auto_calibrate.py")
            return False

        print(f"  Focal length: {conf.f}")
        print(f"  Camera position: ({conf.P_c_x}, {conf.P_c_y}, {conf.P_c_z})")
        print(f"  Screen size: {conf.S_W} x {conf.S_H} cm")
        return True
    except Exception as e:
        print(f"  ERROR reading config: {e}")
        return False


def run_calibration():
    """Run the calibration script."""
    print("\n" + "=" * 60)
    print("Running camera calibration...")
    print("=" * 60)

    response = input("\nDo you want to calibrate now? (y/n): ").lower()
    if response == 'y':
        subprocess.run([sys.executable, "auto_calibrate.py"])
        return True
    return False


def launch_system(args=None):
    """Launch the gaze correction system."""
    print("\n" + "=" * 60)
    print("Launching FLX-Gaze...")
    print("=" * 60)
    print("\nControls:")
    print("  'r' - Toggle gaze correction")
    print("  'q' - Quit")
    print("\nStarting...")

    cmd = [sys.executable, "regz_socket_MP_FD.py"]
    if args:
        cmd.extend(args)

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    print_header()

    # Step 1: Check Python
    print("\n[1/5] Checking Python...")
    if not check_python():
        return False

    # Step 2: Check packages
    print("\n[2/5] Checking packages...")
    if not check_packages():
        print("\nRun setup first: python setup.py")
        return False

    # Step 3: Check model files
    print("\n[3/5] Checking model files...")
    if not check_model_files():
        print("\nModel files missing - download them first")
        return False

    # Step 4: Check config
    print("\n[4/5] Checking configuration...")
    config_ok = check_config()

    # Step 5: Calibrate if needed
    print("\n[5/5] Final checks...")
    if not config_ok:
        run_calibration()

    # Launch
    launch_system()
    return True


if __name__ == "__main__":
    # Handle command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "calibrate":
            run_calibration()
        elif sys.argv[1] == "check":
            print_header()
            check_python()
            check_packages()
            check_model_files()
            check_config()
        elif sys.argv[1] == "help":
            print("Usage:")
            print("  python run.py          - Launch the system")
            print("  python run.py calibrate - Run calibration")
            print("  python run.py check    - Check setup")
            print("  python run.py help     - Show this help")
        else:
            # Pass args to the main system
            print_header()
            launch_system(sys.argv[1:])
    else:
        main()
