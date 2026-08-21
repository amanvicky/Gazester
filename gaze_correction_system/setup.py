#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated setup script for FLX-Gaze system
Installs all dependencies and verifies the installation
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print(f"  OK: {description}")
            return True
        else:
            print(f"  WARNING: {description} had issues")
            if result.stderr:
                print(f"  {result.stderr[:200]}")
            return True  # Continue even with warnings
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {description} took too long")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8+ required")
        return False
    return True


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def main():
    print("=" * 60)
    print("FLX-Gaze Automated Setup")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install core packages
    print("\n" + "=" * 60)
    print("Installing core packages...")
    print("=" * 60)

    packages = [
        ("tensorflow>=2.15.0", "tensorflow"),
        ("opencv-python>=4.8.0", "cv2"),
        ("numpy>=1.24.0", "numpy"),
        ("scipy>=1.10.0", "scipy"),
    ]

    for pkg, import_name in packages:
        if check_package(import_name):
            print(f"  [OK] {pkg} already installed")
        else:
            run_command(f"pip install {pkg}", f"Installing {pkg}")

    # Install dlib (special handling)
    print("\n" + "=" * 60)
    print("Installing dlib...")
    print("=" * 60)

    if check_package("dlib"):
        print("  [OK] dlib already installed")
    else:
        print("  dlib requires cmake and C++ build tools")
        print("  Attempting installation...")

        # Try to install cmake first
        run_command("pip install cmake", "Installing cmake")

        # Try to install dlib
        success = run_command("pip install dlib", "Installing dlib")

        if not success:
            print("\n  dlib installation failed!")
            print("  Please install manually:")
            print("  1. Install Visual C++ Build Tools")
            print("  2. Run: pip install cmake dlib")

    # Verify installation
    print("\n" + "=" * 60)
    print("Verifying installation...")
    print("=" * 60)

    all_ok = True
    for pkg, import_name in [("TensorFlow", "tensorflow"), ("OpenCV", "cv2"),
                              ("NumPy", "numpy"), ("dlib", "dlib")]:
        if check_package(import_name):
            try:
                mod = __import__(import_name)
                version = getattr(mod, '__version__', 'unknown')
                print(f"  [OK] {pkg} {version}")
            except Exception:
                print(f"  [OK] {pkg}")
        else:
            print(f"  [MISSING] {pkg}")
            all_ok = False

    # Check model files
    print("\n" + "=" * 60)
    print("Checking model files...")
    print("=" * 60)

    required_files = [
        "weights/warping_model/flx/12/L/checkpoint",
        "weights/warping_model/flx/12/R/checkpoint",
        "lm_feat/shape_predictor_68_face_landmarks.dat",
    ]

    for f in required_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  [OK] {f} ({size:,} bytes)")
        else:
            print(f"  [MISSING] {f}")
            all_ok = False

    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)

    if all_ok:
        print("  All checks passed!")
        print("\n  Next steps:")
        print("  1. Run: python auto_calibrate.py")
        print("  2. Run: python run.py")
    else:
        print("  Some checks failed - see above for details")
        print("  Install missing packages manually")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
