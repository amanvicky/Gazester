#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated focal length calibration
Uses MediaPipe FaceLandmarker for face detection (cross-platform)
Detects your face, calculates focal length, and saves to config.py
"""

import cv2
import numpy as np
import os
import sys
import re

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_detection import FaceDetector, RIGHT_EYE_INDICES, LEFT_EYE_INDICES

# Default settings
DEFAULT_DISTANCE = 50  # cm
DEFAULT_IPD = 6.3  # cm (interpupillary distance)
VIDEO_RES = [640, 480]


def get_eye_positions(shape):
    """Get eye positions from face landmarks."""
    # Right eye center: between landmarks 33 and 133
    r_outer = shape.part(33)
    r_inner = shape.part(133)
    RE_center = [(r_outer.x + r_inner.x) / 2, (r_outer.y + r_inner.y) / 2]

    # Left eye center: between landmarks 362 and 263
    l_outer = shape.part(362)
    l_inner = shape.part(263)
    LE_center = [(l_outer.x + l_inner.x) / 2, (l_outer.y + l_inner.y) / 2]

    return LE_center, RE_center


def calculate_focal_length(LE_center, RE_center, distance, ipd):
    """Calculate focal length from eye positions."""
    eye_dist_px = np.sqrt(
        (LE_center[0] - RE_center[0])**2 +
        (LE_center[1] - RE_center[1])**2
    )
    if eye_dist_px < 1:
        return None
    return int(eye_dist_px * distance / ipd)


def update_config(focal_length):
    """Update config.py with new focal length."""
    config_path = os.path.join(os.path.dirname(__file__), "config.py")

    with open(config_path, 'r') as f:
        content = f.read()

    pattern = r"('--f',\s*type=eval,\s*default=)\d+"
    replacement = f"\\g<1>{focal_length}"
    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        with open(config_path, 'w') as f:
            f.write(new_content)
        return True
    return False


def main():
    auto_mode = '--auto' in sys.argv

    print("=" * 60)
    print("FLX-Gaze Automated Calibration")
    print("=" * 60)

    if auto_mode:
        print(f"\nAuto mode: Please sit {DEFAULT_DISTANCE} cm from your camera")
        print("Collecting 10 samples (5 seconds)...")
    else:
        print(f"\nPlease sit {DEFAULT_DISTANCE} cm from your camera")
        print("Press 'q' to finish calibration\n")

    # Initialize camera
    vs = cv2.VideoCapture(0)
    if not vs.isOpened():
        print("ERROR: Could not open camera")
        return False

    vs.set(3, VIDEO_RES[0])
    vs.set(4, VIDEO_RES[1])

    # Initialize face detector
    detector = FaceDetector()

    # Calibration loop
    focal_lengths = []
    print("Detecting face... (hold still)")

    while True:
        ret, frame = vs.read()
        if not ret:
            continue

        shape = detector.detect(frame)
        if shape is not None:
            LE_center, RE_center = get_eye_positions(shape)

            f = calculate_focal_length(
                LE_center, RE_center,
                DEFAULT_DISTANCE, DEFAULT_IPD
            )

            if f and 200 < f < 2000:
                focal_lengths.append(f)

                # Display
                cv2.rectangle(frame, (VIDEO_RES[0] - 200, 0),
                            (VIDEO_RES[0], 80), (255, 255, 255), -1)
                cv2.putText(frame, f'f: {f}',
                           (VIDEO_RES[0] - 190, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f'Samples: {len(focal_lengths)}',
                           (VIDEO_RES[0] - 190, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Draw eye markers
                cv2.circle(frame, (int(LE_center[0]), int(LE_center[1])),
                         3, (0, 255, 0), -1)
                cv2.circle(frame, (int(RE_center[0]), int(RE_center[1])),
                         3, (0, 255, 0), -1)

        cv2.imshow("Calibration", frame)
        k = cv2.waitKey(10)

        if not auto_mode and k == ord('q'):
            break

        max_samples = 10 if auto_mode else 30
        if len(focal_lengths) >= max_samples:
            print(f"\nCollected {len(focal_lengths)} samples")
            break

    vs.release()
    cv2.destroyAllWindows()
    detector.close()

    if focal_lengths:
        final_f = int(np.median(focal_lengths))
        print(f"\nCalibration complete!")
        print(f"Collected {len(focal_lengths)} samples")
        print(f"Focal length range: {min(focal_lengths)} - {max(focal_lengths)}")
        print(f"Final focal length: {final_f}")

        if update_config(final_f):
            print(f"\nConfig updated: config.py -> --f {final_f}")
        else:
            print(f"\nPlease manually set --f {final_f} in config.py")

        return True
    else:
        print("\nERROR: Could not detect face")
        print("Make sure your face is visible and well-lit")
        return False


if __name__ == "__main__":
    success = main()
    if success and '--auto' not in sys.argv:
        print("\nNext step: python run.py")
    sys.exit(0 if success else 1)
