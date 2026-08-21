#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated focal length calibration
Detects your face, calculates focal length, and saves to config.py
"""

import cv2
import numpy as np
import dlib
import os
import re


# Default settings
DEFAULT_DISTANCE = 50  # cm (distance from camera)
DEFAULT_IPD = 6.3  # cm (interpupillary distance)
VIDEO_RES = [640, 480]
FACE_DETECT_SIZE = [320, 240]


def get_eye_pos(shape, pos="L"):
    """Get eye position from facial landmarks."""
    if pos == "R":
        lc, rc = 36, 39
    elif pos == "L":
        lc, rc = 42, 45
    else:
        return None, None, None

    eye_cx = (shape.part(rc).x + shape.part(lc).x) * 0.5
    eye_cy = (shape.part(rc).y + shape.part(lc).y) * 0.5
    return [eye_cx, eye_cy], None, None


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
    config_path = "config.py"

    with open(config_path, 'r') as f:
        content = f.read()

    # Update the focal length parameter
    pattern = r"(--f',\s*type=eval,\s*default=)\d+"
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
    detector = dlib.get_frontal_face_detector()

    # Try to load predictor
    predictor_path = "./lm_feat/shape_predictor_68_face_landmarks.dat"
    if os.path.exists(predictor_path):
        predictor = dlib.shape_predictor(predictor_path)
    else:
        print("ERROR: Face landmark model not found")
        print(f"Expected: {predictor_path}")
        vs.release()
        return False

    # Calibration loop
    focal_lengths = []
    print("Detecting face... (hold still)")

    while True:
        ret, frame = vs.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect_gray = cv2.resize(gray,
                                      (FACE_DETECT_SIZE[0], FACE_DETECT_SIZE[1]))

        detections = detector(face_detect_gray, 0)

        for k, bx in enumerate(detections):
            # Scale detection to full resolution
            x_ratio = VIDEO_RES[0] / FACE_DETECT_SIZE[0]
            y_ratio = VIDEO_RES[1] / FACE_DETECT_SIZE[1]

            target_bx = dlib.rectangle(
                left=int(bx.left() * x_ratio),
                right=int(bx.right() * x_ratio),
                top=int(bx.top() * y_ratio),
                bottom=int(bx.bottom() * y_ratio)
            )
            shape = predictor(gray, target_bx)

            # Get eye positions
            LE_center, _, _ = get_eye_pos(shape, pos="L")
            RE_center, _, _ = get_eye_pos(shape, pos="R")

            if LE_center and RE_center:
                # Calculate focal length
                f = calculate_focal_length(
                    LE_center, RE_center,
                    DEFAULT_DISTANCE, DEFAULT_IPD
                )

                if f and 200 < f < 2000:  # Sanity check
                    focal_lengths.append(f)

                    # Display
                    cv2.rectangle(frame,
                                (VIDEO_RES[0] - 200, 0),
                                (VIDEO_RES[0], 80),
                                (255, 255, 255), -1)
                    cv2.putText(frame,
                               f'f: {f}',
                               (VIDEO_RES[0] - 190, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 0, 255), 2)
                    cv2.putText(frame,
                               f'Samples: {len(focal_lengths)}',
                               (VIDEO_RES[0] - 190, 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (0, 0, 0), 1)

                    # Draw eye markers
                    cv2.circle(frame,
                             (int(LE_center[0]), int(LE_center[1])),
                             3, (0, 255, 0), -1)
                    cv2.circle(frame,
                             (int(RE_center[0]), int(RE_center[1])),
                             3, (0, 255, 0), -1)

        cv2.imshow("Calibration", frame)
        k = cv2.waitKey(10)

        if not auto_mode and k == ord('q'):
            break

        # Auto-stop after 10 samples (auto mode) or 30 samples (manual)
        max_samples = 10 if auto_mode else 30
        if len(focal_lengths) >= max_samples:
            if auto_mode:
                print(f"\nCollected {max_samples} samples")
            else:
                print("\nCollected 30 samples, stopping...")
            break

    vs.release()
    cv2.destroyAllWindows()

    # Calculate result
    if focal_lengths:
        # Use median for robustness
        final_f = int(np.median(focal_lengths))
        print(f"\nCalibration complete!")
        print(f"Collected {len(focal_lengths)} samples")
        print(f"Focal length range: {min(focal_lengths)} - {max(focal_lengths)}")
        print(f"Final focal length: {final_f}")

        # Update config
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
    import sys
    success = main()
    if success and '--auto' not in sys.argv:
        print("\nNext step: python run.py")
    sys.exit(0 if success else 1)
