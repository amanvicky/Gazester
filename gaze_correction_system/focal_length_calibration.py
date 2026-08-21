#!/usr/bin/env python
# coding: utf-8
"""
Focal length calibration tool for camera
Uses MediaPipe FaceLandmarker for face detection (cross-platform)
"""

import cv2
import numpy as np
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_detection import FaceDetector

# Settings
d = 50  # cm (distance from camera)
P_IPD = 6.3  # cm (interpupillary distance)
video_res = [640, 480]


def main():
    vs = cv2.VideoCapture(0)

    if not vs.isOpened():
        print("Error: Could not open camera")
        return

    print("Press 'q' to exit calibration")
    print("Place your head about 50 cm from the camera")

    detector = FaceDetector()

    while True:
        ret, recv_frame = vs.read()
        if ret:
            shape = detector.detect(recv_frame)

            if shape is not None:
                # Get eye centers (landmarks 33/133 for right, 362/263 for left)
                r_out = shape.part(33)
                r_in = shape.part(133)
                RE_center = [(r_out.x + r_in.x) / 2, (r_out.y + r_in.y) / 2]

                l_out = shape.part(362)
                l_in = shape.part(263)
                LE_center = [(l_out.x + l_in.x) / 2, (l_out.y + l_in.y) / 2]

                # Calculate focal length
                f = int(np.sqrt(
                    (LE_center[0] - RE_center[0])**2 +
                    (LE_center[1] - RE_center[1])**2
                ) * d / P_IPD)

                # Display focal length
                cv2.rectangle(recv_frame,
                             (video_res[0] - 150, 0),
                             (video_res[0], 40),
                             (255, 255, 255), -1)
                cv2.putText(recv_frame, f'f:{f}',
                           (video_res[0] - 140, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           (0, 0, 255), 1, cv2.LINE_AA)

                # Draw eye markers
                cv2.circle(recv_frame,
                          (int(LE_center[0]), int(LE_center[1])),
                          2, (0, 255, 0), -1)
                cv2.circle(recv_frame,
                          (int(RE_center[0]), int(RE_center[1])),
                          2, (0, 255, 0), -1)

        cv2.imshow("Calibration", recv_frame)
        k = cv2.waitKey(10)
        if k == ord('q'):
            vs.release()
            cv2.destroyAllWindows()
            detector.close()
            break

    print(f"The focal length of your camera is {f}, "
          f"please set the value of f (--f) in the config.py")


if __name__ == "__main__":
    main()
