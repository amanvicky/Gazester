#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-platform face detection wrapper using MediaPipe FaceLandmarker (v1.0+)
Provides a simple interface for face detection and landmark extraction.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# Video resolution
VIDEO_RES = (640, 480)

# MediaPipe face landmarker model path
MODEL_PATH = "./face_landmarker.task"

# MediaPipe 478 face landmark indices for eyes
# Right eye: outer=33, top1=159, top2=158, inner=133, bot1=153, bot2=145
# Left eye:  inner=362, top1=385, top2=386, outer=263, bot1=374, bot2=373
RIGHT_EYE_INDICES = [33, 159, 158, 133, 153, 145]
LEFT_EYE_INDICES = [362, 385, 386, 263, 374, 373]

# For bounding box detection (face oval extremes)
FACE_TOP = 10
FACE_BOTTOM = 152
FACE_LEFT = 234
FACE_RIGHT = 454


class LandmarkPart:
    """Mimics dlib's shape.part() interface for compatibility"""
    __slots__ = ('x', 'y')

    def __init__(self, x, y):
        self.x = x
        self.y = y


class LandmarkShape:
    """Mimics dlib's shape interface with .part(idx) method"""
    def __init__(self, landmarks, image_width, image_height):
        self._parts = {}
        for idx, lm in enumerate(landmarks):
            self._parts[idx] = LandmarkPart(
                int(lm.x * image_width),
                int(lm.y * image_height)
            )

    def part(self, idx):
        return self._parts.get(idx, LandmarkPart(0, 0))


class FaceDetector:
    """Cross-platform face detection using MediaPipe FaceLandmarker"""

    def __init__(self, model_path=MODEL_PATH, num_faces=1):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        """
        Detect face landmarks in a BGR frame.
        Returns LandmarkShape (dlib-compatible) or None.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if result.face_landmarks:
            return LandmarkShape(result.face_landmarks[0], w, h)
        return None

    def detect_bbox(self, frame_bgr):
        """
        Detect face and return bounding box (x1, y1, x2, y2) or None.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            x_coords = [p.x for p in lm]
            y_coords = [p.y for p in lm]
            return (
                int(min(x_coords) * w), int(min(y_coords) * h),
                int(max(x_coords) * w), int(max(y_coords) * h)
            )
        return None

    def close(self):
        self.landmarker.close()
