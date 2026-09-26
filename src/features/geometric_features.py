import cv2
import numpy as np
import math
from typing import Optional


class GeometricFeatureExtractor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.nose_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_mcs_nose.xml"
        )
        self.mouth_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_mcs_mouth.xml"
        )

    def _get_face_bbox(self, gray: np.ndarray) -> Optional[tuple]:
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda r: r[2] * r[3])

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        features = {}

        face_bbox = self._get_face_bbox(gray)
        if face_bbox is None:
            return np.zeros(6)

        fx, fy, fw, fh = face_bbox

        face_roi = gray[fy:fy + fh, fx:fx + fw]
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5, minSize=(10, 10))
        noses = self.nose_cascade.detectMultiScale(face_roi, 1.1, 5, minSize=(10, 10))
        mouths = self.mouth_cascade.detectMultiScale(face_roi, 1.1, 10, minSize=(15, 10))

        features["face_width"] = fw
        features["face_height"] = fh
        features["face_aspect_ratio"] = fw / fh if fh > 0 else 0

        if len(eyes) >= 2:
            eye_centers = []
            for ex, ey, ew, eh in eyes[:2]:
                cx = fx + ex + ew // 2
                cy = fy + ey + eh // 2
                eye_centers.append((cx, cy))
            inter_eye_distance = math.sqrt(
                (eye_centers[0][0] - eye_centers[1][0]) ** 2
                + (eye_centers[0][1] - eye_centers[1][1]) ** 2
            )
            features["inter_eye_distance"] = inter_eye_distance
        else:
            features["inter_eye_distance"] = 0

        if len(noses) > 0:
            nx, ny, nw, nh = noses[0]
            features["nose_width"] = nw
        else:
            features["nose_width"] = 0

        if len(mouths) > 0:
            mx, my, mw, mh = mouths[0]
            features["mouth_width"] = mw
        else:
            features["mouth_width"] = 0

        features["jaw_width"] = fw * 0.85

        return np.array([
            features["inter_eye_distance"],
            features["nose_width"],
            features["face_width"],
            features["face_height"],
            features["jaw_width"],
            features["mouth_width"],
        ])
