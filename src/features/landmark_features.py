import cv2
import numpy as np
from typing import List, Optional, Tuple

from ..preprocessing.cascade_locator import load_cascade


class LandmarkFeatureExtractor:
    def __init__(self, predictor_path: str = "shape_predictor_68_face_landmarks.dat"):
        self.predictor_path = predictor_path
        self.detector = None
        self.predictor = None
        self._load_models()

    def _load_models(self):
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.predictor_path)
        except (ImportError, RuntimeError) as e:
            print(f"dlib landmark model unavailable: {e}")
            self.detector = None
            self.predictor = None

    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.predictor is None:
            return self._get_landmarks_cv2(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.detector(gray, 1)
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)
        return landmarks

    def _get_landmarks_cv2(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        cascade = load_cascade("haarcascade_frontalface_default.xml")
        if cascade is None:
            return None
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces) == 0:
            return None
        return None

    def extract_landmark_distances(self, landmarks: np.ndarray) -> np.ndarray:
        if landmarks is None:
            return np.array([])
        distances = []
        n = len(landmarks)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(landmarks[i] - landmarks[j])
                distances.append(d)
        return np.array(distances)

    def extract_landmark_angles(self, landmarks: np.ndarray) -> np.ndarray:
        if landmarks is None:
            return np.array([])
        angles = []
        n = len(landmarks)
        for i in range(1, n - 1):
            v1 = landmarks[i] - landmarks[i - 1]
            v2 = landmarks[i + 1] - landmarks[i]
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm == 0:
                continue
            angle = np.arccos(np.clip(dot / norm, -1.0, 1.0))
            angles.append(angle)
        return np.array(angles)

    def extract_landmark_ratios(self, landmarks: np.ndarray) -> np.ndarray:
        if landmarks is None:
            return np.array([])
        ratios = []
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])
        face_height = np.linalg.norm(landmarks[8] - np.mean(landmarks[[27, 28, 29, 30]], axis=0))
        if face_height > 0:
            ratios.append(face_width / face_height)
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        eye_dist = np.linalg.norm(left_eye_center - right_eye_center)
        if eye_dist > 0:
            ratios.append(eye_dist / face_width)
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        if mouth_width > 0:
            ratios.append(mouth_width / face_width)
        nose_width = np.linalg.norm(landmarks[31] - landmarks[35])
        if nose_width > 0:
            ratios.append(nose_width / face_width)
        return np.array(ratios)

    def extract(self, image: np.ndarray) -> np.ndarray:
        landmarks = self.get_landmarks(image)
        if landmarks is None:
            return np.array([])
        distances = self.extract_landmark_distances(landmarks)
        angles = self.extract_landmark_angles(landmarks)
        ratios = self.extract_landmark_ratios(landmarks)
        return np.concatenate([distances, angles, ratios])

    def extract_summary(self, image: np.ndarray) -> np.ndarray:
        landmarks = self.get_landmarks(image)
        if landmarks is None:
            return np.array([])
        return self.extract_landmark_ratios(landmarks)
