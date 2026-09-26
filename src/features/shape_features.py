import cv2
import numpy as np
from typing import Optional, Tuple


class ShapeFeatureExtractor:
    def __init__(self, threshold_method: str = "otsu"):
        self.threshold_method = threshold_method

    def _get_face_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        if self.threshold_method == "otsu":
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _get_largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def extract(self, image: np.ndarray) -> np.ndarray:
        mask = self._get_face_mask(image)
        contour = self._get_largest_contour(mask)
        if contour is None:
            return np.zeros(3)

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return np.zeros(3)

        circularity = 4 * np.pi * area / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        return np.array([area, perimeter, aspect_ratio])

    def extract_shape_moments(self, image: np.ndarray) -> np.ndarray:
        mask = self._get_face_mask(image)
        contour = self._get_largest_contour(mask)
        if contour is None:
            return np.zeros(7)

        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        return hu_moments.flatten()
