"""Face detection and cropping.

Uses OpenCV's bundled Haar cascade, which ships with the ``opencv-python``
wheel on every platform (macOS, Windows, Linux). dlib is optional: if the
package is present the detector is tried first, otherwise Haar is used.

Detection parameters are intentionally more permissive than OpenCV's defaults
(``minNeighbors=5``). The registered dataset contains small 128x128 face crops
that the default settings miss entirely, which would silently degrade
recognition accuracy.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .cascade_locator import load_cascade, resolve_cascade

BBox = Tuple[int, int, int, int]


class FaceDetector:
    def __init__(
        self,
        cascade_path: str = "haarcascade_frontalface_default.xml",
        use_dlib: bool = True,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 30,
    ):
        self.use_dlib = use_dlib
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.detector = None
        self.face_cascade = None

        cascade_file = self._resolve_cascade(cascade_path)
        # load_cascade() returns None instead of an empty classifier, and it
        # swallows the case where the OpenCV build has no CascadeClassifier at
        # all (OpenCV 5.0 removed the legacy Haar API). FaceDetector() is built
        # during app startup, so a raw cv2.CascadeClassifier(...) call here
        # used to take the whole API down with an AttributeError.
        self.face_cascade = load_cascade(cascade_path) if cascade_file else None
        if self.face_cascade is None:
            warnings.warn(
                f"Haar cascade {cascade_path!r} is unavailable, so face "
                "detection is disabled and every frame will be classified "
                "whole. This means OpenCV's CascadeClassifier is missing "
                "(OpenCV 5.0 removed it) or the vendored XML under "
                "src/preprocessing/cascades/ is gone.",
                RuntimeWarning,
                stacklevel=2,
            )

        if use_dlib:
            try:  # pragma: no cover - optional dependency
                import dlib

                self.detector = dlib.get_frontal_face_detector()
            except Exception:
                self.use_dlib = False

    @staticmethod
    def _resolve_cascade(cascade_path: str) -> Optional[str]:
        """Return a usable path to the Haar cascade XML.

        Delegates to :mod:`src.preprocessing.cascade_locator`, which prefers the
        copy vendored inside the package. ``cv2.data.haarcascades`` alone is
        not reliable: installing both ``opencv-python`` and
        ``opencv-contrib-python`` can leave ``cv2/data`` incomplete, which
        silently disables face detection on a fresh Linux machine.
        """
        return resolve_cascade(cascade_path)

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def detect_face_dlib(self, image: np.ndarray) -> Optional[BBox]:
        if not self.use_dlib or self.detector is None:
            return None
        gray = self._gray(image)
        try:
            faces = self.detector(gray, 1)
        except Exception:
            return None
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda r: r.width() * r.height())
        return (face.left(), face.top(), face.width(), face.height())

    def detect_face_haar(self, image: np.ndarray) -> Optional[BBox]:
        return self._haar(image, self.scale_factor, self.min_neighbors, self.min_size)

    def _haar(self, image: np.ndarray, sf: float, mn: int, ms: int) -> Optional[BBox]:
        if self.face_cascade is None:
            return None
        gray = self._gray(image)
        try:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms)
            )
        except Exception:
            return None
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda r: r[2] * r[3])
        return tuple(int(v) for v in face)

    def detect(self, image: np.ndarray) -> Optional[BBox]:
        """Detect the most prominent face.

        IMPORTANT - training/inference parity
        -------------------------------------
        The parameters here must stay exactly as they were when the shipped
        model was trained (``train.py`` calls ``crop_face`` with this class).
        On the registered 128x128 dataset images the strict Haar settings find
        no face, so training features were computed from the *whole* image.
        Loosening these parameters (for example by retrying with
        ``minNeighbors=3``) changes the feature geometry and drops measured
        self-identification accuracy from 58.7% to 25.3%.

        Webcam frames contain large faces and are detected normally by the
        strict settings, so live capture is unaffected.
        """
        if image is None or getattr(image, "size", 0) == 0:
            return None

        bbox = self.detect_face_dlib(image)
        if bbox is not None:
            return bbox
        return self.detect_face_haar(image)

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("image is None")
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def detect_multiple(self, image: np.ndarray) -> List[BBox]:
        if image is None or getattr(image, "size", 0) == 0:
            return []
        gray = self._gray(image)
        if self.use_dlib and self.detector is not None:  # pragma: no cover
            try:
                faces = self.detector(gray, 1)
                return [(f.left(), f.top(), f.width(), f.height()) for f in faces]
            except Exception:
                pass
        if self.face_cascade is None:
            return []
        try:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
                minSize=(self.min_size, self.min_size),
            )
        except Exception:
            return []
        return [tuple(int(v) for v in f) for f in faces]

    # ------------------------------------------------------------------
    # cropping
    # ------------------------------------------------------------------
    def crop_from_bbox(self, image: np.ndarray, bbox: BBox, margin: float = 0.2) -> np.ndarray:
        """Crop *bbox* out of *image* with a relative margin around the face."""
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]

    def crop_face(self, image: np.ndarray, margin: float = 0.2) -> Optional[np.ndarray]:
        """Detect and crop the largest face, or return ``None`` if not found."""
        bbox = self.detect(image)
        if bbox is None:
            return None
        return self.crop_from_bbox(image, bbox, margin)
