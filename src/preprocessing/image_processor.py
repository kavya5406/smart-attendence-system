"""Image preprocessing for the face recognition pipeline.

The trained artifacts in ``models/`` were fitted on **64x64** preprocessed
faces. The default constructor therefore reads ``target_size`` from
``config/config.yaml`` instead of hardcoding a value, because instantiating
this class with a 128x128 default silently changes the HOG/LBP feature
geometry and destroys prediction accuracy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..core.settings import target_size as configured_target_size


class ImageProcessor:
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        apply_hist_equalization: Optional[bool] = None,
        apply_gaussian_blur: Optional[bool] = None,
        blur_kernel_size: Optional[int] = None,
    ):
        if target_size is None:
            target_size = configured_target_size()
        self.target_size = (int(target_size[0]), int(target_size[1]))
        if apply_hist_equalization is None:
            apply_hist_equalization = True
        if apply_gaussian_blur is None:
            apply_gaussian_blur = True
        if blur_kernel_size is None:
            blur_kernel_size = 3
        self.apply_hist_equalization = apply_hist_equalization
        self.apply_gaussian_blur = apply_gaussian_blur
        self.blur_kernel_size = int(blur_kernel_size)

    def resize(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Cannot convert None to grayscale")
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        if not self.apply_hist_equalization:
            return image
        return cv2.equalizeHist(image)

    def gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        if not self.apply_gaussian_blur:
            return image
        k = self.blur_kernel_size
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(image, (k, k), 0)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = self.to_grayscale(image)
        resized = self.resize(gray)
        equalized = self.histogram_equalization(resized)
        blurred = self.gaussian_blur(equalized)
        return blurred

    def preprocess_pipeline(self, image: np.ndarray) -> np.ndarray:
        return self.preprocess(image)
