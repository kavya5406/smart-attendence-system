import numpy as np
from skimage.feature import local_binary_pattern


class LBPFeatureExtractor:
    def __init__(
        self,
        radius: int = 1,
        n_points: int = 8,
        method: str = "uniform",
        n_bins: int = None,
    ):
        self.radius = radius
        self.n_points = n_points
        self.method = method
        if method == "uniform":
            self.n_bins = n_bins or n_points + 2
        else:
            self.n_bins = n_bins or 256

    def extract(self, image: np.ndarray) -> np.ndarray:
        lbp = local_binary_pattern(
            image, self.n_points, self.radius, method=self.method
        )
        n_bins = self.n_bins
        if self.method == "uniform":
            n_bins = self.n_points + 2
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True,
        )
        return hist

    def extract_multiscale(
        self, image: np.ndarray, radii: list = None
    ) -> np.ndarray:
        if radii is None:
            radii = [1, 2, 3]
        all_features = []
        for radius in radii:
            n_points = 8 * radius
            lbp = local_binary_pattern(
                image, n_points, radius, method=self.method
            )
            n_bins = n_points + 2 if self.method == "uniform" else 256
            hist, _ = np.histogram(
                lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
            )
            all_features.extend(hist)
        return np.array(all_features)
