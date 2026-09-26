import numpy as np
from skimage.feature import hog


class HOGFeatureExtractor:
    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (2, 2),
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def extract(self, image: np.ndarray) -> np.ndarray:
        features = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True,
        )
        return features

    def get_feature_dimension(self, image_shape: tuple) -> int:
        from skimage.feature import hog
        test = np.zeros(image_shape, dtype=np.uint8)
        features = self.extract(test)
        return len(features)
