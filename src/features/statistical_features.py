import numpy as np
from scipy import stats
from skimage.measure import shannon_entropy


class StatisticalFeatureExtractor:
    def __init__(self):
        pass

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

        mean = np.mean(image)
        std = np.std(image)
        variance = np.var(image)
        min_val = np.min(image)
        max_val = np.max(image)
        range_val = max_val - min_val
        median = np.median(image)
        entropy_val = shannon_entropy(image)

        # On a constant image (e.g. a blank/overexposed camera frame) skew and
        # kurtosis are 0/0 and come back as NaN, which would poison the whole
        # feature vector. Guard those two moments explicitly.
        flat = float(np.std(image))
        if flat < 1e-8:
            skewness = 0.0
            kurtosis = 0.0
        else:
            skewness = float(stats.skew(image.ravel()))
            kurtosis = float(stats.kurtosis(image.ravel()))
        if not np.isfinite(skewness):
            skewness = 0.0
        if not np.isfinite(kurtosis):
            kurtosis = 0.0

        percentiles = np.percentile(image, [10, 25, 50, 75, 90])

        hist, _ = np.histogram(image.ravel(), bins=32, range=(0, 256), density=True)
        hist_entropy = -np.sum(hist * np.log(hist + 1e-10))

        sobel_x = np.abs(np.gradient(image.astype(float), axis=1))
        sobel_y = np.abs(np.gradient(image.astype(float), axis=0))
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        mean_gradient = np.mean(gradient_magnitude)
        std_gradient = np.std(gradient_magnitude)

        features = np.array([
            mean,
            std,
            variance,
            min_val,
            max_val,
            range_val,
            median,
            entropy_val,
            skewness,
            kurtosis,
            percentiles[0],
            percentiles[1],
            percentiles[2],
            percentiles[3],
            percentiles[4],
            hist_entropy,
            mean_gradient,
            std_gradient,
        ])
        return features

    def extract_pixel_stats(self, image: np.ndarray) -> np.ndarray:
        return self.extract(image)
