import numpy as np
import pytest
from src.features.hog_features import HOGFeatureExtractor
from src.features.lbp_features import LBPFeatureExtractor
from src.features.statistical_features import StatisticalFeatureExtractor
from src.features.shape_features import ShapeFeatureExtractor


@pytest.fixture
def sample_gray():
    return np.random.randint(0, 256, (128, 128), dtype=np.uint8)


@pytest.fixture
def sample_color():
    return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)


class TestHOGFeatures:
    def test_extract(self, sample_gray):
        hog = HOGFeatureExtractor()
        features = hog.extract(sample_gray)
        assert len(features) > 0
        assert features.dtype == np.float64


class TestLBPFeatures:
    def test_extract(self, sample_gray):
        lbp = LBPFeatureExtractor()
        features = lbp.extract(sample_gray)
        assert len(features) > 0
        assert np.isclose(features.sum(), 1.0, atol=0.01)

    def test_multiscale(self, sample_gray):
        lbp = LBPFeatureExtractor()
        features = lbp.extract_multiscale(sample_gray)
        assert len(features) > 0


class TestStatisticalFeatures:
    def test_extract(self, sample_gray):
        stat = StatisticalFeatureExtractor()
        features = stat.extract(sample_gray)
        assert len(features) == 18
        assert features[0] > 0


class TestShapeFeatures:
    def test_extract(self, sample_color):
        shape = ShapeFeatureExtractor()
        features = shape.extract(sample_color)
        assert len(features) == 3
