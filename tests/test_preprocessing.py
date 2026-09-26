"""Preprocessing, feature extraction and the real model stack."""

import cv2
import numpy as np
import pytest

from src.preprocessing.image_processor import ImageProcessor


def test_preprocess_uses_configured_64x64():
    processor = ImageProcessor()
    result = processor.preprocess(np.full((128, 128, 3), 128, dtype=np.uint8))
    assert result.shape == (64, 64)
    assert result.dtype == np.uint8


def test_preprocess_accepts_gray_and_colour():
    processor = ImageProcessor()
    gray = np.full((120, 90), 200, dtype=np.uint8)
    colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert processor.preprocess(gray).shape == (64, 64)
    assert processor.preprocess(colour).shape == (64, 64)


def test_face_detector_returns_none_on_noise():
    from src.preprocessing.face_detector import FaceDetector

    detector = FaceDetector()
    noise = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    assert detector.crop_face(noise) is None


def test_blank_image_features_are_finite(pipeline):
    """A blank frame must not produce NaN (this broke inference before)."""
    processor = pipeline["processor"]
    extractor = pipeline["extractor"]
    blank = np.zeros((64, 64), dtype=np.uint8)
    features = extractor.extract_all(cv2.cvtColor(processor.preprocess(blank), cv2.COLOR_GRAY2BGR))
    assert np.atleast_2d(features).shape[1] == 1805
    assert np.isfinite(features).all()


def test_noise_features_are_finite(pipeline):
    processor = pipeline["processor"]
    extractor = pipeline["extractor"]
    noise = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    features = extractor.extract_all(cv2.cvtColor(processor.preprocess(noise), cv2.COLOR_GRAY2BGR))
    assert np.isfinite(features).all()


def test_real_image_produces_expected_feature_width(pipeline):
    import cv2 as _cv2

    from tests.conftest import student_images

    images = student_images(1)
    if not images:
        pytest.skip("no dataset images available")

    image = _cv2.imread(str(images[0]))
    processor = pipeline["processor"]
    features = pipeline["extractor"].extract_all(
        _cv2.cvtColor(processor.preprocess(image), _cv2.COLOR_GRAY2BGR)
    )
    assert np.atleast_2d(features).shape == (1, 1805)
    assert np.isfinite(features).all()


def test_model_exposes_consistent_dimensions(pipeline):
    predictor = pipeline["predictor"]
    info = predictor.model_info()
    assert info["raw_feature_dim"] == 1805
    assert info["model_feature_dim"] == 715
    assert predictor.known_student_ids() == ["S0001", "S0002", "S0003", "S0004", "S0005"]


def test_predict_returns_known_label(pipeline):
    from tests.conftest import student_images

    images = student_images(1)
    if not images:
        pytest.skip("no dataset images available")

    image = cv2.imread(str(images[0]))
    processor = pipeline["processor"]
    features = pipeline["extractor"].extract_all(
        cv2.cvtColor(processor.preprocess(image), cv2.COLOR_GRAY2BGR)
    )
    student_id, confidence = pipeline["predictor"].predict(features)
    assert student_id in pipeline["predictor"].known_student_ids()
    assert 0.0 <= confidence <= 1.0


def test_load_errors_empty_when_artifacts_present(pipeline):
    assert pipeline["predictor"].load_errors == {}
    assert pipeline["predictor"].is_ready is True
