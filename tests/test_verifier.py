"""Unknown-face rejection using the real prototype verifier."""

import cv2
import numpy as np
import pytest

from src.verification.verifier import FaceVerifier
from tests.conftest import student_images


@pytest.fixture(scope="module")
def verifier(pipeline, dataset_dir, sandbox):
    from src.verification.verifier import FaceVerifier

    instance = FaceVerifier(pipeline["predictor"], cache_path=sandbox / "prototypes.npz")
    instance.build(pipeline["extractor"], pipeline["processor"], pipeline["detector"], force=True)
    if not instance.built:
        pytest.skip("verifier could not build prototypes")
    return instance


def _features(pipeline, image):
    processed = pipeline["processor"].preprocess(image)
    return pipeline["extractor"].extract_all(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))


def test_prototypes_cover_model_identities(verifier, pipeline, dataset_dir):
    folders = {p.name for p in dataset_dir.iterdir() if p.is_dir()}
    expected = folders & set(pipeline["predictor"].known_student_ids())
    assert set(verifier.prototypes) == expected
    assert expected, "the sandbox dataset must contain at least one known identity"


def test_threshold_is_configured(verifier):
    assert verifier.threshold == pytest.approx(0.08)
    assert verifier.info()["threshold"] == verifier.threshold


def test_known_face_is_accepted(verifier, pipeline, synthetic_dataset):
    if synthetic_dataset:
        pytest.skip("generated frames are not real faces; acceptance is data dependent")
    images = student_images(3)
    if not images:
        pytest.skip("no dataset images available")

    accepted = 0
    for path in images:
        result = verifier.verify(_features(pipeline, cv2.imread(str(path))), "S0001")
        if result.accepted:
            accepted += 1
            assert result.student_id == "S0001"
    assert accepted >= 2, f"only {accepted}/{len(images)} known faces accepted"


def test_blank_image_is_rejected(verifier, pipeline):
    blank = np.zeros((64, 64), dtype=np.uint8)
    result = verifier.verify(_features(pipeline, blank), "S0001")
    assert result.accepted is False
    assert result.student_id is None
    assert result.similarity < verifier.threshold


def test_noise_is_rejected(verifier, pipeline, synthetic_dataset):
    # With the generated fallback dataset the prototypes are themselves noise, so
    # a fixed threshold cannot separate anything. RECOGNITION_THRESHOLD is
    # calibrated against real faces by scripts/measure_inference.py.
    if synthetic_dataset:
        pytest.skip("threshold calibration only applies to a real dataset")
    rng = np.random.default_rng(7)
    for _ in range(3):
        noise = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        result = verifier.verify(_features(pipeline, noise), "S0001")
        assert result.accepted is False
        assert result.similarity < verifier.threshold


def test_similarities_cover_all_identities(verifier, pipeline):  # noqa: D401
    images = student_images(1)
    if not images:
        pytest.skip("no dataset images available")
    scores = verifier.similarities(_features(pipeline, cv2.imread(str(images[0]))))
    assert set(scores) == set(verifier.prototypes)
    assert all(-1.0001 <= value <= 1.0001 for value in scores.values())


def test_info_does_not_claim_unmeasured_accuracy(verifier):
    info = verifier.info()
    assert "threshold" in info
    assert "method" in info
    # the project must not invent accuracy numbers
    assert "accuracy" not in {k.lower() for k in info}


def test_prototype_cache_round_trips(verifier, pipeline, sandbox):
    """A saved cache must be reusable, otherwise every start rebuilds it."""
    fresh = FaceVerifier(pipeline["predictor"], cache_path=sandbox / "cache-roundtrip.npz")
    assert fresh._load_cache() is False
    assert fresh.build(pipeline["extractor"], pipeline["processor"], pipeline["detector"]) is True
    assert (sandbox / "cache-roundtrip.npz").exists()

    reloaded = FaceVerifier(pipeline["predictor"], cache_path=sandbox / "cache-roundtrip.npz")
    assert reloaded._load_cache() is True
    assert set(reloaded.prototypes) == set(verifier.prototypes)
    assert reloaded.sample_counts == verifier.sample_counts


def test_rebuild_is_fast_when_cached(verifier, pipeline, sandbox):
    import time

    cached = FaceVerifier(pipeline["predictor"], cache_path=sandbox / "cache-roundtrip.npz")
    start = time.time()
    assert cached.build(pipeline["extractor"], pipeline["processor"], pipeline["detector"]) is True
    # loading the cache must not re-read the dataset
    assert time.time() - start < 2.0
