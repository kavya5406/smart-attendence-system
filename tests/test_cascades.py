"""Haar cascade resolution must not depend on how OpenCV was installed.

Regression tests for a real deployment bug: `requirements.txt` pinned both
`opencv-python` and `opencv-contrib-python`. They install into the same `cv2`
directory and overwrite each other, and on a fresh Linux install `cv2/data` was
left without the frontal-face XML. `CascadeClassifier` then returned an *empty*
classifier, so face detection was silently disabled in CI and in Docker while
the app still reported itself healthy.
"""

from __future__ import annotations

import cv2
import pytest

from src.preprocessing import cascade_locator
from src.preprocessing.cascade_locator import (
    CASCADE_DIR,
    available_cascades,
    load_cascade,
    resolve_cascade,
)
from src.preprocessing.face_detector import FaceDetector


class TestCascadeResolution:
    def test_frontalface_cascade_is_vendored_in_the_package(self):
        assert (CASCADE_DIR / "haarcascade_frontalface_default.xml").is_file()

    def test_eye_cascade_is_vendored(self):
        assert (CASCADE_DIR / "haarcascade_eye.xml").is_file()

    def test_cascades_resolve_to_a_real_file(self):
        for name in ("haarcascade_frontalface_default.xml", "haarcascade_eye.xml"):
            path = resolve_cascade(name)
            assert path is not None, name
            assert cascade_locator.Path(path).is_file(), path

    def test_works_when_cv2_data_is_broken(self, monkeypatch, tmp_path):
        """The vendored copy must win over a missing/incomplete cv2/data."""
        monkeypatch.setattr(cv2.data, "haarcascades", f"{tmp_path}/")
        path = resolve_cascade("haarcascade_frontalface_default.xml")
        assert path is not None
        assert str(path).startswith(str(CASCADE_DIR))

    def test_load_cascade_never_returns_an_empty_classifier(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cv2.data, "haarcascades", f"{tmp_path}/")
        cascade = load_cascade("haarcascade_frontalface_default.xml")
        assert cascade is not None
        assert not cascade.empty()

    def test_missing_cascade_returns_none_not_an_empty_classifier(self):
        # A None result lets callers degrade explicitly; an empty classifier
        # would fail deep inside detectMultiScale instead.
        assert load_cascade("haarcascade_does_not_exist.xml") is None

    def test_optional_nose_and_mouth_cascades_report_unavailable(self):
        # They are absent from current OpenCV wheels and were absent when the
        # model was trained. The geometric extractor must cope.
        for name in ("haarcascade_mcs_nose.xml", "haarcascade_mcs_mouth.xml"):
            if resolve_cascade(name) is None:
                assert load_cascade(name) is None

    def test_available_cascades_lists_only_xml(self):
        for name in available_cascades():
            assert name.endswith(".xml")


class TestFaceDetectorUsesResolvedCascade:
    def test_detector_has_a_working_cascade(self, recwarn):
        detector = FaceDetector(use_dlib=False)
        assert detector.face_cascade is not None
        assert not detector.face_cascade.empty()

    def test_detector_works_with_broken_cv2_data(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cv2.data, "haarcascades", f"{tmp_path}/")
        detector = FaceDetector(use_dlib=False)
        assert detector.face_cascade is not None
        assert not detector.face_cascade.empty()

    def test_no_warning_when_the_cascade_is_present(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            FaceDetector(use_dlib=False)

    def test_warns_loudly_when_the_cascade_is_missing(self, monkeypatch):
        import warnings

        from src.preprocessing import face_detector as fd

        # face_detector binds resolve_cascade at import time, so patch it there.
        monkeypatch.setattr(fd, "resolve_cascade", lambda name: None, raising=True)
        monkeypatch.setattr(fd, "load_cascade", lambda name: None, raising=True)
        with pytest.warns(RuntimeWarning, match="face detection is disabled"):
            FaceDetector(use_dlib=False)


class TestFeatureExtractorsDegradeGracefully:
    def test_geometric_extractor_tolerates_missing_optional_cascades(self, monkeypatch, tmp_path):
        from src.features.geometric_features import GeometricFeatureExtractor

        monkeypatch.setattr(cv2.data, "haarcascades", f"{tmp_path}/")
        extractor = GeometricFeatureExtractor()
        assert extractor.face_cascade is not None
        # nose/mouth are None and must not raise during extraction
        image = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8")
        vector = extractor.extract(image)
        assert vector is not None
        assert np.isfinite(np.asarray(vector, dtype=float)).all()

    def test_landmark_extractor_tolerates_missing_cascade(self, monkeypatch, tmp_path):
        from src.features.landmark_features import LandmarkFeatureExtractor

        monkeypatch.setattr(cv2.data, "haarcascades", f"{tmp_path}/")
        extractor = LandmarkFeatureExtractor()
        image = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype("uint8")
        # Must not raise even without a usable cascade.
        extractor.extract(image)


import numpy as np  # noqa: E402  (used by the graceful-degradation tests above)


class TestOpenCVSupportsHaar:
    """Guard the dependency that face detection actually rests on.

    `opencv-python>=4.10.0` resolved to OpenCV 5.0, which removed the legacy
    Haar API: `cv2.CascadeClassifier` disappeared and the `haarcascade_*.xml`
    files stopped shipping. Because `FaceDetector()` is constructed at app
    startup, that took the whole API and every container down with an
    AttributeError. requirements.txt now pins `<5.0.0`; this test makes sure a
    future bump cannot silently reintroduce the outage.
    """

    def test_opencv_version_is_pinned_below_5(self):
        import re
        from pathlib import Path

        req = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text()
        match = re.search(r"^opencv-python([<>=!~0-9.,\s]+)$", req, re.MULTILINE)
        assert match, "opencv-python must be pinned in requirements.txt"
        assert "<5" in match.group(1), (
            "opencv-python must stay below 5.0.0: OpenCV 5.0 removed "
            "cv2.CascadeClassifier and the haarcascade XML files"
        )

    def test_contrib_python_is_not_installed(self):
        from pathlib import Path

        req = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text()
        active = [
            line for line in req.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert not any("opencv-contrib-python" in line for line in active), (
            "opencv-contrib-python conflicts with opencv-python over the cv2 "
            "directory and can strip the cascade data files"
        )

    def test_cascadeclassifier_api_exists(self):
        assert hasattr(cv2, "CascadeClassifier"), (
            "this OpenCV build has no CascadeClassifier - the legacy Haar API "
            "was removed in OpenCV 5.0, so pin opencv-python<5.0.0"
        )

    def test_loaded_cascade_is_functional(self):
        cascade = load_cascade("haarcascade_frontalface_default.xml")
        assert cascade is not None and not cascade.empty()

    def test_face_detector_startup_does_not_raise(self):
        """The API constructs FaceDetector at startup; it must never raise."""
        detector = FaceDetector(use_dlib=False)
        assert detector.face_cascade is not None
