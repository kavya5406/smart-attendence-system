"""Shared pytest fixtures.

Every piece of writable state is redirected into a temporary sandbox *before*
the application is imported, so running the suite never touches the real
``data/raw`` dataset, the real SQLite database or ``models/prototypes.npz``.

The trained model artifacts are read-only, so the real ``models/`` directory is
used and those tests are skipped when the artifacts are absent.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# sandbox: set up BEFORE importing src.*
# ---------------------------------------------------------------------------
SANDBOX = Path(tempfile.mkdtemp(prefix="sas-tests-"))
DATASET_DIR = SANDBOX / "raw"

def _has_images(path: Path) -> bool:
    return any(p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") for p in path.rglob("*"))


def _synthesise_dataset(target: Path) -> None:
    """Create a small stand-in dataset so the suite runs without real faces.

    ``data/raw`` is git-ignored, so a fresh clone (and CI) has no images. The
    tests that need *a* dataset get deterministic synthetic frames instead of
    being skipped. These are noise patterns, not real faces, and they are only
    ever written to the temporary sandbox.
    """
    import cv2
    import numpy as np

    rng = np.random.default_rng(1234)
    for index, student in enumerate(("S0001", "S0002", "S0003")):
        folder = target / student
        folder.mkdir(parents=True)
        base = rng.integers(60, 200, size=(48, 48, 3), dtype=np.uint8)
        for shot in range(4):
            frame = np.clip(base.astype(np.int16) + rng.integers(-12, 12, base.shape), 0, 255)
            frame[:, :, index % 3] = np.clip(frame[:, :, index % 3] + 30, 0, 255)
            cv2.imwrite(str(folder / f"frame_{shot}.jpg"), frame.astype(np.uint8))


_source_raw = ROOT / "data" / "raw"
if _source_raw.is_dir() and _has_images(_source_raw):
    shutil.copytree(_source_raw, DATASET_DIR)
    SYNTHETIC = False
else:
    _synthesise_dataset(DATASET_DIR)
    SYNTHETIC = True

os.environ["DATABASE_URL"] = f"sqlite:///{(SANDBOX / 'test.db').as_posix()}"
os.environ["DATASET_PATH"] = str(DATASET_DIR)


@pytest.fixture(scope="session")
def sandbox() -> Path:
    return SANDBOX


@pytest.fixture(scope="session")
def dataset_dir() -> Path:
    return DATASET_DIR


@pytest.fixture(scope="session")
def synthetic_dataset() -> bool:
    """True when the sandbox uses generated frames instead of real images."""
    return SYNTHETIC


@pytest.fixture(scope="session")
def artifacts_present() -> bool:
    required = (
        "best_model.pkl",
        "scaler.pkl",
        "feature_selector.pkl",
        "best_model_label_encoder.pkl",
    )
    return all((ROOT / "models" / name).exists() for name in required)


@pytest.fixture(scope="session", autouse=True)
def verifier_cache_in_sandbox():
    """Keep the prototype cache out of the project's models/ directory."""
    from src.verification import verifier as verifier_module

    original = verifier_module.FaceVerifier.__init__

    def patched(self, *args, **kwargs):  # noqa: ANN001
        kwargs.setdefault("cache_path", SANDBOX / "prototypes.npz")
        original(self, *args, **kwargs)

    verifier_module.FaceVerifier.__init__ = patched
    try:
        yield
    finally:
        verifier_module.FaceVerifier.__init__ = original


@pytest.fixture(scope="session")
def predictor(artifacts_present):
    if not artifacts_present:
        pytest.skip("trained model artifacts are not present")
    from src.models.predict import AttendancePredictor

    return AttendancePredictor()


@pytest.fixture(scope="session")
def pipeline(predictor):
    """The real preprocessing + feature extraction + model stack."""
    if not predictor.is_ready:
        pytest.skip(f"model failed to load: {predictor.load_errors}")
    from src.features.feature_extractor import FeatureExtractor
    from src.preprocessing.face_detector import FaceDetector
    from src.preprocessing.image_processor import ImageProcessor

    return {
        "detector": FaceDetector(),
        "processor": ImageProcessor(),
        "extractor": FeatureExtractor(),
        "predictor": predictor,
    }


@pytest.fixture(scope="session")
def client(artifacts_present):
    if not artifacts_present:
        pytest.skip("trained model artifacts are not present")
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def student_images(limit: int = 3) -> list:
    """Real images from the first dataset identity (sandbox copy)."""
    folders = sorted(p for p in DATASET_DIR.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    if not folders:
        return []
    images = sorted(
        p for p in folders[0].iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    return images[:limit]
