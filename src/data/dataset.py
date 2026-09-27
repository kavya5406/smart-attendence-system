"""Dataset management and statistics.

Everything reported here is computed by actually walking the dataset folder
and opening the images - no value is hardcoded.

Expected layout::

    data/raw/
        STUDENT_001/image1.jpg ...
        STUDENT_002/image1.jpg ...
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from ..core.settings import get_settings

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Metadata that macOS/Windows drop into zips and that must never be treated as
# student data.
IGNORED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini", ".gitkeep"}
IGNORED_PREFIXES = ("._", "._", ".__")
IGNORED_DIRS = ("__MACOSX",)

# zip-slip protection: reject names that try to escape the destination.
_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


def is_ignored_file(name: str) -> bool:
    if name in IGNORED_NAMES:
        return True
    return name.startswith("._") or name.startswith(".__")


def is_ignored_dir(name: str) -> bool:
    return name in IGNORED_DIRS or name.startswith(".") or name.startswith("__")


def sanitise_student_id(raw: str) -> str:
    """Return a filesystem-safe student id, or ``""`` if nothing survives."""
    cleaned = _SAFE_ID.sub("", str(raw).strip())
    cleaned = cleaned.strip("._-")
    return cleaned


def read_image_size(path: Path) -> Optional[Tuple[int, int]]:
    """Return ``(width, height)`` by parsing the file header only.

    Fully decoding every image makes a dataset scan take minutes because the
    dataset contains multi-megapixel photographs. Parsing the header is enough
    to reject truncated/corrupt files and to report dimensions.
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
            if not header:
                return None

            # PNG: 8-byte signature, then an IHDR chunk with width/height.
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                handle.seek(16)
                raw = handle.read(8)
                if len(raw) < 8:
                    return None
                return int.from_bytes(raw[0:4], "big"), int.from_bytes(raw[4:8], "big")

            # BMP: DIB header width/height at fixed offsets.
            if header[:2] == b"BM":
                handle.seek(18)
                raw = handle.read(8)
                if len(raw) < 8:
                    return None
                width = int.from_bytes(raw[0:4], "little", signed=True)
                height = int.from_bytes(raw[4:8], "little", signed=True)
                return width, abs(height)

            # JPEG: walk the segment markers to the start-of-frame header.
            if header[:2] == b"\xff\xd8":
                handle.seek(2)
                while True:
                    byte = handle.read(1)
                    if not byte:
                        return None
                    if byte != b"\xff":
                        continue
                    marker = handle.read(1)
                    while marker == b"\xff":
                        marker = handle.read(1)
                    if not marker:
                        return None
                    code = marker[0]
                    # SOF0..SOF15, excluding DHT/JPG/DAC which share the range.
                    if code in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        length_bytes = handle.read(2)
                        if len(length_bytes) < 2:
                            return None
                        payload = handle.read(5)
                        if len(payload) < 5:
                            return None
                        height = int.from_bytes(payload[1:3], "big")
                        width = int.from_bytes(payload[3:5], "big")
                        return width, height
                    if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
                        continue
                    length_bytes = handle.read(2)
                    if len(length_bytes) < 2:
                        return None
                    segment_length = int.from_bytes(length_bytes, "big")
                    if segment_length < 2:
                        return None
                    handle.seek(segment_length - 2, 1)
    except Exception:
        return None
    return None


@dataclass
class ImageReport:
    path: str
    valid: bool
    reason: str = ""
    width: int = 0
    height: int = 0
    face_detected: bool = False
    duplicate_of: Optional[str] = None


@dataclass
class StudentDatasetEntry:
    student_id: str
    student_name: str = ""
    image_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    duplicate_count: int = 0
    face_count: int = 0
    registered: bool = False
    images: List[ImageReport] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "student_id": self.student_id,
            "student_name": self.student_name,
            "image_count": self.image_count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "duplicate_count": self.duplicate_count,
            "images_with_faces": self.face_count,
            "registered": self.registered,
            "status": "registered" if self.registered else "not_registered",
        }


class DatasetManager:
    def __init__(self, raw_dir: Optional[Path] = None, config_path: Optional[Path] = None):
        settings = get_settings()
        self.raw_dir = Path(raw_dir) if raw_dir else settings.dataset_path
        self.config_path = Path(config_path) if config_path else settings.config_path
        self.config = self._load_config()
        self.deep = False

    def _load_config(self) -> Dict:
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        return {}

    # ------------------------------------------------------------------
    # scanning
    # ------------------------------------------------------------------
    def student_ids(self) -> List[str]:
        """Return registered student folders (macOS metadata folders excluded)."""
        if not self.raw_dir.exists():
            return []
        return sorted(
            d.name for d in self.raw_dir.iterdir() if d.is_dir() and not is_ignored_dir(d.name)
        )

    def list_images(self, student_id: str) -> List[Path]:
        folder = self.raw_dir / student_id
        if not folder.is_dir():
            return []
        return sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS and not is_ignored_file(p.name)
        )

    @staticmethod
    def _fingerprint(path: Path) -> str:
        """Content hash used for duplicate detection.

        Hashes the file size plus the first and last 64 KiB instead of the whole
        file. The dataset contains multi-megapixel photos, and reading every
        byte in full makes a dataset scan take minutes; this keeps duplicate
        detection fast and still distinguishes different images.
        """
        stat = path.stat()
        digest = hashlib.sha256()
        digest.update(str(stat.st_size).encode())
        with path.open("rb") as handle:
            digest.update(handle.read(65536))
            if stat.st_size > 131072:
                handle.seek(-65536, 2)
                digest.update(handle.read(65536))
        return digest.hexdigest()

    def inspect_student(self, student_id: str, known_hashes: Optional[Dict[str, str]] = None) -> StudentDatasetEntry:
        """Validate every image for one student.

        ``known_hashes`` maps an already-seen content hash to the path it was
        first seen at, so duplicates are detected across the whole dataset.
        ``deep=True`` additionally decodes each image to confirm it is not
        corrupt and to run face detection; the default header-only mode keeps
        large datasets fast.
        """
        entry = StudentDatasetEntry(student_id=student_id)
        seen: Dict[str, str] = known_hashes if known_hashes is not None else {}

        for path in self.list_images(student_id):
            rel = str(path.relative_to(self.raw_dir))
            digest = self._fingerprint(path)
            report = ImageReport(path=rel, valid=False)

            if digest in seen:
                report.duplicate_of = seen[digest]
                report.reason = f"duplicate of {seen[digest]}"
                entry.duplicate_count += 1
                entry.images.append(report)
                continue
            seen[digest] = rel

            size = read_image_size(path)
            if size is None:
                report.reason = "unreadable, corrupt or unsupported image file"
                entry.invalid_count += 1
                entry.images.append(report)
                continue

            report.width, report.height = size
            if report.width < 32 or report.height < 32:
                report.reason = f"image too small ({report.width}x{report.height})"
                entry.invalid_count += 1
                entry.images.append(report)
                continue

            if self.deep:
                image = cv2.imread(str(path))
                if image is None:
                    report.reason = "unreadable or corrupt image file"
                    entry.invalid_count += 1
                    entry.images.append(report)
                    continue

            report.valid = True
            report.reason = "ok"
            entry.valid_count += 1
            entry.images.append(report)

        entry.image_count = len(entry.images)
        return entry

    def scan(self, check_faces: bool = False, deep: bool = False) -> Dict:
        """Full dataset scan returning real statistics."""
        known_hashes: Dict[str, str] = {}
        students: List[StudentDatasetEntry] = []

        for student_id in self.student_ids():
            self.deep = deep
            entry = self.inspect_student(student_id, known_hashes)
            students.append(entry)
        self.deep = False

        if check_faces:
            self._annotate_faces(students)

        total_images = sum(s.image_count for s in students)
        valid_images = sum(s.valid_count for s in students)
        invalid_images = sum(s.invalid_count for s in students)
        duplicate_images = sum(s.duplicate_count for s in students)

        return {
            "dataset_path": str(self.raw_dir),
            "dataset_exists": self.raw_dir.exists(),
            "total_students": len(students),
            "total_images": total_images,
            "valid_images": valid_images,
            "invalid_images": invalid_images,
            "duplicate_images": duplicate_images,
            "students": [s.to_dict() for s in students],
            "_entries": students,
        }

    def _annotate_faces(self, entries: List[StudentDatasetEntry]) -> None:
        from ..preprocessing.face_detector import FaceDetector

        detector = FaceDetector()
        for entry in entries:
            count = 0
            for report in entry.images:
                if not report.valid:
                    continue
                image = cv2.imread(str(self.raw_dir / report.path))
                if image is None:
                    continue
                if detector.detect(image) is not None:
                    report.face_detected = True
                    count += 1
            entry.face_count = count

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def ensure_student_dir(self, student_id: str) -> Path:
        folder = self.raw_dir / student_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def add_image(self, student_id: str, source: Path, filename: Optional[str] = None) -> Path:
        folder = self.ensure_student_dir(student_id)
        name = filename or source.name
        target = folder / name
        if source.resolve() != target.resolve():
            target.write_bytes(Path(source).read_bytes())
        return target

    def remove_student(self, student_id: str) -> bool:
        folder = self.raw_dir / student_id
        if not folder.is_dir():
            return False
        for path in sorted(folder.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        folder.rmdir()
        return True


# ----------------------------------------------------------------------
# train / validation / test splitting
# ----------------------------------------------------------------------
def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ``X``/``y`` into stratified train, validation and test sets.

    Returns ``(X_train, X_val, X_test, y_train, y_val, y_test)``.

    Leakage rules enforced here:

    * The split is performed on the *indices* only, so the returned arrays are
      never shuffled out of sync with their labels.
    * The split is **stratified** on the label so every student is represented
      in all three sets. With 5 students and 30 images each this is the
      difference between a meaningful score and an ``UnboundLocalError``
      further down the pipeline.
    * Splitting happens *before* anything is fitted. The caller receives raw
      ``X_train``/``X_val``/``X_test`` and is responsible for fitting the
      scaler and the feature selector on ``X_train`` alone (see
      ``src/models/train.py``).

    This is a pure function of its inputs: it never touches the filesystem and
    never reads the dataset directory, so it is safe to unit test and to reuse
    from a notebook.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y length mismatch: X has {X.shape[0]} rows, y has {len(y)}"
        )
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

    n_samples = X.shape[0]
    n_classes, class_counts = np.unique(y, return_counts=True)
    min_class = int(class_counts.min())

    # Stratification needs at least `n_splits` members per class. Shrink the
    # test fraction if the dataset is too small, and fall back to a plain
    # (non-stratified) split rather than raising on a tiny dataset.
    if min_class < 2:
        raise ValueError(
            "Every student needs at least 2 usable images to build a test set; "
            f"the smallest class has {min_class}. Re-register more images or "
            "collect them with the Dataset page."
        )

    test_size = float(test_size)
    val_size = float(val_size)
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError(
            f"Invalid split fractions: test_size={test_size}, val_size={val_size} "
            "(both must be > 0 and sum to < 1)"
        )

    # The validation split is carved out of the *training* portion, never out
    # of the test portion. `val_fraction` is therefore expressed relative to
    # what is left after the test set has been held back.
    remaining = 1.0 - test_size
    if remaining <= 0:
        raise ValueError("test_size leaves no data for train/validation")

    val_fraction = val_size / remaining
    # A two-way stratified split of the remainder needs every class to still
    # have >= 2 members in it.
    if min_class >= 3:
        per_class_in_remainder = min_class - int(round(min_class * test_size))
        max_val_fraction = 1.0 - 1.0 / per_class_in_remainder
        val_fraction = min(val_fraction, max_val_fraction)
    else:
        # Fewer than 3 images for the smallest student: a separate validation
        # split is impossible without leaking the training set into it.
        val_fraction = 0.0
        warnings.warn(
            f"The smallest student class has only {min_class} usable image(s), "
            "so no separate validation set can be held out. Validation scores "
            "will be computed on the training data and must NOT be reported as "
            "held-out results. Register more images per student "
            "(>= 3) for a meaningful validation split.",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        # Step 1 - hold the test set back.
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # Step 2 - carve validation out of the training portion.
        if val_fraction > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_tmp, y_tmp,
                test_size=val_fraction,
                random_state=random_state,
                stratify=y_tmp,
            )
        else:
            X_train, y_train = X_tmp, y_tmp
            X_val, y_val = X_tmp, y_tmp
    except ValueError:
        # Too few samples per class for stratification: degrade to an
        # unstratified split instead of failing the whole run.
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        if val_fraction > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_tmp, y_tmp, test_size=val_fraction, random_state=random_state
            )
        else:
            X_train, y_train = X_tmp, y_tmp
            X_val, y_val = X_tmp, y_tmp

    return X_train, X_val, X_test, y_train, y_val, y_test
