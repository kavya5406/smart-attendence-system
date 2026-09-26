"""Dataset manager: metadata filtering, validation and duplicate detection."""

import cv2
import numpy as np
import pytest

from src.data.dataset import (
    VALID_EXTENSIONS,
    DatasetManager,
    is_ignored_dir,
    is_ignored_file,
    sanitise_student_id,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        (".DS_Store", True),
        ("Thumbs.db", True),
        ("._image.jpg", True),
        ("image.jpg", False),
        ("photo.JPEG", False),
    ],
)
def test_is_ignored_file(name, expected):
    assert is_ignored_file(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [("__MACOSX", True), ("__pycache__", True), (".git", True), ("S0001", False)],
)
def test_is_ignored_dir(name, expected):
    assert is_ignored_dir(name) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("S0001", "S0001"),
        ("  s0001 ", "s0001"),
        ("../../etc", "etc"),
        ("a/b", "ab"),
        ("../..", ""),
        ("", ""),
    ],
)
def test_sanitise_student_id(raw, expected):
    assert sanitise_student_id(raw) == expected


def test_scan_reports_real_dataset(dataset_dir):
    manager = DatasetManager(raw_dir=dataset_dir)
    result = manager.scan()

    assert result["dataset_exists"] is True
    assert result["total_students"] >= 1
    assert result["total_images"] >= 1
    assert result["invalid_images"] == 0
    assert result["dataset_path"].endswith("raw")


def test_scan_ignores_macos_metadata(sandbox, dataset_dir):
    import shutil

    raw = sandbox / "macos-raw"
    shutil.copytree(dataset_dir, raw)
    folder = raw / sorted(p.name for p in raw.iterdir() if p.is_dir())[0]
    (folder / ".DS_Store").write_bytes(b"junk")
    (folder / "._face.jpg").write_bytes(b"junk")
    (raw / "__MACOSX").mkdir(exist_ok=True)
    (raw / "__MACOSX" / "x.jpg").write_bytes(b"junk")

    manager = DatasetManager(raw_dir=raw)
    listed = [p.name for sid in manager.student_ids() for p in manager.list_images(sid)]
    assert ".DS_Store" not in listed
    assert "._face.jpg" not in listed
    assert "x.jpg" not in listed
    assert manager.scan()["invalid_images"] == 0


def test_invalid_and_duplicate_detection(sandbox, dataset_dir):
    import shutil

    raw = sandbox / "bad-raw"
    shutil.copytree(dataset_dir, raw)
    folder = raw / sorted(p.name for p in raw.iterdir() if p.is_dir())[0]
    original = sorted(folder.glob("*.jpg"))[0]

    (folder / "broken.jpg").write_bytes(b"not an image")
    shutil.copy(original, folder / "duplicate.jpg")

    result = DatasetManager(raw_dir=raw).scan()
    assert result["invalid_images"] >= 1
    assert result["duplicate_images"] >= 1


def test_missing_dataset_directory(sandbox):
    result = DatasetManager(raw_dir=sandbox / "does-not-exist").scan()
    assert result["dataset_exists"] is False
    assert result["total_students"] == 0
    assert result["total_images"] == 0


def test_valid_extensions_cover_training_formats():
    for ext in (".jpg", ".jpeg", ".png"):
        assert ext in VALID_EXTENSIONS
    assert ".txt" not in VALID_EXTENSIONS
    assert ".DS_Store" not in VALID_EXTENSIONS


def test_add_and_remove_student(sandbox, dataset_dir):
    import shutil

    raw = sandbox / "mutate-raw"
    shutil.copytree(dataset_dir, raw)
    manager = DatasetManager(raw_dir=raw)

    folder = manager.ensure_student_dir("T0001")
    assert folder.name == "T0001"

    image = np.full((96, 96, 3), 200, dtype=np.uint8)
    source = sandbox / "t0001.png"
    cv2.imwrite(str(source), image)

    saved = manager.add_image("T0001", source)
    assert saved.exists() and saved.parent == folder
    assert saved.name.endswith(".png")

    assert manager.remove_student("T0001") is True
    assert manager.remove_student("T0001") is False
