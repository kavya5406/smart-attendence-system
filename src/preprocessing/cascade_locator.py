"""Single source of truth for locating Haar cascade files.

Why this module exists
----------------------
Cascade XML files are *data*, and where they live depends on how OpenCV was
installed. ``cv2.data.haarcascades`` is not reliable:

* ``opencv-python`` and ``opencv-contrib-python`` install into the same ``cv2``
  package directory and overwrite each other. With both pinned, a fresh Linux
  install can end up with an incomplete ``cv2/data`` folder - which is exactly
  what happened in CI: the frontal-face XML was missing, so
  ``CascadeClassifier`` silently returned an *empty* classifier and face
  detection was disabled without any error.
* Current OpenCV wheels no longer ship every cascade. ``haarcascade_mcs_nose``
  and ``haarcascade_mcs_mouth`` are absent, so those two sub-detectors have
  always been empty on modern installs.

The fix is to vendor the cascades that are actually used inside the package and
resolve them with ``pathlib``, so detection behaves identically on macOS,
Windows, Linux and in Docker regardless of wheel contents. ``cv2.data`` stays as
a secondary lookup, and a missing file returns ``None`` instead of an empty
classifier so callers can degrade explicitly rather than silently.

Note on parity: the nose and mouth cascades cannot be restored from the wheel.
The model was trained without them, so they must stay unavailable - adding them
now would change the feature vector and invalidate the shipped artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

#: Cascades vendored inside the package, resolved relative to this file.
CASCADE_DIR = Path(__file__).resolve().parent / "cascades"

#: The only cascades the pipeline relies on.
VENDORED_CASCADES = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml",
)


def resolve_cascade(name: str) -> Optional[str]:
    """Return a usable path to *name*, or ``None`` if it is not available.

    Lookup order:

    1. the copy vendored in :data:`CASCADE_DIR` (deterministic, always wins),
    2. the OpenCV package's own ``data`` directory.
    """
    if not name:
        return None

    vendored = CASCADE_DIR / Path(name).name
    if vendored.is_file():
        return str(vendored)

    try:
        data_dir = Path(cv2.data.haarcascades)
    except Exception:
        return None
    if data_dir.is_dir():
        candidate = data_dir / Path(name).name
        if candidate.is_file():
            return str(candidate)
    return None


def load_cascade(name: str) -> Optional[cv2.CascadeClassifier]:
    """Load *name*, returning ``None`` when the file is genuinely unavailable.

    An empty ``CascadeClassifier`` is never returned: it would make every
    ``detectMultiScale`` call fail at runtime, far from the real cause.
    """
    path = resolve_cascade(name)
    if path is None:
        return None
    try:
        cascade = cv2.CascadeClassifier(path)
    except Exception:
        return None
    if cascade.empty():
        return None
    return cascade


def available_cascades() -> list:
    """Names of the vendored cascades, for diagnostics and tests."""
    return [p.name for p in sorted(CASCADE_DIR.glob("*.xml"))]
