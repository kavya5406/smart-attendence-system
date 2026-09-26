"""Central, cross-platform configuration.

Every path in this project is resolved relative to the project root (the
directory that contains ``config/``, ``models/`` and ``src/``) and can be
overridden with an environment variable. No absolute path of any particular
machine is ever hardcoded.

Resolution order for a setting:

1. Environment variable (e.g. ``MODEL_PATH``)
2. Value in ``config/config.yaml``
3. Built-in default

Project root detection walks upwards from this file looking for a marker
directory, so the code works no matter what the current working directory is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Marker files/directories that identify the project root.
_ROOT_MARKERS = ("config", "models", "src")


def find_project_root() -> Path:
    """Locate the project root by walking upwards from this module.

    Falls back to the current working directory if no marker is found, so the
    import never fails (for example when the package is installed).
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if all((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    return Path.cwd().resolve()


BASE_DIR: Path = find_project_root()


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_path(value: Any, default: Path) -> Path:
    """Resolve *value* against the project root, honouring env overrides.

    Relative paths are anchored to :data:`BASE_DIR` so the result never depends
    on the process working directory. Absolute paths are respected as-is.
    """
    raw = _env(str(value).upper()) if value is not None else None
    candidate = Path(raw or value or default)
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()


@dataclass
class Settings:
    """Runtime settings resolved once at import time."""

    base_dir: Path = BASE_DIR
    config_path: Path = BASE_DIR / "config" / "config.yaml"

    # Data
    dataset_path: Path = BASE_DIR / "data" / "raw"
    processed_path: Path = BASE_DIR / "data" / "processed"
    database_path: Path = BASE_DIR / "data" / "attendance.db"
    database_url: str = ""

    # Model artifacts
    model_path: Path = BASE_DIR / "models" / "best_model.pkl"
    scaler_path: Path = BASE_DIR / "models" / "scaler.pkl"
    selector_path: Path = BASE_DIR / "models" / "feature_selector.pkl"
    label_encoder_path: Path = BASE_DIR / "models" / "best_model_label_encoder.pkl"

    # Frontend build output
    frontend_dist: Path = BASE_DIR / "frontend" / "dist"

    # Serving
    host: str = "0.0.0.0"
    port: int = 8000
    frontend_url: str = "http://localhost:5173"
    # Comma separated list of allowed browser origins. Empty means "any origin",
    # which is fine while the frontend is served from the same origin.
    cors_origins: List[str] = field(default_factory=list)

    # Recognition
    # The trained model's predict_proba is saturated (it returns ~1.0 even for
    # pure noise), so it CANNOT be used to reject unknown faces. Rejection is
    # instead based on distance to per-class feature prototypes built from the
    # registered dataset. See src/verification/verifier.py.
    # The threshold is read from RECOGNITION_THRESHOLD; the figures it was chosen
    # from are measured by scripts/measure_inference.py and stored in
    # models/verification_metrics.json.
    recognition_threshold: float = 0.08

    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def yaml_config(self) -> Dict[str, Any]:
        return self.raw


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and cache the resolved :class:`Settings` object."""
    config_file = Path(_env("CONFIG_PATH") or (BASE_DIR / "config" / "config.yaml"))
    raw = _load_yaml(config_file)

    data_cfg = raw.get("data", {}) or {}
    api_cfg = raw.get("api", {}) or {}
    features_cfg = raw.get("features", {}) or {}
    landmarks_cfg = features_cfg.get("landmarks", {}) or {}

    database_url = _env("DATABASE_URL") or ""
    if not database_url:
        db_path = resolve_path(
            _env("DATABASE_PATH") or data_cfg.get("database_path"),
            BASE_DIR / "data" / "attendance.db",
        )
        database_url = f"sqlite:///{db_path.as_posix()}"

    threshold = _env("RECOGNITION_THRESHOLD")
    try:
        threshold_value = float(threshold) if threshold else 0.08
    except ValueError:
        threshold_value = 0.08

    return Settings(
        config_path=config_file,
        dataset_path=resolve_path(_env("DATASET_PATH") or data_cfg.get("raw_dir"), BASE_DIR / "data" / "raw"),
        processed_path=resolve_path(
            _env("PROCESSED_PATH") or data_cfg.get("processed_dir"), BASE_DIR / "data" / "processed"
        ),
        database_path=Path(database_url.replace("sqlite:///", "")) if database_url.startswith("sqlite:///") else BASE_DIR / "data" / "attendance.db",
        database_url=database_url,
        model_path=resolve_path(_env("MODEL_PATH") or api_cfg.get("model_path"), BASE_DIR / "models" / "best_model.pkl"),
        scaler_path=resolve_path(_env("SCALER_PATH") or api_cfg.get("scaler_path"), BASE_DIR / "models" / "scaler.pkl"),
        selector_path=resolve_path(
            _env("FEATURE_SELECTOR_PATH") or _env("SELECTOR_PATH") or api_cfg.get("selector_path"),
            BASE_DIR / "models" / "feature_selector.pkl",
        ),
        label_encoder_path=resolve_path(
            _env("LABEL_ENCODER_PATH"),
            BASE_DIR / "models" / "best_model_label_encoder.pkl",
        ),
        frontend_dist=Path(_env("FRONTEND_DIST") or (BASE_DIR / "frontend" / "dist")).resolve(),
        host=_env("BACKEND_HOST") or api_cfg.get("host") or "0.0.0.0",
        port=int(_env("BACKEND_PORT") or api_cfg.get("port") or 8000),
        frontend_url=_env("FRONTEND_URL") or "http://localhost:5173",
        cors_origins=[
            origin.strip()
            for origin in (_env("CORS_ORIGINS") or "").split(",")
            if origin.strip()
        ],
        recognition_threshold=threshold_value,
        raw=raw,
    )


def target_size() -> tuple:
    """Return the configured preprocessing target size as ``(w, h)``.

    The trained model was fitted on 64x64 preprocessed faces. Callers must use
    this value; constructing :class:`~src.preprocessing.image_processor.ImageProcessor`
    with its own defaults silently uses 128x128 and destroys accuracy.
    """
    cfg = get_settings().raw.get("preprocessing", {}) or {}
    size = cfg.get("target_size", [64, 64])
    return (int(size[0]), int(size[1]))


def landmark_model_path() -> Path:
    cfg = get_settings().raw.get("features", {}) or {}
    landmarks = cfg.get("landmarks", {}) or {}
    return resolve_path(landmarks.get("model_path"), BASE_DIR / "shape_predictor_68_face_landmarks.dat")
