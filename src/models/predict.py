"""Inference wrapper around the existing trained classical ML artifacts.

The model is the **existing** ``LogisticRegression`` stored in
``models/best_model.pkl``. Nothing here retrains or substitutes the model; the
loader only makes the artifacts reliably loadable from any working directory
and exposes the real diagnostics the API needs.

Artifacts used (all four are required):

* ``best_model.pkl``                  - LogisticRegression classifier
* ``scaler.pkl``                      - StandardScaler, 1805 input features
* ``feature_selector.pkl``            - 1805 -> 715 feature selection
* ``best_model_label_encoder.pkl``    - maps class index -> student id
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from ..core.settings import get_settings


class ModelArtifactsError(RuntimeError):
    """Raised when the trained artifacts are missing or unreadable."""


class AttendancePredictor:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        selector_path: Optional[Path] = None,
        label_encoder_path: Optional[Path] = None,
        strict: bool = False,
    ):
        settings = get_settings()
        self.model_path = Path(model_path) if model_path else settings.model_path
        self.scaler_path = Path(scaler_path) if scaler_path else settings.scaler_path
        self.selector_path = Path(selector_path) if selector_path else settings.selector_path
        self.label_encoder_path = (
            Path(label_encoder_path) if label_encoder_path else settings.label_encoder_path
        )

        self.model = None
        self.scaler = None
        self.selector = None
        self.label_encoder = None
        self.load_errors: Dict[str, str] = {}
        self.expected_features: Optional[int] = None

        self._load_artifacts(strict=strict)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _load_artifacts(self, strict: bool = False) -> None:
        missing: List[str] = []
        for attr, path in (
            ("model", self.model_path),
            ("scaler", self.scaler_path),
            ("selector", self.selector_path),
            ("label_encoder", self.label_encoder_path),
        ):
            if not path.exists():
                missing.append(f"{attr}: {path}")
                self.load_errors[attr] = f"missing file: {path}"
                continue
            try:
                setattr(self, attr, joblib.load(path))
            except Exception as exc:  # pragma: no cover - corrupt artifact
                self.load_errors[attr] = f"failed to load {path.name}: {exc}"

        if missing and strict:
            raise ModelArtifactsError("; ".join(missing))

        # The RAW feature dimension is defined by the scaler (the first
        # transform applied). The model's n_features_in_ is the POST-selection
        # dimension and must not be used to validate raw input.
        raw_features = getattr(self.scaler, "n_features_in_", None)
        if raw_features is None and self.scaler is not None:
            try:
                raw_features = int(self.scaler.mean_.shape[0])
            except Exception:
                raw_features = None
        if raw_features is None:
            raw_features = getattr(self.model, "n_features_in_", None)
        if raw_features:
            self.expected_features = int(raw_features)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """True only when all four artifacts loaded and the model can predict."""
        return all(x is not None for x in (self.model, self.scaler, self.selector, self.label_encoder))

    def known_student_ids(self) -> List[str]:
        if self.label_encoder is None:
            return []
        return [str(c) for c in self.label_encoder.classes_]

    def model_info(self) -> Dict[str, Any]:
        """Descriptive metadata about the loaded model (no invented metrics)."""
        info: Dict[str, Any] = {
            "loaded": self.is_ready,
            "model_type": type(self.model).__name__ if self.model is not None else None,
            "supports_predict_proba": bool(self.model is not None and hasattr(self.model, "predict_proba")),
            "raw_feature_dim": self.expected_features,
            "model_feature_dim": getattr(self.model, "n_features_in_", None),
            "known_students": self.known_student_ids(),
            "known_student_count": len(self.known_student_ids()),
            "artifacts": {
                "model": str(self.model_path),
                "scaler": str(self.scaler_path),
                "selector": str(self.selector_path),
                "label_encoder": str(self.label_encoder_path),
            },
            "errors": self.load_errors,
        }
        metrics = self.model_path.parent / "training_metrics.json"
        if metrics.exists():
            try:
                info["training_metrics"] = json.loads(metrics.read_text(encoding="utf-8"))
            except Exception:
                info["training_metrics"] = None
        else:
            info["training_metrics"] = None
            info["training_metrics_note"] = (
                "No training metrics were recorded for the existing artifacts; "
                "accuracy/F1 are not available and are not estimated at runtime."
            )
        return info

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _transform(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            features = self.scaler.transform(features)
        if self.selector is not None:
            features = self.selector.transform(features)
        return features

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Return ``(student_id, confidence)`` for a single raw feature vector."""
        student_id, confidence, _ = self.predict_detailed(features)
        return student_id, confidence

    def predict_detailed(self, features: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Return ``(student_id, confidence, diagnostics)``.

        ``confidence`` is the model's own ``max(predict_proba)``. Note that for
        the existing artifact this value is saturated near 1.0 and is therefore
        NOT usable on its own to reject unknown faces; rejection is handled by
        :mod:`src.verification`.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Provide a trained artifact.")

        vector = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if self.expected_features and vector.shape[1] != self.expected_features:
            raise ValueError(
                f"Expected {self.expected_features} raw features, received {vector.shape[1]}. "
                "Feature extraction is out of sync with the trained artifacts."
            )

        transformed = self._transform(vector)
        proba = self.model.predict_proba(transformed)[0]
        index = int(np.argmax(proba))
        confidence = float(np.max(proba))

        classes = self.known_student_ids()
        student_id = classes[index] if index < len(classes) else str(index)

        diagnostics = {
            "class_index": index,
            "probabilities": {cls: float(p) for cls, p in zip(classes, proba)},
            "margin": float(np.sort(proba)[-1] - np.sort(proba)[-2]) if len(proba) > 1 else 1.0,
        }
        return student_id, confidence, diagnostics

    def predict_proba_vector(self, features: np.ndarray) -> Dict[str, float]:
        """Return the full probability distribution for a single feature vector."""
        vector = np.asarray(features, dtype=np.float64).reshape(1, -1)
        proba = self.model.predict_proba(self._transform(vector))[0]
        return {cls: float(p) for cls, p in zip(self.known_student_ids(), proba)}

    def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not loaded.")
        matrix = np.asarray(features, dtype=np.float64)
        transformed = self._transform(matrix)
        preds = self.model.predict(transformed)
        confidences = np.max(self.model.predict_proba(transformed), axis=1)
        if self.label_encoder is not None:
            preds = self.label_encoder.inverse_transform(preds)
        return preds, confidences
