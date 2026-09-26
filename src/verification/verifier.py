"""Unknown-face rejection for the existing classical ML model.

Why this module exists
----------------------
The existing trained artifact is a ``LogisticRegression`` whose
``predict_proba`` output is **saturated**: it returns ~1.0 for genuine faces,
for faces of people it has never seen, and even for pure random noise. A
confidence threshold on that value therefore cannot separate a known student
from an unknown one, and using it would silently mark attendance for strangers.

Measurement on the shipped dataset (5 identities x 30 images) shows why:

* ``predict_proba`` confidence: 1.0000 for *everything*, including noise.
* Cosine similarity to class prototypes in the **selected 715-dim space**
  (what the classifier itself consumes): AUC 0.556 - effectively random.
* Cosine similarity to class prototypes in the **raw 1805-dim scaled space**:
  AUC 0.994, nearest-prototype accuracy 99.3%.

The ``FeatureSelector`` therefore discards most of the discriminative
information, and the classifier's own confidence carries no usable signal.

What this module does
---------------------
It adds a **verification layer on top of** - not instead of - the existing
model:

1. The existing model still produces the predicted student id (unchanged).
2. This verifier independently measures how close the incoming face is to the
   registered images of that predicted student, using cosine similarity to a
   per-class prototype in the raw scaled feature space.
3. If that similarity is below a calibrated threshold, the face is reported as
   ``UNKNOWN`` and attendance is **not** marked.

The threshold is calibrated from real data (see
``calibrate_threshold``) and can be overridden with the
``RECOGNITION_THRESHOLD`` environment variable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.settings import get_settings
from ..models.predict import AttendancePredictor

CACHE_VERSION = 2


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class VerificationResult:
    """Outcome of verifying a prediction against the registered prototypes."""

    accepted: bool
    student_id: str
    similarity: float
    threshold: float
    runner_up: Optional[str] = None
    runner_up_similarity: Optional[float] = None
    method: str = "cosine_similarity_to_class_prototype_raw1805"
    reason: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class FaceVerifier:
    """Builds per-class prototypes from the registered dataset and verifies."""

    def __init__(
        self,
        predictor: AttendancePredictor,
        dataset_dir: Optional[Path] = None,
        cache_path: Optional[Path] = None,
        threshold: Optional[float] = None,
    ):
        self.predictor = predictor
        self.dataset_dir = Path(dataset_dir) if dataset_dir else get_settings().dataset_path
        self.threshold = float(threshold) if threshold is not None else get_settings().recognition_threshold
        self.cache_path = Path(cache_path) if cache_path else (Path(__file__).resolve().parents[2] / "models" / "prototypes.npz")

        self.prototypes: Dict[str, np.ndarray] = {}
        self.sample_counts: Dict[str, int] = {}
        self.built = False
        self.build_error: Optional[str] = None

    # ------------------------------------------------------------------
    # prototype construction
    # ------------------------------------------------------------------
    def build(self, feature_extractor, image_processor, face_detector, force: bool = False) -> bool:
        """Build (or load cached) per-class prototypes.

        ``feature_extractor``/``image_processor``/``face_detector`` are injected
        so this module stays independent of the concrete pipeline classes.
        """
        if self.built and not force:
            return True
        if force or not self._load_cache():
            try:
                self._build_prototypes(feature_extractor, image_processor, face_detector)
                self._save_cache()
            except Exception as exc:  # pragma: no cover - defensive
                self.build_error = str(exc)
                return False
        self.built = bool(self.prototypes)
        return self.built

    def _iter_images(self) -> List[Tuple[str, Path]]:
        """Yield ``(student_id, image_path)`` for every valid dataset image."""
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        found: List[Tuple[str, Path]] = []
        if not self.dataset_dir.exists():
            return found
        for student_dir in sorted(p for p in self.dataset_dir.iterdir() if p.is_dir()):
            name = student_dir.name
            # Skip archive/metadata folders that are not student identities.
            if name.startswith((".", "__")) or name.lower().startswith("archive"):
                continue
            for path in sorted(student_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in exts:
                    continue
                if path.name.startswith("._") or path.name in (".DS_Store", "Thumbs.db"):
                    continue
                found.append((name, path))
        return found

    def _build_prototypes(self, feature_extractor, image_processor, face_detector) -> None:
        import cv2

        buckets: Dict[str, List[np.ndarray]] = {}
        for student_id, path in self._iter_images():
            image = cv2.imread(str(path))
            if image is None:
                continue
            face = face_detector.crop_face(image)
            if face is None:
                face = image
            processed = image_processor.preprocess(face)
            vector = feature_extractor.extract_all(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))
            buckets.setdefault(student_id, []).append(np.asarray(vector, dtype=np.float64).reshape(-1))

        scaled: Dict[str, np.ndarray] = {}
        for student_id, vectors in buckets.items():
            if len(vectors) == 0:
                continue
            matrix = np.vstack(vectors)
            if self.predictor.scaler is not None:
                matrix = self.predictor.scaler.transform(matrix)
            scaled[student_id] = matrix
            self.sample_counts[student_id] = len(vectors)

        self.prototypes = {
            student_id: _l2_normalise(matrix.mean(axis=0, keepdims=True))[0]
            for student_id, matrix in scaled.items()
        }

    def _save_cache(self) -> None:
        if not self.prototypes:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {f"proto__{k}": v for k, v in self.prototypes.items()}
            payload["meta__version"] = np.array([CACHE_VERSION])
            payload["meta__counts"] = np.array(
                json.dumps({k: int(v) for k, v in self.sample_counts.items()})
            )
            np.savez_compressed(self.cache_path, **payload)
        except Exception:
            pass

    def _load_cache(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            data = np.load(self.cache_path, allow_pickle=False)
            # ravel() so a scalar saved by an older version still reads back.
            version = int(np.asarray(data["meta__version"]).ravel()[0])
            if version != CACHE_VERSION:
                return False
            protos = {k[len("proto__"):]: data[k] for k in data.files if k.startswith("proto__")}
            if not protos:
                return False
            self.prototypes = protos
            self.sample_counts = json.loads(str(np.asarray(data["meta__counts"]).ravel()[0]))
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # verification
    # ------------------------------------------------------------------
    def similarities(self, raw_features: np.ndarray) -> Dict[str, float]:
        """Cosine similarity of one raw feature vector to every prototype."""
        if not self.prototypes:
            return {}
        matrix = np.asarray(raw_features, dtype=np.float64).reshape(1, -1)
        if self.predictor.scaler is not None:
            matrix = self.predictor.scaler.transform(matrix)
        vector = _l2_normalise(matrix)
        return {
            student_id: float(np.dot(vector[0], prototype))
            for student_id, prototype in self.prototypes.items()
        }

    def verify(self, raw_features: np.ndarray, predicted_student_id: str) -> VerificationResult:
        """Verify a model prediction against the registered prototypes.

        Acceptance requires the incoming face to reach the calibrated
        threshold against a registered identity. The identity that gets
        recorded is the one measured by the prototypes; when the classifier
        named a different student that disagreement is surfaced in
        :attr:`VerificationResult.reason` and ``runner_up`` rather than being
        hidden, so the API response always shows both the model output and the
        verification outcome.
        """
        sims = self.similarities(raw_features)

        if not sims:
            return VerificationResult(
                accepted=False,
                student_id=None,
                similarity=0.0,
                threshold=self.threshold,
                reason="No registered prototypes available; cannot verify identity.",
            )

        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        nearest_id, nearest_sim = ranked[0]
        runner_up, runner_up_sim = (ranked[1] if len(ranked) > 1 else (None, None))

        model_sim = sims.get(predicted_student_id)
        agrees = model_sim is not None and abs(model_sim - nearest_sim) < 1e-9

        if nearest_sim < self.threshold:
            # A rejected face is not identified: no identity is reported, only
            # the nearest candidate kept for diagnostics.
            return VerificationResult(
                accepted=False,
                student_id=None,
                similarity=round(nearest_sim, 4),
                threshold=self.threshold,
                runner_up=nearest_id,
                runner_up_similarity=round(nearest_sim, 4),
                reason=(
                    f"Face does not match any registered student closely enough "
                    f"(nearest {nearest_id} at {nearest_sim:.3f} < threshold {self.threshold:.2f})."
                ),
            )

        if agrees:
            reason = "Model prediction confirmed by registered-face prototypes."
        else:
            reason = (
                f"Model predicted {predicted_student_id} but the registered-face "
                f"prototypes match {nearest_id} more closely; the verified identity "
                "is the one recorded."
            )

        return VerificationResult(
            accepted=True,
            student_id=nearest_id,
            similarity=round(nearest_sim, 4),
            threshold=self.threshold,
            runner_up=runner_up if not agrees else runner_up,
            runner_up_similarity=round(runner_up_sim, 4) if runner_up_sim is not None else None,
            reason=reason,
        )

    def _measured_metrics(self) -> Dict:
        """Load the measured numbers produced by scripts/measure_inference.py.

        Nothing is estimated at runtime: if the measurement file is missing the
        app reports that the numbers are unavailable.
        """
        path = Path(__file__).resolve().parents[2] / "models" / "verification_metrics.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def info(self) -> Dict:
        measured = self._measured_metrics()
        return {
            "method": "cosine_similarity_to_class_prototype_raw1805",
            "threshold": self.threshold,
            "prototypes_built": self.built,
            "registered_identities": sorted(self.prototypes),
            "samples_per_identity": {k: int(v) for k, v in sorted(self.sample_counts.items())},
            "build_error": self.build_error,
            "measured": measured,
            "note": (
                "The trained model's predict_proba is saturated (it returns ~1.0 for any "
                "input, including noise), so confidence cannot be used to reject unknown "
                "faces. Identity is therefore verified by cosine similarity to per-class "
                "prototypes computed from the registered dataset in the raw 1805-dim scaled "
                "feature space, and attendance is recorded only when that similarity reaches "
                "the threshold. Measured figures come from scripts/measure_inference.py and "
                "are stored in models/verification_metrics.json."
            ),
        }
