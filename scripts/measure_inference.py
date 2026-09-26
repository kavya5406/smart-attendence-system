"""Measure the real inference numbers for the existing artifacts.

Run:  python3 scripts/measure_inference.py

Writes ``models/verification_metrics.json`` so every number the app quotes can
be traced back to a measurement. Nothing here is estimated or hardcoded.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import DatasetManager  # noqa: E402
from src.features.feature_extractor import FeatureExtractor  # noqa: E402
from src.models.predict import AttendancePredictor  # noqa: E402
from src.preprocessing.face_detector import FaceDetector  # noqa: E402
from src.preprocessing.image_processor import ImageProcessor  # noqa: E402
from src.verification.verifier import FaceVerifier  # noqa: E402


def features_for(path, detector, processor, extractor):
    image = cv2.imread(str(path))
    face = detector.crop_face(image)
    processed = processor.preprocess(face if face is not None else image)
    return extractor.extract_all(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))


def auc(positives, negatives):
    """Rank-based AUC (ties averaged)."""
    combined = [(v, 1) for v in positives] + [(v, 0) for v in negatives]
    combined.sort(key=lambda kv: kv[0])
    ranks, i = {}, 0
    values = [c[0] for c in combined]
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = rank
        i = j + 1
    pos_rank_sum = sum(ranks[k] for k, (_, label) in enumerate(combined) if label == 1)
    n_pos, n_neg = len(positives), len(negatives)
    if n_pos == 0 or n_neg == 0:
        return None
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    detector, processor, extractor = FaceDetector(), ImageProcessor(), FeatureExtractor()
    predictor = AttendancePredictor()
    if not predictor.is_ready:
        raise SystemExit(f"model not loaded: {predictor.load_errors}")

    verifier = FaceVerifier(predictor)
    verifier.build(extractor, processor, detector, force=True)
    if not verifier.built:
        raise SystemExit(f"prototypes not built: {verifier.build_error}")

    manager = DatasetManager()
    registered = []  # (truth, model_id, verifier_id, similarity)
    for sid in sorted(manager.student_ids()):
        for path in manager.list_images(sid):
            feats = features_for(path, detector, processor, extractor)
            model_id, _ = predictor.predict(feats)
            sims = verifier.similarities(feats)
            result = verifier.verify(feats, model_id)
            registered.append(
                (sid, model_id, result.student_id, max(sims.values()) if sims else 0.0)
            )

    # negatives: deterministic noise + blank frames
    rng = np.random.default_rng(20240501)
    negatives = []
    for _ in range(24):
        noise = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        processed = processor.preprocess(noise)
        feats = extractor.extract_all(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))
        sims = verifier.similarities(feats)
        negatives.append(max(sims.values()) if sims else 0.0)
    for _ in range(24):
        blank = np.zeros((128, 128, 3), dtype=np.uint8)
        processed = processor.preprocess(blank)
        feats = extractor.extract_all(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))
        sims = verifier.similarities(feats)
        negatives.append(max(sims.values()) if sims else 0.0)

    n = len(registered)
    model_correct = sum(1 for t, m, _, _ in registered if m == t)
    verifier_correct = sum(1 for t, _, v, _ in registered if v == t)
    accepted = sum(1 for _, _, v, s in registered if v is not None and s >= verifier.threshold)
    false_accept = sum(1 for s in negatives if s >= verifier.threshold)

    report = {
        "measured_on": "registered dataset in data/raw",
        "registered_images": n,
        "negative_samples": len(negatives),
        "classifier_top1_accuracy": round(model_correct / n, 4) if n else None,
        "prototype_top1_accuracy": round(verifier_correct / n, 4) if n else None,
        "prototype_acceptance_rate_on_registered": round(accepted / n, 4) if n else None,
        "false_acceptance_rate_on_negatives": round(false_accept / len(negatives), 4) if negatives else None,
        "similarity_auc_registered_vs_negative": (
            round(auc([s for *_, s in registered], negatives), 4) if negatives else None
        ),
        "recognition_threshold": verifier.threshold,
        "registered_similarity_min": round(min((s for *_, s in registered), default=0.0), 4),
        "registered_similarity_mean": round(float(np.mean([s for *_, s in registered])), 4) if n else None,
        "negative_similarity_max": round(max(negatives), 4) if negatives else None,
    }

    out = ROOT / "models" / "verification_metrics.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
