import os
import sys
import argparse
import mlflow
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.train import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.mlops.mlflow_tracker import MLflowTracker
from src.mlops.model_versioning import ModelVersioning
from src.data.dataset import FaceDataset
from src.features.feature_extractor import FeatureExtractor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.image_processor import ImageProcessor
import cv2
import numpy as np
from tqdm import tqdm


def train(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Smart Attendance System - Training Pipeline")
    print("=" * 60)

    dataset_dir = Path(config["data"]["raw_dir"])
    if not dataset_dir.exists():
        print(f"Dataset directory {dataset_dir} not found.")
        return

    student_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if not student_dirs:
        print(f"No student directories found in {dataset_dir}")
        return

    print(f"\nStudents found: {len(student_dirs)}")
    for sd in student_dirs:
        imgs = list(sd.glob("*"))
        print(f"  {sd.name}: {len(imgs)} images")

    print("\n[1] Preprocessing...")
    detector = FaceDetector(
        cascade_path=config["preprocessing"]["face_detection_model"],
        use_dlib=config["preprocessing"]["use_dlib_detector"],
    )
    processor = ImageProcessor(
        target_size=tuple(config["preprocessing"]["target_size"]),
        apply_hist_equalization=config["preprocessing"]["apply_hist_equalization"],
        apply_gaussian_blur=config["preprocessing"]["apply_gaussian_blur"],
        blur_kernel_size=config["preprocessing"]["blur_kernel_size"],
    )

    all_imgs, all_labels = [], []
    for sd in sorted(student_dirs):
        sid = sd.name
        for ip in sd.glob("*"):
            if ip.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = cv2.imread(str(ip))
            if img is None:
                continue
            face = detector.crop_face(img)
            if face is not None:
                proc = processor.preprocess(face)
            else:
                proc = processor.preprocess(img)
            all_imgs.append(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))
            all_labels.append(sid)

    print(f"  Preprocessed {len(all_imgs)} face images")

    if len(all_imgs) == 0:
        print("No images found. Exiting.")
        return

    print("\n[2] Extracting handcrafted features...")
    fe = FeatureExtractor(config_path=config_path)
    features = []
    for img in tqdm(all_imgs, desc="Feature extraction"):
        features.append(fe.extract_all(img))

    X = np.array(features)
    y = np.array(all_labels)
    print(f"  Feature matrix: {X.shape}")

    print("\n[3] Training models with hyperparameter tuning...")

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_tracker = MLflowTracker(config_path=config_path)
    with mlflow_tracker.start_run(run_name="full_pipeline"):
        mlflow_tracker.log_params({
            "num_students": len(student_dirs),
            "num_samples": len(all_imgs),
            "feature_dim": X.shape[1],
            "models": ",".join(config["model_training"]["models"]),
        })

        trainer = ModelTrainer(config_path=config_path)
        results = trainer.train_all(X, y)

        trainer.save_model("models")

        best_model_name = trainer.best_model_name
        best_report = trainer.evaluator.results.get(best_model_name, {})

        if best_report:
            mlflow_tracker.log_metrics({
                f"val_{k}": v for k, v in best_report.items()
                if isinstance(v, (int, float))
            })

            test_key = f"{best_model_name}_test"
            test_report = trainer.evaluator.results.get(test_key, {})
            if test_report:
                mlflow_tracker.log_metrics({
                    f"test_{k}": v for k, v in test_report.items()
                    if isinstance(v, (int, float))
                })

        model_versioner = ModelVersioning()
        model_versioner.register_model(
            model_path="models/best_model.pkl",
            model_name=best_model_name,
            metrics=best_report,
        )

        print(f"\nBest model: {best_model_name}")
        print(f"Weighted F1: {best_report.get('f1_weighted', 'N/A'):.4f}")

        comparison = trainer.evaluator.compare_models()
        print("\nModel comparison:")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    train(args.config)
