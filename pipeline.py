import os
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.image_processor import ImageProcessor
from src.features.feature_extractor import FeatureExtractor
from src.features.feature_selector import FeatureSelector
from src.models.train import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.mlops.mlflow_tracker import MLflowTracker
from src.mlops.model_versioning import ModelVersioning


def run_pipeline(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Smart Attendance System - Training Pipeline")
    print("=" * 60)

    raw_dir = Path(config["data"]["raw_dir"])

    if not raw_dir.exists():
        print(f"Error: Raw data directory '{raw_dir}' does not exist.")
        print("Place student face images in data/raw/<student_id>/ directories.")
        return

    student_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not student_dirs:
        print(f"Error: No student directories found in {raw_dir}.")
        print("Structure: data/raw/<student_id>/image1.jpg")
        return

    print(f"\nFound {len(student_dirs)} students in dataset")
    for sd in student_dirs:
        imgs = list(sd.glob("*"))
        print(f"  {sd.name}: {len(imgs)} images")

    print("\n[1/5] Detecting faces and preprocessing...")
    face_detector = FaceDetector(
        cascade_path=config["preprocessing"]["face_detection_model"],
        use_dlib=config["preprocessing"]["use_dlib_detector"],
    )
    image_processor = ImageProcessor(
        target_size=tuple(config["preprocessing"]["target_size"]),
        apply_hist_equalization=config["preprocessing"]["apply_hist_equalization"],
        apply_gaussian_blur=config["preprocessing"]["apply_gaussian_blur"],
        blur_kernel_size=config["preprocessing"]["blur_kernel_size"],
    )

    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_images, all_labels = [], []
    failed = 0

    for student_dir in tqdm(sorted(student_dirs), desc="Processing students"):
        student_id = student_dir.name
        for img_path in student_dir.glob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue
            face = face_detector.crop_face(img)
            if face is None:
                failed += 1
                continue
            processed = image_processor.preprocess(face)
            save_path = processed_dir / student_id / img_path.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), processed)

            processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            all_images.append(processed_color)
            all_labels.append(student_id)

    print(f"  Preprocessed {len(all_images)} faces, {failed} skipped")

    print("\n[2/5] Extracting handcrafted features...")
    feature_extractor = FeatureExtractor(config_path=config_path)
    all_features = []
    for img in tqdm(all_images, desc="Extracting features"):
        feats = feature_extractor.extract_all(img)
        all_features.append(feats)

    X = np.array(all_features)
    y = np.array(all_labels)
    print(f"  Feature matrix shape: {X.shape}")

    np.save(os.path.join(processed_dir, "features.npy"), X)
    np.save(os.path.join(processed_dir, "labels.npy"), y)
    print(f"  Features saved to {processed_dir / 'features.npy'}")

    print("\n[3/5] Feature selection...")
    trainer = ModelTrainer(config_path=config_path)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_processed = trainer.feature_selector.fit_transform(X, y_enc)
    print(f"  After selection: {X_processed.shape[1]} features (from {X.shape[1]})")

    print("\n[4/5] Training and comparing models...")
    results = trainer.train_all(X, y)

    print("\n[5/5] Saving artifacts...")
    os.makedirs("models", exist_ok=True)
    trainer.save_model("models")

    model_versioner = ModelVersioning()
    model_versioner.register_model(
        model_path="models/best_model.pkl",
        model_name=trainer.best_model_name,
        metrics=trainer.evaluator.results.get(trainer.best_model_name, {}),
    )
    print(f"  Model versioned in registry")

    print("\n" + "=" * 60)
    print(f"Pipeline complete! Best model: {trainer.best_model_name}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Attendance System Training Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    run_pipeline(args.config)
