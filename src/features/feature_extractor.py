import cv2
import yaml
import numpy as np
from typing import List, Optional
from .hog_features import HOGFeatureExtractor
from .lbp_features import LBPFeatureExtractor
from .landmark_features import LandmarkFeatureExtractor
from .geometric_features import GeometricFeatureExtractor
from .statistical_features import StatisticalFeatureExtractor
from .shape_features import ShapeFeatureExtractor


class FeatureExtractor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        feat_config = self.config["features"]
        self.hog_extractor = HOGFeatureExtractor(
            orientations=feat_config["hog"]["orientations"],
            pixels_per_cell=tuple(feat_config["hog"]["pixels_per_cell"]),
            cells_per_block=tuple(feat_config["hog"]["cells_per_block"]),
        )
        self.lbp_extractor = LBPFeatureExtractor(
            radius=feat_config["lbp"]["radius"],
            n_points=feat_config["lbp"]["n_points"],
            method=feat_config["lbp"]["method"],
        )
        self.landmark_extractor = (
            LandmarkFeatureExtractor(
                predictor_path=feat_config["landmarks"]["model_path"]
            )
            if feat_config["landmarks"]["use_landmarks"]
            else None
        )
        self.geometric_extractor = (
            GeometricFeatureExtractor()
            if feat_config["geometric"]["enabled"]
            else None
        )
        self.statistical_extractor = (
            StatisticalFeatureExtractor()
            if feat_config["statistical"]["enabled"]
            else None
        )
        self.shape_extractor = (
            ShapeFeatureExtractor()
            if feat_config["shape"]["enabled"]
            else None
        )
        self.target_size = tuple(self.config["preprocessing"]["target_size"])

    def extract_all(self, image: np.ndarray) -> np.ndarray:
        features = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)

        hog_feat = self.hog_extractor.extract(gray_resized)
        features.extend(hog_feat)

        lbp_feat = self.lbp_extractor.extract(gray_resized)
        features.extend(lbp_feat)

        if self.landmark_extractor is not None:
            try:
                landmark_feat = self.landmark_extractor.extract_summary(image)
                if len(landmark_feat) > 0:
                    features.extend(landmark_feat)
                else:
                    features.extend([0.0] * 4)
            except Exception:
                features.extend([0.0] * 4)

        if self.geometric_extractor is not None:
            try:
                geom_feat = self.geometric_extractor.extract(image)
                features.extend(geom_feat)
            except Exception:
                features.extend([0.0] * 6)

        if self.statistical_extractor is not None:
            try:
                stat_feat = self.statistical_extractor.extract(gray_resized)
                features.extend(stat_feat)
            except Exception:
                features.extend([0.0] * 18)

        if self.shape_extractor is not None:
            try:
                shape_feat = self.shape_extractor.extract(image)
                features.extend(shape_feat)
            except Exception:
                features.extend([0.0] * 3)

        vector = np.array(features, dtype=np.float64)
        # A degenerate frame (blank / fully saturated) can still produce NaN or
        # inf in individual feature blocks. Sanitise here so one bad frame
        # cannot silently corrupt prediction or prototype comparison.
        if not np.all(np.isfinite(vector)):
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        return vector

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        feature_list = []
        for img in images:
            feat = self.extract_all(img)
            feature_list.append(feat)
        return np.array(feature_list)

    def get_feature_dimension(self) -> int:
        dummy = np.zeros((*self.target_size, 3), dtype=np.uint8)
        return len(self.extract_all(dummy))
