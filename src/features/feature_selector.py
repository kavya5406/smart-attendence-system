import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


class FeatureSelector:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        fs_config = self.config["feature_selection"]
        self.correlation_threshold = fs_config["correlation_threshold"]
        self.variance_threshold = fs_config["variance_threshold"]
        self.use_pca = fs_config["use_pca"]
        self.pca_variance_ratio = fs_config["pca_variance_ratio"]

        self.scaler = StandardScaler()
        self.var_selector = VarianceThreshold(threshold=self.variance_threshold)
        self.pca = None
        self._correlated_columns_to_drop = None

    def _find_correlated_columns(self, X: np.ndarray):
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self._correlated_columns_to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]

    def _remove_correlated(self, X: np.ndarray) -> np.ndarray:
        if self._correlated_columns_to_drop is None:
            return X
        df = pd.DataFrame(X)
        df = df.drop(columns=self._correlated_columns_to_drop, errors="ignore")
        return df.values

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        X_scaled = self.scaler.fit_transform(X)
        X_filtered = self.var_selector.fit_transform(X_scaled)
        self._find_correlated_columns(X_filtered)
        X_uncorrelated = self._remove_correlated(X_filtered)

        if self.use_pca:
            self.pca = PCA(n_components=self.pca_variance_ratio)
            X_reduced = self.pca.fit_transform(X_uncorrelated)
            return X_reduced
        return X_uncorrelated

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_filtered = self.var_selector.transform(X_scaled)
        X_uncorrelated = self._remove_correlated(X_filtered)
        if self.use_pca and self.pca is not None:
            return self.pca.transform(X_uncorrelated)
        return X_uncorrelated

    def get_feature_importance(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        model.fit(X, y)
        return model.feature_importances_
