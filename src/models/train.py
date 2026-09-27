import os
import yaml
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .evaluator import ModelEvaluator
from ..features.feature_selector import FeatureSelector


MODEL_REGISTRY = {
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {"random_state": 42, "n_jobs": -1, "class_weight": "balanced"},
    },
    "decision_tree": {
        "class": DecisionTreeClassifier,
        "params": {"random_state": 42, "class_weight": "balanced"},
    },
    "logistic_regression": {
        "class": LogisticRegression,
        "params": {"random_state": 42, "n_jobs": -1, "multi_class": "multinomial"},
    },
    "knn": {
        "class": KNeighborsClassifier,
        "params": {"n_jobs": -1},
    },
}


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.models_config = self.config["model_training"]
        self.hp_grids = self.config["hyperparameter_grids"]
        self.cv_folds = self.models_config["cv_folds"]
        self.scoring = self.models_config["scoring"]
        self.n_iter = self.models_config["n_iter_search"]
        self.random_state = self.models_config["random_state"]
        self.evaluator = ModelEvaluator()
        self.feature_selector = FeatureSelector(config_path)
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.best_params = None

    def _get_xgb_params(self):
        try:
            from xgboost import XGBClassifier
            return {"class": XGBClassifier, "params": {"random_state": self.random_state, "n_jobs": -1, "verbosity": 0}}
        except ImportError:
            return None

    def _get_lgbm_params(self):
        try:
            import lightgbm as lgb
            return {"class": lgb.LGBMClassifier, "params": {"random_state": self.random_state, "n_jobs": -1, "verbose": -1}}
        except ImportError:
            return None

    def _get_catboost_params(self):
        try:
            from catboost import CatBoostClassifier
            return {"class": CatBoostClassifier, "params": {"random_state": self.random_state, "verbose": 0, "allow_writing_files": False}}
        except ImportError:
            return None

    def build_model(self, model_name: str):
        if model_name in MODEL_REGISTRY:
            entry = MODEL_REGISTRY[model_name]
            return entry["class"](**entry["params"])

        registry = {
            "xgboost": self._get_xgb_params,
            "lightgbm": self._get_lgbm_params,
            "catboost": self._get_catboost_params,
        }

        if model_name in registry:
            entry = registry[model_name]()
            if entry:
                return entry["class"](**entry["params"])
        return None

    def get_param_grid(self, model_name: str) -> Dict:
        return self.hp_grids.get(model_name, {})

    def train_with_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str,
    ) -> Tuple:
        model = self.build_model(model_name)
        if model is None:
            return None, None, None

        param_grid = self.get_param_grid(model_name)
        if not param_grid:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            report = self.evaluator.evaluate(y_val, y_pred, model_name)
            return model, report, {}

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=min(self.n_iter, 30),
            cv=cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X_train, y_train)
        best = search.best_estimator_
        y_pred = best.predict(X_val)
        report = self.evaluator.evaluate(y_val, y_pred, model_name)
        return best, report, search.best_params_

    def train_all(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict:
        from ..data.dataset import split_dataset

        # Split first, fit second: the scaler and the feature selector are
        # fitted on X_train only, so no test/validation statistics can leak
        # into training.
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            X,
            y,
            test_size=self.models_config["test_size"],
            val_size=self.models_config["val_size"],
            random_state=self.random_state,
        )
        print(
            f"  Split: train={X_train.shape[0]} val={X_val.shape[0]} "
            f"test={X_test.shape[0]} (stratified on {len(np.unique(y))} students)"
        )

        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        y_test_enc = self.label_encoder.transform(y_test)

        X_train_processed = self.feature_selector.fit_transform(
            X_train, y_train_enc
        )
        X_val_processed = self.feature_selector.transform(X_val)
        X_test_processed = self.feature_selector.transform(X_test)

        results = {}
        models_to_train = self.models_config["models"]

        for model_name in models_to_train:
            print(f"Training {model_name}...")
            model, report, best_params = self.train_with_tuning(
                X_train_processed, y_train_enc,
                X_val_processed, y_val_enc,
                model_name,
            )
            if model is None:
                print(f"  Skipping {model_name} (not available)")
                continue

            results[model_name] = {
                "model": model,
                "report": report,
                "best_params": best_params,
            }
            print(f"  Val F1 (weighted): {report['f1_weighted']:.4f}")

            test_pred = model.predict(X_test_processed)
            test_report = self.evaluator.evaluate(
                y_test_enc, test_pred, f"{model_name}_test"
            )
            results[model_name]["test_report"] = test_report
            print(f"  Test F1 (weighted): {test_report['f1_weighted']:.4f}")

        comparison = self.evaluator.compare_models(include_test=False)
        print("\n" + "=" * 60)
        print("Model Comparison (validation split only - the held-out test")
        print("set is reported below but never used to pick a model)")
        print("=" * 60)
        print(comparison.to_string(index=False))

        if comparison.empty:
            raise RuntimeError(
                "No model produced a validation report; nothing to select from."
            )

        # Select on validation F1 only. The test set is evaluated once, after
        # the winner is fixed, so the reported test score stays honest.
        best_row = comparison.iloc[0]
        best_name = best_row["model"]
        if best_name not in results:
            raise RuntimeError(
                f"Comparison picked {best_name!r} which is not a trained model."
            )
        self.best_model_name = best_name
        self.best_model = results[best_name]["model"]

        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Val F1: {best_row['f1_weighted']:.4f}")
        print(
            f"Test F1 (held out, {results[best_name]['test_report'].get('model', best_name + '_test')}): "
            f"{results[best_name]['test_report']['f1_weighted']:.4f}"
        )

        return results

    def save_model(self, output_dir: str = "models") -> Dict[str, str]:
        """Write the four artifacts and return their paths.

        ``output_dir`` defaults to ``models/`` so the historical behaviour is
        unchanged, but callers retraining over the shipped artifacts should
        pass a different directory - see ``train.py --output-dir``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_path = out / "best_model.pkl"
        joblib.dump(self.best_model, model_path)

        scaler_path = out / "scaler.pkl"
        joblib.dump(self.feature_selector.scaler, scaler_path)

        selector_path = out / "feature_selector.pkl"
        joblib.dump(self.feature_selector, selector_path)

        le_path = out / "best_model_label_encoder.pkl"
        joblib.dump(self.label_encoder, le_path)

        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Feature selector saved to {selector_path}")
        print(f"Label encoder saved to {le_path}")

        return {
            "model": str(model_path),
            "scaler": str(scaler_path),
            "feature_selector": str(selector_path),
            "label_encoder": str(le_path),
        }
