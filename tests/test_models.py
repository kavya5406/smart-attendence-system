import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.models.evaluator import ModelEvaluator


class TestModelEvaluator:
    def test_evaluate(self):
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        report = evaluator.evaluate(y_true, y_pred, "test_model")
        assert report["accuracy"] == 1.0
        assert report["f1_weighted"] == 1.0

    def test_evaluate_imperfect(self):
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 1, 1])
        report = evaluator.evaluate(y_true, y_pred, "test_model")
        assert report["accuracy"] < 1.0

    def test_compare_models(self):
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 0, 1])
        evaluator.evaluate(y_true, y_true, "model_a")
        evaluator.evaluate(y_true, y_true, "model_b")
        df = evaluator.compare_models()
        assert len(df) == 2
        assert "model" in df.columns

    def test_confusion_matrix(self):
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = evaluator.get_confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2
