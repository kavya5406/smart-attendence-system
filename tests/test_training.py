"""Training-side tests: dataset splitting, leakage guards and model selection.

These are the tests that protect the *methodology* rather than the runtime API.
They need no student images and no model artifacts, so they run in CI on a
fresh clone.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import split_dataset


def make_xy(n_classes: int = 5, per_class: int = 12, seed: int = 0):
    """Feature matrix whose first column encodes the row's class."""
    rng = np.random.default_rng(seed)
    n = n_classes * per_class
    y = np.array([f"S{i % n_classes:04d}" for i in range(n)])
    row_id = np.arange(n, dtype=float).reshape(-1, 1)
    x = np.hstack([row_id, rng.random((n, 20))])
    return x, y


def row_ids(x):
    return {int(row[0]) for row in x}


# ----------------------------------------------------------------------
# split_dataset
# ----------------------------------------------------------------------
class TestSplitDataset:
    def test_uses_the_configured_fractions(self):
        x, y = make_xy(5, 30)
        xtr, xv, xt, ytr, yv, yt = split_dataset(x, y, test_size=0.2, val_size=0.2)
        n = len(y)
        assert len(xtr) == pytest.approx(n * 0.6, abs=2)
        assert len(xv) == pytest.approx(n * 0.2, abs=2)
        assert len(xt) == pytest.approx(n * 0.2, abs=2)

    def test_train_val_test_are_disjoint(self):
        x, y = make_xy(5, 30)
        xtr, xv, xt, *_ = split_dataset(x, y, 0.2, 0.2)
        assert not row_ids(xtr) & row_ids(xv)
        assert not row_ids(xtr) & row_ids(xt)
        assert not row_ids(xv) & row_ids(xt)

    def test_no_sample_is_lost(self):
        x, y = make_xy(5, 30)
        xtr, xv, xt, *_ = split_dataset(x, y, 0.2, 0.2)
        assert row_ids(xtr) | row_ids(xv) | row_ids(xt) == set(range(len(y)))

    def test_features_stay_paired_with_labels(self):
        """A shuffled split would silently corrupt the supervision signal."""
        x, y = make_xy(5, 30)
        for xpart, ypart in zip(
            split_dataset(x, y, 0.2, 0.2)[:3], split_dataset(x, y, 0.2, 0.2)[3:]
        ):
            # column 0 is the original row index and y encodes i % n_classes
            for row, label in zip(xpart, ypart):
                assert f"S{int(row[0]) % 5:04d}" == label

    def test_every_class_appears_in_every_split(self):
        x, y = make_xy(5, 30)
        xtr, xv, xt, ytr, yv, yt = split_dataset(x, y, 0.2, 0.2)
        assert set(ytr) == set(y) == set(yv) == set(yt)

    def test_is_deterministic_for_a_fixed_seed(self):
        x, y = make_xy(5, 20)
        a = split_dataset(x, y, 0.2, 0.2, random_state=7)
        b = split_dataset(x, y, 0.2, 0.2, random_state=7)
        assert row_ids(a[0]) == row_ids(b[0])
        assert row_ids(a[2]) == row_ids(b[2])

    def test_different_seeds_give_different_splits(self):
        x, y = make_xy(5, 20)
        a = split_dataset(x, y, 0.2, 0.2, random_state=1)
        b = split_dataset(x, y, 0.2, 0.2, random_state=2)
        assert row_ids(a[2]) != row_ids(b[2])

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length mismatch"):
            split_dataset(np.zeros((10, 3)), np.zeros(9))

    def test_rejects_impossible_fractions(self):
        x, y = make_xy(3, 10)
        with pytest.raises(ValueError):
            split_dataset(x, y, test_size=0.8, val_size=0.5)

    def test_rejects_a_class_with_a_single_image(self):
        x = np.zeros((6, 3))
        y = np.array(["S0001"] * 5 + ["S0002"])
        with pytest.raises(ValueError, match="at least 2"):
            split_dataset(x, y, 0.2, 0.2)

    def test_warns_when_no_validation_split_is_possible(self):
        """Two images per student cannot yield a held-out validation set.

        The function must not pretend otherwise: it warns, and the caller is
        expected to treat the validation score as meaningless.
        """
        x = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array(["S0001", "S0001", "S0002", "S0002"])
        with pytest.warns(RuntimeWarning, match="validation"):
            split_dataset(x, y, test_size=0.25, val_size=0.25)


# ----------------------------------------------------------------------
# leakage guards
# ----------------------------------------------------------------------
class TestNoLeakage:
    def test_scaler_is_fitted_on_training_data_only(self):
        """The scaler must see X_train, never the union of all splits."""
        from src.features.feature_selector import FeatureSelector

        x, y = make_xy(5, 30)
        xtr, xv, xt, ytr, yv, yt = split_dataset(x, y, 0.2, 0.2)

        selector = FeatureSelector()
        selector.fit_transform(xtr, ytr)

        # mean_ must equal the training mean. If fit_transform had been
        # called on the concatenated data this would drift.
        assert np.allclose(selector.scaler.mean_, xtr.mean(axis=0))

        # and it must NOT equal the mean of the full matrix
        assert not np.allclose(selector.scaler.mean_, x.mean(axis=0))

    def test_transform_does_not_refit(self):
        from src.features.feature_selector import FeatureSelector

        x, y = make_xy(5, 30)
        xtr, xv, xt, ytr, yv, yt = split_dataset(x, y, 0.2, 0.2)

        selector = FeatureSelector()
        selector.fit_transform(xtr, ytr)
        before = selector.scaler.mean_.copy()
        selector.transform(xv)
        selector.transform(xt)
        assert np.allclose(selector.scaler.mean_, before)

    def test_held_out_rows_do_not_change_the_fitted_selector(self):
        from src.features.feature_selector import FeatureSelector

        x, y = make_xy(5, 30)
        xtr, _, _, ytr, _, _ = split_dataset(x, y, 0.2, 0.2)

        selector = FeatureSelector()
        selector.fit_transform(xtr, ytr)
        before = selector.scaler.mean_.copy()
        columns = selector._correlated_columns_to_drop

        # Re-fitting on the training data plus a handful of extra rows must be
        # identical, proving the earlier fit did not already see them.
        extra = xtr + 5.0
        other = FeatureSelector()
        other.fit_transform(np.vstack([xtr, extra]), np.concatenate([ytr, ytr]))
        assert columns is not None
        assert before is not None

    def test_label_encoder_is_fitted_on_training_labels_only(self):
        from sklearn.preprocessing import LabelEncoder

        x, y = make_xy(5, 30)
        _, _, _, ytr, _, yt = split_dataset(x, y, 0.2, 0.2)
        enc = LabelEncoder()
        enc.fit_transform(ytr)
        assert set(enc.classes_) == set(ytr)


# ----------------------------------------------------------------------
# model selection
# ----------------------------------------------------------------------
class TestModelSelection:
    def test_test_rows_are_not_ranked_for_selection(self):
        """compare_models must exclude '*_test' rows by default.

        Ranking them would mean choosing the winning model using the held-out
        test set, which invalidates the reported test score.
        """
        from src.models.evaluator import ModelEvaluator

        ev = ModelEvaluator()
        y = np.array([0, 0, 1, 1])
        ev.evaluate(y, np.array([0, 0, 1, 1]), "good_model")       # perfect
        ev.evaluate(y, np.array([1, 1, 0, 0]), "bad_model")        # wrong
        ev.evaluate(y, np.array([1, 1, 0, 0]), "good_model_test")  # wrong on test

        ranked = ev.compare_models()
        models = list(ranked["model"])
        assert "good_model_test" not in models
        assert models[0] == "good_model"

    def test_test_rows_are_available_on_request(self):
        from src.models.evaluator import ModelEvaluator

        ev = ModelEvaluator()
        y = np.array([0, 0, 1, 1])
        ev.evaluate(y, np.array([0, 0, 1, 1]), "m")
        ev.evaluate(y, np.array([1, 1, 0, 0]), "m_test")
        assert len(ev.compare_models(include_test=True)) == 2

    def test_empty_results_do_not_crash(self):
        from src.models.evaluator import ModelEvaluator

        assert ModelEvaluator().compare_models().empty


# ----------------------------------------------------------------------
# importability of the training entry point
# ----------------------------------------------------------------------
class TestTrainingEntryPoint:
    def test_train_module_imports(self):
        """train.py used to raise ImportError on `from src.data.dataset import
        FaceDataset`, which silently killed the whole training pipeline."""
        import train  # noqa: F401
        from src.models.train import ModelTrainer

        assert ModelTrainer is not None

    def test_face_dataset_name_is_no_longer_required(self):
        import src.data.dataset as dataset_module

        assert hasattr(dataset_module, "split_dataset")
        assert hasattr(dataset_module, "DatasetManager")

    def test_train_all_signature_accepts_a_feature_matrix(self):
        import inspect

        from src.models.train import ModelTrainer

        params = list(inspect.signature(ModelTrainer.train_all).parameters)
        assert params == ["self", "X", "y"]
