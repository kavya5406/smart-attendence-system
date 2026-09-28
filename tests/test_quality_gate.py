from src.mlops.quality_gate import ModelQualityGate


def test_first_model_passes():
    gate = ModelQualityGate()

    assert gate.check(
        {"f1_weighted": 0.70},
        None
    )


def test_better_model_passes():
    gate = ModelQualityGate()

    assert gate.check(
        {"f1_weighted": 0.80},
        {"f1_weighted": 0.70}
    )


def test_equal_model_passes():
    gate = ModelQualityGate()

    assert gate.check(
        {"f1_weighted": 0.70},
        {"f1_weighted": 0.70}
    )


def test_worse_model_fails():
    gate = ModelQualityGate()

    assert not gate.check(
        {"f1_weighted": 0.60},
        {"f1_weighted": 0.70}
    )