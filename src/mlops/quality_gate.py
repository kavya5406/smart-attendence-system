from typing import Dict


class ModelQualityGate:
    """Compare a candidate model against the latest registered model."""

    def __init__(self, min_improvement: float = 0.0):
        self.min_improvement = min_improvement

    def check(
        self,
        new_metrics: Dict,
        current_metrics: Dict | None = None,
    ) -> bool:
        new_f1 = float(new_metrics.get("f1_weighted", 0.0))

        # No previous model: allow the first model.
        if not current_metrics:
            return True

        current_f1 = float(current_metrics.get("f1_weighted", 0.0))

        required_f1 = current_f1 + self.min_improvement

        return new_f1 >= required_f1