import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Tuple


class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> Dict:
        report = {
            "model": model_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }
        self.results[model_name] = report
        return report

    def get_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None
    ) -> np.ndarray:
        return confusion_matrix(y_true, y_pred, labels=labels)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None,
        title: str = "Confusion Matrix",
        save_path: str = None,
    ):
        cm = self.get_confusion_matrix(y_true, y_pred, labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def compare_models(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results.values())
        return df.sort_values("f1_weighted", ascending=False)

    def plot_comparison(
        self, save_path: str = None
    ):
        df = self.compare_models()
        metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
        plt.figure(figsize=(14, 6))
        x = np.arange(len(df))
        width = 0.2
        for i, metric in enumerate(metrics):
            plt.bar(
                x + i * width, df[metric].values, width, label=metric.replace("_", " ").title()
            )
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.title("Model Performance Comparison")
        plt.xticks(x + width * 1.5, df["model"].values, rotation=45, ha="right")
        plt.legend(loc="lower right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
