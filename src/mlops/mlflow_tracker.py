import yaml
import mlflow
import mlflow.sklearn
from pathlib import Path


class MLflowTracker:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        mlflow_config = self.config["mlflow"]
        self.tracking_uri = mlflow_config["tracking_uri"]
        self.experiment_name = mlflow_config["experiment_name"]
        self.run_name_prefix = mlflow_config["run_name_prefix"]

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def log_artifacts(self, artifact_dir: str):
        mlflow.log_artifacts(artifact_dir)

    def log_model(self, model, model_name: str):
        mlflow.sklearn.log_model(model, model_name)

    def log_feature_importance(self, importance: dict):
        for feat_name, imp_value in importance.items():
            mlflow.log_metric(f"feature_importance_{feat_name}", imp_value)

    def start_run(self, run_name: str = None):
        if run_name is None:
            run_name = f"{self.run_name_prefix}"
        return mlflow.start_run(run_name=run_name)

    def end_run(self):
        mlflow.end_run()

    def log_training_results(
        self,
        model,
        model_name: str,
        params: dict,
        metrics: dict,
        confusion_matrix_path: str = None,
        comparison_path: str = None,
    ):
        with self.start_run(run_name=f"{self.run_name_prefix}_{model_name}"):
            self.log_params(params)
            self.log_metrics(metrics)
            self.log_model(model, model_name)

            if confusion_matrix_path and Path(confusion_matrix_path).exists():
                mlflow.log_artifact(confusion_matrix_path)

            if comparison_path and Path(comparison_path).exists():
                mlflow.log_artifact(comparison_path)
