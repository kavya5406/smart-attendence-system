import os
import yaml
import hashlib
import joblib
import mlflow
import mlflow.sklearn


def register_existing_model(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000"))
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config.get("mlflow", {}).get("experiment_name", "Smart_Attendance_System")
    mlflow.set_experiment(experiment_name)

    model_path = config["api"]["model_path"]
    scaler_path = config["api"]["scaler_path"]
    selector_path = config["api"]["selector_path"]
    le_path = model_path.replace(".pkl", "_label_encoder.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    selector = joblib.load(selector_path) if os.path.exists(selector_path) else None
    le = joblib.load(le_path) if os.path.exists(le_path) else None

    model_hash = hashlib.sha256(open(model_path, "rb").read()).hexdigest()[:16]

    with mlflow.start_run(run_name=f"existing_model_{model_hash}"):
        mlflow.log_params({
            "model_type": type(model).__name__,
            "n_features_in": getattr(model, "n_features_in_", None),
            "n_classes": len(getattr(model, "classes_", [])),
            "classes": str(list(getattr(model, "classes_", []))),
            "feature_extraction": "HOG+LBP+Landmark+Geometric+Statistical+Shape",
            "preprocessing": "HaarCascade+Grayscale+Resize64x64+HistEq+GBlur",
            "model_hash": model_hash,
        })

        if hasattr(model, "get_params"):
            params = model.get_params()
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(f"model_{k}", v)

        mlflow.sklearn.log_model(model, "model")
        if scaler:
            mlflow.sklearn.log_model(scaler, "scaler")
        if selector:
            mlflow.sklearn.log_model(selector, "feature_selector")
        if le:
            mlflow.sklearn.log_model(le, "label_encoder")

        mlflow.set_tag("status", "registered")
        mlflow.set_tag("source", "existing_trained_model")
        print(f"Model registered: {model_path} ({type(model).__name__})")
        print(f"  Hash: {model_hash}")
        print(f"  Features: {getattr(model, 'n_features_in_', 'N/A')}")
        print(f"  Classes: {getattr(model, 'classes_', [])}")

    print("MLflow registration complete.")


if __name__ == "__main__":
    register_existing_model()
