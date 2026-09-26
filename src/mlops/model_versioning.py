import os
import json
import joblib
import hashlib
import datetime
from pathlib import Path
from typing import Dict


class ModelVersioning:
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "model_registry.json"
        self._init_registry()

    def _init_registry(self):
        if not self.metadata_file.exists():
            with open(self.metadata_file, "w") as f:
                json.dump({"models": [], "latest": {}}, f)

    def _load_registry(self) -> Dict:
        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def _save_registry(self, registry: Dict):
        with open(self.metadata_file, "w") as f:
            json.dump(registry, f, indent=2)

    def _compute_hash(self, filepath: str) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def register_model(
        self,
        model_path: str,
        model_name: str,
        version: str = None,
        metrics: Dict = None,
        params: Dict = None,
    ) -> str:
        if version is None:
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        model_hash = self._compute_hash(model_path)
        registry = self._load_registry()

        entry = {
            "version": version,
            "model_name": model_name,
            "path": model_path,
            "hash": model_hash,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics or {},
            "params": params or {},
        }

        registry["models"].append(entry)
        registry["latest"][model_name] = version
        self._save_registry(registry)
        return version

    def get_model_path(self, model_name: str, version: str = None) -> str:
        registry = self._load_registry()

        if version is None:
            version = registry["latest"].get(model_name)
            if version is None:
                raise ValueError(f"No version found for {model_name}")

        for entry in registry["models"]:
            if entry["model_name"] == model_name and entry["version"] == version:
                return entry["path"]

        raise ValueError(f"Model {model_name} version {version} not found")

    def list_versions(self, model_name: str = None) -> list:
        registry = self._load_registry()
        if model_name:
            return [
                e
                for e in registry["models"]
                if e["model_name"] == model_name
            ]
        return registry["models"]
