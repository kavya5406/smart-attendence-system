import os
import yaml
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(dirs: List[str]):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def save_json(data: Any, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def list_student_dirs(data_dir: str) -> List[Path]:
    base = Path(data_dir)
    return sorted([d for d in base.iterdir() if d.is_dir()])
