from pathlib import Path

import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)