import json
from pathlib import Path


def load_config(path: Path):
    with open(path, "r") as f:
        config = json.load(f)
    return config
