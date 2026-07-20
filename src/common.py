from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_splits(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def device_from_system() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

