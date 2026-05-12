from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: Path | str) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)
