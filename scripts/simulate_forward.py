#!/usr/bin/env python3
"""Simulate encoded measurements from hyperspectral cubes (optical pipeline Sec. 3)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    _ = parser.parse_args()
    raise SystemExit("Implement dataset iterator + forward_model.forward_measurement.")


if __name__ == "__main__":
    main()
