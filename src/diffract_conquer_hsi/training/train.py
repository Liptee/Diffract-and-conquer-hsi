from __future__ import annotations

import argparse
from pathlib import Path

from diffract_conquer_hsi.processing.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Train GGPIR / cmKAN++ on encoded HSI")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = p.parse_args()
    _ = load_config(args.config)
    raise SystemExit("Training script placeholder: wire datasets and optical forward.")


if __name__ == "__main__":
    main()
