from __future__ import annotations

import argparse
from pathlib import Path

from diffract_conquer_hsi.processing.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate reconstruction (SAM / PSNR / SSIM)")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = p.parse_args()
    _ = load_config(args.config)
    raise SystemExit("Evaluation script placeholder.")


if __name__ == "__main__":
    main()
