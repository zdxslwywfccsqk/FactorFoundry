from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_foundry.config import AppConfig
from alpha_foundry.pipeline import configure_logging, ensure_market_data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--universe-size", type=int, default=None)
    args = parser.parse_args()
    config = AppConfig.load(args.config)
    config.ensure_dirs()
    configure_logging(config.paths.output_dir)
    if args.universe_size:
        config.data.universe_size = args.universe_size
    ensure_market_data(config, force_download=args.force)
    logging.getLogger(__name__).info("Market data ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
