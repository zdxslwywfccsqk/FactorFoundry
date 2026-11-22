from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from .agents import AGENT_SEQUENCE, AgentContext
from .config import AppConfig
from .data import build_fields, data_fingerprint, load_fields, save_fields
from .datasource import (
    download_prices,
    download_prices_baostock,
    download_universe,
    load_cached_market_data,
)
from .memory import ExperimentMemory
from .report import write_report


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "pipeline.log", mode="a", encoding="utf-8"),
        ],
        force=True,
    )


def ensure_market_data(config: AppConfig, force_download: bool = False) -> None:
    universe_path = config.paths.raw_dir / "universe.csv"
    prices_path = config.paths.raw_dir / "prices" / "prices.parquet"
    if universe_path.exists() and prices_path.exists() and not force_download:
        logging.getLogger(__name__).info("Using cached market data")
        return
    universe = download_universe(
        output_dir=config.paths.raw_dir,
        size=config.data.universe_size,
        force_refresh=force_download,
    )
    try:
        download_prices_baostock(
            universe=universe,
            start=config.data.start_date,
            end=config.data.end_date,
            cache_dir=config.paths.raw_dir / "baostock",
            index_symbol=config.data.index_symbol,
            force_refresh=force_download,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        logging.getLogger(__name__).warning(
            "Baostock download failed (%s); falling back to Tencent", exc
        )
        download_prices(
            universe=universe,
            start=config.data.start_date,
            end=config.data.end_date,
            cache_dir=config.paths.raw_dir / "prices",
            index_symbol=config.data.index_symbol,
            max_workers=2,
            force_refresh=force_download,
        )


def ensure_fields(config: AppConfig, force_build: bool = False) -> tuple[dict, dict]:
    fields_dir = config.paths.processed_dir / "fields"
    if fields_dir.exists() and any(fields_dir.glob("*.parquet")) and not force_build:
        logging.getLogger(__name__).info("Using cached processed fields")
        return load_fields(config.paths.processed_dir)
    universe, prices, index_frame = load_cached_market_data(config.paths.raw_dir)
    del universe
    fields, metadata = build_fields(prices, index_frame, config)
    save_fields(fields, metadata, config.paths.processed_dir)
    return fields, metadata


def run_research(config: AppConfig, fields: dict, metadata: dict) -> AgentContext:
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    fingerprint = data_fingerprint(config)
    metadata = dict(metadata)
    metadata["data_fingerprint"] = fingerprint
    memory = ExperimentMemory(config.paths.experiments_dir)
    memory.start_run(run_id, config.raw, fingerprint)
    context = AgentContext(
        config=config,
        memory=memory,
        run_id=run_id,
        fields=fields,
        metadata=metadata,
    )
    for agent_cls in AGENT_SEQUENCE:
        agent = agent_cls()
        agent.run(context)
    memory.finish_run(run_id, "completed")

    with (config.paths.output_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config.raw, fh, ensure_ascii=False, indent=2)
    with (config.paths.output_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    return context


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FactorFoundry multi-agent alpha research pipeline")
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--stage",
        choices=["all", "download", "build", "run", "report"],
        default="all",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--universe-size", type=int, default=None)
    parser.add_argument("--factor-limit", type=int, default=None)
    parser.add_argument("--min-names", type=int, default=None)
    parser.add_argument("--quantile", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = AppConfig.load(args.config)
    config.ensure_dirs()
    configure_logging(config.paths.output_dir)
    logger = logging.getLogger(__name__)

    if args.universe_size is not None:
        config.data.universe_size = int(args.universe_size)
        config.raw.setdefault("data", {})["universe_size"] = int(args.universe_size)
    if args.factor_limit is not None:
        config.raw.setdefault("runtime", {})["factor_limit"] = int(args.factor_limit)
    if args.min_names is not None:
        config.backtest.min_names = int(args.min_names)
        config.raw.setdefault("backtest", {})["min_names"] = int(args.min_names)
    if args.quantile is not None:
        config.backtest.quantile = float(args.quantile)
        config.raw.setdefault("backtest", {})["quantile"] = float(args.quantile)

    logger.info("Starting %s stage=%s", config.project_name, args.stage)
    if args.stage in {"all", "download"}:
        ensure_market_data(config, force_download=args.force_download)
    if args.stage in {"all", "build"}:
        fields, metadata = ensure_fields(config, force_build=args.force_build or args.force_download)
    elif args.stage in {"run", "report"}:
        fields, metadata = ensure_fields(config, force_build=False)
    else:
        return 0

    if args.stage in {"all", "run"}:
        context = run_research(config, fields, metadata)
        report_path, summary_path = write_report(context)
        logger.info("Wrote report to %s", report_path)
        logger.info("Wrote summary to %s", summary_path)
    elif args.stage == "report":
        # Report stage re-runs the lightweight in-memory analysis if no context exists.
        context = run_research(config, fields, metadata)
        report_path, summary_path = write_report(context)
        logger.info("Wrote report to %s", report_path)
        logger.info("Wrote summary to %s", summary_path)
    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
