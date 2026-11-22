from __future__ import annotations

import argparse
import json
import logging
import sys

from ..config import AppConfig
from ..scorecard import score_run
from .orchestrator import ResearchOrchestrator


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("mini_agent").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _run(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    config.ensure_dirs()
    if args.max_audits is not None:
        config.raw.setdefault("agentic", {})["max_audits"] = int(args.max_audits)
    orchestrator = ResearchOrchestrator.create(
        config,
        mode=args.mode,
        factor_limit=args.factor_limit,
        release_holdout=args.release_holdout,
        run_id=args.run_id,
    )
    report_path, summary_path = orchestrator.run()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    portfolio = summary.get("portfolio") or {}
    validation = portfolio.get("validation_metrics", {})
    holdout = (summary.get("holdout") or {}).get("metrics", {})
    print("\nFactorFoundry agentic run completed")
    print(f"  run id:       {summary.get('run_id')}")
    print(f"  report:       {report_path}")
    print(f"  summary:      {summary_path}")
    print(f"  artifacts:    {summary.get('artifact_count')}")
    print(f"  hypotheses:   {summary.get('n_hypotheses')}")
    print(f"  implemented:  {summary.get('n_implemented')}")
    if validation:
        print(
            "  validation:   "
            f"net Sharpe={validation.get('net_sharpe'):.4f}, "
            f"net annual={validation.get('net_ann_return'):.4f}, "
            f"max drawdown={validation.get('max_drawdown'):.4f}"
        )
    if holdout:
        print(
            "  holdout:      "
            f"net Sharpe={holdout.get('net_sharpe'):.4f}, "
            f"net annual={holdout.get('net_ann_return'):.4f}, "
            f"max drawdown={holdout.get('max_drawdown'):.4f}"
        )
    return 0


def _list(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    runs_dir = config.paths.runs_dir
    if not runs_dir.exists():
        print(f"No runs directory: {runs_dir}")
        return 0
    for path in sorted((item for item in runs_dir.iterdir() if item.is_dir()), reverse=True):
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"{manifest.get('run_id')}  status={manifest.get('status')}  "
            f"mode={manifest.get('mode')}  artifacts={manifest.get('artifact_count')}"
        )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    run_dir = config.paths.runs_dir / args.run_id
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    if not summary_path.exists():
        print(f"Missing summary for run: {args.run_id}", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report_path.exists():
        print(f"\nReport: {report_path}")
    return 0


def _score(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    run_dir = config.paths.runs_dir / args.run_id
    if not (run_dir / "summary.json").exists():
        print(f"Missing summary for run: {args.run_id}", file=sys.stderr)
        return 2
    scorecard = score_run(run_dir)
    rendered = json.dumps(scorecard.to_dict(), ensure_ascii=False, indent=2)
    if args.format == "markdown":
        rendered = scorecard.to_markdown()
    if args.output:
        output_path = config.root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Scorecard written to {output_path}")
    else:
        print(rendered)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorFoundry agentic research runtime"
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one research campaign")
    run_parser.add_argument("--config", default=argparse.SUPPRESS, help="Path to config.json")
    run_parser.add_argument(
        "--mode",
        choices=["deterministic", "mini-agent"],
        default=None,
        help="Deterministic offline reference mode or Mini-Agent role mode",
    )
    run_parser.add_argument("--factor-limit", type=int, default=None)
    run_parser.add_argument("--max-audits", type=int, default=None)
    run_parser.add_argument(
        "--release-holdout",
        action="store_true",
        help="Explicitly unlock and evaluate the test split after portfolio construction",
    )
    run_parser.add_argument("--run-id", default=None)
    run_parser.set_defaults(func=_run)

    list_parser = subparsers.add_parser("list", help="List runs")
    list_parser.add_argument("--config", default=argparse.SUPPRESS, help="Path to config.json")
    list_parser.set_defaults(func=_list)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a run summary")
    inspect_parser.add_argument("--config", default=argparse.SUPPRESS, help="Path to config.json")
    inspect_parser.add_argument("run_id")
    inspect_parser.set_defaults(func=_inspect)

    score_parser = subparsers.add_parser(
        "score", help="Score a run against the Alpha Evidence Scorecard"
    )
    score_parser.add_argument("--config", default=argparse.SUPPRESS, help="Path to config.json")
    score_parser.add_argument("run_id")
    score_parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )
    score_parser.add_argument(
        "--output",
        default=None,
        help="Optional output path, relative to the repository root",
    )
    score_parser.set_defaults(func=_score)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging()
    if args.command == "run" and args.mode is None:
        config = AppConfig.load(args.config)
        args.mode = str(config.raw.get("agentic", {}).get("default_mode", "deterministic"))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
