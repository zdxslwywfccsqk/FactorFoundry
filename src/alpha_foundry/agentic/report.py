from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .schemas import ArtifactKind, ArtifactRecord, RunManifest, SplitName
from .session import ResearchSession
from .utils import to_jsonable
from .workspace import RunWorkspace


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _latest_by_factor(
    store: ArtifactStore,
    kind: ArtifactKind,
    split: SplitName | None = None,
) -> dict[str, ArtifactRecord]:
    output: dict[str, ArtifactRecord] = {}
    for record in store.list(kind=kind):
        if split is not None and record.split != split:
            continue
        factor_id = record.payload.get("factor_id") or record.payload.get("spec_id")
        if factor_id:
            output[str(factor_id)] = record
    return output


def write_agent_report(
    session: ResearchSession,
    workspace: RunWorkspace,
    manifest: RunManifest,
) -> tuple[Path, Path]:
    store = session.store
    hypotheses = store.list(ArtifactKind.HYPOTHESIS_CARD)
    implementations = store.list(ArtifactKind.FACTOR_IMPLEMENTATION)
    validation_reports = _latest_by_factor(store, ArtifactKind.BACKTEST_REPORT, SplitName.VALIDATION)
    critic_record = store.latest(ArtifactKind.CRITIC_REVIEW)
    portfolio_record = store.latest(ArtifactKind.PORTFOLIO_DECISION)
    holdout_record = store.latest(ArtifactKind.HOLDOUT_REPORT)
    critic = critic_record.payload if critic_record else {}
    portfolio = portfolio_record.payload if portfolio_record else {}
    holdout = holdout_record.payload if holdout_record else {}
    implementation = session.summary().get("implementability", {})
    decisions = {item["factor_id"]: item for item in critic.get("decisions", [])}

    lines: list[str] = []
    lines.append(f"# FactorFoundry agentic run `{manifest.run_id}`")
    lines.append("")
    lines.append("## Run contract")
    lines.append("")
    lines.append(f"- Mode: `{manifest.mode}`")
    lines.append(f"- Status: `{manifest.status}`")
    lines.append(f"- Config hash: `{manifest.config_hash}`")
    lines.append(f"- Data fingerprint: `{manifest.data_fingerprint}`")
    lines.append(f"- Git commit: `{manifest.git_commit or 'not available'}`")
    lines.append(f"- Holdout released: `{manifest.holdout_released}`")
    lines.append(f"- Artifacts: `{store.count()}`")
    lines.append("")
    lines.append(
        "When enabled, the language-model layer proposes and critiques. All performance numbers "
        "below come from deterministic tools, and the test split is gated by the run-level split guard."
    )
    lines.append("")

    lines.append("## Research funnel")
    lines.append("")
    funnel_rows = [
        ["hypothesis cards", len(hypotheses)],
        [
            "implemented factors",
            sum(1 for record in implementations if record.payload.get("status") == "implemented"),
        ],
        [
            "rejected implementations",
            sum(1 for record in implementations if record.payload.get("status") == "rejected"),
        ],
        ["critic candidates", critic.get("n_candidates", 0)],
        ["critic accepted", critic.get("n_accepted", 0)],
        ["portfolio factors", len(portfolio.get("selected_factors", []))],
    ]
    lines.append(_table(["Stage", "Count"], funnel_rows))
    lines.append("")

    lines.append("## Factor evidence")
    lines.append("")
    factor_rows: list[list[Any]] = []
    for implementation in implementations:
        spec_id = str(implementation.payload.get("spec_id", ""))
        validation = validation_reports.get(spec_id)
        metrics = validation.payload.get("metrics", {}) if validation else {}
        decision = decisions.get(spec_id, {})
        status = decision.get("status", implementation.payload.get("status", ""))
        factor_rows.append(
            [
                f"`{spec_id}`",
                status,
                _fmt(metrics.get("net_sharpe")),
                _fmt(metrics.get("net_ann_return")),
                _fmt(metrics.get("mean_ic")),
                _fmt(metrics.get("icir")),
                _fmt(decision.get("q_value")),
                _fmt(decision.get("dsr")),
                "; ".join(decision.get("reasons", []))[:90],
            ]
        )
    lines.append(
        _table(
            ["Factor", "Status", "Val net SR", "Val ann ret", "Val IC", "Val ICIR", "FDR q", "DSR", "Reason"],
            factor_rows,
        )
    )
    lines.append("")

    lines.append("## Critic and multiple testing")
    lines.append("")
    if critic:
        pbo = critic.get("pbo", {})
        lines.append(
            f"- Candidates: `{critic.get('n_candidates', 0)}`; accepted: `{critic.get('n_accepted', 0)}`."
        )
        lines.append(
            f"- PBO: `{_fmt(pbo.get('pbo'))}` across `{pbo.get('n_trials', 0)}` trials."
        )
        lines.append(f"- Reviewer notes: {critic.get('reviewer_notes', '-')}")
    else:
        lines.append("_No critic review was recorded._")
    lines.append("")

    lines.append("## Portfolio")
    lines.append("")
    if portfolio:
        validation = portfolio.get("validation_metrics", {})
        lines.append(f"- Selected factors: `{', '.join(portfolio.get('selected_factors', []))}`")
        lines.append(f"- Fallback used: `{portfolio.get('fallback')}`")
        lines.append(
            f"- Validation net Sharpe `{_fmt(validation.get('net_sharpe'))}`, "
            f"annualized net return `{_fmt(validation.get('net_ann_return'))}`, "
            f"max drawdown `{_fmt(validation.get('max_drawdown'))}`, "
            f"annual turnover `{_fmt(validation.get('turnover_annual'))}`."
        )
        if portfolio.get("warnings"):
            lines.append(f"- Warnings: {'; '.join(portfolio['warnings'])}")
    else:
        lines.append("_No portfolio decision was recorded._")
    lines.append("")

    lines.append("## Implementability")
    lines.append("")
    long_only = implementation.get("long_only", {}) if isinstance(implementation, dict) else {}
    capacity = implementation.get("capacity", {}) if isinstance(implementation, dict) else {}
    if long_only:
        lines.append("A separate long-only implementation is evaluated from the same signal.")
        for split in ("validation", "test"):
            item = long_only.get(split, {})
            metrics = item.get("metrics", {}) if isinstance(item, dict) else {}
            if metrics:
                lines.append(
                    f"- Long-only {split}: net Sharpe `{_fmt(metrics.get('net_sharpe'))}`, "
                    f"annualized net return `{_fmt(metrics.get('net_ann_return'))}`, "
                    f"annual turnover `{_fmt(metrics.get('turnover_annual'))}`, "
                    f"max drawdown `{_fmt(metrics.get('max_drawdown'))}`."
                )
        if capacity.get("capacity_estimate"):
            lines.append(
                f"- Estimated 10% ADV capacity: median AUM `{capacity.get('estimated_aum_median'):,.0f}`, "
                f"p10 AUM `{capacity.get('estimated_aum_p10'):,.0f}`."
            )
        lines.append(
            "- Long-only returns retain market beta; this is an implementation check, not standalone alpha evidence."
        )
    else:
        lines.append("_No long-only implementation result was recorded._")
    lines.append("")

    lines.append("## Holdout")
    lines.append("")
    if holdout:
        metrics = holdout.get("metrics", {})
        lines.append(f"- Portfolio: `{holdout.get('portfolio_id')}`")
        lines.append(
            f"- Test net Sharpe `{_fmt(metrics.get('net_sharpe'))}`, "
            f"annualized net return `{_fmt(metrics.get('net_ann_return'))}`, "
            f"max drawdown `{_fmt(metrics.get('max_drawdown'))}`, "
            f"annual turnover `{_fmt(metrics.get('turnover_annual'))}`."
        )
        lines.append(f"- Released by: `{holdout.get('released_by')}`; reason: {holdout.get('reason')}")
        for item in holdout.get("limitations", []):
            lines.append(f"- Limitation: {item}")
    else:
        lines.append("_Holdout was not released in this run._")
    lines.append("")

    lines.append("## Known limitations")
    lines.append("")
    limitations = [
        "The current A-share universe is built from a public data source and may retain survivorship or point-in-time limitations.",
        "The long-short research portfolio does not fully model A-share borrow availability or short-sale constraints.",
        "Transaction costs are a constant per-side assumption; capacity and market impact require separate stress testing.",
        "The historical test window may have been observed during earlier development. Treat the holdout number as a benchmark, not prospective evidence.",
        "The initial implementation uses a deterministic factor-template library; arbitrary agent-authored code is intentionally disabled.",
    ]
    for item in limitations:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Artifact index")
    lines.append("")
    counts: dict[str, int] = {}
    for record in store.list():
        counts[record.kind.value] = counts.get(record.kind.value, 0) + 1
    lines.append(_table(["Artifact kind", "Count"], [[key, value] for key, value in sorted(counts.items())]))
    lines.append("")
    lines.append("All artifacts live under `artifacts/`; the event stream is in `events.jsonl`.")
    lines.append("")

    report_path = workspace.run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary_path = workspace.run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(to_jsonable(session.summary()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path, summary_path
