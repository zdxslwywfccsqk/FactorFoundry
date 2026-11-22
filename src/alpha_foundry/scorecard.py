from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCORECARD_VERSION = "1.0.0"
MIN_PERIODS = 24
ALPHA_CANDIDATE_SCORE = 80.0
PAPER_ALPHA_SCORE = 90.0


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    name: str
    passed: bool
    evidence: str


@dataclass(frozen=True)
class DimensionScore:
    key: str
    name: str
    earned: float
    maximum: float
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        return self.earned / self.maximum if self.maximum else 0.0


@dataclass(frozen=True)
class Scorecard:
    run_id: str
    score: float
    grade: str
    alpha_candidate: bool
    certification: str
    gates: list[GateResult]
    dimensions: list[DimensionScore]
    metrics: dict[str, Any]
    evidence_gaps: list[str]
    version: str = SCORECARD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "version": self.version,
            "run_id": self.run_id,
            "score": round(self.score, 2),
            "grade": self.grade,
            "alpha_candidate": self.alpha_candidate,
            "certification": self.certification,
            "gates": [
                {
                    "gate_id": gate.gate_id,
                    "name": gate.name,
                    "passed": gate.passed,
                    "evidence": gate.evidence,
                }
                for gate in self.gates
            ],
            "dimensions": [
                {
                    "key": dimension.key,
                    "name": dimension.name,
                    "earned": round(dimension.earned, 2),
                    "maximum": dimension.maximum,
                    "ratio": round(dimension.ratio, 4),
                    "details": dimension.details,
                }
                for dimension in self.dimensions
            ],
            "metrics": self.metrics,
            "evidence_gaps": self.evidence_gaps,
        })

    def to_markdown(self) -> str:
        verdict = "YES" if self.alpha_candidate else "NO"
        lines = [
            f"# Alpha Evidence Scorecard: `{self.run_id}`",
            "",
            f"**{self.score:.1f} / 100 · Grade {self.grade} · Alpha Candidate: {verdict}**",
            "",
            f"Certification: **{self.certification}**",
            "",
            f"Scorecard version: `{self.version}`",
            "",
            "## Hard gates",
            "",
            "| Gate | Result | Evidence |",
            "|---|---|---|",
        ]
        for gate in self.gates:
            result = "PASS" if gate.passed else "FAIL"
            lines.append(f"| `{gate.gate_id}` {gate.name} | **{result}** | {gate.evidence} |")

        lines.extend(
            [
                "",
                "## Dimension scores",
                "",
                "| Dimension | Earned | Maximum |",
                "|---|---:|---:|",
            ]
        )
        for dimension in self.dimensions:
            lines.append(
                f"| {dimension.name} | {dimension.earned:.1f} | {dimension.maximum:.0f} |"
            )

        lines.extend(
            [
                "",
                "## Key metrics",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        for name, value in self.metrics.items():
            lines.append(f"| `{name}` | {_format_metric(value)} |")

        if self.evidence_gaps:
            lines.extend(["", "## Blocking evidence gaps", ""])
            for gap in self.evidence_gaps:
                lines.append(f"- {gap}")

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                _interpretation(self.alpha_candidate, self.grade),
                "",
                "This scorecard is a research-evidence standard, not investment advice. "
                "A run without the Alpha Candidate certification must be described as a "
                "benchmark or exploratory result, even if its headline Sharpe is high.",
                "",
            ]
        )
        return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "-"
    if value is None:
        return "-"
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _interpretation(alpha_candidate: bool, grade: str) -> str:
    if alpha_candidate and grade in {"S", "A"}:
        return (
            "The run clears the hard gates and has strong evidence. It is still a research "
            "candidate until paper trading, capacity, and live-data checks are complete."
        )
    if alpha_candidate:
        return (
            "The run clears the hard gates but does not yet have top-tier evidence quality. "
            "It can be called an Alpha Candidate, not a production strategy."
        )
    if grade in {"S", "A", "B"}:
        return (
            "The run has useful research-quality evidence, but it fails at least one hard gate. "
            "It must not be presented as a deployable or statistically certified Alpha Candidate."
        )
    return (
        "The run is exploratory or incomplete. Its headline performance should be treated as a "
        "benchmark artifact until data, execution, cost, and multiple-testing gaps are repaired."
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _load_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.exists():
        return []
    artifacts: list[dict[str, Any]] = []
    for path in sorted(artifacts_dir.glob("*.json")):
        artifact = _read_json(path)
        if artifact:
            artifacts.append(artifact)
    artifacts.sort(key=lambda item: (str(item.get("created_at", "")), str(item.get("artifact_id", ""))))
    return artifacts


def _latest_artifact(
    artifacts: list[dict[str, Any]],
    kind: str,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    candidates = [item for item in artifacts if item.get("kind") == kind]
    if split is not None:
        candidates = [item for item in candidates if item.get("split") == split]
    return candidates[-1] if candidates else {}


def _load_scorecard_evidence(
    run_dir: Path,
    summary: dict[str, Any],
    run_config: dict[str, Any],
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    if override is not None:
        return override
    summary_evidence = _mapping(summary.get("scorecard_evidence"))
    if summary_evidence:
        return summary_evidence
    file_evidence = _read_json(run_dir / "scorecard_evidence.json")
    if file_evidence:
        return file_evidence
    return _mapping(run_config.get("scorecard_evidence"))


def _selected_factor_ids(summary: dict[str, Any], critic: dict[str, Any]) -> list[str]:
    portfolio = _mapping(summary.get("portfolio"))
    selected = [str(item) for item in _list(portfolio.get("selected_factors"))]
    if selected:
        return selected
    decisions = [_mapping(item) for item in _list(critic.get("decisions"))]
    return [str(item.get("factor_id")) for item in decisions if item.get("accepted")]


def _selected_decisions(
    critic: dict[str, Any],
    selected_ids: list[str],
) -> list[dict[str, Any]]:
    decisions = [_mapping(item) for item in _list(critic.get("decisions"))]
    by_id = {str(item.get("factor_id")): item for item in decisions}
    return [by_id[factor_id] for factor_id in selected_ids if factor_id in by_id]


def _component_points(
    value: float,
    thresholds: list[tuple[float, float]],
    *,
    higher_is_better: bool = True,
) -> float:
    if not math.isfinite(value):
        return 0.0
    for threshold, points in thresholds:
        if higher_is_better and value >= threshold:
            return points
        if not higher_is_better and value <= threshold:
            return points
    return 0.0


def _grade_for_score(score: float, gates: list[GateResult]) -> str:
    if score >= 90.0:
        grade = "S"
    elif score >= 80.0:
        grade = "A"
    elif score >= 65.0:
        grade = "B"
    elif score >= 50.0:
        grade = "C"
    else:
        grade = "D"
    if any(not gate.passed for gate in gates) and grade in {"S", "A"}:
        return "B"
    return grade


def _certification(score: float, gates: list[GateResult]) -> tuple[bool, str]:
    all_gates = all(gate.passed for gate in gates)
    if all_gates and score >= PAPER_ALPHA_SCORE:
        return True, "Paper Alpha"
    if all_gates and score >= ALPHA_CANDIDATE_SCORE:
        return True, "Alpha Candidate"
    if all_gates:
        return False, "Research Candidate"
    return False, "Benchmark only"


def _statistical_dimension(
    summary: dict[str, Any],
    critic: dict[str, Any],
    selected_decisions: list[dict[str, Any]],
) -> tuple[DimensionScore, dict[str, Any]]:
    valuation = _mapping(_nested(summary, "portfolio", "validation_metrics", default={}))
    holdout = _mapping(_nested(summary, "holdout", "metrics", default={}))
    q_values = [_as_float(item.get("q_value")) for item in selected_decisions]
    dsr_values = [_as_float(item.get("dsr")) for item in selected_decisions]
    q_value = max(q_values) if q_values and all(math.isfinite(item) for item in q_values) else math.nan
    dsr = min(dsr_values) if dsr_values and all(math.isfinite(item) for item in dsr_values) else math.nan
    pbo = _as_float(_nested(critic, "pbo", "pbo", default=math.nan))

    fdr_points = _component_points(
        q_value,
        [(0.05, 8.0), (0.10, 6.0), (0.20, 4.0), (0.30, 2.0)],
        higher_is_better=False,
    )
    dsr_points = _component_points(
        dsr,
        [(0.95, 8.0), (0.90, 6.0), (0.75, 4.0), (0.50, 2.0)],
    )
    pbo_points = _component_points(
        pbo,
        [(0.10, 7.0), (0.20, 5.0), (0.35, 3.0), (0.50, 1.0)],
        higher_is_better=False,
    )

    holdout_low = _as_float(holdout.get("sharpe_ci_low"))
    holdout_high = _as_float(holdout.get("sharpe_ci_high"))
    validation_low = _as_float(valuation.get("sharpe_ci_low"))
    validation_high = _as_float(valuation.get("sharpe_ci_high"))
    if math.isfinite(holdout_low) and holdout_low > 0:
        bootstrap_points = 5.0
    elif math.isfinite(validation_low) and validation_low > 0:
        bootstrap_points = 4.0
    elif (math.isfinite(holdout_high) and holdout_high > 0) or (
        math.isfinite(validation_high) and validation_high > 0
    ):
        bootstrap_points = 2.0
    elif any(
        math.isfinite(item)
        for item in (holdout_low, holdout_high, validation_low, validation_high)
    ):
        bootstrap_points = 1.0
    else:
        bootstrap_points = 0.0

    mean_ic = _as_float(valuation.get("mean_ic"))
    icir = _as_float(valuation.get("icir"))
    subperiod = _as_float(valuation.get("subperiod_ic_sign_ratio"))
    hit_rate = _as_float(valuation.get("ic_hit_rate"))
    stability_points = 0.0
    stability_points += 2.0 if math.isfinite(mean_ic) and mean_ic > 0 else 0.0
    stability_points += 2.0 if math.isfinite(icir) and icir >= 0.30 else 0.0
    stability_points += 2.0 if math.isfinite(subperiod) and subperiod >= 0.66 else 0.0
    stability_points += 1.0 if math.isfinite(hit_rate) and hit_rate >= 0.50 else 0.0

    earned = fdr_points + dsr_points + pbo_points + bootstrap_points + stability_points
    dimension = DimensionScore(
        key="statistical_evidence",
        name="Statistical evidence",
        earned=earned,
        maximum=35.0,
        details={
            "fdr": fdr_points,
            "dsr": dsr_points,
            "pbo": pbo_points,
            "bootstrap": bootstrap_points,
            "ic_stability": stability_points,
        },
    )
    metrics = {
        "selected_q_value": q_value,
        "selected_dsr": dsr,
        "pbo": pbo,
        "bootstrap_ci_low": holdout_low if math.isfinite(holdout_low) else validation_low,
        "bootstrap_ci_high": holdout_high if math.isfinite(holdout_high) else validation_high,
        "validation_mean_ic": mean_ic,
        "validation_icir": icir,
        "validation_subperiod_sign_ratio": subperiod,
    }
    return dimension, metrics


def _execution_dimension(
    evidence: dict[str, Any],
    run_config: dict[str, Any],
    validation_protocol: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[DimensionScore, dict[str, Any]]:
    execution = _mapping(evidence.get("execution"))
    cost = _mapping(evidence.get("cost"))
    backtest_config = _mapping(run_config.get("backtest"))
    lag = _as_int(
        execution.get("execution_lag_bars", backtest_config.get("execution_lag_bars")),
        default=-1,
    )
    lag_points = 7.0 if lag >= 1 else 0.0

    components = [str(item) for item in _list(cost.get("components"))]
    initial_charged = bool(cost.get("initial_position_charged", False))
    rebalance_charged = bool(cost.get("rebalance_charged", False))
    cost_mode = str(cost.get("mode", ""))
    if cost_mode and initial_charged and rebalance_charged and len(set(components)) >= 4:
        cost_points = 7.0
    elif cost_mode and initial_charged and rebalance_charged and len(set(components)) >= 2:
        cost_points = 5.0
    else:
        reported_bps = _as_float(validation_protocol.get("cost_bps_per_side"))
        cost_points = 2.0 if math.isfinite(reported_bps) and reported_bps > 0 else 0.0

    validation = _mapping(_nested(summary, "portfolio", "validation_metrics", default={}))
    holdout = _mapping(_nested(summary, "holdout", "metrics", default={}))
    annual_turnover = _as_float(
        holdout.get("turnover_annual", validation.get("turnover_annual"))
    )
    if math.isfinite(annual_turnover) and annual_turnover <= 12.0:
        turnover_points = 3.0
    elif math.isfinite(annual_turnover) and annual_turnover <= 20.0:
        turnover_points = 2.0
    elif math.isfinite(annual_turnover) and annual_turnover <= 35.0:
        turnover_points = 1.0
    else:
        turnover_points = 0.0
    capacity_points = 2.0 if bool(execution.get("capacity_estimate", False)) else 0.0

    mode = str(execution.get("implementability", "research_long_short"))
    borrow_cost = _as_float(execution.get("borrow_cost_bps"))
    if mode == "long_only":
        implementability_points = 6.0
    elif mode == "constrained_short" and math.isfinite(borrow_cost) and borrow_cost > 0:
        implementability_points = 6.0
    else:
        implementability_points = 0.0

    earned = lag_points + cost_points + turnover_points + capacity_points + implementability_points
    dimension = DimensionScore(
        key="execution_and_cost",
        name="Execution and cost",
        earned=earned,
        maximum=25.0,
        details={
            "execution_lag": lag_points,
            "cost_completeness": cost_points,
            "turnover_and_capacity": turnover_points + capacity_points,
            "implementability": implementability_points,
        },
    )
    metrics = {
        "execution_lag_bars": lag if lag >= 0 else None,
        "annual_turnover": annual_turnover,
        "implementability_mode": mode,
    }
    return dimension, metrics


def _data_dimension(
    evidence: dict[str, Any],
    validation_protocol: dict[str, Any],
) -> tuple[DimensionScore, dict[str, Any]]:
    data = _mapping(evidence.get("data"))
    leakage = _mapping(evidence.get("leakage"))

    pit_points = 6.0 if bool(leakage.get("pit_universe", False)) else 0.0
    survivorship_points = 5.0 if bool(data.get("delisting_control", False)) else 0.0

    constraint_points = 0.0
    constraint_points += 2.0 if bool(data.get("t_plus_one", False)) else 0.0
    constraint_points += 1.0 if bool(data.get("limit_control", False)) else 0.0
    constraint_points += 1.0 if bool(data.get("st_control", False)) else 0.0
    constraint_points += 1.0 if bool(data.get("suspension_control", False)) else 0.0
    if constraint_points == 0.0 and bool(validation_protocol.get("tradable_filter", False)):
        constraint_points = 1.0

    missing_points = 4.0 if bool(data.get("selection_missing_window", False)) else 0.0
    earned = pit_points + survivorship_points + constraint_points + missing_points
    dimension = DimensionScore(
        key="data_integrity",
        name="Data integrity",
        earned=earned,
        maximum=20.0,
        details={
            "pit_universe": pit_points,
            "survivorship": survivorship_points,
            "trading_constraints": constraint_points,
            "pit_missingness": missing_points,
        },
    )
    return dimension, {"tradable_filter": bool(validation_protocol.get("tradable_filter", False))}


def _robustness_dimension(
    summary: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[DimensionScore, dict[str, Any]]:
    validation = _mapping(_nested(summary, "portfolio", "validation_metrics", default={}))
    holdout = _mapping(_nested(summary, "holdout", "metrics", default={}))
    robustness = _mapping(evidence.get("robustness"))
    statistics = _mapping(evidence.get("statistics"))

    validation_sign = _as_float(validation.get("subperiod_ic_sign_ratio"))
    holdout_sign = _as_float(holdout.get("subperiod_ic_sign_ratio"))
    subperiod_points = 0.0
    subperiod_points += 2.0 if math.isfinite(validation_sign) and validation_sign >= 0.66 else 0.0
    subperiod_points += 2.0 if math.isfinite(holdout_sign) and holdout_sign >= 0.66 else 0.0

    cost_sensitivity = _list(robustness.get("cost_sensitivity"))
    parameter_perturbation = _list(robustness.get("parameter_perturbation"))
    regime_results = _list(robustness.get("regime_results"))
    cost_points = 4.0 if len(cost_sensitivity) >= 3 else 0.0
    parameter_points = 3.0 if len(parameter_perturbation) >= 3 else 0.0
    regime_points = 2.0 if len(regime_results) >= 2 else 0.0
    pristine_points = 2.0 if bool(statistics.get("holdout_pristine", False)) else 0.0

    earned = subperiod_points + cost_points + parameter_points + regime_points + pristine_points
    dimension = DimensionScore(
        key="robustness",
        name="Robustness",
        earned=earned,
        maximum=15.0,
        details={
            "subperiod_stability": subperiod_points,
            "cost_sensitivity": cost_points,
            "parameter_perturbation": parameter_points,
            "regime_stability": regime_points,
            "pristine_holdout": pristine_points,
        },
    )
    return dimension, {"holdout_pristine": bool(statistics.get("holdout_pristine", False))}


def _engineering_dimension(
    run_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    critic: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> tuple[DimensionScore, dict[str, Any]]:
    config_points = 1.0 if (run_dir / "run_config.json").exists() else 0.0
    code_revision_points = 1.0 if str(manifest.get("git_commit") or "").strip() else 0.0
    artifact_count = _as_int(manifest.get("artifact_count"), _as_int(summary.get("artifact_count")))
    lineage_points = 1.0 if artifact_count > 0 and any(item.get("content_hash") for item in artifacts) else 0.0
    decisions = [_mapping(item) for item in _list(critic.get("decisions"))]
    rejected_points = 1.0 if any(not bool(item.get("accepted", False)) for item in decisions) else 0.0
    audit_count = sum(1 for item in artifacts if item.get("kind") == "audit_report")
    governance_points = 1.0 if bool(summary.get("holdout_released", False)) and audit_count > 0 else 0.0
    earned = (
        config_points
        + code_revision_points
        + lineage_points
        + rejected_points
        + governance_points
    )
    dimension = DimensionScore(
        key="research_governance",
        name="Reproducibility and research governance",
        earned=earned,
        maximum=5.0,
        details={
            "config_snapshot": config_points,
            "code_revision": code_revision_points,
            "artifact_lineage": lineage_points,
            "rejected_trials_retained": rejected_points,
            "holdout_and_audit_governance": governance_points,
        },
    )
    return dimension, {"audit_count": audit_count, "artifact_count": artifact_count}


def _hard_gates(
    run_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    critic: dict[str, Any],
    evidence: dict[str, Any],
    selected_ids: list[str],
    selected_decisions: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[GateResult]:
    leakage = _mapping(evidence.get("leakage"))
    execution = _mapping(evidence.get("execution"))
    cost = _mapping(evidence.get("cost"))
    validation = _mapping(_nested(summary, "portfolio", "validation_metrics", default={}))
    holdout = _mapping(_nested(summary, "holdout", "metrics", default={}))

    manifest_keys = [
        bool(str(manifest.get("run_id") or "").strip()),
        bool(str(manifest.get("config_hash") or "").strip()),
        bool(str(manifest.get("data_fingerprint") or "").strip()),
        str(manifest.get("status") or "") == "completed",
        _as_int(manifest.get("artifact_count"), _as_int(summary.get("artifact_count"))) > 0,
    ]
    config_snapshot = (run_dir / "run_config.json").exists()
    gate_reproducible = config_snapshot and all(manifest_keys)

    selected_audits = [
        _mapping(item.get("payload"))
        for item in artifacts
        if item.get("kind") == "audit_report"
        and str(_mapping(item.get("payload")).get("factor_id")) in set(selected_ids)
    ]
    audits_passed = bool(selected_audits) and all(
        bool(item.get("passed", False)) for item in selected_audits
    )
    leakage_flags = [
        bool(leakage.get("label_boundary_guard", False)),
        bool(leakage.get("no_lookahead_full_prefix", False)),
        bool(leakage.get("label_shuffle_same_universe", False)),
        bool(leakage.get("pit_universe", False)),
    ]
    gate_leakage = all(leakage_flags) and audits_passed

    lag = _as_int(
        execution.get("execution_lag_bars"),
        default=-1,
    )
    trade_price = str(execution.get("trade_price", ""))
    gate_execution = lag >= 1 and trade_price not in {"", "same_close"}

    components = {str(item) for item in _list(cost.get("components"))}
    gate_cost = (
        str(cost.get("mode", "")) == "turnover_l1"
        and bool(cost.get("initial_position_charged", False))
        and bool(cost.get("rebalance_charged", False))
        and len(components) >= 4
    )

    q_values = [_as_float(item.get("q_value")) for item in selected_decisions]
    dsr_values = [_as_float(item.get("dsr")) for item in selected_decisions]
    max_q = max(q_values) if q_values else math.nan
    min_dsr = min(dsr_values) if dsr_values else math.nan
    pbo = _as_float(_nested(critic, "pbo", "pbo", default=math.nan))
    n_candidates = _as_int(critic.get("n_candidates"))
    n_decisions = len(_list(critic.get("decisions")))
    all_trials = n_candidates > 0 and n_candidates == n_decisions
    holdout_ci_low = _as_float(holdout.get("sharpe_ci_low"))
    gate_statistics = (
        all_trials
        and math.isfinite(max_q)
        and max_q <= 0.10
        and math.isfinite(min_dsr)
        and min_dsr >= 0.90
        and math.isfinite(pbo)
        and pbo <= 0.20
        and math.isfinite(holdout_ci_low)
        and holdout_ci_low > 0
    )

    implementation_mode = str(execution.get("implementability", "research_long_short"))
    borrow_cost = _as_float(execution.get("borrow_cost_bps"))
    implementable = implementation_mode == "long_only" or (
        implementation_mode == "constrained_short"
        and math.isfinite(borrow_cost)
        and borrow_cost > 0
    )
    gate_implementability = implementable and bool(execution.get("capacity_estimate", False))

    validation_periods = _as_int(validation.get("n_periods"))
    holdout_periods = _as_int(holdout.get("n_periods"))
    gate_sample = validation_periods >= MIN_PERIODS and holdout_periods >= MIN_PERIODS

    return [
        GateResult(
            "G1",
            "Reproducible run contract",
            gate_reproducible,
            (
                f"config_snapshot={config_snapshot}, manifest_fields={sum(manifest_keys)}/5, "
                f"artifacts={_as_int(manifest.get('artifact_count'))}"
            ),
        ),
        GateResult(
            "G2",
            "Leakage control",
            gate_leakage,
            f"leakage_flags={sum(leakage_flags)}/4, selected_audits_passed={audits_passed}",
        ),
        GateResult(
            "G3",
            "Execution causality",
            gate_execution,
            f"execution_lag_bars={lag}, trade_price={trade_price or 'missing'}",
        ),
        GateResult(
            "G4",
            "Cost integrity",
            gate_cost,
            f"mode={cost.get('mode', 'missing')}, components={len(components)}, "
            f"initial={bool(cost.get('initial_position_charged', False))}, "
            f"rebalance={bool(cost.get('rebalance_charged', False))}",
        ),
        GateResult(
            "G5",
            "Statistical survival",
            gate_statistics,
            f"all_trials={all_trials}, max_q={_format_metric(max_q)}, "
            f"min_dsr={_format_metric(min_dsr)}, pbo={_format_metric(pbo)}, "
            f"holdout_ci_low={_format_metric(holdout_ci_low)}",
        ),
        GateResult(
            "G6",
            "Implementability",
            gate_implementability,
            f"mode={implementation_mode}, borrow_cost_bps={_format_metric(borrow_cost)}, "
            f"capacity_estimate={bool(execution.get('capacity_estimate', False))}",
        ),
        GateResult(
            "G7",
            "Sample sufficiency",
            gate_sample,
            f"validation_periods={validation_periods}, holdout_periods={holdout_periods}, "
            f"minimum={MIN_PERIODS}",
        ),
    ]


def score_run(
    run_dir: str | Path,
    *,
    evidence_override: dict[str, Any] | None = None,
) -> Scorecard:
    run_path = Path(run_dir)
    manifest = _read_json(run_path / "manifest.json")
    summary = _read_json(run_path / "summary.json")
    run_config = _read_json(run_path / "run_config.json")
    artifacts = _load_artifacts(run_path)
    evidence = _load_scorecard_evidence(run_path, summary, run_config, evidence_override)

    critic = _mapping(summary.get("critic"))
    selected_ids = _selected_factor_ids(summary, critic)
    selected_decisions = _selected_decisions(critic, selected_ids)
    validation_artifact = _latest_artifact(artifacts, "backtest_report", split="validation")
    validation_protocol = _mapping(_mapping(validation_artifact.get("payload")).get("protocol"))

    statistical, statistical_metrics = _statistical_dimension(
        summary, critic, selected_decisions
    )
    execution, execution_metrics = _execution_dimension(
        evidence, run_config, validation_protocol, summary
    )
    data, data_metrics = _data_dimension(evidence, validation_protocol)
    robustness, robustness_metrics = _robustness_dimension(summary, evidence)
    governance, governance_metrics = _engineering_dimension(
        run_path, manifest, summary, critic, artifacts
    )
    dimensions = [statistical, execution, data, robustness, governance]
    score = round(sum(dimension.earned for dimension in dimensions), 4)

    gates = _hard_gates(
        run_path,
        manifest,
        summary,
        critic,
        evidence,
        selected_ids,
        selected_decisions,
        artifacts,
    )
    grade = _grade_for_score(score, gates)
    alpha_candidate, certification = _certification(score, gates)
    evidence_gaps = [f"{gate.gate_id} {gate.name}: {gate.evidence}" for gate in gates if not gate.passed]

    metrics = {
        **statistical_metrics,
        **execution_metrics,
        **data_metrics,
        **robustness_metrics,
        **governance_metrics,
        "validation_net_sharpe": _as_float(
            _nested(summary, "portfolio", "validation_metrics", "net_sharpe")
        ),
        "holdout_net_sharpe": _as_float(
            _nested(summary, "holdout", "metrics", "net_sharpe")
        ),
        "holdout_net_ann_return": _as_float(
            _nested(summary, "holdout", "metrics", "net_ann_return")
        ),
    }
    return Scorecard(
        run_id=str(manifest.get("run_id") or summary.get("run_id") or run_path.name),
        score=score,
        grade=grade,
        alpha_candidate=alpha_candidate,
        certification=certification,
        gates=gates,
        dimensions=dimensions,
        metrics=metrics,
        evidence_gaps=evidence_gaps,
    )
