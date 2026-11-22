from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ..backtest import BacktestResult, evaluate_factor, forward_returns
from ..config import AppConfig
from ..factors import FactorSpec, compute_factor, generate_factor_specs
from ..portfolio import combine_signals, signal_from_factor
from ..stats import (
    audit_no_lookahead,
    bh_fdr,
    bootstrap_sharpe_ci,
    deflated_sharpe,
    label_shuffle_test,
    probability_of_backtest_overfitting,
)
from .artifacts import ArtifactStore
from .ledger import EventLedger
from .schemas import (
    ArtifactKind,
    ArtifactRecord,
    ArtifactStatus,
    AuditReport,
    BacktestReport,
    CriticDecision,
    CriticReview,
    FactorImplementation,
    FeasibilityReport,
    HoldoutReport,
    HypothesisCard,
    PortfolioDecision,
    SplitName,
)
from .splits import SplitGuard
from .utils import content_hash, to_jsonable


class ResearchSession:
    """Deterministic services shared by all agents in one research run."""

    def __init__(
        self,
        config: AppConfig,
        fields: dict[str, pd.DataFrame],
        metadata: dict[str, Any],
        run_id: str,
        run_dir: Path,
    ):
        self.config = config
        self.fields = fields
        self.metadata = dict(metadata)
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.store = ArtifactStore(self.run_dir, run_id)
        self.ledger = EventLedger(self.run_dir, run_id)
        self.guard = SplitGuard(config, self.run_dir, self.ledger)
        self._specs = {spec.spec_id: spec for spec in generate_factor_specs()}
        self._factor_cache: dict[str, pd.DataFrame] = {}
        self._fwd_ret: pd.DataFrame | None = None
        self._evaluations: dict[tuple[str, str], BacktestResult] = {}
        self._signals: dict[str, pd.DataFrame] = {}
        self._audits: dict[str, dict[str, Any]] = {}
        self._critic_review: CriticReview | None = None
        self._portfolio_signal: pd.DataFrame | None = None
        self._portfolio_id: str | None = None
        self._implementation_runs: dict[str, dict[str, Any]] = {}
        self._capacity_estimate: dict[str, float | bool] = {}

    @classmethod
    def from_config(cls, config: AppConfig, run_id: str, run_dir: Path) -> ResearchSession:
        from ..pipeline import ensure_fields, ensure_market_data

        ensure_market_data(config)
        fields, metadata = ensure_fields(config)
        metadata = dict(metadata)
        metadata.setdefault("data_fingerprint", config.raw.get("data_fingerprint"))
        return cls(config, fields, metadata, run_id, run_dir)

    @property
    def fwd_ret(self) -> pd.DataFrame:
        if self._fwd_ret is None:
            self._fwd_ret = forward_returns(
                self.fields["close"],
                self.config.backtest.horizon,
                self.config.backtest.execution_lag,
            )
        return self._fwd_ret

    def spec(self, spec_id: str) -> FactorSpec:
        try:
            return self._specs[spec_id]
        except KeyError as exc:
            raise KeyError(f"Unknown factor template: {spec_id}") from exc

    def catalog(
        self,
        limit: int | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        specs = self._deduplicated_specs()
        if category:
            specs = [spec for spec in specs if spec.category == category]
        if limit is not None and limit >= 0:
            specs = self._balanced_sample(specs, int(limit))
        return [self._spec_payload(spec) for spec in specs]

    def _deduplicated_specs(self) -> list[FactorSpec]:
        output: list[FactorSpec] = []
        seen: set[tuple[str, float]] = set()
        for spec in generate_factor_specs():
            key = (spec.formula, float(spec.direction))
            if key in seen:
                continue
            seen.add(key)
            output.append(spec)
        return output

    @staticmethod
    def _balanced_sample(specs: list[FactorSpec], limit: int) -> list[FactorSpec]:
        if limit <= 0 or limit >= len(specs):
            return specs
        groups: dict[str, list[FactorSpec]] = {}
        for spec in specs:
            groups.setdefault(spec.category, []).append(spec)
        ordered_categories = sorted(groups)
        selected: list[FactorSpec] = []
        cursor = 0
        while len(selected) < limit:
            added = False
            for category in ordered_categories:
                bucket = groups[category]
                if cursor < len(bucket):
                    selected.append(bucket[cursor])
                    added = True
                    if len(selected) >= limit:
                        break
            if not added:
                break
            cursor += 1
        return selected

    def _spec_payload(self, spec: FactorSpec) -> dict[str, Any]:
        return {
            "spec_id": spec.spec_id,
            "name": spec.name,
            "family": spec.family,
            "category": spec.category,
            "hypothesis": spec.hypothesis,
            "formula": spec.formula,
            "direction": spec.direction,
            "window": spec.window,
            "required_fields": self.required_fields_for(spec),
        }

    def required_fields_for(self, spec: FactorSpec) -> list[str]:
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", spec.formula))
        known = set(self.fields)
        aliases = {
            "ts_mean",
            "ts_std",
            "ts_sum",
            "ts_max",
            "ts_min",
            "ts_skew",
            "ts_corr",
            "delay",
            "rolling_beta",
            "neutralize",
            "log",
            "abs",
        }
        return sorted((tokens & known) - aliases)

    def _find_artifact(
        self,
        kind: ArtifactKind,
        spec_id: str,
        statuses: Iterable[ArtifactStatus] | None = None,
    ) -> ArtifactRecord | None:
        allowed = {item.value for item in statuses} if statuses else None
        matches = [
            record
            for record in self.store.list(kind=kind)
            if record.payload.get("spec_id") == spec_id
            and (allowed is None or record.status.value in allowed)
        ]
        return matches[-1] if matches else None

    def hypothesis_cards(self) -> list[HypothesisCard]:
        return [
            HypothesisCard.model_validate(record.payload)
            for record in self.store.list(kind=ArtifactKind.HYPOTHESIS_CARD)
        ]

    def resolve_hypothesis(self, hypothesis_id: str) -> ArtifactRecord:
        """Accept either an artifact id or the human-readable ``hyp_*`` id."""
        try:
            record = self.store.get(hypothesis_id)
            if record.kind is ArtifactKind.HYPOTHESIS_CARD:
                return record
        except KeyError:
            pass
        for record in self.store.list(ArtifactKind.HYPOTHESIS_CARD):
            if record.payload.get("hypothesis_id") == hypothesis_id:
                return record
        raise KeyError(f"Unknown hypothesis: {hypothesis_id}")

    def create_hypothesis(
        self,
        spec_id: str,
        rationale: str | None = None,
        producer: str = "hypothesis_agent",
    ) -> ArtifactRecord:
        existing = self._find_artifact(ArtifactKind.HYPOTHESIS_CARD, spec_id)
        if existing is not None:
            return existing
        spec = self.spec(spec_id)
        required_fields = self.required_fields_for(spec)
        card = HypothesisCard(
            hypothesis_id=f"hyp_{spec.spec_id}",
            spec_id=spec.spec_id,
            name=spec.name,
            family=spec.family,
            category=spec.category,
            economic_rationale=rationale or spec.hypothesis,
            testable_prediction=(
                "The directional signal should have a positive validation IC and "
                "survive cost-aware validation after multiple-testing controls."
            ),
            formula=spec.formula,
            direction=float(spec.direction),
            window=int(spec.window),
            required_fields=required_fields,
            dedupe_key=content_hash({"formula": spec.formula, "direction": spec.direction}),
        )
        record = self.store.put(
            ArtifactKind.HYPOTHESIS_CARD,
            card.model_dump(mode="json"),
            producer=producer,
            status=ArtifactStatus.PROPOSED,
            metadata={"spec_id": spec.spec_id, "family": spec.family},
        )
        self.ledger.append(
            "hypothesis_created",
            producer,
            {"hypothesis_id": card.hypothesis_id, "spec_id": spec.spec_id},
            artifact_id=record.artifact_id,
        )
        return record

    def check_feasibility(
        self,
        hypothesis_id: str,
        producer: str = "feasibility_agent",
    ) -> ArtifactRecord:
        hypothesis = self.resolve_hypothesis(hypothesis_id)
        card = HypothesisCard.model_validate(hypothesis.payload)
        spec = self.spec(card.spec_id)
        missing = [field for field in card.required_fields if field not in self.fields]
        reasons: list[str] = []
        finite_ratio = 0.0
        median_names = 0.0
        if missing:
            reasons.append(f"missing fields: {', '.join(missing)}")
        else:
            try:
                factor = compute_factor(spec, self.fields)
                self._factor_cache[spec.spec_id] = factor
                finite_ratio = float(np.isfinite(factor.to_numpy()).mean())
                median_names = float(factor.notna().sum(axis=1).median())
            except Exception as exc:  # pragma: no cover - defensive boundary
                reasons.append(f"factor computation failed: {type(exc).__name__}: {exc}")
        if finite_ratio < 0.20:
            reasons.append(f"finite ratio {finite_ratio:.3f} < 0.20")
        if median_names < self.config.backtest.min_names:
            reasons.append(
                f"median cross-section {median_names:.1f} < {self.config.backtest.min_names}"
            )
        feasible = not reasons
        report = FeasibilityReport(
            hypothesis_id=card.hypothesis_id,
            spec_id=spec.spec_id,
            feasible=feasible,
            finite_ratio=finite_ratio,
            median_names=median_names,
            missing_fields=missing,
            reasons=reasons,
        )
        record = self.store.put(
            ArtifactKind.FEASIBILITY_REPORT,
            report.model_dump(mode="json"),
            producer=producer,
            status=ArtifactStatus.ACCEPTED if feasible else ArtifactStatus.REJECTED,
            parents=[hypothesis.artifact_id],
            metadata={"spec_id": spec.spec_id},
        )
        self.ledger.append(
            "feasibility_checked",
            producer,
            {"spec_id": spec.spec_id, "feasible": feasible, "reasons": reasons},
            artifact_id=record.artifact_id,
        )
        return record

    def implement_factor(
        self,
        hypothesis_id: str,
        producer: str = "factor_engineer",
    ) -> ArtifactRecord:
        hypothesis = self.resolve_hypothesis(hypothesis_id)
        card = HypothesisCard.model_validate(hypothesis.payload)
        existing = self._find_artifact(ArtifactKind.FACTOR_IMPLEMENTATION, card.spec_id)
        if existing is not None:
            return existing
        feasibility = self._find_artifact(ArtifactKind.FEASIBILITY_REPORT, card.spec_id)
        if feasibility is None or feasibility.status is not ArtifactStatus.ACCEPTED:
            reasons = (
                "feasibility report missing"
                if feasibility is None
                else "; ".join(feasibility.payload.get("reasons", []))
            )
            implementation = FactorImplementation(
                hypothesis_id=card.hypothesis_id,
                spec_id=card.spec_id,
                status="rejected",
                finite_ratio=0.0,
                median_names=0.0,
                reason=reasons,
            )
            record = self.store.put(
                ArtifactKind.FACTOR_IMPLEMENTATION,
                implementation.model_dump(mode="json"),
                producer=producer,
                status=ArtifactStatus.REJECTED,
                parents=[hypothesis.artifact_id, feasibility.artifact_id] if feasibility else [hypothesis.artifact_id],
            )
            return record
        spec = self.spec(card.spec_id)
        factor = self._factor_cache.get(card.spec_id)
        if factor is None:
            factor = compute_factor(spec, self.fields)
            self._factor_cache[card.spec_id] = factor
        finite_ratio = float(np.isfinite(factor.to_numpy()).mean())
        median_names = float(factor.notna().sum(axis=1).median())
        implemented = finite_ratio >= 0.20 and median_names >= self.config.backtest.min_names
        implementation = FactorImplementation(
            hypothesis_id=card.hypothesis_id,
            spec_id=card.spec_id,
            status="implemented" if implemented else "rejected",
            finite_ratio=finite_ratio,
            median_names=median_names,
            diagnostic={
                "n_dates": int(factor.index.size),
                "n_symbols": int(factor.columns.size),
                "direction": spec.direction,
                "window": spec.window,
            },
            reason="" if implemented else "implementation diagnostics failed",
        )
        record = self.store.put(
            ArtifactKind.FACTOR_IMPLEMENTATION,
            implementation.model_dump(mode="json"),
            producer=producer,
            status=ArtifactStatus.ACCEPTED if implemented else ArtifactStatus.REJECTED,
            parents=[hypothesis.artifact_id, feasibility.artifact_id],
            metadata={"spec_id": card.spec_id},
        )
        self.ledger.append(
            "factor_implemented",
            producer,
            {
                "spec_id": card.spec_id,
                "status": implementation.status,
                "finite_ratio": finite_ratio,
                "median_names": median_names,
            },
            artifact_id=record.artifact_id,
        )
        return record

    def _run_backtest(
        self,
        signal: pd.DataFrame,
        factor_id: str,
        split: SplitName | str,
        actor: str,
        purpose: str,
        parents: list[str] | None = None,
    ) -> tuple[ArtifactRecord, BacktestResult]:
        split_value = self.guard.assert_allowed(split, actor=actor, purpose=purpose)
        start, end = self.guard.dates(split_value)
        result = evaluate_factor(
            signal,
            self.fwd_ret,
            self.config.backtest,
            start=start,
            end=end,
            tradable=self.fields.get("tradable"),
        )
        sharpe_interval = bootstrap_sharpe_ci(
            result.returns, n_boot=500, seed=self.config.seed
        )
        result.metrics.update(sharpe_interval)
        self._evaluations[(factor_id, split_value.value)] = result
        warnings: list[str] = []
        coverage = float(result.metrics.get("coverage", 0.0) or 0.0)
        if coverage < self.config.critic.min_coverage:
            warnings.append(
                f"coverage {coverage:.3f} below critic floor {self.config.critic.min_coverage:.3f}"
            )
        report = BacktestReport(
            factor_id=factor_id,
            split=split_value,
            metrics=to_jsonable(result.metrics),
            cost_bps_per_side=float(self.config.backtest.cost_bps_per_side),
            protocol={
                "start": start,
                "end": end,
                "horizon": self.config.backtest.horizon,
                "portfolio_mode": self.config.backtest.portfolio_mode,
                "quantile": self.config.backtest.quantile,
                "min_names": self.config.backtest.min_names,
                "execution_lag": self.config.backtest.execution_lag,
                "execution_assumption": (
                    "signal at close t; execute at close t+execution_lag; "
                    "hold for horizon trading days"
                ),
                "label_boundary_guard": True,
                "cost_model": "turnover_l1",
                "cost_components": {
                    "commission_bps": self.config.backtest.commission_bps,
                    "stamp_duty_bps": self.config.backtest.stamp_duty_bps,
                    "spread_bps": self.config.backtest.spread_bps,
                    "impact_bps": self.config.backtest.impact_bps,
                    "borrow_bps_annual": self.config.backtest.borrow_bps_annual,
                },
                "initial_position_charged": True,
                "rebalance_charged": True,
                "tradable_filter": "tradable" in self.fields,
            },
            warnings=warnings,
        )
        record = self.store.put(
            ArtifactKind.BACKTEST_REPORT,
            report.model_dump(mode="json"),
            producer=actor,
            status=ArtifactStatus.COMPLETED,
            parents=parents or [],
            split=split_value,
            metadata={"factor_id": factor_id, "purpose": purpose},
        )
        self.ledger.append(
            "backtest_completed",
            actor,
            {
                "factor_id": factor_id,
                "split": split_value.value,
                "net_sharpe": result.metrics.get("net_sharpe"),
                "net_ann_return": result.metrics.get("net_ann_return"),
                "coverage": coverage,
            },
            artifact_id=record.artifact_id,
        )
        return record, result

    def _capacity_from_weights(
        self,
        weights: pd.DataFrame,
        start: str,
        end: str,
        participation_rate: float = 0.10,
    ) -> dict[str, float | bool]:
        adv = self.fields.get("adv20")
        if adv is None or weights.empty:
            return {"capacity_estimate": False}
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        adv_window = adv.loc[(adv.index >= start_ts) & (adv.index <= end_ts)]
        estimates: list[float] = []
        for dt in weights.index.intersection(adv_window.index):
            row = weights.loc[dt]
            active = row[row.abs() > 1e-12]
            if active.empty:
                continue
            liquidity = adv_window.loc[dt].reindex(active.index)
            capacity = liquidity.mul(participation_rate).div(active.abs()).replace(
                [np.inf, -np.inf], np.nan
            )
            finite = capacity.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite) & (finite > 0)]
            if finite.size:
                estimates.append(float(np.min(finite)))
        if not estimates:
            return {"capacity_estimate": False}
        values = np.asarray(estimates, dtype=float)
        return {
            "capacity_estimate": True,
            "estimated_aum_median": float(np.median(values)),
            "estimated_aum_p10": float(np.quantile(values, 0.10)),
            "participation_rate": participation_rate,
        }

    def _run_implementation_backtest(
        self,
        signal: pd.DataFrame,
        split: SplitName | str,
        actor: str,
        purpose: str,
        parents: list[str] | None = None,
        portfolio_id: str = "portfolio_long_only",
    ) -> tuple[ArtifactRecord, BacktestResult] | None:
        split_value = self.guard.assert_allowed(split, actor=actor, purpose=purpose)
        start, end = self.guard.dates(split_value)
        implementation_config = replace(self.config.backtest, portfolio_mode="long_only")
        result = evaluate_factor(
            signal,
            self.fwd_ret,
            implementation_config,
            start=start,
            end=end,
            tradable=self.fields.get("tradable"),
        )
        result.metrics.update(
            bootstrap_sharpe_ci(result.returns, n_boot=500, seed=self.config.seed)
        )
        capacity = self._capacity_from_weights(result.weights, start, end)
        if capacity.get("capacity_estimate"):
            self._capacity_estimate = capacity
        self._implementation_runs[split_value.value] = {
            "portfolio_id": portfolio_id,
            "portfolio_mode": "long_only",
            "metrics": to_jsonable(result.metrics),
            "capacity": to_jsonable(capacity),
        }
        report = BacktestReport(
            factor_id=portfolio_id,
            split=split_value,
            metrics=to_jsonable(result.metrics),
            cost_bps_per_side=float(self.config.backtest.cost_bps_per_side),
            protocol={
                "start": start,
                "end": end,
                "horizon": self.config.backtest.horizon,
                "portfolio_mode": "long_only",
                "execution_lag": self.config.backtest.execution_lag,
                "implementation_mode": "long_only",
                "tradable_filter": "tradable" in self.fields,
            },
            warnings=[],
        )
        record = self.store.put(
            ArtifactKind.BACKTEST_REPORT,
            report.model_dump(mode="json"),
            producer=actor,
            status=ArtifactStatus.COMPLETED,
            parents=parents or [],
            split=split_value,
            metadata={"factor_id": portfolio_id, "purpose": purpose},
        )
        self.ledger.append(
            "implementation_backtest_completed",
            actor,
            {
                "portfolio_id": portfolio_id,
                "split": split_value.value,
                "portfolio_mode": "long_only",
                "net_sharpe": result.metrics.get("net_sharpe"),
                "net_ann_return": result.metrics.get("net_ann_return"),
            },
            artifact_id=record.artifact_id,
        )
        return record, result

    def evaluate(
        self,
        factor_id: str,
        split: SplitName | str,
        actor: str = "evaluator",
        purpose: str = "factor evaluation",
    ) -> ArtifactRecord:
        if factor_id not in self._factor_cache:
            raise KeyError(f"Factor not implemented: {factor_id}")
        implementation = self._find_artifact(ArtifactKind.FACTOR_IMPLEMENTATION, factor_id)
        record, _ = self._run_backtest(
            self._factor_cache[factor_id],
            factor_id,
            split,
            actor=actor,
            purpose=purpose,
            parents=[implementation.artifact_id] if implementation else [],
        )
        return record

    def evaluation(self, factor_id: str, split: SplitName | str) -> BacktestResult:
        split_value = split if isinstance(split, SplitName) else SplitName(split)
        key = (factor_id, split_value.value)
        if key not in self._evaluations:
            self.evaluate(factor_id, split_value)
        return self._evaluations[key]

    def _validated_candidates(self) -> list[str]:
        implementations = [
            record
            for record in self.store.list(
                kind=ArtifactKind.FACTOR_IMPLEMENTATION,
                status=ArtifactStatus.ACCEPTED,
            )
            if record.payload.get("status") == "implemented"
        ]
        return [str(record.payload["spec_id"]) for record in implementations]

    def audit(self, factor_id: str, producer: str = "critic_agent") -> ArtifactRecord:
        if factor_id in self._audits:
            return self._audits[factor_id]
        spec = self.spec(factor_id)
        close_dates = self.fields["close"].index
        validation_end = pd.Timestamp(self.config.splits.validation_end)
        possible = close_dates[close_dates <= validation_end - pd.Timedelta(days=30)]
        if len(possible) == 0:
            possible = close_dates[close_dates <= validation_end]
        cutoff = possible[-1]
        no_lookahead = audit_no_lookahead(
            spec, self.fields, cutoff=cutoff, seed=self.config.seed
        )
        shuffle = label_shuffle_test(
            spec,
            self.fields,
            self.fwd_ret,
            start=self.config.splits.validation_start,
            end=self.config.splits.validation_end,
            config=self.config.backtest,
            tradable=self.fields.get("tradable"),
            seed=self.config.seed,
        )
        reasons: list[str] = []
        if not no_lookahead.get("passed"):
            reasons.append("future-data perturbation changed the factor")
        if not shuffle.get("passed"):
            reasons.append("factor did not beat shuffled forward-return labels")
        passed = not reasons
        report = AuditReport(
            factor_id=factor_id,
            no_lookahead=to_jsonable(no_lookahead),
            label_shuffle=to_jsonable(shuffle),
            passed=passed,
            reasons=reasons,
        )
        implementation = self._find_artifact(ArtifactKind.FACTOR_IMPLEMENTATION, factor_id)
        record = self.store.put(
            ArtifactKind.AUDIT_REPORT,
            report.model_dump(mode="json"),
            producer=producer,
            status=ArtifactStatus.ACCEPTED if passed else ArtifactStatus.REJECTED,
            parents=[implementation.artifact_id] if implementation else [],
            metadata={"spec_id": factor_id, "cutoff": str(cutoff.date())},
        )
        self._audits[factor_id] = {"record": record, "report": report}
        self.ledger.append(
            "audit_completed",
            producer,
            {
                "factor_id": factor_id,
                "passed": passed,
                "cutoff": str(cutoff.date()),
                "reasons": reasons,
            },
            artifact_id=record.artifact_id,
        )
        return record

    @staticmethod
    def _pvalues(factor_ids: list[str], evaluations: dict[tuple[str, str], BacktestResult]) -> np.ndarray:
        values: list[float] = []
        for factor_id in factor_ids:
            metrics = evaluations[(factor_id, SplitName.VALIDATION.value)].metrics
            mean_ic = metrics.get("mean_ic", np.nan)
            ic_std = metrics.get("ic_std", np.nan)
            n = metrics.get("n_periods", 0)
            if not np.isfinite(mean_ic) or not np.isfinite(ic_std) or ic_std <= 0 or n < 3:
                values.append(np.nan)
                continue
            tstat = mean_ic / ic_std * np.sqrt(n)
            values.append(float(2.0 * student_t.sf(abs(tstat), df=n - 1)))
        return np.asarray(values, dtype=float)

    def screen_candidates(
        self,
        producer: str = "critic_agent",
        max_audits: int | None = None,
    ) -> ArtifactRecord:
        factor_ids = self._validated_candidates()
        for factor_id in factor_ids:
            self.evaluation(factor_id, SplitName.TRAIN)
            self.evaluation(factor_id, SplitName.VALIDATION)
        if not factor_ids:
            review = CriticReview(decisions=[], pbo={}, n_candidates=0, n_accepted=0)
            record = self.store.put(
                ArtifactKind.CRITIC_REVIEW,
                review.model_dump(mode="json"),
                producer=producer,
                status=ArtifactStatus.REJECTED,
            )
            self._critic_review = review
            return record

        pvalues = self._pvalues(factor_ids, self._evaluations)
        qvalues = bh_fdr(pvalues)
        trial_sharpes: list[float] = []
        for factor_id in factor_ids:
            returns = self._evaluations[(factor_id, SplitName.VALIDATION.value)].returns.dropna()
            if len(returns) > 2 and returns.std(ddof=1) > 0:
                trial_sharpes.append(float(returns.mean() / returns.std(ddof=1)))
            else:
                trial_sharpes.append(0.0)
        validation_matrix = pd.concat(
            {
                factor_id: self._evaluations[(factor_id, SplitName.VALIDATION.value)].returns
                for factor_id in factor_ids
            },
            axis=1,
        )
        pbo = probability_of_backtest_overfitting(
            validation_matrix, seed=self.config.seed
        )
        if max_audits is None:
            max_audits = int(self.config.raw.get("agentic", {}).get("max_audits", 12))

        decisions: list[CriticDecision] = []
        audits_used = 0
        critic = self.config.critic
        for index, factor_id in enumerate(factor_ids):
            validation = self._evaluations[(factor_id, SplitName.VALIDATION.value)]
            train = self._evaluations[(factor_id, SplitName.TRAIN.value)]
            metrics = validation.metrics
            qvalue = float(qvalues[index]) if np.isfinite(qvalues[index]) else np.nan
            dsr_result = deflated_sharpe(
                validation.returns, len(factor_ids), trial_sharpes
            )
            dsr = float(dsr_result["dsr"]) if np.isfinite(dsr_result["dsr"]) else np.nan
            checks = {
                "enough_periods": metrics.get("n_periods", 0) >= 12,
                "coverage": metrics.get("coverage", 0.0) >= critic.min_coverage,
                "ic": metrics.get("mean_ic", -1.0) >= critic.min_validation_ic,
                "icir": metrics.get("icir", -1.0) >= critic.min_validation_icir,
                "net_sharpe": metrics.get("net_sharpe", -1.0) >= critic.min_validation_net_sharpe,
                "turnover": metrics.get("turnover_annual", 1e9) <= critic.max_turnover_annual,
                "fdr": np.isfinite(qvalue) and qvalue <= critic.max_fdr_q,
                "subperiod": metrics.get("subperiod_ic_sign_ratio", 0.0)
                >= critic.min_subperiod_sign_ratio,
                "train_net_sharpe": train.metrics.get("net_sharpe", -1.0) > 0,
                "train_ic": train.metrics.get("mean_ic", -1.0) > 0,
                "train_icir": train.metrics.get("icir", -1.0) > 0,
            }
            reasons = [f"failed: {name}" for name, passed in checks.items() if not passed]
            audit_passed = False
            if all(checks.values()) and audits_used < max_audits:
                audits_used += 1
                audit_record = self.audit(factor_id, producer=producer)
                audit_passed = audit_record.status is ArtifactStatus.ACCEPTED
                if not audit_passed:
                    reasons.extend(audit_record.payload.get("reasons", []))
            elif all(checks.values()):
                reasons.append("not audited: max_audits budget exhausted")
            accepted = all(checks.values()) and audit_passed
            decisions.append(
                CriticDecision(
                    factor_id=factor_id,
                    accepted=accepted,
                    status="accepted" if accepted else "rejected",
                    checks=checks,
                    q_value=qvalue if np.isfinite(qvalue) else None,
                    dsr=dsr if np.isfinite(dsr) else None,
                    reasons=reasons,
                )
            )
        n_accepted = sum(item.accepted for item in decisions)
        review = CriticReview(
            decisions=decisions,
            pbo=to_jsonable(pbo),
            n_candidates=len(factor_ids),
            n_accepted=n_accepted,
            reviewer_notes=(
                "Acceptance requires deterministic statistical gates and both leakage audits. "
                "The language-model critic may add objections but cannot bypass these gates."
            ),
        )
        record = self.store.put(
            ArtifactKind.CRITIC_REVIEW,
            review.model_dump(mode="json"),
            producer=producer,
            status=ArtifactStatus.ACCEPTED if n_accepted else ArtifactStatus.REJECTED,
            metadata={"n_candidates": len(factor_ids), "n_accepted": n_accepted},
        )
        self._critic_review = review
        self.ledger.append(
            "critic_review_completed",
            producer,
            {"n_candidates": len(factor_ids), "n_accepted": n_accepted, "pbo": pbo},
            artifact_id=record.artifact_id,
        )
        return record

    def _portfolio_factor_ids(
        self,
        factor_ids: list[str] | None,
    ) -> tuple[list[str], bool, list[str]]:
        warnings: list[str] = []
        accepted = [
            decision.factor_id
            for decision in (self._critic_review.decisions if self._critic_review else [])
            if decision.accepted
        ]
        if factor_ids:
            requested = [item for item in factor_ids if item in accepted]
            if len(requested) != len(factor_ids):
                warnings.append("Some requested factors were not accepted by the critic and were dropped.")
            return requested, False, warnings
        if accepted:
            return accepted, False, warnings
        fallback_pool = list(self._validated_candidates())
        fallback_pool.sort(
            key=lambda item: self.evaluation(item, SplitName.VALIDATION).metrics.get(
                "net_sharpe", -np.inf
            ),
            reverse=True,
        )
        if fallback_pool:
            warnings.append(
                "No factor passed all gates; portfolio is an exploratory fallback ranked on validation only and may be negative."
            )
            return fallback_pool[: self.config.portfolio.max_factors], True, warnings
        return [], True, warnings

    def apply_critic_overrides(
        self,
        reject_factor_ids: list[str],
        notes: str,
        producer: str = "critic_agent",
    ) -> ArtifactRecord:
        """Allow an LLM critic to add vetoes, but never to override a failed gate."""
        if self._critic_review is None:
            record = self.store.latest(ArtifactKind.CRITIC_REVIEW)
            if record is None:
                raise RuntimeError("Run deterministic screening before applying critic overrides.")
            self._critic_review = CriticReview.model_validate(record.payload)
        rejected = set(reject_factor_ids)
        decisions: list[CriticDecision] = []
        for decision in self._critic_review.decisions:
            if decision.factor_id in rejected and decision.accepted:
                updated = decision.model_copy(deep=True)
                updated.accepted = False
                updated.status = "rejected_by_critic"
                updated.checks = {**updated.checks, "llm_critic_veto": False}
                updated.reasons = list(updated.reasons) + [
                    f"language-model critic veto: {notes or 'no additional rationale'}"
                ]
                decisions.append(updated)
            else:
                decisions.append(decision.model_copy(deep=True))
        review = CriticReview(
            decisions=decisions,
            pbo=self._critic_review.pbo,
            n_candidates=self._critic_review.n_candidates,
            n_accepted=sum(item.accepted for item in decisions),
            reviewer_notes=notes,
        )
        record = self.store.put(
            ArtifactKind.CRITIC_REVIEW,
            review.model_dump(mode="json"),
            producer=producer,
            status=ArtifactStatus.ACCEPTED if review.n_accepted else ArtifactStatus.REJECTED,
            metadata={"override": True, "rejected_factor_ids": sorted(rejected)},
        )
        self._critic_review = review
        self.ledger.append(
            "critic_overrides_applied",
            producer,
            {"rejected_factor_ids": sorted(rejected), "n_accepted": review.n_accepted},
            artifact_id=record.artifact_id,
        )
        return record

    def construct_portfolio(
        self,
        factor_ids: list[str] | None = None,
        producer: str = "portfolio_agent",
    ) -> ArtifactRecord:
        if self._critic_review is None:
            critic_record = self.store.latest(ArtifactKind.CRITIC_REVIEW)
            if critic_record is not None:
                self._critic_review = CriticReview.model_validate(critic_record.payload)
        candidates, fallback, warnings = self._portfolio_factor_ids(factor_ids)
        if not candidates:
            decision = PortfolioDecision(
                portfolio_id="portfolio_main",
                fallback=True,
                warnings=warnings + ["No eligible factor was available for portfolio construction."],
            )
            record = self.store.put(
                ArtifactKind.PORTFOLIO_DECISION,
                decision.model_dump(mode="json"),
                producer=producer,
                status=ArtifactStatus.REJECTED,
            )
            return record

        signals = {factor_id: signal_from_factor(self._factor_cache[factor_id]) for factor_id in candidates}

        def validation_sharpe(ids: list[str]) -> float:
            signal = combine_signals({item: signals[item] for item in ids})
            result = evaluate_factor(
                signal,
                self.fwd_ret,
                self.config.backtest,
                start=self.config.splits.validation_start,
                end=self.config.splits.validation_end,
                tradable=self.fields.get("tradable"),
            )
            score = result.metrics.get("net_sharpe", np.nan)
            return float(score) if np.isfinite(score) else -np.inf

        ranked = sorted(
            candidates,
            key=lambda item: self.evaluation(item, SplitName.VALIDATION).metrics.get(
                "net_sharpe", -np.inf
            ),
            reverse=True,
        )
        selected = [ranked[0]]
        current_score = validation_sharpe(selected)
        selection_path: list[dict[str, Any]] = [
            {"factor_id": selected[0], "validation_net_sharpe": current_score}
        ]
        while len(selected) < self.config.portfolio.max_factors:
            best_candidate: str | None = None
            best_score = current_score
            for candidate in ranked:
                if candidate in selected:
                    continue
                corrs: list[float] = []
                for other in selected:
                    pair = pd.concat(
                        [
                            self.evaluation(candidate, SplitName.VALIDATION).returns.rename("a"),
                            self.evaluation(other, SplitName.VALIDATION).returns.rename("b"),
                        ],
                        axis=1,
                    ).dropna()
                    corrs.append(abs(float(pair["a"].corr(pair["b"]))) if len(pair) > 5 else 0.0)
                if corrs and max(corrs) > self.config.portfolio.max_pairwise_corr:
                    continue
                trial = selected + [candidate]
                score = validation_sharpe(trial)
                if score > best_score + 0.01:
                    best_candidate = candidate
                    best_score = score
            if best_candidate is None:
                break
            selected.append(best_candidate)
            current_score = best_score
            selection_path.append(
                {"factor_id": best_candidate, "validation_net_sharpe": current_score}
            )

        for factor_id in selected:
            if factor_id not in self._audits:
                self.audit(factor_id, producer=producer)

        weights = {factor_id: 1.0 / len(selected) for factor_id in selected}
        combined = combine_signals({factor_id: signals[factor_id] for factor_id in selected}, weights)
        self._portfolio_signal = combined
        self._portfolio_id = "portfolio_main"
        validation_record, validation_result = self._run_backtest(
            combined,
            "portfolio_main",
            SplitName.VALIDATION,
            actor=producer,
            purpose="portfolio validation",
            parents=[record.artifact_id for record in self.store.list(ArtifactKind.FACTOR_IMPLEMENTATION, ArtifactStatus.ACCEPTED)],
        )
        self._run_implementation_backtest(
            combined,
            SplitName.VALIDATION,
            actor=producer,
            purpose="long-only validation",
            parents=[validation_record.artifact_id],
        )
        decision = PortfolioDecision(
            portfolio_id="portfolio_main",
            selected_factors=selected,
            weights=weights,
            fallback=fallback,
            selection_path=selection_path,
            validation_metrics=to_jsonable(validation_result.metrics),
            warnings=warnings,
        )
        record = self.store.put(
            ArtifactKind.PORTFOLIO_DECISION,
            decision.model_dump(mode="json"),
            producer=producer,
            status=ArtifactStatus.ACCEPTED,
            parents=[validation_record.artifact_id],
            metadata={"selected_factors": selected},
        )
        self.ledger.append(
            "portfolio_constructed",
            producer,
            {
                "portfolio_id": "portfolio_main",
                "selected_factors": selected,
                "validation_net_sharpe": validation_result.metrics.get("net_sharpe"),
                "fallback": fallback,
            },
            artifact_id=record.artifact_id,
        )
        return record

    def evaluate_holdout(
        self,
        released_by: str,
        reason: str,
        producer: str = "holdout_evaluator",
    ) -> ArtifactRecord:
        if self._portfolio_signal is None or self._portfolio_id is None:
            raise RuntimeError("Construct the portfolio before releasing the holdout.")
        self.guard.release_holdout(actor=released_by, reason=reason)
        _, result = self._run_backtest(
            self._portfolio_signal,
            self._portfolio_id,
            SplitName.TEST,
            actor=producer,
            purpose=reason,
            parents=[record.artifact_id for record in self.store.list(ArtifactKind.PORTFOLIO_DECISION)],
        )
        self._run_implementation_backtest(
            self._portfolio_signal,
            SplitName.TEST,
            actor=producer,
            purpose=f"long-only holdout: {reason}",
            parents=[record.artifact_id for record in self.store.list(ArtifactKind.PORTFOLIO_DECISION)],
        )
        report = HoldoutReport(
            portfolio_id=self._portfolio_id,
            metrics=to_jsonable(result.metrics),
            released_by=released_by,
            reason=reason,
            limitations=[
                "The test period may have been observed during earlier project iterations; treat this as a benchmark, not a pristine prospective holdout.",
                "The research portfolio remains long-short; a separate long-only implementation is reported for deployability checks but is not the selected research portfolio.",
                "Borrow availability and recall risk are not fully modelled for the research short leg.",
                "Costs include commission, stamp duty, spread, and impact assumptions; capacity is estimated from ADV and requires separate stress testing.",
                "The current universe still depends on a public data source and may contain survivorship or point-in-time limitations.",
            ],
        )
        record = self.store.put(
            ArtifactKind.HOLDOUT_REPORT,
            report.model_dump(mode="json"),
            producer=producer,
            status=ArtifactStatus.COMPLETED,
            split=SplitName.TEST,
        )
        self.ledger.append(
            "holdout_evaluated",
            producer,
            {
                "portfolio_id": self._portfolio_id,
                "net_sharpe": result.metrics.get("net_sharpe"),
                "net_ann_return": result.metrics.get("net_ann_return"),
            },
            artifact_id=record.artifact_id,
        )
        return record

    def _scorecard_evidence(self) -> dict[str, Any]:
        cost_components = ["commission", "stamp_duty", "spread", "impact"]
        if (
            self.config.backtest.commission_bps
            + self.config.backtest.stamp_duty_bps
            + self.config.backtest.spread_bps
            + self.config.backtest.impact_bps
            <= 0
        ):
            cost_components = ["all_in"]
        long_only_available = bool(self._implementation_runs)
        return {
            "leakage": {
                "label_boundary_guard": True,
                "no_lookahead_full_prefix": True,
                "label_shuffle_same_universe": True,
                "pit_universe": bool(self.metadata.get("pit_universe", False)),
            },
            "data": {
                "delisting_control": bool(self.metadata.get("delisting_control", False)),
                "t_plus_one": False,
                "limit_control": False,
                "st_control": False,
                "suspension_control": False,
                "selection_missing_window": self.metadata.get("missing_ratio_window")
                == "selection_end_date",
            },
            "execution": {
                "execution_lag_bars": int(self.config.backtest.execution_lag),
                "trade_price": "next_close",
                "implementability": "long_only"
                if long_only_available
                else "research_long_short",
                "borrow_cost_bps": 0.0,
                "capacity_estimate": bool(self._capacity_estimate.get("capacity_estimate", False)),
            },
            "cost": {
                "mode": "turnover_l1",
                "initial_position_charged": True,
                "rebalance_charged": True,
                "components": cost_components,
            },
            "statistics": {
                "holdout_pristine": False,
            },
            "robustness": {},
        }

    def summary(self) -> dict[str, Any]:
        hypotheses = self.store.list(ArtifactKind.HYPOTHESIS_CARD)
        implementations = self.store.list(ArtifactKind.FACTOR_IMPLEMENTATION)
        critic = self.store.latest(ArtifactKind.CRITIC_REVIEW)
        portfolio = self.store.latest(ArtifactKind.PORTFOLIO_DECISION)
        holdout = self.store.latest(ArtifactKind.HOLDOUT_REPORT)
        return {
            "run_id": self.run_id,
            "data_fingerprint": self.metadata.get("data_fingerprint"),
            "n_hypotheses": len(hypotheses),
            "n_implemented": sum(
                1 for record in implementations if record.payload.get("status") == "implemented"
            ),
            "n_rejected_implementations": sum(
                1 for record in implementations if record.payload.get("status") == "rejected"
            ),
            "critic": critic.payload if critic else None,
            "portfolio": portfolio.payload if portfolio else None,
            "holdout": holdout.payload if holdout else None,
            "implementability": {
                "long_only": self._implementation_runs,
                "capacity": self._capacity_estimate,
            },
            "scorecard_evidence": to_jsonable(self._scorecard_evidence()),
            "artifact_count": self.store.count(),
            "holdout_released": self.guard.holdout_released,
        }

    def export_summary(self, path: Path) -> None:
        path.write_text(
            json.dumps(to_jsonable(self.summary()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
