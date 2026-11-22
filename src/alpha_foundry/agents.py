from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .backtest import BacktestResult, evaluate_factor, forward_returns
from .config import AppConfig
from .factors import FactorSpec, generate_factor_specs
from .memory import ExperimentMemory
from .portfolio import combine_signals, signal_from_factor
from .stats import (
    audit_no_lookahead,
    bh_fdr,
    deflated_sharpe,
    label_shuffle_test,
    probability_of_backtest_overfitting,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class FactorRun:
    spec: FactorSpec
    factor: pd.DataFrame
    train: BacktestResult
    validation: BacktestResult
    test: BacktestResult
    status: str = "candidate"
    q_value: float = np.nan
    dsr: float = np.nan
    audit: dict[str, Any] = field(default_factory=dict)
    shuffle: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""

    @property
    def validation_metrics(self) -> dict:
        return self.validation.metrics

    @property
    def test_metrics(self) -> dict:
        return self.test.metrics


@dataclass
class AgentContext:
    config: AppConfig
    memory: ExperimentMemory
    run_id: str
    fields: dict[str, pd.DataFrame]
    metadata: dict[str, Any]
    fwd_ret: pd.DataFrame | None = None
    specs: list[FactorSpec] = field(default_factory=list)
    factor_values: dict[str, pd.DataFrame] = field(default_factory=dict)
    results: list[FactorRun] = field(default_factory=list)
    accepted_ids: list[str] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)
    portfolio_metrics: dict[str, Any] = field(default_factory=dict)
    critic_summary: dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    name = "base"

    def log(self, context: AgentContext, action: str, payload: dict | None = None) -> None:
        LOGGER.info("[%s] %s %s", self.name, action, payload or {})
        context.memory.log_event(context.run_id, self.name, action, payload)

    def run(self, context: AgentContext) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class DataStewardAgent(BaseAgent):
    name = "data_steward"

    def run(self, context: AgentContext) -> None:
        meta = context.metadata
        self.log(
            context,
            "validate_panel",
            {
                "n_dates": meta.get("n_dates"),
                "n_symbols": meta.get("n_selected_symbols"),
                "start": meta.get("start_date"),
                "end": meta.get("end_date"),
            },
        )
        required = {"open", "high", "low", "close", "volume", "ret", "tradable"}
        missing = required.difference(context.fields)
        if missing:
            raise RuntimeError(f"Panel missing required fields: {sorted(missing)}")


class PlannerAgent(BaseAgent):
    name = "planner"

    def run(self, context: AgentContext) -> None:
        plan = {
            "objective": "Generate, implement and validate cross-sectional A-share factor candidates.",
            "stages": [
                "hypothesis_generation",
                "factor_implementation",
                "backtest",
                "adversarial_validation",
                "portfolio_construction",
            ],
            "train": [context.config.splits.train_start, context.config.splits.train_end],
            "validation": [
                context.config.splits.validation_start,
                context.config.splits.validation_end,
            ],
            "test": [context.config.splits.test_start, context.config.splits.test_end],
            "selection_rule": "Select on train + validation only; test is a locked holdout.",
        }
        self.log(context, "create_research_plan", plan)


class HypothesisAgent(BaseAgent):
    name = "hypothesis"

    def run(self, context: AgentContext) -> None:
        specs = generate_factor_specs()
        limit = context.config.raw.get("runtime", {}).get("factor_limit")
        if limit:
            specs = specs[: int(limit)]
        context.specs = specs
        families = sorted({spec.family for spec in specs})
        categories = sorted({spec.category for spec in specs})
        self.log(
            context,
            "generate_hypotheses",
            {
                "n_specs": len(specs),
                "n_families": len(families),
                "families": families,
                "categories": categories,
                "mode": "deterministic_economic_template_library",
                "llm_enabled": bool(context.config.llm.get("enabled", False)),
            },
        )


class FactorEngineerAgent(BaseAgent):
    name = "factor_engineer"

    def run(self, context: AgentContext) -> None:
        from .factors import compute_factor

        requested = len(context.specs)
        valid_specs: list[FactorSpec] = []
        for idx, spec in enumerate(context.specs, start=1):
            factor = compute_factor(spec, context.fields)
            finite_ratio = float(np.isfinite(factor.to_numpy()).mean())
            cross_sectional = float(factor.notna().sum(axis=1).median())
            if finite_ratio < 0.20 or cross_sectional < context.config.backtest.min_names:
                self.log(
                    context,
                    "reject_unimplementable",
                    {
                        "factor_id": spec.spec_id,
                        "finite_ratio": finite_ratio,
                        "median_names": cross_sectional,
                    },
                )
                continue
            context.factor_values[spec.spec_id] = factor
            valid_specs.append(spec)
            if idx % 20 == 0 or idx == len(context.specs):
                LOGGER.info("Implemented %d/%d factors", idx, len(context.specs))
        context.specs = valid_specs
        self.log(
            context,
            "implement_factors",
            {"implemented": len(valid_specs), "requested": requested},
        )


class BacktestAgent(BaseAgent):
    name = "backtest"

    def run(self, context: AgentContext) -> None:
        close = context.fields["close"]
        horizon = context.config.backtest.horizon
        context.fwd_ret = forward_returns(close, horizon)
        tradable = context.fields.get("tradable")
        splits = context.config.splits
        for idx, spec in enumerate(context.specs, start=1):
            factor = context.factor_values[spec.spec_id]
            train = evaluate_factor(
                factor,
                context.fwd_ret,
                context.config.backtest,
                start=splits.train_start,
                end=splits.train_end,
                tradable=tradable,
            )
            validation = evaluate_factor(
                factor,
                context.fwd_ret,
                context.config.backtest,
                start=splits.validation_start,
                end=splits.validation_end,
                tradable=tradable,
            )
            test = evaluate_factor(
                factor,
                context.fwd_ret,
                context.config.backtest,
                start=splits.test_start,
                end=splits.test_end,
                tradable=tradable,
            )
            context.results.append(FactorRun(spec=spec, factor=factor, train=train, validation=validation, test=test))
            if idx % 10 == 0 or idx == len(context.specs):
                LOGGER.info("Backtested %d/%d factors", idx, len(context.specs))
        self.log(
            context,
            "run_backtests",
            {
                "n_factors": len(context.results),
                "horizon": horizon,
                "cost_bps_per_side": context.config.backtest.cost_bps_per_side,
            },
        )


class CriticAgent(BaseAgent):
    name = "critic"

    def _pvalues(self, results: list[FactorRun]) -> np.ndarray:
        pvalues = []
        for result in results:
            metrics = result.validation.metrics
            mean_ic = metrics.get("mean_ic", np.nan)
            ic_std = metrics.get("ic_std", np.nan)
            n = metrics.get("n_periods", 0)
            if not np.isfinite(mean_ic) or not np.isfinite(ic_std) or ic_std <= 0 or n < 3:
                pvalues.append(np.nan)
                continue
            tstat = mean_ic / ic_std * np.sqrt(n)
            pvalues.append(float(2.0 * student_t.sf(abs(tstat), df=n - 1)))
        return np.asarray(pvalues, dtype=float)

    def _correlation_filter(
        self, candidates: list[FactorRun], max_corr: float, max_accept: int
    ) -> tuple[list[FactorRun], list[dict[str, Any]]]:
        accepted: list[FactorRun] = []
        notes: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_returns = candidate.validation.returns
            max_abs_corr = 0.0
            if accepted:
                corrs = []
                for other in accepted:
                    pair = pd.concat(
                        [candidate_returns.rename("a"), other.validation.returns.rename("b")],
                        axis=1,
                    ).dropna()
                    corr = float(pair["a"].corr(pair["b"])) if len(pair) > 5 else 0.0
                    corrs.append(abs(corr))
                max_abs_corr = max(corrs) if corrs else 0.0
            if max_abs_corr > max_corr:
                candidate.status = "rejected_correlated"
                candidate.rejection_reason = f"max_abs_corr={max_abs_corr:.3f}"
                notes.append({"factor_id": candidate.spec.spec_id, "reason": candidate.rejection_reason})
                continue
            accepted.append(candidate)
            if len(accepted) >= max_accept:
                break
        return accepted, notes

    def run(self, context: AgentContext) -> None:
        results = context.results
        pvalues = self._pvalues(results)
        qvalues = bh_fdr(pvalues)
        trial_sharpes = []
        for result in results:
            returns = result.validation.returns.dropna()
            if len(returns) > 2 and returns.std(ddof=1) > 0:
                trial_sharpes.append(float(returns.mean() / returns.std(ddof=1)))
            else:
                trial_sharpes.append(0.0)

        validation_returns = pd.concat(
            {result.spec.spec_id: result.validation.returns for result in results}, axis=1
        )
        pbo = probability_of_backtest_overfitting(validation_returns, seed=context.config.seed)

        for result, qvalue in zip(results, qvalues, strict=True):
            result.q_value = float(qvalue) if np.isfinite(qvalue) else np.nan
            dsr = deflated_sharpe(result.validation.returns, len(results), trial_sharpes)
            result.dsr = float(dsr["dsr"]) if np.isfinite(dsr["dsr"]) else np.nan
            metrics = result.validation.metrics
            critic = context.config.critic
            checks = {
                "enough_periods": metrics.get("n_periods", 0) >= 12,
                "coverage": metrics.get("coverage", 0.0) >= critic.min_coverage,
                "ic": metrics.get("mean_ic", -1.0) >= critic.min_validation_ic,
                "icir": metrics.get("icir", -1.0) >= critic.min_validation_icir,
                "net_sharpe": metrics.get("net_sharpe", -1.0) >= critic.min_validation_net_sharpe,
                "turnover": metrics.get("turnover_annual", 1e9) <= critic.max_turnover_annual,
                "fdr": result.q_value <= critic.max_fdr_q if np.isfinite(result.q_value) else False,
                "subperiod": metrics.get("subperiod_ic_sign_ratio", 0.0)
                >= critic.min_subperiod_sign_ratio,
                "train_net_sharpe": result.train.metrics.get("net_sharpe", -1.0) > 0,
                "train_ic": result.train.metrics.get("mean_ic", -1.0) > 0,
                "train_icir": result.train.metrics.get("icir", -1.0) > 0,
            }
            result.audit["screening_checks"] = checks
            if all(checks.values()):
                result.status = "candidate"
            else:
                result.status = "rejected_screening"
                failed = [name for name, passed in checks.items() if not passed]
                result.rejection_reason = "failed:" + ",".join(failed)

        candidates = [r for r in results if r.status == "candidate"]
        candidates.sort(
            key=lambda r: r.validation.metrics.get("net_sharpe", -np.inf), reverse=True
        )
        accepted, correlation_notes = self._correlation_filter(
            candidates,
            max_corr=0.95,
            max_accept=len(candidates),
        )

        validation_dates = context.fields["close"].index
        validation_end = pd.Timestamp(context.config.splits.validation_end) - pd.Timedelta(days=30)
        possible_cutoffs = validation_dates[validation_dates <= validation_end]
        cutoff = possible_cutoffs[-1] if len(possible_cutoffs) else validation_end

        final_accepted: list[FactorRun] = []
        for result in accepted:
            audit = audit_no_lookahead(result.spec, context.fields, cutoff=cutoff, seed=context.config.seed)
            result.audit.update(audit)
            if not audit["passed"]:
                result.status = "rejected_leakage"
                result.rejection_reason = "future-data perturbation changed factor"
                continue
            shuffle = label_shuffle_test(
                result.spec,
                context.fields,
                context.fwd_ret,
                start=context.config.splits.validation_start,
                end=context.config.splits.validation_end,
                config=context.config.backtest,
                seed=context.config.seed,
            )
            result.shuffle = shuffle
            if not shuffle["passed"]:
                result.status = "rejected_label_shuffle"
                result.rejection_reason = "factor did not beat shuffled labels"
                continue
            result.status = "accepted"
            final_accepted.append(result)

        for result in results:
            if result.status == "candidate" and result not in final_accepted:
                result.status = "rejected_audit"
                result.rejection_reason = "not passed after correlation/audit review"

        context.accepted_ids = [result.spec.spec_id for result in final_accepted]
        context.critic_summary = {
            "n_candidates": len(results),
            "n_screened_pass": len(candidates),
            "n_accepted": len(final_accepted),
            "pbo": pbo.get("pbo", np.nan),
            "pbo_trials": pbo.get("n_trials", 0),
            "pbo_median_oos_rank": pbo.get("median_oos_rank", np.nan),
            "correlation_notes": correlation_notes,
            "audit_cutoff": str(pd.Timestamp(cutoff).date()),
        }
        self.log(context, "adversarial_audit", context.critic_summary)


class PortfolioAgent(BaseAgent):
    name = "portfolio"

    def run(self, context: AgentContext) -> None:
        accepted = [r for r in context.results if r.status == "accepted"]
        fallback = False
        if not accepted:
            accepted = sorted(
                [r for r in context.results if r.validation.metrics.get("mean_ic", -1.0) > 0],
                key=lambda r: r.validation.metrics.get("icir", -np.inf),
                reverse=True,
            )[: context.config.portfolio.max_factors]
            fallback = True

        if not accepted:
            self.log(context, "portfolio_skipped", {"reason": "no candidates"})
            return

        tradable = context.fields.get("tradable")
        signals = {
            result.spec.spec_id: signal_from_factor(result.factor) for result in accepted
        }

        def validation_sharpe(factor_ids: list[str]) -> float:
            signal = combine_signals({factor_id: signals[factor_id] for factor_id in factor_ids})
            result = evaluate_factor(
                signal,
                context.fwd_ret,
                context.config.backtest,
                start=context.config.splits.validation_start,
                end=context.config.splits.validation_end,
                tradable=tradable,
            )
            score = result.metrics.get("net_sharpe", np.nan)
            return float(score) if np.isfinite(score) else -np.inf

        singles = sorted(
            accepted,
            key=lambda result: result.validation.metrics.get("net_sharpe", -np.inf),
            reverse=True,
        )
        selected = [singles[0]]
        current_score = validation_sharpe([singles[0].spec.spec_id])
        selection_path = [
            {
                "factor_id": singles[0].spec.spec_id,
                "validation_net_sharpe": current_score,
            }
        ]
        min_improvement = 0.01
        while len(selected) < context.config.portfolio.max_factors:
            best_candidate: FactorRun | None = None
            best_score = current_score
            selected_ids = {result.spec.spec_id for result in selected}
            for candidate in accepted:
                if candidate.spec.spec_id in selected_ids:
                    continue
                corrs = []
                for other in selected:
                    pair = pd.concat(
                        [candidate.validation.returns.rename("a"), other.validation.returns.rename("b")],
                        axis=1,
                    ).dropna()
                    corrs.append(abs(float(pair["a"].corr(pair["b"]))) if len(pair) > 5 else 0.0)
                if corrs and max(corrs) > context.config.portfolio.max_pairwise_corr:
                    continue
                combo_ids = [result.spec.spec_id for result in selected] + [candidate.spec.spec_id]
                score = validation_sharpe(combo_ids)
                if score > best_score + min_improvement:
                    best_score = score
                    best_candidate = candidate
            if best_candidate is None:
                break
            selected.append(best_candidate)
            current_score = best_score
            selection_path.append(
                {
                    "factor_id": best_candidate.spec.spec_id,
                    "validation_net_sharpe": current_score,
                }
            )

        # Equal-risk weighting is deliberately simple and validated out of sample.
        weight = 1.0 / len(selected)
        combined = combine_signals(
            {result.spec.spec_id: signal_from_factor(result.factor) for result in selected}
        )
        validation = evaluate_factor(
            combined,
            context.fwd_ret,
            context.config.backtest,
            start=context.config.splits.validation_start,
            end=context.config.splits.validation_end,
            tradable=tradable,
        )
        test = evaluate_factor(
            combined,
            context.fwd_ret,
            context.config.backtest,
            start=context.config.splits.test_start,
            end=context.config.splits.test_end,
            tradable=tradable,
        )
        context.selected_ids = [result.spec.spec_id for result in selected]
        selected_test_sharpe = float(test.metrics.get("net_sharpe", np.nan))
        pool = sorted(
            [r for r in context.results if r.validation.metrics.get("mean_ic", -1.0) > 0],
            key=lambda r: r.validation.metrics.get("icir", -np.inf),
            reverse=True,
        )[:20]
        random_sharpes: list[float] = []
        if len(pool) >= len(selected):
            rng = np.random.default_rng(context.config.seed)
            pool_signals = {
                result.spec.spec_id: signal_from_factor(result.factor) for result in pool
            }
            pool_ids = list(pool_signals)
            for _ in range(50):
                sample_ids = rng.choice(pool_ids, size=len(selected), replace=False).tolist()
                sampled_signal = combine_signals(
                    {factor_id: pool_signals[factor_id] for factor_id in sample_ids}
                )
                sampled_result = evaluate_factor(
                    sampled_signal,
                    context.fwd_ret,
                    context.config.backtest,
                    start=context.config.splits.test_start,
                    end=context.config.splits.test_end,
                    tradable=tradable,
                )
                random_sharpes.append(float(sampled_result.metrics.get("net_sharpe", np.nan)))
        random_sharpes_array = np.asarray([x for x in random_sharpes if np.isfinite(x)])
        random_percentile = (
            float(np.mean(random_sharpes_array < selected_test_sharpe))
            if len(random_sharpes_array) and np.isfinite(selected_test_sharpe)
            else np.nan
        )
        no_critic_pool = sorted(
            context.results,
            key=lambda r: r.validation.metrics.get("icir", -np.inf),
            reverse=True,
        )[: context.config.portfolio.max_factors]
        no_critic_signal = combine_signals(
            {
                result.spec.spec_id: signal_from_factor(result.factor)
                for result in no_critic_pool
            }
        )
        no_critic_test = evaluate_factor(
            no_critic_signal,
            context.fwd_ret,
            context.config.backtest,
            start=context.config.splits.test_start,
            end=context.config.splits.test_end,
            tradable=tradable,
        )
        context.portfolio_metrics = {
            "portfolio_id": "multi_agent_selected",
            "fallback": fallback,
            "selected_factors": context.selected_ids,
            "weights": {result.spec.spec_id: weight for result in selected},
            "validation": validation.metrics,
            "test": test.metrics,
            "validation_returns": validation.returns,
            "test_returns": test.returns,
            "combined_factor": combined,
            "selection_path": selection_path,
            "random_baseline": {
                "n_portfolios": int(len(random_sharpes_array)),
                "mean_test_net_sharpe": float(random_sharpes_array.mean())
                if len(random_sharpes_array)
                else np.nan,
                "median_test_net_sharpe": float(np.median(random_sharpes_array))
                if len(random_sharpes_array)
                else np.nan,
                "selected_percentile": random_percentile,
            },
            "ablation_no_critic": {
                "selected_factors": [result.spec.spec_id for result in no_critic_pool],
                "test": no_critic_test.metrics,
            },
        }
        self.log(
            context,
            "construct_portfolio",
            {
                "portfolio_id": "multi_agent_selected",
                "fallback": fallback,
                "selected_factors": context.selected_ids,
                "validation_net_sharpe": validation.metrics.get("net_sharpe"),
                "test_net_sharpe": test.metrics.get("net_sharpe"),
                "random_baseline": context.portfolio_metrics.get("random_baseline"),
            },
        )


class ExperimentLibrarianAgent(BaseAgent):
    name = "librarian"

    def run(self, context: AgentContext) -> None:
        records = []
        for result in context.results:
            record = {
                "factor_id": result.spec.spec_id,
                "name": result.spec.name,
                "family": result.spec.family,
                "category": result.spec.category,
                "hypothesis": result.spec.hypothesis,
                "formula": result.spec.formula,
                "direction": result.spec.direction,
                "window": result.spec.window,
                "status": result.status,
                "rejection_reason": result.rejection_reason,
                "q_value": result.q_value,
                "dsr": result.dsr,
                "audit_passed": result.audit.get("passed", None),
                "audit_max_abs_diff": result.audit.get("max_abs_diff", np.nan),
                "shuffle_passed": result.shuffle.get("passed", None),
                "shuffle_p_value": result.shuffle.get("p_value", np.nan),
            }
            for split_name, split_result in (
                ("train", result.train),
                ("validation", result.validation),
                ("test", result.test),
            ):
                for key, value in split_result.metrics.items():
                    record[f"{split_name}_{key}"] = value
            records.append(record)
        frame = pd.DataFrame(records)
        frame.to_csv(context.config.paths.experiments_dir / "factor_results.csv", index=False)

        for result in context.results:
            context.memory.save_factor(
                context.run_id,
                result.spec,
                result.status,
                {"q_value": result.q_value, "dsr": result.dsr, **result.validation.metrics},
                result.train.metrics,
                result.validation.metrics,
                result.test.metrics,
            )

        if context.portfolio_metrics:
            context.memory.save_portfolio(
                context.run_id,
                "multi_agent_selected",
                context.selected_ids,
                {
                    "fallback": context.portfolio_metrics.get("fallback"),
                    "validation": context.portfolio_metrics.get("validation"),
                    "test": context.portfolio_metrics.get("test"),
                    "ablation_no_critic": context.portfolio_metrics.get("ablation_no_critic"),
                },
            )
            portfolio_frame = pd.DataFrame(
                [
                    {
                        "portfolio_id": "multi_agent_selected",
                        "fallback": context.portfolio_metrics.get("fallback"),
                        "selected_factors": ",".join(context.selected_ids),
                        "validation_net_sharpe": context.portfolio_metrics["validation"].get("net_sharpe"),
                        "validation_mean_ic": context.portfolio_metrics["validation"].get("mean_ic"),
                        "test_net_sharpe": context.portfolio_metrics["test"].get("net_sharpe"),
                        "test_mean_ic": context.portfolio_metrics["test"].get("mean_ic"),
                        "test_ann_return": context.portfolio_metrics["test"].get("net_ann_return"),
                        "test_max_drawdown": context.portfolio_metrics["test"].get("max_drawdown"),
                        "test_turnover_annual": context.portfolio_metrics["test"].get("turnover_annual"),
                        "no_critic_test_net_sharpe": context.portfolio_metrics.get(
                            "ablation_no_critic", {}
                        ).get("test", {}).get("net_sharpe"),
                    }
                ]
            )
            portfolio_frame.to_csv(
                context.config.paths.experiments_dir / "portfolio_results.csv", index=False
            )

        if context.selected_ids:
            sample = {
                factor_id: context.factor_values[factor_id].iloc[-250:]
                for factor_id in context.selected_ids
                if factor_id in context.factor_values
            }
            if sample:
                pd.concat(sample, names=["factor_id", "date"]).to_parquet(
                    context.config.paths.experiments_dir / "selected_factor_values.parquet"
                )

        self.log(
            context,
            "persist_experiments",
            {
                "factor_results": str(context.config.paths.experiments_dir / "factor_results.csv"),
                "n_records": len(frame),
                "selected_factors": context.selected_ids,
            },
        )


AGENT_SEQUENCE = [
    DataStewardAgent,
    PlannerAgent,
    HypothesisAgent,
    FactorEngineerAgent,
    BacktestAgent,
    CriticAgent,
    PortfolioAgent,
    ExperimentLibrarianAgent,
]
