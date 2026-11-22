from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..data import data_fingerprint
from .report import write_agent_report
from .roles import ROLE_SPECS
from .runtime import MiniAgentRuntime, MiniAgentUnavailable
from .schemas import (
    AgentResult,
    AgentTask,
    ArtifactKind,
    ArtifactStatus,
    RoleName,
    TaskStatus,
)
from .session import ResearchSession
from .utils import to_jsonable
from .workspace import RunWorkspace

LOGGER = logging.getLogger(__name__)


class ResearchOrchestrator:
    """State-machine orchestration over a shared artifact blackboard.

    The deterministic path is the reference implementation and always works
    offline. The Mini-Agent path uses separate role agents but keeps the same
    tools, artifacts, split guard, and critic gates.
    """

    def __init__(
        self,
        config: AppConfig,
        workspace: RunWorkspace,
        session: ResearchSession,
        mode: str = "deterministic",
        factor_limit: int | None = None,
        release_holdout: bool = False,
        fallback_on_agent_failure: bool = True,
    ):
        self.config = config
        self.workspace = workspace
        self.session = session
        self.mode = mode
        self.factor_limit = factor_limit
        self.release_holdout = release_holdout
        self.fallback_on_agent_failure = fallback_on_agent_failure
        self.runtime: MiniAgentRuntime | None = None
        self.report_path: Path | None = None
        self.summary_path: Path | None = None

    @classmethod
    def create(
        cls,
        config: AppConfig,
        mode: str = "deterministic",
        factor_limit: int | None = None,
        release_holdout: bool = False,
        run_id: str | None = None,
    ) -> ResearchOrchestrator:
        workspace = RunWorkspace(config.paths.runs_dir, run_id=run_id)
        fingerprint = data_fingerprint(config)
        workspace.create_manifest(
            project=config.project_name,
            mode=mode,
            raw_config=config.raw,
            data_fingerprint=fingerprint,
            repo_root=config.root,
        )
        session = ResearchSession.from_config(
            config,
            run_id=workspace.run_id,
            run_dir=workspace.run_dir,
        )
        session.metadata["data_fingerprint"] = fingerprint
        agentic = config.raw.get("agentic", {})
        if factor_limit is None:
            factor_limit = int(agentic.get("factor_limit", 20))
        fallback = bool(agentic.get("fallback_on_agent_failure", True))
        return cls(
            config=config,
            workspace=workspace,
            session=session,
            mode=mode,
            factor_limit=factor_limit,
            release_holdout=release_holdout,
            fallback_on_agent_failure=fallback,
        )

    def run(self) -> tuple[Path, Path]:
        self._initialize_runtime()
        try:
            self._write_plan()
            self._stage("hypothesis", self._hypothesis_stage)
            self._stage("feasibility", self._feasibility_stage)
            self._stage("engineering", self._engineering_stage)
            self._stage("evaluation", self._evaluation_stage)
            self._stage("critic", self._critic_stage)
            self._stage("portfolio", self._portfolio_stage)
            if self.release_holdout:
                self._stage("holdout", self._holdout_stage)
            self._stage("report", self._report_stage)
            summary = self.session.summary()
            summary["report_path"] = str(self.report_path) if self.report_path else None
            summary["summary_path"] = str(self.summary_path) if self.summary_path else None
            self.workspace.update_manifest(
                holdout_released=self.session.guard.holdout_released,
                artifact_count=self.session.store.count(),
                summary=to_jsonable(summary),
            )
            self.workspace.finish(status="completed", summary=summary)
            # Re-render once more so the report carries the final manifest status,
            # artifact count, and completed timestamp rather than a stale "running" view.
            self.report_path, self.summary_path = write_agent_report(
                self.session, self.workspace, self.workspace.load_manifest()
            )
        except Exception as exc:
            self.session.ledger.append(
                "run_failed",
                "orchestrator",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            self.workspace.finish(
                status="failed",
                summary={"error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        return self.report_path, self.summary_path  # type: ignore[return-value]

    def _initialize_runtime(self) -> None:
        if self.mode != "mini-agent":
            return
        if not MiniAgentRuntime.available():
            message = "Mini-Agent runtime is not importable; falling back to deterministic mode."
            if not self.fallback_on_agent_failure:
                raise MiniAgentUnavailable(message)
            LOGGER.warning(message)
            self.session.ledger.append("runtime_fallback", "orchestrator", {"reason": message})
            self.workspace.update_manifest(mode=f"{self.mode}_fallback")
            return
        try:
            self.runtime = MiniAgentRuntime(
                self.config,
                self.session,
                self.workspace.agent_logs_dir,
            )
            self.runtime._llm_settings()
        except MiniAgentUnavailable as exc:
            if not self.fallback_on_agent_failure:
                raise
            LOGGER.warning("Mini-Agent unavailable: %s", exc)
            self.session.ledger.append(
                "runtime_fallback", "orchestrator", {"reason": str(exc)}
            )
            self.runtime = None
            self.workspace.update_manifest(mode=f"{self.mode}_fallback")

    def _write_plan(self) -> None:
        plan = {
            "objective": "Convert economically motivated hypotheses into cost-aware, audited A-share factor portfolios.",
            "mode": self.mode,
            "factor_limit": self.factor_limit,
            "splits": {
                "train": [self.config.splits.train_start, self.config.splits.train_end],
                "validation": [
                    self.config.splits.validation_start,
                    self.config.splits.validation_end,
                ],
                "test": [self.config.splits.test_start, self.config.splits.test_end],
            },
            "selection_rule": "Select on train + validation only. Holdout is locked unless explicitly released.",
            "acceptance_rule": "Deterministic statistical gates plus leakage audits. LLM critic may veto but cannot accept failed factors.",
            "portfolio_rule": "Greedy low-correlation combination selected on validation only.",
            "limitations": [
                "Research backtest, not investment advice.",
                "Short-sale constraints and capacity are not fully modelled.",
                "Current public-data universe may retain survivorship bias.",
            ],
        }
        record = self.session.store.put(
            ArtifactKind.RUN_PLAN,
            plan,
            producer="orchestrator",
            status=ArtifactStatus.COMPLETED,
        )
        self.session.ledger.append(
            "run_plan_created", "orchestrator", plan, artifact_id=record.artifact_id
        )
        self.workspace.add_stage("plan", "completed", {"artifact_id": record.artifact_id})

    def _stage(self, name: str, fn: Callable[[], dict[str, Any] | None]) -> None:
        self.session.ledger.append("stage_started", "orchestrator", {"stage": name})
        self.workspace.add_stage(name, "running")
        detail = fn() or {}
        self.session.ledger.append(
            "stage_completed", "orchestrator", {"stage": name, **detail}
        )
        self.workspace.add_stage(name, "completed", detail)
        self.workspace.update_manifest(artifact_count=self.session.store.count())

    def _run_role(
        self,
        role: RoleName,
        objective: str,
        inputs: dict[str, Any],
        expected_kind: ArtifactKind,
    ) -> AgentResult | None:
        if self.runtime is None:
            return None
        before = len(self.session.store.list(expected_kind))
        spec = ROLE_SPECS[role]
        task = AgentTask(
            task_id=f"{self.workspace.run_id}_{role.value}",
            role=role,
            objective=objective,
            inputs=inputs,
            allowed_tools=spec.allowed_tools,
            max_steps=spec.max_steps,
            metadata={"run_id": self.workspace.run_id, "mode": self.mode},
        )
        try:
            result = self.runtime.run_task(task)
        except Exception as exc:  # pragma: no cover - API dependent
            if not self.fallback_on_agent_failure:
                raise
            LOGGER.warning("Role %s failed: %s", role.value, exc)
            self.session.ledger.append(
                "agent_fallback", "orchestrator", {"role": role.value, "error": str(exc)}
            )
            return None
        after = len(self.session.store.list(expected_kind))
        if result.status is not TaskStatus.COMPLETED or after <= before:
            if not self.fallback_on_agent_failure:
                raise RuntimeError(
                    f"Role {role.value} did not produce a {expected_kind.value} artifact"
                )
            self.session.ledger.append(
                "agent_fallback",
                "orchestrator",
                {
                    "role": role.value,
                    "reason": "no expected artifact",
                    "status": result.status.value,
                    "summary": result.summary[:500],
                },
            )
        return result

    def _hypothesis_stage(self) -> dict[str, Any]:
        self._run_role(
            RoleName.HYPOTHESIS,
            (
                f"Propose up to {self.factor_limit} diverse, economically motivated factor hypotheses. "
                "Call list_factor_templates, then submit_hypothesis for each selected spec_id."
            ),
            {
                "factor_limit": self.factor_limit,
                "catalog": self.session.catalog(limit=max(self.factor_limit, 12)),
            },
            ArtifactKind.HYPOTHESIS_CARD,
        )
        cards = self.session.hypothesis_cards()
        if not cards:
            for item in self.session.catalog(limit=self.factor_limit):
                self.session.create_hypothesis(
                    str(item["spec_id"]), producer="deterministic_hypothesis_agent"
                )
        cards = self.session.hypothesis_cards()
        return {"n_hypotheses": len(cards), "hypothesis_ids": [card.hypothesis_id for card in cards]}

    def _feasibility_stage(self) -> dict[str, Any]:
        cards = self.session.hypothesis_cards()
        if self.runtime is not None:
            self._run_role(
                RoleName.FEASIBILITY,
                "Check data fields and cross-sectional coverage for every hypothesis. Call check_feasibility for each hypothesis_id.",
                {"hypotheses": [card.model_dump(mode="json") for card in cards]},
                ArtifactKind.FEASIBILITY_REPORT,
            )
        checked = 0
        feasible = 0
        for card in cards:
            existing = self.session._find_artifact(ArtifactKind.FEASIBILITY_REPORT, card.spec_id)
            if existing is None:
                record = self.session.check_feasibility(
                    card.hypothesis_id, producer="deterministic_feasibility_agent"
                )
            else:
                record = existing
            checked += 1
            feasible += int(record.status is ArtifactStatus.ACCEPTED)
        return {"checked": checked, "feasible": feasible}

    def _engineering_stage(self) -> dict[str, Any]:
        cards = self.session.hypothesis_cards()
        if self.runtime is not None:
            self._run_role(
                RoleName.ENGINEER,
                "Implement every feasible hypothesis with the deterministic DSL. Call implement_factor for each hypothesis_id.",
                {"hypotheses": [card.model_dump(mode="json") for card in cards]},
                ArtifactKind.FACTOR_IMPLEMENTATION,
            )
        implemented = 0
        rejected = 0
        for card in cards:
            feasibility = self.session._find_artifact(
                ArtifactKind.FEASIBILITY_REPORT, card.spec_id
            )
            if feasibility is None or feasibility.status is not ArtifactStatus.ACCEPTED:
                continue
            existing = self.session._find_artifact(
                ArtifactKind.FACTOR_IMPLEMENTATION, card.spec_id
            )
            record = existing or self.session.implement_factor(
                card.hypothesis_id, producer="deterministic_factor_engineer"
            )
            if record.status is ArtifactStatus.ACCEPTED:
                implemented += 1
            else:
                rejected += 1
        return {"implemented": implemented, "rejected": rejected}

    def _evaluation_stage(self) -> dict[str, Any]:
        factor_ids = self.session._validated_candidates()
        for factor_id in factor_ids:
            self.session.evaluate(
                factor_id, "train", actor="deterministic_evaluator", purpose="train evaluation"
            )
            self.session.evaluate(
                factor_id,
                "validation",
                actor="deterministic_evaluator",
                purpose="validation evaluation",
            )
        return {"evaluated": len(factor_ids)}

    def _critic_stage(self) -> dict[str, Any]:
        record = self.session.screen_candidates(producer="deterministic_critic")
        review = record.payload
        if self.runtime is not None:
            self._run_role(
                RoleName.CRITIC,
                "Run deterministic screening, inspect the evidence packet, and veto accepted factors only with a concrete objection. Record notes with apply_critic_overrides.",
                {
                    "screening_artifact_id": record.artifact_id,
                    "review": review,
                },
                ArtifactKind.CRITIC_REVIEW,
            )
            latest = self.session.store.latest(ArtifactKind.CRITIC_REVIEW)
            if latest is not None:
                record = latest
                review = latest.payload
        return {
            "n_candidates": review.get("n_candidates", 0),
            "n_accepted": review.get("n_accepted", 0),
            "pbo": review.get("pbo", {}).get("pbo"),
        }

    def _portfolio_stage(self) -> dict[str, Any]:
        before = len(self.session.store.list(ArtifactKind.PORTFOLIO_DECISION))
        if self.runtime is not None:
            critic_record = self.session.store.latest(ArtifactKind.CRITIC_REVIEW)
            self._run_role(
                RoleName.PORTFOLIO,
                "Construct a low-correlation portfolio from critic-accepted factors. Call construct_portfolio.",
                {"critic_review": critic_record.payload if critic_record else {}},
                ArtifactKind.PORTFOLIO_DECISION,
            )
        after = len(self.session.store.list(ArtifactKind.PORTFOLIO_DECISION))
        if after <= before:
            self.session.construct_portfolio(producer="deterministic_portfolio_agent")
        record = self.session.store.latest(ArtifactKind.PORTFOLIO_DECISION)
        payload = record.payload if record else {}
        return {
            "selected_factors": payload.get("selected_factors", []),
            "fallback": payload.get("fallback"),
            "validation_net_sharpe": payload.get("validation_metrics", {}).get("net_sharpe"),
        }

    def _holdout_stage(self) -> dict[str, Any]:
        record = self.session.evaluate_holdout(
            released_by="orchestrator",
            reason="Explicit release after frozen train/validation selection.",
            producer="deterministic_holdout_evaluator",
        )
        self.workspace.update_manifest(holdout_released=True)
        metrics = record.payload.get("metrics", {})
        return {
            "net_sharpe": metrics.get("net_sharpe"),
            "net_ann_return": metrics.get("net_ann_return"),
            "max_drawdown": metrics.get("max_drawdown"),
        }

    def _report_stage(self) -> dict[str, Any]:
        manifest = self.workspace.load_manifest()
        self.report_path, self.summary_path = write_agent_report(
            self.session, self.workspace, manifest
        )
        summary = self.session.summary()
        record = self.session.store.put(
            ArtifactKind.RUN_SUMMARY,
            summary,
            producer="librarian",
            status=ArtifactStatus.COMPLETED,
            metadata={"report_path": str(self.report_path)},
        )
        return {
            "report_path": str(self.report_path),
            "summary_path": str(self.summary_path),
            "summary_artifact_id": record.artifact_id,
        }
