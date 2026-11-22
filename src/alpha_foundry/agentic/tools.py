from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

try:  # pragma: no cover - exercised only when the optional Mini-Agent extra is installed
    from mini_agent.tools.base import Tool as MiniAgentTool
    from mini_agent.tools.base import ToolResult as MiniAgentToolResult
except Exception:  # pragma: no cover - fallback is the normal lightweight install path
    class MiniAgentToolResult(BaseModel):
        success: bool
        content: str = ""
        error: str | None = None

    class MiniAgentTool:  # type: ignore[no-redef]
        @property
        def name(self) -> str:
            raise NotImplementedError

        @property
        def description(self) -> str:
            raise NotImplementedError

        @property
        def parameters(self) -> dict[str, Any]:
            raise NotImplementedError

        def to_schema(self) -> dict[str, Any]:
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": self.parameters,
            }

        def to_openai_schema(self) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }


from .schemas import ArtifactKind, ArtifactStatus, RoleName
from .session import ResearchSession
from .splits import SplitGuardError
from .utils import to_jsonable


def _object_schema(
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


class FunctionTool(MiniAgentTool):
    """Mini-Agent-compatible tool backed by one deterministic Python function."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any],
        session: ResearchSession,
        actor: str,
    ):
        self._name = name
        self._description = description
        self._parameters = parameters
        self._fn = fn
        self.session = session
        self.actor = actor

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> MiniAgentToolResult:
        started = time.perf_counter()
        try:
            value = self._fn(**kwargs)
            payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else to_jsonable(value)
            content = json.dumps(payload, ensure_ascii=False, indent=2)
            self.session.ledger.append(
                "tool_called",
                self.actor,
                {
                    "tool": self.name,
                    "arguments": to_jsonable(kwargs),
                    "success": True,
                    "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    "result_preview": content[:1000],
                },
            )
            return MiniAgentToolResult(success=True, content=content)
        except Exception as exc:
            self.session.ledger.append(
                "tool_failed",
                self.actor,
                {
                    "tool": self.name,
                    "arguments": to_jsonable(kwargs),
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
                },
            )
            return MiniAgentToolResult(
                success=False,
                content="",
                error=f"{type(exc).__name__}: {exc}",
            )


ROLE_TOOL_ALLOWLIST: dict[str, set[str]] = {
    RoleName.DIRECTOR.value: {
        "get_run_state",
        "list_factor_templates",
        "get_critic_packet",
    },
    RoleName.HYPOTHESIS.value: {
        "get_run_state",
        "list_factor_templates",
        "submit_hypothesis",
    },
    RoleName.FEASIBILITY.value: {
        "list_hypotheses",
        "check_feasibility",
        "get_run_state",
    },
    RoleName.ENGINEER.value: {
        "list_hypotheses",
        "check_feasibility",
        "implement_factor",
        "get_run_state",
    },
    RoleName.EVALUATOR.value: {
        "list_implemented_factors",
        "run_backtest",
        "get_run_state",
    },
    RoleName.CRITIC.value: {
        "get_critic_packet",
        "run_screening",
        "run_audit",
        "apply_critic_overrides",
        "get_run_state",
    },
    RoleName.PORTFOLIO.value: {
        "get_critic_packet",
        "construct_portfolio",
        "get_run_state",
    },
    RoleName.LIBRARIAN.value: {
        "get_run_state",
        "get_critic_packet",
    },
}


def build_tools(session: ResearchSession, role: RoleName | str) -> list[FunctionTool]:
    role_value = role.value if isinstance(role, RoleName) else str(role)
    actor = f"{role_value}_agent"

    def list_factor_templates(limit: int = 20, category: str | None = None) -> list[dict[str, Any]]:
        return session.catalog(limit=limit, category=category)

    def submit_hypothesis(spec_id: str, rationale: str | None = None) -> dict[str, Any]:
        record = session.create_hypothesis(spec_id, rationale=rationale, producer=actor)
        return {"artifact_id": record.artifact_id, "payload": record.payload, "status": record.status.value}

    def list_hypotheses() -> list[dict[str, Any]]:
        return [
            {
                "artifact_id": record.artifact_id,
                "status": record.status.value,
                **record.payload,
            }
            for record in session.store.list(ArtifactKind.HYPOTHESIS_CARD)
        ]

    def check_feasibility(hypothesis_id: str) -> dict[str, Any]:
        record = session.check_feasibility(hypothesis_id, producer=actor)
        return {"artifact_id": record.artifact_id, "status": record.status.value, **record.payload}

    def implement_factor(hypothesis_id: str) -> dict[str, Any]:
        hypothesis = session.resolve_hypothesis(hypothesis_id)
        spec_id = str(hypothesis.payload["spec_id"])
        feasibility = session._find_artifact(ArtifactKind.FEASIBILITY_REPORT, spec_id)
        if feasibility is None:
            session.check_feasibility(hypothesis_id, producer=actor)
        record = session.implement_factor(hypothesis_id, producer=actor)
        return {"artifact_id": record.artifact_id, "status": record.status.value, **record.payload}

    def list_implemented_factors() -> list[dict[str, Any]]:
        return [
            {
                "artifact_id": record.artifact_id,
                "status": record.status.value,
                **record.payload,
            }
            for record in session.store.list(
                ArtifactKind.FACTOR_IMPLEMENTATION, ArtifactStatus.ACCEPTED
            )
            if record.payload.get("status") == "implemented"
        ]

    def run_backtest(factor_id: str, split: str = "train") -> dict[str, Any]:
        if split == "test":
            raise SplitGuardError("The holdout split is not available to research agents.")
        record = session.evaluate(factor_id, split, actor=actor, purpose=f"agent-requested {split} backtest")
        return {"artifact_id": record.artifact_id, **record.payload}

    def get_critic_packet() -> dict[str, Any]:
        factors: list[dict[str, Any]] = []
        for factor_id in session._validated_candidates():
            row: dict[str, Any] = {"factor_id": factor_id}
            for split in ("train", "validation"):
                try:
                    row[split] = to_jsonable(session.evaluation(factor_id, split).metrics)
                except Exception as exc:
                    row[split] = {"error": str(exc)}
            factors.append(row)
        return {
            "factors": factors,
            "critic_config": to_jsonable(session.config.critic.__dict__),
            "note": "Deterministic gates are authoritative. The critic may veto accepted factors but cannot accept a failed one.",
        }

    def run_screening() -> dict[str, Any]:
        record = session.screen_candidates(producer=actor)
        return {"artifact_id": record.artifact_id, "status": record.status.value, **record.payload}

    def run_audit(factor_id: str) -> dict[str, Any]:
        record = session.audit(factor_id, producer=actor)
        return {"artifact_id": record.artifact_id, "status": record.status.value, **record.payload}

    def apply_critic_overrides(reject_factor_ids: list[str], notes: str) -> dict[str, Any]:
        record = session.apply_critic_overrides(reject_factor_ids, notes, producer=actor)
        return {"artifact_id": record.artifact_id, "status": record.status.value, **record.payload}

    def construct_portfolio(factor_ids: list[str] | None = None) -> dict[str, Any]:
        record = session.construct_portfolio(factor_ids=factor_ids, producer=actor)
        return {"artifact_id": record.artifact_id, "status": record.status.value, **record.payload}

    def get_run_state() -> dict[str, Any]:
        return session.summary()

    definitions: dict[str, tuple[str, dict[str, Any], Callable[..., Any]]] = {
        "get_run_state": (
            "Return a compact summary of this research run: hypothesis counts, critic decision, portfolio, and holdout status.",
            _object_schema(),
            get_run_state,
        ),
        "list_factor_templates": (
            "List deterministic factor templates available for hypothesis proposals. Use this instead of inventing arbitrary code.",
            _object_schema(
                {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                    "category": {"type": "string"},
                }
            ),
            list_factor_templates,
        ),
        "submit_hypothesis": (
            "Turn one factor template into an explicit, auditable hypothesis card. Duplicate submissions are idempotent.",
            _object_schema(
                {
                    "spec_id": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                required=["spec_id"],
            ),
            submit_hypothesis,
        ),
        "list_hypotheses": (
            "List hypothesis cards and their current status.",
            _object_schema(),
            list_hypotheses,
        ),
        "check_feasibility": (
            "Check data fields, cross-sectional coverage, and implementability for a hypothesis.",
            _object_schema({"hypothesis_id": {"type": "string"}}, required=["hypothesis_id"]),
            check_feasibility,
        ),
        "implement_factor": (
            "Implement a feasible hypothesis as a deterministic DSL factor and record diagnostics.",
            _object_schema({"hypothesis_id": {"type": "string"}}, required=["hypothesis_id"]),
            implement_factor,
        ),
        "list_implemented_factors": (
            "List factors that were successfully implemented and are ready for evaluation.",
            _object_schema(),
            list_implemented_factors,
        ),
        "run_backtest": (
            "Run a cost-aware backtest on train or validation only. Test access is forbidden.",
            _object_schema(
                {
                    "factor_id": {"type": "string"},
                    "split": {"type": "string", "enum": ["train", "validation"]},
                },
                required=["factor_id"],
            ),
            run_backtest,
        ),
        "get_critic_packet": (
            "Return train and validation evidence for implemented factors so the critic can form objections.",
            _object_schema(),
            get_critic_packet,
        ),
        "run_screening": (
            "Run the deterministic multiple-testing, train/validation, and audit gates. This is authoritative.",
            _object_schema(),
            run_screening,
        ),
        "run_audit": (
            "Run future-data perturbation and label-shuffle audits for one factor.",
            _object_schema({"factor_id": {"type": "string"}}, required=["factor_id"]),
            run_audit,
        ),
        "apply_critic_overrides": (
            "Add vetoes to the deterministic critic review. May reject accepted factors but may not accept rejected ones.",
            _object_schema(
                {
                    "reject_factor_ids": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                required=["reject_factor_ids", "notes"],
            ),
            apply_critic_overrides,
        ),
        "construct_portfolio": (
            "Construct a low-correlation portfolio from critic-accepted factors and evaluate it on validation.",
            _object_schema(
                {
                    "factor_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                }
            ),
            construct_portfolio,
        ),
    }
    allowed = ROLE_TOOL_ALLOWLIST.get(role_value, set())
    return [
        FunctionTool(name, description, parameters, fn, session, actor)
        for name, (description, parameters, fn) in definitions.items()
        if name in allowed
    ]
