from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactKind(str, Enum):
    RUN_PLAN = "run_plan"
    HYPOTHESIS_CARD = "hypothesis_card"
    FEASIBILITY_REPORT = "feasibility_report"
    FACTOR_IMPLEMENTATION = "factor_implementation"
    BACKTEST_REPORT = "backtest_report"
    AUDIT_REPORT = "audit_report"
    CRITIC_REVIEW = "critic_review"
    PORTFOLIO_DECISION = "portfolio_decision"
    HOLDOUT_REPORT = "holdout_report"
    AGENT_TRANSCRIPT = "agent_transcript"
    RUN_SUMMARY = "run_summary"


class ArtifactStatus(str, Enum):
    PROPOSED = "proposed"
    COMPLETED = "completed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class SplitName(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class RoleName(str, Enum):
    DIRECTOR = "director"
    HYPOTHESIS = "hypothesis"
    FEASIBILITY = "feasibility"
    ENGINEER = "engineer"
    EVALUATOR = "evaluator"
    CRITIC = "critic"
    PORTFOLIO = "portfolio"
    LIBRARIAN = "librarian"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class HypothesisCard(BaseModel):
    hypothesis_id: str
    spec_id: str
    name: str
    family: str
    category: str
    economic_rationale: str
    testable_prediction: str
    formula: str
    direction: float
    window: int
    required_fields: list[str] = Field(default_factory=list)
    source: str = "economic_template_library"
    dedupe_key: str


class FeasibilityReport(BaseModel):
    hypothesis_id: str
    spec_id: str
    feasible: bool
    finite_ratio: float
    median_names: float
    missing_fields: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class FactorImplementation(BaseModel):
    hypothesis_id: str
    spec_id: str
    implementation: Literal["dsl_template"] = "dsl_template"
    status: Literal["implemented", "rejected"] = "implemented"
    finite_ratio: float
    median_names: float
    diagnostic: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class BacktestReport(BaseModel):
    factor_id: str
    split: SplitName
    metrics: dict[str, Any]
    cost_bps_per_side: float
    protocol: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AuditReport(BaseModel):
    factor_id: str
    no_lookahead: dict[str, Any]
    label_shuffle: dict[str, Any]
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class CriticDecision(BaseModel):
    factor_id: str
    accepted: bool
    status: str
    checks: dict[str, bool] = Field(default_factory=dict)
    q_value: float | None = None
    dsr: float | None = None
    reasons: list[str] = Field(default_factory=list)


class CriticReview(BaseModel):
    decisions: list[CriticDecision]
    pbo: dict[str, Any] = Field(default_factory=dict)
    n_candidates: int
    n_accepted: int
    reviewer_notes: str = ""


class PortfolioDecision(BaseModel):
    portfolio_id: str
    selected_factors: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    fallback: bool = False
    selection_path: list[dict[str, Any]] = Field(default_factory=list)
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    test_metrics: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class HoldoutReport(BaseModel):
    portfolio_id: str
    metrics: dict[str, Any]
    released_by: str
    reason: str
    limitations: list[str] = Field(default_factory=list)


class AgentTask(BaseModel):
    task_id: str
    role: RoleName
    objective: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    max_steps: int = 20
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    task_id: str
    role: RoleName
    status: TaskStatus
    summary: str = ""
    artifact_ids: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    tokens: int = 0
    error: str | None = None


class ArtifactRecord(BaseModel):
    artifact_id: str
    run_id: str
    kind: ArtifactKind
    status: ArtifactStatus
    producer: str
    parents: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION
    split: SplitName | None = None
    content_hash: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    run_id: str
    project: str
    mode: str
    status: str = "running"
    started_at: str = Field(default_factory=utc_now)
    finished_at: str | None = None
    config_hash: str
    data_fingerprint: str
    git_commit: str | None = None
    holdout_released: bool = False
    stages: list[dict[str, Any]] = Field(default_factory=list)
    artifact_count: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)
