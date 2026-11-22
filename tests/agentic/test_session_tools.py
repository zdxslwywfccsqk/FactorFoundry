from __future__ import annotations

import asyncio

import pytest
from conftest import make_fields

from alpha_foundry.agentic.schemas import ArtifactStatus, RoleName
from alpha_foundry.agentic.session import ResearchSession
from alpha_foundry.agentic.splits import SplitGuardError
from alpha_foundry.agentic.tools import build_tools
from alpha_foundry.agentic.workspace import RunWorkspace


def _session(app_config):
    workspace = RunWorkspace(app_config.paths.runs_dir, run_id="run_session")
    fields = make_fields(n_days=320, n_symbols=40)
    return ResearchSession(
        app_config,
        fields,
        {"data_fingerprint": "synthetic", "n_selected_symbols": 40},
        run_id=workspace.run_id,
        run_dir=workspace.run_dir,
    )


def test_hypothesis_to_validation_pipeline(app_config) -> None:
    session = _session(app_config)
    catalog = session.catalog(limit=3)
    assert len(catalog) == 3
    hypothesis = session.create_hypothesis(catalog[0]["spec_id"])
    feasibility = session.check_feasibility(hypothesis.artifact_id)
    assert feasibility.payload["feasible"]
    implementation = session.implement_factor(hypothesis.artifact_id)
    assert implementation.status is ArtifactStatus.ACCEPTED
    validation = session.evaluate(catalog[0]["spec_id"], "validation")
    assert validation.split.value == "validation"
    assert "net_sharpe" in validation.payload["metrics"]


def test_agents_cannot_request_test_split(app_config) -> None:
    session = _session(app_config)
    hypothesis = session.create_hypothesis(session.catalog(limit=1)[0]["spec_id"])
    session.check_feasibility(hypothesis.artifact_id)
    session.implement_factor(hypothesis.artifact_id)
    with pytest.raises(SplitGuardError):
        session.evaluate(hypothesis.payload["spec_id"], "test")


def test_role_tool_allowlists_are_scoped(app_config) -> None:
    session = _session(app_config)
    hypothesis_tools = {tool.name for tool in build_tools(session, RoleName.HYPOTHESIS)}
    critic_tools = {tool.name for tool in build_tools(session, RoleName.CRITIC)}
    assert "submit_hypothesis" in hypothesis_tools
    assert "run_screening" in critic_tools
    assert "construct_portfolio" not in hypothesis_tools
    tool = next(item for item in build_tools(session, RoleName.HYPOTHESIS) if item.name == "list_factor_templates")
    result = asyncio.run(tool.execute(limit=2))
    assert result.success
    assert "spec_id" in result.content
