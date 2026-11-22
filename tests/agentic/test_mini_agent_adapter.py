from __future__ import annotations

import pytest
from conftest import make_fields

pytest.importorskip("mini_agent")

from mini_agent.schema import FunctionCall, LLMResponse, ToolCall

from alpha_foundry.agentic.runtime import MiniAgentRuntime
from alpha_foundry.agentic.schemas import AgentTask, ArtifactKind, RoleName, TaskStatus
from alpha_foundry.agentic.session import ResearchSession
from alpha_foundry.agentic.workspace import RunWorkspace


class ScriptedLLM:
    def __init__(self, spec_id: str):
        self.spec_id = spec_id
        self.calls = 0

    async def generate(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="submit_hypothesis",
                            arguments={"spec_id": self.spec_id, "rationale": "test hypothesis"},
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )
        return LLMResponse(content="submitted one hypothesis", finish_reason="stop")


def test_mini_agent_loop_can_call_factor_tools(app_config) -> None:
    workspace = RunWorkspace(app_config.paths.runs_dir, run_id="run_mini")
    session = ResearchSession(
        app_config,
        make_fields(),
        {"data_fingerprint": "synthetic"},
        run_id=workspace.run_id,
        run_dir=workspace.run_dir,
    )
    spec_id = session.catalog(limit=1)[0]["spec_id"]
    runtime = MiniAgentRuntime(
        app_config,
        session,
        workspace.agent_logs_dir,
        llm_factory=lambda: ScriptedLLM(spec_id),
    )
    task = AgentTask(
        task_id="task_hypothesis",
        role=RoleName.HYPOTHESIS,
        objective="Submit one hypothesis using tools.",
        inputs={"factor_limit": 1},
        allowed_tools=["submit_hypothesis"],
        max_steps=3,
    )
    result = runtime.run_task(task, timeout_sec=30)
    assert result.status is TaskStatus.COMPLETED
    assert session.store.list(ArtifactKind.HYPOTHESIS_CARD)
    assert session.store.list(ArtifactKind.AGENT_TRANSCRIPT)
