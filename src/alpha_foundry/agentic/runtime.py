from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..config import AppConfig
from .roles import build_system_prompt
from .schemas import (
    AgentResult,
    AgentTask,
    ArtifactKind,
    ArtifactStatus,
    RoleName,
    TaskStatus,
)
from .session import ResearchSession
from .tools import build_tools


class MiniAgentUnavailable(RuntimeError):
    pass


def _import_mini_agent() -> tuple[Any, Any, Any]:
    try:
        from mini_agent import LLMClient
        from mini_agent.agent import Agent
        from mini_agent.schema import LLMProvider

        return Agent, LLMClient, LLMProvider
    except Exception:
        root = os.environ.get("MINI_AGENT_ROOT")
        if root:
            root_path = str(Path(root).expanduser().resolve())
            if root_path not in sys.path:
                sys.path.insert(0, root_path)
            try:
                from mini_agent import LLMClient
                from mini_agent.agent import Agent
                from mini_agent.schema import LLMProvider

                return Agent, LLMClient, LLMProvider
            except Exception:
                pass
    raise MiniAgentUnavailable(
        "Mini-Agent is not installed. Install the optional runtime or set MINI_AGENT_ROOT."
    )


class MiniAgentRuntime:
    """Thin adapter from FactorFoundry roles to Mini-Agent's single-agent loop."""

    def __init__(
        self,
        config: AppConfig,
        session: ResearchSession,
        agent_logs_dir: Path,
        llm_factory: Any | None = None,
    ):
        self.config = config
        self.session = session
        self.agent_logs_dir = Path(agent_logs_dir)
        self.agent_logs_dir.mkdir(parents=True, exist_ok=True)
        self.llm_factory = llm_factory

    @staticmethod
    def available() -> bool:
        try:
            _import_mini_agent()
            return True
        except MiniAgentUnavailable:
            return False

    def _llm_settings(self) -> dict[str, Any]:
        llm = dict(self.config.llm or {})
        key_env = str(llm.get("api_key_env", "FACTORFOUNDRY_API_KEY"))
        api_key = os.environ.get(key_env)
        if not api_key:
            raise MiniAgentUnavailable(
                f"Missing API key. Set {key_env} before running --mode mini-agent."
            )
        if not llm.get("enabled", False):
            raise MiniAgentUnavailable(
                "LLM is disabled in config.json. Set llm.enabled=true for mini-agent mode."
            )
        return {
            "api_key": api_key,
            "provider": str(llm.get("provider", "openai")),
            "api_base": str(llm.get("api_base", "https://api.openai.com/v1")),
            "model": str(llm.get("model", "gpt-4.1-mini")),
            "max_steps": int(llm.get("max_steps", 30)),
            "token_limit": int(llm.get("token_limit", 80000)),
        }

    def run_task(self, task: AgentTask, timeout_sec: int = 900) -> AgentResult:
        Agent, LLMClient, LLMProvider = _import_mini_agent()
        settings = self._llm_settings() if self.llm_factory is None else {
            "max_steps": int((self.config.llm or {}).get("max_steps", 30)),
            "token_limit": int((self.config.llm or {}).get("token_limit", 80000)),
        }
        role = task.role if isinstance(task.role, RoleName) else RoleName(task.role)
        tools = build_tools(self.session, role)
        workspace = self.agent_logs_dir / f"{task.task_id}_{role.value}"
        workspace.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MINI_AGENT_LOG_DIR", str(self.agent_logs_dir / "mini_agent_logs"))
        before = {record.artifact_id for record in self.session.store.list()}
        if self.llm_factory is None:
            llm_client = LLMClient(
                api_key=settings["api_key"],
                provider=LLMProvider(settings["provider"]),
                api_base=settings["api_base"],
                model=settings["model"],
            )
        else:
            llm_client = self.llm_factory()
        agent = Agent(
            llm_client=llm_client,
            system_prompt=build_system_prompt(role),
            tools=tools,
            max_steps=min(task.max_steps, settings["max_steps"]),
            workspace_dir=str(workspace),
            token_limit=settings["token_limit"],
        )
        prompt = self._task_prompt(task)
        agent.add_user_message(prompt)
        self.session.ledger.append(
            "agent_task_started",
            f"{role.value}_agent",
            {
                "task_id": task.task_id,
                "objective": task.objective,
                "allowed_tools": [tool.name for tool in tools],
                "max_steps": agent.max_steps,
            },
            task_id=task.task_id,
        )
        try:
            summary = asyncio.run(asyncio.wait_for(agent.run(), timeout=timeout_sec))
            status = TaskStatus.COMPLETED
            error = None
        except Exception as exc:  # pragma: no cover - API dependent
            summary = ""
            status = TaskStatus.FAILED
            error = f"{type(exc).__name__}: {exc}"
        after = {record.artifact_id for record in self.session.store.list()}
        created = sorted(after - before)
        transcript = {
            "task": task.model_dump(mode="json"),
            "messages": [message.model_dump(mode="json") for message in agent.messages],
            "summary": summary,
            "status": status.value,
            "error": error,
            "tool_names": [tool.name for tool in tools],
            "created_artifacts": created,
            "tokens": int(getattr(agent, "api_total_tokens", 0) or 0),
        }
        transcript_record = self.session.store.put(
            ArtifactKind.AGENT_TRANSCRIPT,
            transcript,
            producer=f"{role.value}_agent",
            status=ArtifactStatus.COMPLETED if status is TaskStatus.COMPLETED else ArtifactStatus.FAILED,
            metadata={"task_id": task.task_id, "role": role.value},
        )
        tool_calls = sum(len(message.tool_calls or []) for message in agent.messages)
        result = AgentResult(
            task_id=task.task_id,
            role=role,
            status=status,
            summary=summary,
            artifact_ids=created + [transcript_record.artifact_id],
            tool_calls=tool_calls,
            tokens=int(getattr(agent, "api_total_tokens", 0) or 0),
            error=error,
        )
        self.session.ledger.append(
            "agent_task_finished",
            f"{role.value}_agent",
            result.model_dump(mode="json"),
            task_id=task.task_id,
            artifact_id=transcript_record.artifact_id,
        )
        return result

    @staticmethod
    def _task_prompt(task: AgentTask) -> str:
        inputs = json.dumps(task.inputs, ensure_ascii=False, indent=2)
        return (
            f"Task id: {task.task_id}\n"
            f"Role: {task.role.value}\n"
            f"Objective: {task.objective}\n\n"
            f"Structured inputs:\n{inputs}\n\n"
            "Use the allowed tools to complete the objective. "
            "Do not stop with a plan; execute the required tool calls and then return a short evidence summary."
        )
