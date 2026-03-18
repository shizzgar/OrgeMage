from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import DownstreamAgentConfig, PlanTask, TaskStatus, WorkerResult


class DownstreamClient(Protocol):
    def execute_task(
        self,
        *,
        session_id: str,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
        agent: DownstreamAgentConfig,
    ) -> WorkerResult:
        ...


@dataclass(slots=True)
class MockDownstreamClient:
    """Deterministic stand-in used for local development and testing."""

    def execute_task(
        self,
        *,
        session_id: str,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
        agent: DownstreamAgentConfig,
    ) -> WorkerResult:
        summary = (
            f"[{agent.name}/{selected_model}] completed '{task.title}' for session {session_id}. "
            f"Directive: {coordinator_prompt[:120]}"
        )
        if task.required_capabilities.get("needsTerminal"):
            summary += " Used terminal-capable worker."
        if task.required_capabilities.get("needsFilesystem"):
            summary += " Used filesystem-capable worker."
        return WorkerResult(
            task_id=task.task_id,
            agent_id=agent.agent_id,
            status=TaskStatus.COMPLETED,
            summary=summary,
            raw_output=summary,
            metadata={"mock": True},
        )
