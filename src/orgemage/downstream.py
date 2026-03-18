from __future__ import annotations

from dataclasses import dataclass
import json

from .models import DownstreamAgentConfig, PlanTask, TaskStatus, WorkerResult


@dataclass(slots=True)
class MockDownstreamClient:
    """Deterministic stand-in used only for explicit dev/test fallback runs."""

    def execute_task(
        self,
        *,
        session_id: str,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
        agent: DownstreamAgentConfig,
    ) -> WorkerResult:
        if task._meta.get("phase") == "planning" or task.title == "Generate structured orchestration plan":
            raw_output = json.dumps(
                {
                    "tasks": [
                        {
                            "title": "Analyze request and produce execution plan",
                            "details": "Coordinator decomposes the request and confirms execution scope.",
                            "dependencies": [],
                            "required_capabilities": {"planner": True, "needsPermissions": True},
                            "assignee_hints": [agent.agent_id],
                            "acceptable_models": [f"{agent.agent_id}::{selected_model}"],
                            "priority": 100,
                            "_meta": {"planner": "mock", "reason": "bootstrap planning task"},
                        },
                        {
                            "title": "Inspect repository and gather context",
                            "details": "Read project files, identify impacted modules, and capture constraints.",
                            "dependencies": ["Analyze request and produce execution plan"],
                            "required_capabilities": {"needsFilesystem": True, "commands": ["read", "search"]},
                            "assignee_hints": [agent.agent_id],
                            "acceptable_models": [],
                            "priority": 80,
                            "_meta": {"planner": "mock", "reason": "repo inspection"},
                        },
                        {
                            "title": "Implement orchestrator changes",
                            "details": "Edit project files, add code paths, and wire core orchestrator behavior.",
                            "dependencies": ["Inspect repository and gather context"],
                            "required_capabilities": {"needsFilesystem": True, "needsTerminal": True, "commands": ["edit", "test"]},
                            "assignee_hints": [],
                            "acceptable_models": [],
                            "priority": 70,
                            "_meta": {"planner": "mock", "reason": "implementation"},
                        },
                        {
                            "title": "Validate behavior",
                            "details": "Run tests or dry-run validations and summarize execution results.",
                            "dependencies": ["Implement orchestrator changes"],
                            "required_capabilities": {"needsTerminal": True, "commands": ["test"]},
                            "assignee_hints": [],
                            "acceptable_models": [],
                            "priority": 60,
                            "_meta": {"planner": "mock", "reason": "validation"},
                        },
                    ],
                    "_meta": {"planner": "mock_downstream", "provenance": "Deterministic structured plan for tests."},
                }
            )
            return WorkerResult(
                task_id=task.task_id,
                agent_id=agent.agent_id,
                status=TaskStatus.COMPLETED,
                summary="Generated structured orchestration plan.",
                raw_output=raw_output,
                metadata={"mock": True, "phase": "planning"},
            )
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
