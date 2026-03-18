from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import textwrap
import uuid

from .acp.manager import DownstreamConnectorManager
from .catalog import FederatedModelCatalog
from .models import DownstreamAgentConfig, PlanTask, SessionSnapshot, TaskStatus, ToolEvent
from .scheduler import Scheduler
from .state import SQLiteSessionStore


class Orchestrator:
    def __init__(
        self,
        agents: list[DownstreamAgentConfig],
        store: SQLiteSessionStore | None = None,
        connector_manager: DownstreamConnectorManager | None = None,
    ) -> None:
        if not agents:
            raise ValueError("At least one downstream agent configuration is required")
        self.catalog = FederatedModelCatalog(agents)
        self.store = store or SQLiteSessionStore()
        self.connector_manager = connector_manager or DownstreamConnectorManager(agents)
        self.scheduler = Scheduler()

    def create_session(self, cwd: str, selected_model: str | None = None) -> SessionSnapshot:
        normalized_cwd = str(Path(cwd).resolve())
        snapshot = SessionSnapshot(
            session_id=f"orch-{uuid.uuid4().hex[:12]}",
            cwd=normalized_cwd,
            selected_model=selected_model,
            title=f"OrgeMage: {Path(normalized_cwd).name or normalized_cwd}",
        )
        if selected_model:
            resolved = self.catalog.resolve(selected_model)
            snapshot.coordinator_agent_id = resolved.agent.agent_id
        self.store.save(snapshot)
        return snapshot

    def list_model_options(self) -> list[dict[str, str]]:
        return self.catalog.northbound_model_options()

    def set_selected_model(self, session_id: str, composite_model: str) -> SessionSnapshot:
        snapshot = self._require_session(session_id)
        resolved = self.catalog.resolve(composite_model)
        snapshot.selected_model = composite_model
        snapshot.coordinator_agent_id = resolved.agent.agent_id
        self.store.save(snapshot)
        return snapshot

    def load_session(self, session_id: str, selected_model: str | None = None) -> SessionSnapshot:
        snapshot = self._require_session(session_id)
        if selected_model is not None:
            return self.set_selected_model(session_id, selected_model)
        return snapshot

    def orchestrate(self, session_id: str, user_prompt: str) -> dict[str, object]:
        snapshot = self._require_session(session_id)
        if not snapshot.selected_model:
            default_option = self.list_model_options()[0]["value"]
            snapshot = self.set_selected_model(session_id, default_option)
        resolved = self.catalog.resolve(snapshot.selected_model or "")

        plan = self._build_plan(user_prompt, resolved.agent.agent_id)
        plan = self.scheduler.assign_tasks(plan, self.catalog.agents, resolved.agent.agent_id)
        tool_events = self._run_tasks(
            snapshot=snapshot,
            selected_model=resolved.option.value,
            coordinator_agent=resolved.agent,
            user_prompt=user_prompt,
            tasks=plan,
        )
        snapshot.task_graph = [task.to_dict() for task in plan]
        self.store.save(snapshot)
        summary = self._final_summary(plan)
        return {
            "session": asdict(snapshot),
            "coordinator": {
                "agent_id": resolved.agent.agent_id,
                "agent_name": resolved.agent.name,
                "model": resolved.option.value,
                "composite_model": resolved.composite_value,
            },
            "plan": [task.to_dict() for task in plan],
            "tool_events": [event.to_dict() for event in tool_events],
            "summary": summary,
        }

    def cancel(self, session_id: str, agent_id: str | None = None) -> SessionSnapshot:
        snapshot = self._require_session(session_id)
        self.connector_manager.cancel_session(snapshot, agent_id=agent_id)
        self.store.save(snapshot)
        return snapshot

    def _build_plan(self, user_prompt: str, coordinator_agent_id: str) -> list[PlanTask]:
        normalized = user_prompt.strip()
        title = normalized.splitlines()[0][:72] if normalized else "Orchestration request"
        return [
            PlanTask(
                title="Analyze request and produce execution plan",
                details=f"Coordinator decomposes the request: {title}",
                required_capabilities={"planner": True, "needsPermissions": True},
                assignee=coordinator_agent_id,
                priority=100,
            ),
            PlanTask(
                title="Inspect repository and gather context",
                details="Read project files, identify impacted modules, and capture constraints.",
                required_capabilities={"needsFilesystem": True, "commands": ["read", "search"]},
                dependency_ids=[],
                priority=80,
            ),
            PlanTask(
                title="Implement orchestrator changes",
                details="Edit project files, add code paths, and wire core orchestrator behavior.",
                required_capabilities={"needsFilesystem": True, "needsTerminal": True, "commands": ["edit", "test"]},
                dependency_ids=[],
                priority=70,
            ),
            PlanTask(
                title="Validate behavior",
                details="Run tests or dry-run validations and summarize execution results.",
                required_capabilities={"needsTerminal": True, "commands": ["test"]},
                dependency_ids=[],
                priority=60,
            ),
        ]

    def _run_tasks(
        self,
        *,
        snapshot: SessionSnapshot,
        selected_model: str,
        coordinator_agent: DownstreamAgentConfig,
        user_prompt: str,
        tasks: list[PlanTask],
    ) -> list[ToolEvent]:
        events: list[ToolEvent] = []
        coordinator_prompt = self._coordinator_instruction(user_prompt)
        agents_by_id = {agent.agent_id: agent for agent in self.catalog.agents}
        for task in tasks:
            task.status = TaskStatus.IN_PROGRESS
            events.append(
                ToolEvent(
                    tool_call_id=task.task_id,
                    title=task.title,
                    kind="delegate",
                    status=TaskStatus.IN_PROGRESS,
                    content=f"Delegating to {task.assignee}.",
                )
            )
            agent = agents_by_id[task.assignee or coordinator_agent.agent_id]
            result = self.connector_manager.execute_task(
                session=snapshot,
                task=task,
                coordinator_prompt=coordinator_prompt,
                selected_model=selected_model,
                agent=agent,
            )
            task.status = result.status
            task.output = result.summary
            events.append(
                ToolEvent(
                    tool_call_id=task.task_id,
                    title=task.title,
                    kind="delegate",
                    status=result.status,
                    content=result.summary,
                )
            )
        return events

    def _coordinator_instruction(self, user_prompt: str) -> str:
        return textwrap.dedent(
            f"""
            You are the selected coordinator model for OrgeMage.
            Decompose the request into ACP-native plan entries, choose the best workers,
            and keep permissions/filesystem/terminal access constrained to the upstream cwd.

            User request:
            {user_prompt.strip()}
            """
        ).strip()

    def _final_summary(self, tasks: list[PlanTask]) -> str:
        completed = [task for task in tasks if task.status == TaskStatus.COMPLETED]
        return f"Completed {len(completed)}/{len(tasks)} orchestrated tasks."

    def _require_session(self, session_id: str) -> SessionSnapshot:
        snapshot = self.store.load(session_id)
        if snapshot is None:
            raise KeyError(f"Unknown session: {session_id}")
        return snapshot
