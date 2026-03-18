from __future__ import annotations

from pathlib import Path
import textwrap
import uuid
from typing import Any, Callable

from .acp.manager import DownstreamConnectorManager
from .catalog import FederatedModelCatalog
from .models import DownstreamAgentConfig, OrchestrationTurnState, PlanTask, SessionHistoryEntry, SessionSnapshot, TaskExecutionState, TaskStatus, ToolEvent, TraceCorrelationState
from .scheduler import Scheduler
from .state import SQLiteSessionStore

SessionUpdateCallback = Callable[[dict[str, Any]], None]


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

    def list_sessions(self) -> list[SessionHistoryEntry]:
        return [SessionHistoryEntry.from_snapshot(snapshot) for snapshot in self.store.list_sessions()]

    def session_info(self, session_id: str) -> dict[str, Any]:
        snapshot = self._require_session(session_id)
        return {
            "session_id": snapshot.session_id,
            "title": snapshot.title,
            "cwd": snapshot.cwd,
            "selected_model": snapshot.selected_model,
            "coordinator_agent_id": snapshot.coordinator_agent_id,
            "created_at": snapshot.created_at,
            "updated_at": snapshot.updated_at,
            "task_count": len(snapshot.task_states),
        }

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
        return self.orchestrate_turn(session_id, user_prompt)

    def orchestrate_turn(
        self,
        session_id: str,
        user_prompt: str,
        *,
        emit_update: SessionUpdateCallback | None = None,
    ) -> dict[str, object]:
        snapshot = self._require_session(session_id)
        if not snapshot.selected_model:
            default_option = self.list_model_options()[0]["value"]
            snapshot = self.set_selected_model(session_id, default_option)
        resolved = self.catalog.resolve(snapshot.selected_model or "")

        turn = OrchestrationTurnState(status="running", turn_id=f"turn-{uuid.uuid4().hex[:12]}")
        snapshot.turns.append(turn)
        self.store.create_or_update_turn_state(snapshot.session_id, turn)

        plan = self._build_plan(user_prompt, resolved.agent.agent_id)
        plan = self.scheduler.assign_tasks(plan, self.catalog.agents, resolved.agent.agent_id)
        snapshot.task_states = [TaskExecutionState.from_plan_task(task, parent_turn_id=turn.turn_id) for task in plan]
        for task_state in snapshot.task_states:
            self.store.persist_task_update(snapshot.session_id, task_state)
        self._emit_plan_update(snapshot, plan, emit_update)
        tool_events = self._run_tasks(
            snapshot=snapshot,
            selected_model=resolved.option.value,
            coordinator_agent=resolved.agent,
            user_prompt=user_prompt,
            tasks=plan,
            turn=turn,
            emit_update=emit_update,
        )
        turn.status = "completed"
        turn.stop_reason = "end_turn"
        self.store.create_or_update_turn_state(snapshot.session_id, turn)
        snapshot = self._require_session(session_id)
        summary = self._final_summary(plan)
        result = {
            "session": snapshot.to_dict(),
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
        self._emit_message_update(snapshot, summary, result, emit_update)
        return result

    def cancel(self, session_id: str, agent_id: str | None = None) -> SessionSnapshot:
        snapshot = self._require_session(session_id)
        self.connector_manager.cancel_session(snapshot, agent_id=agent_id)
        snapshot.metadata["cancelled"] = True
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
        turn: OrchestrationTurnState,
        emit_update: SessionUpdateCallback | None,
    ) -> list[ToolEvent]:
        events: list[ToolEvent] = []
        coordinator_prompt = self._coordinator_instruction(user_prompt)
        agents_by_id = {agent.agent_id: agent for agent in self.catalog.agents}
        for task in tasks:
            task.status = TaskStatus.IN_PROGRESS
            task_state = snapshot.get_task_state(task.task_id) or TaskExecutionState.from_plan_task(task, parent_turn_id=turn.turn_id)
            task_state.status = task.status
            task_state.assignee = task.assignee
            task_state.dependency_state = "blocked" if task.dependency_ids else "ready"
            self.store.persist_task_update(snapshot.session_id, task_state)
            snapshot.upsert_task_state(task_state)
            started_event = ToolEvent(
                tool_call_id=task.task_id,
                title=task.title,
                kind="delegate",
                status=TaskStatus.IN_PROGRESS,
                content=f"Delegating to {task.assignee}.",
            )
            events.append(started_event)
            self._emit_tool_call_update(snapshot, task, started_event, emit_update)
            agent = agents_by_id[task.assignee or coordinator_agent.agent_id]
            result = self.connector_manager.execute_task(
                session=snapshot,
                task=task,
                coordinator_prompt=coordinator_prompt,
                selected_model=selected_model,
                agent=agent,
            )
            downstream_session_id = result.metadata.get("downstream_session_id")
            if isinstance(downstream_session_id, str):
                self.store.save_downstream_session_mapping(snapshot.session_id, agent.agent_id, downstream_session_id)
            negotiated = snapshot.metadata.get("downstream_negotiated", {}).get(agent.agent_id)
            if negotiated is not None:
                self.store.persist_trace_metadata(
                    snapshot.session_id,
                    TraceCorrelationState(
                        trace_key=f"negotiated:{agent.agent_id}",
                        task_id=task.task_id,
                        turn_id=turn.turn_id,
                        metadata={"agent_id": agent.agent_id, "negotiated": negotiated},
                    ),
                )
            task.status = result.status
            task.output = result.summary
            task_state = snapshot.get_task_state(task.task_id) or TaskExecutionState.from_plan_task(task, parent_turn_id=turn.turn_id)
            task_state.status = result.status
            task_state.output = result.summary
            task_state.assignee = task.assignee
            task_state.plan_metadata.update({"worker_result": dict(result.metadata)})
            self.store.persist_task_update(snapshot.session_id, task_state)
            snapshot.upsert_task_state(task_state)
            finished_event = ToolEvent(
                tool_call_id=task.task_id,
                title=task.title,
                kind="delegate",
                status=result.status,
                content=result.summary,
            )
            events.append(finished_event)
            self._emit_tool_call_update(snapshot, task, finished_event, emit_update)
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

    def _emit_plan_update(
        self,
        snapshot: SessionSnapshot,
        tasks: list[PlanTask],
        emit_update: SessionUpdateCallback | None,
    ) -> None:
        if emit_update is None:
            return
        emit_update(
            {
                "sessionUpdate": "plan",
                "session_id": snapshot.session_id,
                "plan": [task.to_dict() for task in tasks],
            }
        )

    def _emit_tool_call_update(
        self,
        snapshot: SessionSnapshot,
        task: PlanTask,
        event: ToolEvent,
        emit_update: SessionUpdateCallback | None,
    ) -> None:
        if emit_update is None:
            return
        emit_update(
            {
                "sessionUpdate": "tool_call",
                "session_id": snapshot.session_id,
                "task": task.to_dict(),
                "tool_call": event.to_dict(),
            }
        )

    def _emit_message_update(
        self,
        snapshot: SessionSnapshot,
        summary: str,
        result: dict[str, object],
        emit_update: SessionUpdateCallback | None,
    ) -> None:
        if emit_update is None:
            return
        emit_update(
            {
                "sessionUpdate": "message",
                "session_id": snapshot.session_id,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": summary}],
                },
                "result": result,
            }
        )

    def _final_summary(self, tasks: list[PlanTask]) -> str:
        completed = [task for task in tasks if task.status == TaskStatus.COMPLETED]
        return f"Completed {len(completed)}/{len(tasks)} orchestrated tasks."

    def _require_session(self, session_id: str) -> SessionSnapshot:
        snapshot = self.store.load(session_id)
        if snapshot is None:
            raise KeyError(f"Unknown session: {session_id}")
        return snapshot
