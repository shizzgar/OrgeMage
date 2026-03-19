from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import textwrap
import threading
import uuid
from typing import Any, Callable

from .acp.manager import DownstreamConnectorManager
from .catalog import FederatedModelCatalog
from .debug import debug_event, get_logger
from .execution_graph import ExecutionGraphRunner
from .metadata import build_turn_metadata, event_metadata, propagate_task_metadata, session_title, summarize_session
from .models import (
    DownstreamAgentConfig,
    OrchestrationTurnState,
    PlanTask,
    SessionHistoryEntry,
    SessionSnapshot,
    TaskExecutionState,
    TaskStatus,
    TerminalMapping,
    ToolEvent,
    TraceCorrelationState,
    TurnStatus,
)
from .planning import PlanParseResult, parse_coordinator_plan, synthesize_local_fallback_plan
from .scheduler import Scheduler
from .state import SQLiteSessionStore

SessionUpdateCallback = Callable[[dict[str, Any]], None]
_LOG = get_logger(__name__)


class OrchestrationEventNormalizer:
    def __init__(self, store: SQLiteSessionStore) -> None:
        self.store = store
        self._rebased_tool_ids: dict[tuple[str, str], str] = {}
        self._rebased_counters: dict[str, int] = {}

    def plan_update(self, snapshot: SessionSnapshot, tasks: list[PlanTask]) -> dict[str, Any]:
        return {
            "sessionUpdate": "plan",
            "session_id": snapshot.session_id,
            "plan": [task.to_dict() for task in tasks],
            "globalPlan": [task.to_acp_plan_item() for task in tasks],
            "planning": dict(snapshot.metadata.get("planning", {})),
            "_meta": {
                **dict(snapshot.metadata.get("turn_context", {})),
                "planningProvenance": dict(snapshot.metadata.get("planning", {}).get("normalized_plan", {}).get("_meta", {})),
            },
        }

    def tool_call_update(
        self,
        snapshot: SessionSnapshot,
        task: PlanTask,
        event: ToolEvent,
        *,
        include_task: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "sessionUpdate": "tool_call",
            "session_id": snapshot.session_id,
            "tool_call": event.to_dict(),
            "toolCall": event.to_acp_tool_call(),
            "globalPlan": [plan_task.to_acp_plan_item() for plan_task in self._current_plan(snapshot)],
            "_meta": dict(event.metadata),
        }
        if include_task:
            payload["task"] = task.to_dict()
        return payload

    def normalize_worker_updates(
        self,
        *,
        snapshot: SessionSnapshot,
        task: PlanTask,
        agent: DownstreamAgentConfig,
        updates: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        turn_id = str(task._meta.get("turnId") or snapshot.metadata.get("turn_context", {}).get("turnId") or "")
        for index, update in enumerate(updates):
            tool_call_id = self._resolve_tool_call_id(task, update, index=index)
            locations = self._extract_locations(update)
            terminal = self._extract_terminal(update)
            if terminal is not None:
                mapping = TerminalMapping(
                    upstream_terminal_id=terminal["terminalId"],
                    downstream_terminal_id=str(terminal.get("downstreamTerminalId") or terminal["terminalId"]),
                    owner_task_id=task.task_id,
                    owner_agent_id=agent.agent_id,
                    metadata={"source_update": update, "workerCorrelationId": task._meta.get("workerCorrelationId")},
                )
                self.store.persist_terminal_event(snapshot.session_id, mapping)
                snapshot.terminal_mappings = [
                    current
                    for current in snapshot.terminal_mappings
                    if not (
                        current.upstream_terminal_id == mapping.upstream_terminal_id
                        and current.downstream_terminal_id == mapping.downstream_terminal_id
                    )
                ]
                snapshot.terminal_mappings.append(mapping)
            content = self._extract_text(update)
            if not content and not locations and terminal is None:
                continue
            event = ToolEvent(
                tool_call_id=tool_call_id,
                title=task.title,
                kind="delegate",
                status=TaskStatus.IN_PROGRESS,
                content=content,
                locations=locations,
                terminal=terminal,
                metadata=event_metadata(
                    session_id=snapshot.session_id,
                    turn_id=turn_id,
                    task_id=task.task_id,
                    task_meta=task._meta,
                    assignee=agent.agent_id,
                    source="downstream_session_update",
                    extra={
                        "taskId": task.task_id,
                        "workerAgentId": agent.agent_id,
                        "rawUpdate": update,
                        "toolCallId": tool_call_id,
                    },
                ),
            )
            normalized.append(self.tool_call_update(snapshot, task, event))
        return normalized

    def message_update(
        self,
        snapshot: SessionSnapshot,
        summary: str,
        result: dict[str, object],
        tasks: list[PlanTask],
    ) -> dict[str, Any]:
        blocks = [{"type": "text", "text": summary}]
        for task in tasks:
            if task.output:
                blocks.append({"type": "text", "text": f"{task.title}: {task.output}"})
        return {
            "sessionUpdate": "message",
            "session_id": snapshot.session_id,
            "message": {
                "role": "assistant",
                "content": blocks,
            },
            "result": result,
            "_meta": {
                **dict(snapshot.metadata.get("turn_context", {})),
                "sessionSummary": snapshot.metadata.get("session_summary", summary),
            },
        }

    def delegate_started_event(self, snapshot: SessionSnapshot, task: PlanTask) -> ToolEvent:
        turn_id = str(task._meta.get("turnId") or snapshot.metadata.get("turn_context", {}).get("turnId") or "")
        return ToolEvent(
            tool_call_id=self._orchestrator_tool_id(task),
            title=task.title,
            kind="delegate",
            status=TaskStatus.IN_PROGRESS,
            content=f"Delegating to {task.assignee}.",
            metadata=event_metadata(
                session_id=snapshot.session_id,
                turn_id=turn_id,
                task_id=task.task_id,
                task_meta=task._meta,
                assignee=task.assignee,
                source="orchestrator",
                extra={"taskId": task.task_id},
            ),
        )

    def delegate_finished_event(self, snapshot: SessionSnapshot, task: PlanTask, *, status: TaskStatus, content: str) -> ToolEvent:
        turn_id = str(task._meta.get("turnId") or snapshot.metadata.get("turn_context", {}).get("turnId") or "")
        return ToolEvent(
            tool_call_id=self._orchestrator_tool_id(task),
            title=task.title,
            kind="delegate",
            status=status,
            content=content,
            metadata=event_metadata(
                session_id=snapshot.session_id,
                turn_id=turn_id,
                task_id=task.task_id,
                task_meta=task._meta,
                assignee=task.assignee,
                source="orchestrator",
                extra={"taskId": task.task_id},
            ),
        )

    def _current_plan(self, snapshot: SessionSnapshot) -> list[PlanTask]:
        return [task_state.apply_to_plan_task() for task_state in snapshot.task_states]

    def _orchestrator_tool_id(self, task: PlanTask) -> str:
        return f"orch-tool-{task.task_id}"

    def _resolve_tool_call_id(self, task: PlanTask, update: dict[str, Any], *, index: int) -> str:
        downstream_id = self._extract_tool_call_id(update)
        if not downstream_id:
            return self._orchestrator_tool_id(task)
        key = (task.task_id, downstream_id)
        if key not in self._rebased_tool_ids:
            next_value = self._rebased_counters.get(task.task_id, 0) + 1
            self._rebased_counters[task.task_id] = next_value
            self._rebased_tool_ids[key] = f"{self._orchestrator_tool_id(task)}:worker-{next_value}"
        return self._rebased_tool_ids[key]

    def _extract_tool_call_id(self, payload: Any) -> str | None:
        if isinstance(payload, dict):
            for key in ("toolCallId", "tool_call_id", "id"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value
            for nested_key in ("tool_call", "toolCall", "message", "update"):
                nested = self._extract_tool_call_id(payload.get(nested_key))
                if nested:
                    return nested
            for value in payload.values():
                nested = self._extract_tool_call_id(value)
                if nested:
                    return nested
        if isinstance(payload, list):
            for item in payload:
                nested = self._extract_tool_call_id(item)
                if nested:
                    return nested
        return None

    def _extract_locations(self, payload: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        self._collect_locations(payload, found)
        unique: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()
        for location in found:
            key = tuple(sorted(location.items()))
            if key in seen:
                continue
            seen.add(key)
            unique.append(location)
        return unique

    def _collect_locations(self, payload: Any, found: list[dict[str, Any]]) -> None:
        if isinstance(payload, dict):
            locations = payload.get("locations")
            if isinstance(locations, list):
                for item in locations:
                    if isinstance(item, dict):
                        found.append(dict(item))
            location = payload.get("location")
            if isinstance(location, dict):
                found.append(dict(location))
            path = payload.get("path")
            if isinstance(path, str) and path:
                candidate = {"path": path}
                if isinstance(payload.get("line"), int):
                    candidate["line"] = payload["line"]
                found.append(candidate)
            for value in payload.values():
                self._collect_locations(value, found)
        elif isinstance(payload, list):
            for item in payload:
                self._collect_locations(item, found)

    def _extract_terminal(self, payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            candidate = payload.get("terminal") or payload.get("terminal_output") or payload.get("terminalOutput")
            if isinstance(candidate, dict):
                terminal_id = candidate.get("terminalId") or candidate.get("terminal_id") or candidate.get("id")
                content = candidate.get("content") or candidate.get("text") or candidate.get("output")
                if isinstance(terminal_id, str) and terminal_id and isinstance(content, str):
                    return {
                        "terminalId": f"up-{terminal_id}",
                        "downstreamTerminalId": terminal_id,
                        "content": content,
                    }
            for value in payload.values():
                terminal = self._extract_terminal(value)
                if terminal is not None:
                    return terminal
        elif isinstance(payload, list):
            for item in payload:
                terminal = self._extract_terminal(item)
                if terminal is not None:
                    return terminal
        return None

    def _extract_text(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            text = payload.get("text")
            if isinstance(text, str) and text:
                return text
            for key in ("content", "message"):
                extracted = self._extract_text(payload.get(key))
                if extracted:
                    return extracted
            return ""
        if isinstance(payload, list):
            parts = [self._extract_text(item) for item in payload]
            return "\n".join(part for part in parts if part)
        return ""


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
        self.connector_manager = connector_manager or DownstreamConnectorManager(agents, store=self.store)
        self.scheduler = Scheduler()
        self.normalizer = OrchestrationEventNormalizer(self.store)
        self._turn_control_lock = threading.Lock()
        self._turn_cancel_events: dict[tuple[str, str], threading.Event] = {}
        self._active_turn_by_session: dict[str, str] = {}

    def create_session(
        self,
        cwd: str,
        selected_model: str | None = None,
        *,
        mcp_servers: list[dict[str, Any]] | list[Any] | None = None,
    ) -> SessionSnapshot:
        normalized_cwd = str(Path(cwd).resolve())
        snapshot = SessionSnapshot(
            session_id=f"orch-{uuid.uuid4().hex[:12]}",
            cwd=normalized_cwd,
            selected_model=selected_model,
            title=f"OrgeMage: {Path(normalized_cwd).name or normalized_cwd}",
        )
        snapshot.set_mcp_servers(mcp_servers)
        if selected_model:
            self._refresh_catalog_for_model(selected_model)
            resolved = self.catalog.resolve(selected_model)
            snapshot.coordinator_agent_id = resolved.agent.agent_id
        self.store.save(snapshot)
        return snapshot

    def list_model_options(self, *, refresh: bool = True) -> list[dict[str, object]]:
        if refresh:
            self.connector_manager.refresh_catalog(self.catalog)
        return self.catalog.northbound_model_options()

    def list_sessions(self) -> list[SessionHistoryEntry]:
        return self.store.list_session_history()

    def session_info(self, session_id: str) -> dict[str, Any]:
        snapshot = self._require_session(session_id)
        active_turn = snapshot.active_turn()
        return {
            "session_id": snapshot.session_id,
            "title": snapshot.title,
            "summary": snapshot.metadata.get("session_summary", ""),
            "cwd": snapshot.cwd,
            "selected_model": snapshot.selected_model,
            "coordinator_agent_id": snapshot.coordinator_agent_id,
            "active_turn_id": active_turn.turn_id if active_turn is not None else None,
            "active_turn_status": (
                active_turn.status.value if active_turn is not None and isinstance(active_turn.status, TurnStatus) else active_turn.status
            ) if active_turn is not None else None,
            "created_at": snapshot.created_at,
            "updated_at": snapshot.updated_at,
            "task_count": len(snapshot.task_states),
            "mcp_servers": list(snapshot.mcp_servers),
            "history": SessionHistoryEntry.from_snapshot(snapshot).to_dict(),
        }

    def set_selected_model(self, session_id: str, composite_model: str) -> SessionSnapshot:
        snapshot = self._require_session(session_id)
        self._refresh_catalog_for_model(composite_model)
        resolved = self.catalog.resolve(composite_model)
        snapshot.selected_model = composite_model
        snapshot.coordinator_agent_id = resolved.agent.agent_id
        self.store.save(snapshot)
        return snapshot

    def load_session(
        self,
        session_id: str,
        selected_model: str | None = None,
        *,
        mcp_servers: list[dict[str, Any]] | list[Any] | None = None,
    ) -> SessionSnapshot:
        snapshot = self._require_session(session_id)
        if mcp_servers is not None:
            snapshot.set_mcp_servers(mcp_servers)
            self.store.save(snapshot)
        if selected_model is not None:
            return self.set_selected_model(session_id, selected_model)
        return snapshot

    def orchestrate(
        self,
        session_id: str,
        user_prompt: str,
        *,
        emit_update: SessionUpdateCallback | None = None,
        prompt_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._execute_turn(session_id, user_prompt, emit_update=emit_update, prompt_metadata=prompt_metadata)

    def orchestrate_turn(
        self,
        session_id: str,
        user_prompt: str,
        *,
        emit_update: SessionUpdateCallback | None = None,
        prompt_metadata: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        updates: list[dict[str, Any]] = []

        def collector(update: dict[str, Any]) -> None:
            updates.append(update)
            if emit_update is not None:
                emit_update(update)

        result = self._execute_turn(session_id, user_prompt, emit_update=collector, prompt_metadata=prompt_metadata)
        result["updates"] = updates
        return result

    def _execute_turn(
        self,
        session_id: str,
        user_prompt: str,
        *,
        emit_update: SessionUpdateCallback | None,
        prompt_metadata: dict[str, Any] | None,
    ) -> dict[str, object]:
        snapshot = self._require_session(session_id)
        if not snapshot.selected_model:
            default_option = self.list_model_options()[0]["value"]
            snapshot = self.set_selected_model(session_id, default_option)
        self._refresh_catalog_for_model(snapshot.selected_model or "")
        resolved = self.catalog.resolve(snapshot.selected_model or "")

        turn = OrchestrationTurnState(status=TurnStatus.RUNNING, turn_id=f"turn-{uuid.uuid4().hex[:12]}")
        history_fields = summarize_session(user_prompt)
        turn.metadata["prompt"] = {"summary": history_fields["summary"]}
        turn.metadata["_meta"] = build_turn_metadata(
            session_id=snapshot.session_id,
            turn_id=turn.turn_id,
            prompt_metadata=prompt_metadata,
        )
        snapshot.metadata["turn_context"] = dict(turn.metadata["_meta"])
        snapshot.metadata.setdefault("prompt_metadata", {}).update(dict(prompt_metadata or {}))
        snapshot.metadata["session_summary"] = history_fields["summary"]
        snapshot.title = session_title(snapshot.title, history_fields["title"])
        snapshot.turns.append(turn)
        self.store.create_or_update_turn_state(snapshot.session_id, turn)
        self.store.save(snapshot)
        cancel_event = self._register_turn(session_id, turn.turn_id)

        try:
            plan_parse_result, planner_record = self._generate_plan(
                snapshot=snapshot,
                selected_model=resolved.option.value,
                coordinator_agent=resolved.agent,
                user_prompt=user_prompt,
                turn=turn,
            )
            if planner_record["planner_result"]["status"] == TaskStatus.CANCELLED.value:
                cancel_event.set()
            if cancel_event.is_set():
                plan = []
                tool_events = []
            else:
                plan = self.scheduler.assign_tasks(plan_parse_result.tasks, self.catalog.agents, resolved.agent.agent_id)
                self._apply_task_metadata(snapshot=snapshot, turn=turn, tasks=plan, planner_record=planner_record)
                snapshot.task_states = [TaskExecutionState.from_plan_task(task, parent_turn_id=turn.turn_id) for task in plan]
                for task_state in snapshot.task_states:
                    self.store.persist_task_update(snapshot.session_id, task_state)
                self.store.save(snapshot)
                self._emit_update(self.normalizer.plan_update(snapshot, plan), emit_update)

                tool_events = self._run_tasks(
                    snapshot=snapshot,
                    selected_model=resolved.option.value,
                    coordinator_agent=resolved.agent,
                    user_prompt=user_prompt,
                    tasks=plan,
                    turn=turn,
                    emit_update=emit_update,
                    cancel_event=cancel_event,
                )
            if cancel_event.is_set():
                turn.status = TurnStatus.CANCELLED
                turn.stop_reason = "cancelled"
            else:
                turn.status = TurnStatus.COMPLETED
                turn.stop_reason = "end_turn"
            self.store.create_or_update_turn_state(snapshot.session_id, turn)
            summary = self._final_summary(plan, cancelled=cancel_event.is_set())
            self._update_session_history(snapshot, user_prompt=user_prompt, final_summary=summary)
            self.store.save(snapshot)
            snapshot = self._require_session(session_id)
            result = {
                "session": snapshot.to_dict(),
                "coordinator": {
                    "agent_id": resolved.agent.agent_id,
                    "agent_name": resolved.agent.name,
                    "model": resolved.option.value,
                    "composite_model": resolved.composite_value,
                },
                "planning": planner_record,
                "plan": [task.to_dict() for task in plan],
                "global_plan": [task.to_acp_plan_item() for task in plan],
                "tool_events": [event.to_dict() for event in tool_events],
                "summary": summary,
                "stop_reason": turn.stop_reason,
            }
            self._emit_update(self.normalizer.message_update(snapshot, summary, result, plan), emit_update)
            return result
        except Exception:
            turn.status = TurnStatus.FAILED
            turn.stop_reason = "failed"
            self.store.create_or_update_turn_state(snapshot.session_id, turn)
            self._update_session_history(snapshot, user_prompt=user_prompt, final_summary="Turn failed.")
            self.store.save(snapshot)
            raise
        finally:
            self._clear_turn_registration(session_id, turn.turn_id)

    def _refresh_catalog_for_model(self, composite_model: str) -> None:
        agent_id, _, _ = composite_model.partition("::")
        if agent_id and agent_id in {agent.agent_id for agent in self.catalog.agents}:
            self.connector_manager.refresh_catalog(self.catalog, agent_id=agent_id)

    def cancel(self, session_id: str, agent_id: str | None = None) -> SessionSnapshot:
        snapshot = self._require_session(session_id)
        turn = snapshot.active_turn()
        active_task_ids: set[str] | None = None
        if turn is not None:
            turn.status = TurnStatus.CANCELLING
            turn.stop_reason = "cancel_requested"
            self.store.create_or_update_turn_state(snapshot.session_id, turn)
            active_task_ids = {
                task.task_id
                for task in snapshot.task_states
                if task.parent_turn_id == turn.turn_id and task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
            }
            self._set_turn_cancelled(session_id, turn.turn_id)
        debug_event(
            _LOG,
            "orchestrator.cancel",
            session_id=session_id,
            agent_id=agent_id,
            turn_id=turn.turn_id if turn is not None else None,
            active_task_ids=sorted(active_task_ids or []),
        )
        self.connector_manager.cancel_session(snapshot, agent_id=agent_id)
        self.store.cancel_permission_requests(
            snapshot.session_id,
            owner_task_ids=active_task_ids,
            metadata={"cancel_reason": "session_cancel"},
        )
        self.store.mark_terminal_mappings_cancelled(
            snapshot.session_id,
            owner_task_ids=active_task_ids,
            owner_agent_id=agent_id,
            metadata={"cleanup_reason": "session_cancel"},
        )
        self.store.update_session_metadata(snapshot.session_id, {"cancelled": True, "session_summary": "Turn cancelled."})
        return self._require_session(session_id)

    def _generate_plan(
        self,
        *,
        snapshot: SessionSnapshot,
        selected_model: str,
        coordinator_agent: DownstreamAgentConfig,
        user_prompt: str,
        turn: OrchestrationTurnState,
    ) -> tuple[PlanParseResult, dict[str, Any]]:
        planning_task = PlanTask(
            title="Generate structured orchestration plan",
            details="Analyze the user request and return a structured orchestration plan JSON contract.",
            required_capabilities={"planner": True, "needsPermissions": True},
            assignee=coordinator_agent.agent_id,
            priority=100,
            _meta=propagate_task_metadata(
                {"source": "coordinator", "phase": "planning", "execution_role": "coordinator"},
                session_id=snapshot.session_id,
                turn_id=turn.turn_id,
                task_id=f"planning:{turn.turn_id}",
                assignee=coordinator_agent.agent_id,
                assignee_hints=[coordinator_agent.agent_id],
                planning_provenance={},
                prompt_metadata=dict(snapshot.metadata.get("turn_context", {})),
            ),
        )
        debug_event(
            _LOG,
            "planning.generate.start",
            session_id=snapshot.session_id,
            turn_id=turn.turn_id,
            coordinator_agent_id=coordinator_agent.agent_id,
            selected_model=selected_model,
        )
        planning_result = self.connector_manager.execute_task(
            session=snapshot,
            task=planning_task,
            coordinator_prompt=self._coordinator_instruction(user_prompt),
            selected_model=selected_model,
            agent=coordinator_agent,
        )
        raw_output = planning_result.raw_output or planning_result.summary
        validation_errors: list[str] = []
        if planning_result.status == TaskStatus.CANCELLED:
            plan_parse_result = PlanParseResult(
                tasks=[],
                normalized_plan={"tasks": [], "_meta": {"source": "cancelled", "coordinator_agent_id": coordinator_agent.agent_id}},
                errors=[],
            )
        else:
            parsed_plan = parse_coordinator_plan(raw_output, coordinator_agent_id=coordinator_agent.agent_id)
            validation_errors = list(parsed_plan.errors)
            if not parsed_plan.is_valid:
                fallback_plan = synthesize_local_fallback_plan(user_prompt, coordinator_agent_id=coordinator_agent.agent_id)
                fallback_plan.normalized_plan["_meta"]["coordinator_errors"] = list(parsed_plan.errors)
                plan_parse_result = fallback_plan
            else:
                plan_parse_result = parsed_plan

        planner_record = {
            "raw_coordinator_output": raw_output,
            "normalized_plan": plan_parse_result.normalized_plan,
            "planner_task": planning_task.to_dict(),
            "planner_result": {
                "status": planning_result.status.value,
                "summary": planning_result.summary,
                "metadata": dict(planning_result.metadata),
            },
            "validation_errors": validation_errors,
        }
        debug_event(
            _LOG,
            "planning.generate.complete",
            session_id=snapshot.session_id,
            turn_id=turn.turn_id,
            planner_status=planning_result.status.value,
            validation_errors=validation_errors,
            normalized_plan_meta=plan_parse_result.normalized_plan.get("_meta", {}),
        )
        snapshot.metadata["planning"] = planner_record
        turn.metadata["planning"] = {
            "source": plan_parse_result.normalized_plan.get("_meta", {}).get("source"),
            "planner_task_id": planning_task.task_id,
            "validation_errors": validation_errors,
        }
        turn.metadata["_meta"] = {
            **dict(turn.metadata.get("_meta", {})),
            "planningProvenance": dict(plan_parse_result.normalized_plan.get("_meta", {})),
        }
        snapshot.metadata["turn_context"] = dict(turn.metadata["_meta"])
        self.store.create_or_update_turn_state(snapshot.session_id, turn)
        self.store.save(snapshot)
        return plan_parse_result, planner_record

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
        cancel_event: threading.Event,
    ) -> list[ToolEvent]:
        coordinator_prompt = self._coordinator_instruction(user_prompt)
        runner = ExecutionGraphRunner(
            snapshot=snapshot,
            tasks=tasks,
            turn=turn,
            coordinator_agent=coordinator_agent,
            agents=self.catalog.agents,
            selected_model=selected_model,
            coordinator_prompt=coordinator_prompt,
            persist_task_state=lambda task, reason, result=None, dependency_state=None: self._persist_task_state(
                snapshot=snapshot,
                task=task,
                turn=turn,
                status_reason=reason,
                result=result,
                dependency_state=dependency_state,
            ),
            execute_task=lambda current_snapshot, task, prompt, model, agent: self.connector_manager.execute_task(
                session=current_snapshot,
                task=task,
                coordinator_prompt=prompt,
                selected_model=model,
                agent=agent,
            ),
            normalize_worker_updates=lambda current_snapshot, task, agent, updates: self.normalizer.normalize_worker_updates(
                snapshot=current_snapshot,
                task=task,
                agent=agent,
                updates=updates,
            ),
            emit_plan_update=lambda: self._emit_update(self.normalizer.plan_update(snapshot, tasks), emit_update),
            emit_generic_update=lambda update: self._emit_update(update, emit_update),
            emit_tool_event=lambda task, event: self._emit_update(self.normalizer.tool_call_update(snapshot, task, event), emit_update),
            create_started_event=lambda task: self.normalizer.delegate_started_event(snapshot, task),
            create_finished_event=lambda task, status, content: self.normalizer.delegate_finished_event(snapshot, task, status=status, content=content),
            persist_trace=self.store.persist_trace_metadata,
            save_downstream_mapping=self.store.save_downstream_session_mapping,
            is_cancel_requested=cancel_event.is_set,
            cancel_active_work=lambda: self.connector_manager.cancel_session(snapshot),
        )
        return runner.run()

    def _apply_task_metadata(
        self,
        *,
        snapshot: SessionSnapshot,
        turn: OrchestrationTurnState,
        tasks: list[PlanTask],
        planner_record: dict[str, Any],
    ) -> None:
        planning_provenance = dict(planner_record.get("normalized_plan", {}).get("_meta", {}))
        prompt_metadata = dict(snapshot.metadata.get("turn_context", {}))
        for task in tasks:
            task._meta = propagate_task_metadata(
                task._meta,
                session_id=snapshot.session_id,
                turn_id=turn.turn_id,
                task_id=task.task_id,
                assignee=task.assignee,
                assignee_hints=task.assignee_hints,
                planning_provenance=planning_provenance,
                prompt_metadata=prompt_metadata,
            )

    def _persist_task_state(
        self,
        *,
        snapshot: SessionSnapshot,
        task: PlanTask,
        turn: OrchestrationTurnState,
        status_reason: str | None = None,
        result: Any | None = None,
        dependency_state: str | None = None,
    ) -> TaskExecutionState:
        task_state = snapshot.get_task_state(task.task_id) or TaskExecutionState.from_plan_task(task, parent_turn_id=turn.turn_id)
        task_state.status = task.status
        task_state.output = task.output
        task_state.assignee = task.assignee
        task_state.dependency_state = dependency_state or ("blocked" if task.dependency_ids else "ready")
        task_state.plan_metadata.update(
            {
                "assignee_hints": list(task.assignee_hints),
                "_meta": dict(task._meta),
            }
        )
        if status_reason is not None:
            task_state.plan_metadata["status_reason"] = status_reason
        if result is not None:
            task_state.plan_metadata["worker_result"] = {
                **dict(result.metadata),
                "raw_output": result.raw_output,
            }
        self.store.persist_task_update(snapshot.session_id, task_state)
        snapshot.upsert_task_state(task_state)
        return task_state

    def _update_session_history(self, snapshot: SessionSnapshot, *, user_prompt: str, final_summary: str) -> None:
        fields = summarize_session(user_prompt, final_summary)
        snapshot.title = session_title(snapshot.title, fields["title"])
        snapshot.metadata["session_summary"] = fields["summary"]

    def _coordinator_instruction(self, user_prompt: str) -> str:
        return textwrap.dedent(
            f"""
            You are the selected coordinator model for OrgeMage.
            Your first responsibility is planning: return a structured orchestration plan as JSON.
            Do not return prose before or after the JSON object.

            Contract:
            {{
              "tasks": [
                {{
                  "title": "string",
                  "details": "string",
                  "dependencies": ["task title dependency"],
                  "required_capabilities": {{"needsFilesystem": true, "commands": ["read"]}},
                  "assignee_hints": ["optional-agent-id"],
                  "acceptable_models": ["optional-agent::model or model"],
                  "priority": 80,
                  "_meta": {{"why": "brief provenance or reasoning"}}
                }}
              ],
              "_meta": {{
                "planner": "coordinator",
                "provenance": "explain how the plan was produced"
              }}
            }}

            Rules:
            - Every task must include title, details, dependencies, required_capabilities, priority, and _meta.
            - Dependencies must reference other task titles from the same response.
            - Keep permissions/filesystem/terminal access constrained to the upstream cwd.
            - Use assignee_hints and acceptable_models only when you have a concrete reason.
            - Prefer the smallest task graph that still covers implementation and validation.

            User request:
            {user_prompt.strip()}
            """
        ).strip()

    def _emit_update(self, update: dict[str, Any], emit_update: SessionUpdateCallback | None) -> None:
        if emit_update is None:
            return
        emit_update(update)

    def _final_summary(self, tasks: list[PlanTask], *, cancelled: bool = False) -> str:
        if cancelled:
            return "Turn cancelled."
        completed = [task for task in tasks if task.status == TaskStatus.COMPLETED]
        return f"Completed {len(completed)}/{len(tasks)} orchestrated tasks."

    def _register_turn(self, session_id: str, turn_id: str) -> threading.Event:
        with self._turn_control_lock:
            event = threading.Event()
            self._turn_cancel_events[(session_id, turn_id)] = event
            self._active_turn_by_session[session_id] = turn_id
            return event

    def _clear_turn_registration(self, session_id: str, turn_id: str) -> None:
        with self._turn_control_lock:
            self._turn_cancel_events.pop((session_id, turn_id), None)
            if self._active_turn_by_session.get(session_id) == turn_id:
                self._active_turn_by_session.pop(session_id, None)

    def _set_turn_cancelled(self, session_id: str, turn_id: str) -> None:
        with self._turn_control_lock:
            event = self._turn_cancel_events.get((session_id, turn_id))
        if event is not None:
            event.set()

    def _require_session(self, session_id: str) -> SessionSnapshot:
        snapshot = self.store.load(session_id)
        if snapshot is None:
            raise KeyError(f"Unknown session: {session_id}")
        return snapshot
