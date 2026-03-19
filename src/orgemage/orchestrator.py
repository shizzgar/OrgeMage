from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import textwrap
import uuid
from typing import Any, Callable

from .acp.manager import DownstreamConnectorManager
from .catalog import FederatedModelCatalog
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
)
from .planning import PlanParseResult, parse_coordinator_plan, synthesize_local_fallback_plan
from .scheduler import Scheduler
from .state import SQLiteSessionStore

SessionUpdateCallback = Callable[[dict[str, Any]], None]


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
                    metadata={"source_update": update},
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
                metadata={
                    "taskId": task.task_id,
                    "source": "downstream_session_update",
                    "workerAgentId": agent.agent_id,
                    "rawUpdate": update,
                },
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
        }

    def delegate_started_event(self, task: PlanTask) -> ToolEvent:
        return ToolEvent(
            tool_call_id=self._orchestrator_tool_id(task),
            title=task.title,
            kind="delegate",
            status=TaskStatus.IN_PROGRESS,
            content=f"Delegating to {task.assignee}.",
            metadata={"taskId": task.task_id, "source": "orchestrator"},
        )

    def delegate_finished_event(self, task: PlanTask, *, status: TaskStatus, content: str) -> ToolEvent:
        return ToolEvent(
            tool_call_id=self._orchestrator_tool_id(task),
            title=task.title,
            kind="delegate",
            status=status,
            content=content,
            metadata={"taskId": task.task_id, "source": "orchestrator"},
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

    def create_session(self, cwd: str, selected_model: str | None = None) -> SessionSnapshot:
        normalized_cwd = str(Path(cwd).resolve())
        snapshot = SessionSnapshot(
            session_id=f"orch-{uuid.uuid4().hex[:12]}",
            cwd=normalized_cwd,
            selected_model=selected_model,
            title=f"OrgeMage: {Path(normalized_cwd).name or normalized_cwd}",
        )
        if selected_model:
            self._refresh_catalog_for_model(selected_model)
            resolved = self.catalog.resolve(selected_model)
            snapshot.coordinator_agent_id = resolved.agent.agent_id
        self.store.save(snapshot)
        return snapshot

    def list_model_options(self) -> list[dict[str, object]]:
        self.connector_manager.refresh_catalog(self.catalog)
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
        self._refresh_catalog_for_model(composite_model)
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

    def orchestrate(
        self,
        session_id: str,
        user_prompt: str,
        *,
        emit_update: SessionUpdateCallback | None = None,
    ) -> None:
        self._execute_turn(session_id, user_prompt, emit_update=emit_update)

    def orchestrate_turn(
        self,
        session_id: str,
        user_prompt: str,
        *,
        emit_update: SessionUpdateCallback | None = None,
    ) -> dict[str, object]:
        updates: list[dict[str, Any]] = []

        def collector(update: dict[str, Any]) -> None:
            updates.append(update)
            if emit_update is not None:
                emit_update(update)

        result = self._execute_turn(session_id, user_prompt, emit_update=collector)
        result["updates"] = updates
        return result

    def _execute_turn(
        self,
        session_id: str,
        user_prompt: str,
        *,
        emit_update: SessionUpdateCallback | None,
    ) -> dict[str, object]:
        snapshot = self._require_session(session_id)
        if not snapshot.selected_model:
            default_option = self.list_model_options()[0]["value"]
            snapshot = self.set_selected_model(session_id, default_option)
        self._refresh_catalog_for_model(snapshot.selected_model or "")
        resolved = self.catalog.resolve(snapshot.selected_model or "")

        turn = OrchestrationTurnState(status="running", turn_id=f"turn-{uuid.uuid4().hex[:12]}")
        snapshot.turns.append(turn)
        self.store.create_or_update_turn_state(snapshot.session_id, turn)

        plan_parse_result, planner_record = self._generate_plan(
            snapshot=snapshot,
            selected_model=resolved.option.value,
            coordinator_agent=resolved.agent,
            user_prompt=user_prompt,
            turn=turn,
        )
        plan = self.scheduler.assign_tasks(plan_parse_result.tasks, self.catalog.agents, resolved.agent.agent_id)
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
        )
        turn.status = "completed"
        turn.stop_reason = "end_turn"
        self.store.create_or_update_turn_state(snapshot.session_id, turn)
        self.store.save(snapshot)
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
            "planning": planner_record,
            "plan": [task.to_dict() for task in plan],
            "global_plan": [task.to_acp_plan_item() for task in plan],
            "tool_events": [event.to_dict() for event in tool_events],
            "summary": summary,
        }
        self._emit_update(self.normalizer.message_update(snapshot, summary, result, plan), emit_update)
        return result

    def _refresh_catalog_for_model(self, composite_model: str) -> None:
        agent_id, _, _ = composite_model.partition("::")
        if agent_id and agent_id in {agent.agent_id for agent in self.catalog.agents}:
            self.connector_manager.refresh_catalog(self.catalog, agent_id=agent_id)

    def cancel(self, session_id: str, agent_id: str | None = None) -> SessionSnapshot:
        snapshot = self._require_session(session_id)
        self.connector_manager.cancel_session(snapshot, agent_id=agent_id)
        snapshot.metadata["cancelled"] = True
        self.store.save(snapshot)
        return snapshot

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
            _meta={"source": "coordinator", "phase": "planning"},
        )
        planning_result = self.connector_manager.execute_task(
            session=snapshot,
            task=planning_task,
            coordinator_prompt=self._coordinator_instruction(user_prompt),
            selected_model=selected_model,
            agent=coordinator_agent,
        )
        raw_output = planning_result.raw_output or planning_result.summary
        parsed_plan = parse_coordinator_plan(raw_output, coordinator_agent_id=coordinator_agent.agent_id)
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
            "validation_errors": list(parsed_plan.errors),
        }
        snapshot.metadata["planning"] = planner_record
        turn.metadata["planning"] = {
            "source": plan_parse_result.normalized_plan.get("_meta", {}).get("source"),
            "planner_task_id": planning_task.task_id,
            "validation_errors": list(parsed_plan.errors),
        }
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
            self._emit_update(self.normalizer.plan_update(snapshot, tasks), emit_update)
            started_event = self.normalizer.delegate_started_event(task)
            events.append(started_event)
            self._emit_update(self.normalizer.tool_call_update(snapshot, task, started_event), emit_update)
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
            rebased_updates = self.normalizer.normalize_worker_updates(
                snapshot=snapshot,
                task=task,
                agent=agent,
                updates=result.metadata.get("updates", []),
            )
            for update in rebased_updates:
                self._emit_update(update, emit_update)
            task.status = result.status
            task.output = result.summary
            task_state = snapshot.get_task_state(task.task_id) or TaskExecutionState.from_plan_task(task, parent_turn_id=turn.turn_id)
            task_state.status = result.status
            task_state.output = result.summary
            task_state.assignee = task.assignee
            task_state.plan_metadata.update(
                {
                    "assignee_hints": list(task.assignee_hints),
                    "_meta": dict(task._meta),
                    "worker_result": {
                        **dict(result.metadata),
                        "raw_output": result.raw_output,
                    },
                }
            )
            self.store.persist_task_update(snapshot.session_id, task_state)
            snapshot.upsert_task_state(task_state)
            self._emit_update(self.normalizer.plan_update(snapshot, tasks), emit_update)
            finished_event = self.normalizer.delegate_finished_event(task, status=result.status, content=result.summary)
            events.append(finished_event)
            self._emit_update(self.normalizer.tool_call_update(snapshot, task, finished_event), emit_update)
        return events

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

    def _final_summary(self, tasks: list[PlanTask]) -> str:
        completed = [task for task in tasks if task.status == TaskStatus.COMPLETED]
        return f"Completed {len(completed)}/{len(tasks)} orchestrated tasks."

    def _require_session(self, session_id: str) -> SessionSnapshot:
        snapshot = self.store.load(session_id)
        if snapshot is None:
            raise KeyError(f"Unknown session: {session_id}")
        return snapshot
