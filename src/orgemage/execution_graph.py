from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from .models import (
    DownstreamAgentConfig,
    OrchestrationTurnState,
    PlanTask,
    SessionSnapshot,
    TaskExecutionState,
    TaskStatus,
    ToolEvent,
    TraceCorrelationState,
    WorkerResult,
)

PersistTaskState = Callable[[PlanTask, str | None, WorkerResult | None, str | None], TaskExecutionState]
ExecuteTask = Callable[[SessionSnapshot, PlanTask, str, str, DownstreamAgentConfig], WorkerResult]
NormalizeWorkerUpdates = Callable[[SessionSnapshot, PlanTask, DownstreamAgentConfig, list[dict[str, Any]]], list[dict[str, Any]]]
EmitPlanUpdate = Callable[[], None]
EmitGenericUpdate = Callable[[dict[str, Any]], None]
EmitToolEvent = Callable[[PlanTask, ToolEvent], None]
CreateToolEvent = Callable[[PlanTask], ToolEvent]
CreateFinishedToolEvent = Callable[[PlanTask, TaskStatus, str], ToolEvent]
PersistTrace = Callable[[str, TraceCorrelationState], None]
SaveDownstreamMapping = Callable[[str, str, str], None]
IsCancelRequested = Callable[[], bool]
CancelActiveWork = Callable[[], None]


@dataclass(slots=True)
class _RuntimeNode:
    task: PlanTask
    state: TaskExecutionState


class ExecutionGraphRunner:
    def __init__(
        self,
        *,
        snapshot: SessionSnapshot,
        tasks: list[PlanTask],
        turn: OrchestrationTurnState,
        coordinator_agent: DownstreamAgentConfig,
        agents: list[DownstreamAgentConfig],
        selected_model: str,
        coordinator_prompt: str,
        persist_task_state: PersistTaskState,
        execute_task: ExecuteTask,
        normalize_worker_updates: NormalizeWorkerUpdates,
        emit_plan_update: EmitPlanUpdate,
        emit_generic_update: EmitGenericUpdate,
        emit_tool_event: EmitToolEvent,
        create_started_event: CreateToolEvent,
        create_finished_event: CreateFinishedToolEvent,
        persist_trace: PersistTrace,
        save_downstream_mapping: SaveDownstreamMapping,
        is_cancel_requested: IsCancelRequested,
        cancel_active_work: CancelActiveWork,
    ) -> None:
        self.snapshot = snapshot
        self.tasks = tasks
        self.turn = turn
        self.coordinator_agent = coordinator_agent
        self.agents_by_id = {agent.agent_id: agent for agent in agents}
        self.selected_model = selected_model
        self.coordinator_prompt = coordinator_prompt
        self.persist_task_state = persist_task_state
        self.execute_task = execute_task
        self.normalize_worker_updates = normalize_worker_updates
        self.emit_plan_update = emit_plan_update
        self.emit_generic_update = emit_generic_update
        self.emit_tool_event = emit_tool_event
        self.create_started_event = create_started_event
        self.create_finished_event = create_finished_event
        self.persist_trace = persist_trace
        self.save_downstream_mapping = save_downstream_mapping
        self.is_cancel_requested = is_cancel_requested
        self.cancel_active_work = cancel_active_work
        self.events: list[ToolEvent] = []
        self._nodes: dict[str, _RuntimeNode] = {}
        self._inflight: dict[asyncio.Task[WorkerResult], PlanTask] = {}
        self._stop_scheduling = False

        for task in self.tasks:
            state = self.snapshot.get_task_state(task.task_id) or TaskExecutionState.from_plan_task(task, parent_turn_id=turn.turn_id)
            state.assignee = task.assignee
            self._nodes[task.task_id] = _RuntimeNode(task=task, state=state)

    def run(self) -> list[ToolEvent]:
        asyncio.run(self._run())
        return list(self.events)

    async def _run(self) -> None:
        self._refresh_dependency_states()
        self.emit_plan_update()
        while True:
            if self.is_cancel_requested():
                await self._handle_cancellation()
                break
            launched = await self._launch_ready_tasks()
            if not self._inflight:
                if not launched:
                    break
                continue
            completed, _ = await asyncio.wait(self._inflight.keys(), timeout=0.05, return_when=asyncio.FIRST_COMPLETED)
            if not completed:
                continue
            for future in completed:
                task = self._inflight.pop(future)
                if future.cancelled():
                    continue
                result = future.result()
                self._handle_result(task, result)

    async def _launch_ready_tasks(self) -> bool:
        launched = False
        for task in self.tasks:
            if self._stop_scheduling:
                break
            if task.status != TaskStatus.PENDING:
                continue
            decision = self._readiness(task)
            if decision == "blocked":
                continue
            if decision == "cancelled":
                task.status = TaskStatus.CANCELLED
                self.persist_task_state(task, "dependency_failure", None, "blocked")
                self._refresh_dependency_states()
                self.emit_plan_update()
                event = self.create_finished_event(task, TaskStatus.CANCELLED, "Cancelled because a dependency failed or was cancelled.")
                self.events.append(event)
                self.emit_tool_event(task, event)
                launched = True
                continue

            task.status = TaskStatus.IN_PROGRESS
            self.persist_task_state(task, None, None, "ready")
            self._refresh_dependency_states()
            self.emit_plan_update()
            started_event = self.create_started_event(task)
            self.events.append(started_event)
            self.emit_tool_event(task, started_event)
            agent = self.agents_by_id[task.assignee or self.coordinator_agent.agent_id]
            future = asyncio.create_task(
                asyncio.to_thread(
                    self.execute_task,
                    self.snapshot,
                    task,
                    self.coordinator_prompt,
                    self.selected_model,
                    agent,
                )
            )
            self._inflight[future] = task
            launched = True
        return launched

    def _handle_result(self, task: PlanTask, result: WorkerResult) -> None:
        if self.is_cancel_requested():
            return
        task.status = result.status
        task.output = result.summary
        self.persist_task_state(task, None, result, self._dependency_state(task))

        downstream_session_id = result.metadata.get("downstream_session_id")
        if isinstance(downstream_session_id, str):
            self.save_downstream_mapping(self.snapshot.session_id, result.agent_id, downstream_session_id)

        negotiated = self.snapshot.metadata.get("downstream_negotiated", {}).get(result.agent_id)
        if negotiated is not None:
            self.persist_trace(
                self.snapshot.session_id,
                TraceCorrelationState(
                    trace_key=f"negotiated:{result.agent_id}",
                    task_id=task.task_id,
                    turn_id=self.turn.turn_id,
                    metadata={"agent_id": result.agent_id, "negotiated": negotiated},
                ),
            )

        agent = self.agents_by_id[result.agent_id]
        for update in self.normalize_worker_updates(self.snapshot, task, agent, list(result.metadata.get("updates", []))):
            self.emit_generic_update(update)

        self._apply_failure_policy(task)
        self._refresh_dependency_states()
        self.emit_plan_update()
        event = self.create_finished_event(task, result.status, result.summary)
        self.events.append(event)
        self.emit_tool_event(task, event)

    async def _handle_cancellation(self) -> None:
        self._stop_scheduling = True
        self.cancel_active_work()
        for future in self._inflight:
            future.cancel()
        for task in self.tasks:
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                continue
            task.status = TaskStatus.CANCELLED
            self.persist_task_state(task, "turn_cancelled", None, self._dependency_state(task))
            event = self.create_finished_event(task, TaskStatus.CANCELLED, "Task cancelled.")
            self.events.append(event)
            self.emit_tool_event(task, event)
        self._refresh_dependency_states()
        self.emit_plan_update()
        if self._inflight:
            await asyncio.gather(*self._inflight.keys(), return_exceptions=True)
        self._inflight.clear()

    def _refresh_dependency_states(self) -> None:
        for task in self.tasks:
            dependency_state = self._dependency_state(task)
            self._nodes[task.task_id].state.dependency_state = dependency_state
            self.persist_task_state(task, None, None, dependency_state)

    def _dependency_state(self, task: PlanTask) -> str:
        if not task.dependency_ids:
            return "ready"
        statuses = self._dependency_statuses(task)
        if len(statuses) != len(task.dependency_ids):
            return "blocked"
        if all(status == TaskStatus.COMPLETED for status in statuses):
            return "ready"
        if any(status in {TaskStatus.FAILED, TaskStatus.CANCELLED} for status in statuses):
            return "ready" if self._dependency_failure_policy(task) == "continue" else "blocked"
        return "blocked"

    def _readiness(self, task: PlanTask) -> str:
        if not task.dependency_ids:
            return "ready"
        statuses = self._dependency_statuses(task)
        if len(statuses) != len(task.dependency_ids):
            return "blocked"
        if all(status == TaskStatus.COMPLETED for status in statuses):
            return "ready"
        if any(status in {TaskStatus.FAILED, TaskStatus.CANCELLED} for status in statuses):
            return "ready" if self._dependency_failure_policy(task) == "continue" else "cancelled"
        return "blocked"

    def _dependency_statuses(self, task: PlanTask) -> list[TaskStatus]:
        return [self._nodes[dependency_id].task.status for dependency_id in task.dependency_ids if dependency_id in self._nodes]

    def _dependency_failure_policy(self, task: PlanTask) -> str:
        value = task._meta.get("dependency_failure_policy") or task._meta.get("dependencyFailurePolicy")
        return value if isinstance(value, str) and value == "continue" else "block"

    def _apply_failure_policy(self, task: PlanTask) -> None:
        if task.status not in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
            return
        if self._failure_policy(task) != "fail_fast":
            return
        self._stop_scheduling = True
        for candidate in self.tasks:
            if candidate.task_id == task.task_id or candidate.status != TaskStatus.PENDING:
                continue
            candidate.status = TaskStatus.CANCELLED
            self.persist_task_state(candidate, "fail_fast", None, "blocked")

    def _failure_policy(self, task: PlanTask) -> str:
        raw = task._meta.get("failure_policy") or task._meta.get("failurePolicy")
        if raw in {"fail_fast", "continue"}:
            return str(raw)
        if task._meta.get("coordinator_critical"):
            return "fail_fast"
        if task.required_capabilities.get("planner"):
            return "fail_fast"
        if task.assignee == self.coordinator_agent.agent_id and task._meta.get("execution_role") == "coordinator":
            return "fail_fast"
        return "continue"
