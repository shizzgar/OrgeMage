from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
import time
import uuid


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class ModelOption:
    value: str
    name: str
    description: str = ""


@dataclass(slots=True)
class AgentCapabilities:
    supports_terminal: bool = False
    supports_filesystem: bool = False
    supports_permissions: bool = True
    supports_plan_updates: bool = True
    supports_images: bool = False
    supports_mcp: bool = False
    commands: list[str] = field(default_factory=list)

    def score_for_task(self, task: "PlanTask") -> int:
        score = 0
        if task.required_capabilities.get("needsTerminal") and self.supports_terminal:
            score += 4
        if task.required_capabilities.get("needsFilesystem") and self.supports_filesystem:
            score += 4
        if task.required_capabilities.get("needsPermissions") and self.supports_permissions:
            score += 2
        if task.required_capabilities.get("needsMcp") and self.supports_mcp:
            score += 2
        preferred_commands = task.required_capabilities.get("commands", [])
        score += len(set(preferred_commands).intersection(self.commands))
        return score


@dataclass(slots=True)
class DownstreamAgentConfig:
    agent_id: str
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    models: list[ModelOption] = field(default_factory=list)
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    description: str = ""
    default_model: str | None = None
    runtime: str = "acp"
    metadata: dict[str, Any] = field(default_factory=dict)

    def composite_model_values(self) -> dict[str, ModelOption]:
        result: dict[str, ModelOption] = {}
        for option in self.models:
            result[f"{self.agent_id}::{option.value}"] = option
        return result

    def resolve_model(self, composite_value: str) -> str | None:
        prefix = f"{self.agent_id}::"
        if composite_value.startswith(prefix):
            return composite_value[len(prefix) :]
        return None


@dataclass(slots=True)
class DownstreamNegotiatedState:
    agent_id: str
    agent_info: dict[str, Any] = field(default_factory=dict)
    agent_capabilities: dict[str, Any] = field(default_factory=dict)
    auth_methods: list[Any] = field(default_factory=list)
    protocol_version: int | None = None
    session_capabilities: dict[str, dict[str, Any]] = field(default_factory=dict)
    config_options: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def load_session_supported(self) -> bool:
        return bool(self.agent_capabilities.get("loadSession") or self.agent_capabilities.get("load_session"))

    def record_session(
        self,
        *,
        session_id: str,
        capabilities: dict[str, Any] | None,
        config_options: list[dict[str, Any]] | None,
    ) -> None:
        self.session_capabilities[session_id] = capabilities or {}
        self.config_options[session_id] = config_options or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_info": self.agent_info,
            "agent_capabilities": self.agent_capabilities,
            "auth_methods": self.auth_methods,
            "protocol_version": self.protocol_version,
            "session_capabilities": self.session_capabilities,
            "config_options": self.config_options,
        }


@dataclass(slots=True)
class PlanTask:
    title: str
    details: str
    required_capabilities: dict[str, Any] = field(default_factory=dict)
    acceptable_models: list[str] = field(default_factory=list)
    dependency_ids: list[str] = field(default_factory=list)
    assignee_hints: list[str] = field(default_factory=list)
    _meta: dict[str, Any] = field(default_factory=dict)
    assignee: str | None = None
    task_id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:12]}")
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(slots=True)
class ToolEvent:
    tool_call_id: str
    title: str
    kind: str
    status: TaskStatus
    content: str = ""
    locations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "toolCallId": self.tool_call_id,
            "title": self.title,
            "kind": self.kind,
            "status": self.status.value,
            "content": self.content,
            "locations": self.locations,
        }


@dataclass(slots=True)
class DownstreamSessionMapping:
    agent_id: str
    downstream_session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OrchestrationTurnState:
    turn_id: str
    status: str
    stop_reason: str | None = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskExecutionState:
    task_id: str
    title: str
    details: str
    parent_turn_id: str | None = None
    assignee: str | None = None
    dependency_state: str = "ready"
    plan_metadata: dict[str, Any] = field(default_factory=dict)
    required_capabilities: dict[str, Any] = field(default_factory=dict)
    acceptable_models: list[str] = field(default_factory=list)
    dependency_ids: list[str] = field(default_factory=list)
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    output: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_plan_task(cls, task: PlanTask, *, parent_turn_id: str | None = None) -> "TaskExecutionState":
        return cls(
            task_id=task.task_id,
            title=task.title,
            details=task.details,
            parent_turn_id=parent_turn_id,
            assignee=task.assignee,
            dependency_state="blocked" if task.dependency_ids else "ready",
            plan_metadata={
                **({"assignee_hints": list(task.assignee_hints)} if task.assignee_hints else {}),
                **({"_meta": dict(task._meta)} if task._meta else {}),
            },
            required_capabilities=dict(task.required_capabilities),
            acceptable_models=list(task.acceptable_models),
            dependency_ids=list(task.dependency_ids),
            priority=task.priority,
            status=task.status,
            output=task.output,
        )

    def apply_to_plan_task(self) -> PlanTask:
        return PlanTask(
            title=self.title,
            details=self.details,
            required_capabilities=dict(self.required_capabilities),
            acceptable_models=list(self.acceptable_models),
            dependency_ids=list(self.dependency_ids),
            assignee_hints=list(self.plan_metadata.get("assignee_hints", [])),
            _meta=dict(self.plan_metadata.get("_meta", {})),
            assignee=self.assignee,
            task_id=self.task_id,
            priority=self.priority,
            status=self.status,
            output=self.output,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(slots=True)
class TerminalMapping:
    upstream_terminal_id: str
    downstream_terminal_id: str
    owner_task_id: str | None = None
    owner_agent_id: str | None = None
    refcount: int = 1
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PermissionRequestState:
    request_id: str
    owner_task_id: str | None = None
    status: str = "requested"
    decision: str | None = None
    requested_at: float = field(default_factory=time.time)
    decided_at: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TraceCorrelationState:
    trace_key: str
    turn_id: str | None = None
    task_id: str | None = None
    parent_trace_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SessionSnapshot:
    session_id: str
    cwd: str
    selected_model: str | None = None
    coordinator_agent_id: str | None = None
    title: str = "OrgeMage Session"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    downstream_session_mappings: list[DownstreamSessionMapping] = field(default_factory=list)
    turns: list[OrchestrationTurnState] = field(default_factory=list)
    task_states: list[TaskExecutionState] = field(default_factory=list)
    terminal_mappings: list[TerminalMapping] = field(default_factory=list)
    permission_requests: list[PermissionRequestState] = field(default_factory=list)
    trace_metadata: list[TraceCorrelationState] = field(default_factory=list)

    @property
    def task_graph(self) -> list[dict[str, Any]]:
        return [task.apply_to_plan_task().to_dict() for task in self.task_states]

    @task_graph.setter
    def task_graph(self, value: list[dict[str, Any]]) -> None:
        states: list[TaskExecutionState] = []
        for index, task_payload in enumerate(value):
            status_value = task_payload.get("status", TaskStatus.PENDING.value)
            try:
                status = TaskStatus(status_value)
            except ValueError:
                status = TaskStatus.PENDING
            states.append(
                TaskExecutionState(
                    task_id=str(task_payload.get("task_id", f"legacy-task-{index}")),
                    title=str(task_payload.get("title", task_payload.get("task_id", f"Task {index + 1}"))),
                    details=str(task_payload.get("details", "")),
                    parent_turn_id=task_payload.get("parent_turn_id"),
                    assignee=task_payload.get("assignee"),
                    dependency_state=str(task_payload.get("dependency_state", "blocked" if task_payload.get("dependency_ids") else "ready")),
                    plan_metadata={
                        **dict(task_payload.get("plan_metadata", {})),
                        **({"assignee_hints": list(task_payload.get("assignee_hints", []))} if task_payload.get("assignee_hints") else {}),
                        **({"_meta": dict(task_payload.get("_meta", {}))} if task_payload.get("_meta") else {}),
                    },
                    required_capabilities=dict(task_payload.get("required_capabilities", {})),
                    acceptable_models=list(task_payload.get("acceptable_models", [])),
                    dependency_ids=list(task_payload.get("dependency_ids", [])),
                    priority=int(task_payload.get("priority", 0)),
                    status=status,
                    output=str(task_payload.get("output", "")),
                )
            )
        self.task_states = states

    def downstream_session_map(self) -> dict[str, str]:
        return {mapping.agent_id: mapping.downstream_session_id for mapping in self.downstream_session_mappings}

    def get_downstream_session_id(self, agent_id: str) -> str | None:
        for mapping in self.downstream_session_mappings:
            if mapping.agent_id == agent_id:
                return mapping.downstream_session_id
        return None

    def set_downstream_session_mapping(
        self,
        agent_id: str,
        downstream_session_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> DownstreamSessionMapping:
        now = timestamp or time.time()
        for mapping in self.downstream_session_mappings:
            if mapping.agent_id == agent_id:
                mapping.downstream_session_id = downstream_session_id
                if metadata is not None:
                    mapping.metadata = dict(metadata)
                mapping.updated_at = now
                return mapping
        mapping = DownstreamSessionMapping(
            agent_id=agent_id,
            downstream_session_id=downstream_session_id,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        self.downstream_session_mappings.append(mapping)
        return mapping

    def get_task_state(self, task_id: str) -> TaskExecutionState | None:
        for task_state in self.task_states:
            if task_state.task_id == task_id:
                return task_state
        return None

    def upsert_task_state(self, task_state: TaskExecutionState) -> TaskExecutionState:
        for index, current in enumerate(self.task_states):
            if current.task_id == task_state.task_id:
                self.task_states[index] = task_state
                return task_state
        self.task_states.append(task_state)
        return task_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "cwd": self.cwd,
            "selected_model": self.selected_model,
            "coordinator_agent_id": self.coordinator_agent_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
            "downstream_session_mappings": [mapping.to_dict() for mapping in self.downstream_session_mappings],
            "turns": [turn.to_dict() for turn in self.turns],
            "task_states": [task.to_dict() for task in self.task_states],
            "task_graph": self.task_graph,
            "terminal_mappings": [mapping.to_dict() for mapping in self.terminal_mappings],
            "permission_requests": [request.to_dict() for request in self.permission_requests],
            "trace_metadata": [trace.to_dict() for trace in self.trace_metadata],
        }


@dataclass(slots=True)
class SessionHistoryEntry:
    session_id: str
    title: str
    cwd: str
    selected_model: str | None
    coordinator_agent_id: str | None
    created_at: float
    updated_at: float
    task_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, snapshot: SessionSnapshot) -> "SessionHistoryEntry":
        return cls(
            session_id=snapshot.session_id,
            title=snapshot.title,
            cwd=snapshot.cwd,
            selected_model=snapshot.selected_model,
            coordinator_agent_id=snapshot.coordinator_agent_id,
            created_at=snapshot.created_at,
            updated_at=snapshot.updated_at,
            task_count=len(snapshot.task_states),
            metadata=dict(snapshot.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkerResult:
    task_id: str
    agent_id: str
    status: TaskStatus
    summary: str
    raw_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
