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
class SessionSnapshot:
    session_id: str
    cwd: str
    selected_model: str | None = None
    coordinator_agent_id: str | None = None
    title: str = "OrgeMage Session"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    task_graph: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkerResult:
    task_id: str
    agent_id: str
    status: TaskStatus
    summary: str
    raw_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
