from __future__ import annotations

from typing import Callable

from ..catalog import FederatedModelCatalog
from ..codex_app_server import CodexAppServerConnector
from ..debug import debug_event, get_logger
from ..downstream import MockDownstreamClient
from ..models import DownstreamAgentConfig, SessionSnapshot, PlanTask, TaskStatus, WorkerResult
from ..state import SQLiteSessionStore
from .downstream_client import AcpDownstreamConnector, DownstreamConnector, DownstreamPromptResult

ConnectorFactory = Callable[[DownstreamAgentConfig], DownstreamConnector]
_LOG = get_logger(__name__)


class _MockConnectorAdapter:
    def __init__(self, agent: DownstreamAgentConfig) -> None:
        self.agent = agent
        self.negotiated_state = None
        self._client = MockDownstreamClient()

    def discover_catalog(self, *, force: bool = False) -> dict[str, object]:
        return {
            "agent_id": self.agent.agent_id,
            "config_options": [
                {
                    "id": "model",
                    "name": "Model",
                    "category": "model",
                    "type": "select",
                    "options": [
                        {
                            "value": option.value,
                            "name": option.name,
                            "description": option.description,
                        }
                        for option in self.agent.models
                    ],
                }
            ],
            "capabilities": {
                "supports_terminal": self.agent.capabilities.supports_terminal,
                "supports_filesystem": self.agent.capabilities.supports_filesystem,
                "supports_permissions": self.agent.capabilities.supports_permissions,
                "supports_plan_updates": self.agent.capabilities.supports_plan_updates,
                "supports_images": self.agent.capabilities.supports_images,
                "supports_mcp": self.agent.capabilities.supports_mcp,
            },
            "command_advertisements": list(self.agent.capabilities.commands),
        }

    def mark_catalog_refresh_required(self) -> None:
        return None

    def execute_task(
        self,
        *,
        orchestrator_session_id: str,
        downstream_session_id: str | None,
        cwd: str,
        mcp_servers: list[dict[str, object]] | list[object] | None,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
    ) -> DownstreamPromptResult:
        del mcp_servers
        result = self._client.execute_task(
            session_id=orchestrator_session_id,
            task=task,
            coordinator_prompt=coordinator_prompt,
            selected_model=selected_model,
            agent=self.agent,
        )
        return DownstreamPromptResult(
            downstream_session_id=downstream_session_id or f"mock-{self.agent.agent_id}-{orchestrator_session_id}",
            status=result.status,
            summary=result.summary,
            raw_output=result.raw_output,
            response={"mock": True},
        )

    def cancel(self, downstream_session_id: str) -> None:
        return None


class DownstreamConnectorManager:
    def __init__(
        self,
        agents: list[DownstreamAgentConfig],
        connector_factory: ConnectorFactory | None = None,
        *,
        store: SQLiteSessionStore | None = None,
        headless_policy: Callable[[str, dict[str, object]], str] | None = None,
    ) -> None:
        self._agents = {agent.agent_id: agent for agent in agents}
        self._connector_factory = connector_factory or self._build_connector
        self._connectors: dict[str, DownstreamConnector] = {}
        self._store = store
        self._headless_policy = headless_policy
        self._upstream_client_connection: object | None = None
        self._upstream_capabilities: object | None = None
        self._active_planning_sessions: dict[str, dict[str, str]] = {}

    def bind_upstream_client_connection(self, connection: object | None, *, capabilities: object | None = None) -> None:
        self._upstream_client_connection = connection
        self._upstream_capabilities = capabilities

    def get_connector(self, agent_id: str) -> DownstreamConnector:
        connector = self._connectors.get(agent_id)
        if connector is None:
            agent = self._agents[agent_id]
            connector = self._connector_factory(agent)
            debug_event(_LOG, "connector.manager.create", agent_id=agent_id, runtime=agent.runtime)
            self._connectors[agent_id] = connector
        return connector

    def refresh_catalog(
        self,
        catalog: FederatedModelCatalog,
        *,
        agent_id: str | None = None,
        force: bool = False,
    ) -> None:
        target_ids = [agent_id] if agent_id else list(self._agents.keys())
        for current_agent_id in target_ids:
            connector = self.get_connector(current_agent_id)
            try:
                debug_event(_LOG, "connector.manager.refresh_catalog", agent_id=current_agent_id, force=force)
                payload = connector.discover_catalog(force=force)
            except Exception as exc:
                catalog.record_discovery_failure(current_agent_id, str(exc))
                continue
            catalog.record_discovery(
                agent_id=current_agent_id,
                config_options=list(payload.get("config_options", [])),
                capabilities=dict(payload.get("capabilities", {})),
                command_advertisements=list(payload.get("command_advertisements", [])),
            )

    def execute_task(
        self,
        *,
        session: SessionSnapshot,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
        agent: DownstreamAgentConfig,
    ) -> WorkerResult:
        connector = self.get_connector(agent.agent_id)
        is_planning_task = task._meta.get("phase") == "planning"
        
        # Try to get session ID: 
        # 1. From active planning sessions (if we just finished planning)
        # 2. From the session snapshot (if it was already saved)
        downstream_session_id = session.get_downstream_session_id(agent.agent_id)
        if downstream_session_id is None and not is_planning_task:
            downstream_session_id = self._active_planning_sessions.get(session.session_id, {}).get(agent.agent_id)

        debug_event(_LOG, "connector.manager.execute", session_id=session.session_id, task_id=task.task_id, agent_id=agent.agent_id, downstream_session_id=downstream_session_id)
        try:
            prompt_result = connector.execute_task(
                orchestrator_session_id=session.session_id,
                downstream_session_id=downstream_session_id,
                cwd=session.cwd,
                mcp_servers=session.mcp_servers,
                task=task,
                coordinator_prompt=coordinator_prompt,
                selected_model=selected_model,
            )
        except Exception as exc:
            summary = _exception_summary(exc)
            metadata = {
                "promptMetadata": dict(task._meta),
                "error": {
                    "type": type(exc).__name__,
                    "summary": summary,
                    "code": getattr(exc, "code", None),
                    "method": getattr(exc, "method", None),
                    "message": getattr(exc, "message", None),
                    "data": getattr(exc, "data", None),
                },
            }
            if downstream_session_id is not None:
                metadata["downstream_session_id"] = downstream_session_id
            debug_event(
                _LOG,
                "connector.manager.execute.failed",
                session_id=session.session_id,
                task_id=task.task_id,
                agent_id=agent.agent_id,
                downstream_session_id=downstream_session_id,
                error_summary=summary,
                error_type=type(exc).__name__,
            )
            return WorkerResult(
                task_id=task.task_id,
                agent_id=agent.agent_id,
                status=TaskStatus.FAILED,
                summary=summary,
                raw_output=summary,
                metadata=metadata,
            )
        if not is_planning_task:
            session.set_downstream_session_mapping(agent.agent_id, prompt_result.downstream_session_id)
            if self._store is not None:
                self._store.save_downstream_session_mapping(
                    session.session_id,
                    agent.agent_id,
                    prompt_result.downstream_session_id,
                )
        else:
            self._active_planning_sessions.setdefault(session.session_id, {})[agent.agent_id] = prompt_result.downstream_session_id
        negotiated = getattr(connector, "negotiated_state", None)
        if negotiated is not None:
            session.metadata.setdefault("downstream_negotiated", {})[agent.agent_id] = negotiated.to_dict()
        return WorkerResult(
            task_id=task.task_id,
            agent_id=agent.agent_id,
            status=prompt_result.status,
            summary=prompt_result.summary,
            raw_output=prompt_result.raw_output,
            metadata={
                "downstream_session_id": prompt_result.downstream_session_id,
                "updates": prompt_result.updates,
                "response": prompt_result.response,
                "promptMetadata": dict(task._meta),
            },
        )

    def cancel_session(self, session: SessionSnapshot, agent_id: str | None = None) -> None:
        mappings = session.downstream_session_map()
        planning_mappings = self._active_planning_sessions.get(session.session_id, {})
        target_ids = [agent_id] if agent_id else sorted(set(mappings.keys()) | set(planning_mappings.keys()))
        for current_agent_id in target_ids:
            downstream_session_id = mappings.get(current_agent_id) or planning_mappings.get(current_agent_id)
            if downstream_session_id is None:
                continue
            debug_event(_LOG, "connector.manager.cancel", session_id=session.session_id, agent_id=current_agent_id, downstream_session_id=downstream_session_id)
            self.get_connector(current_agent_id).cancel(downstream_session_id)

    def mark_catalog_refresh_required(self, agent_id: str) -> None:
        connector = self.get_connector(agent_id)
        connector.mark_catalog_refresh_required()

    def _build_connector(self, agent: DownstreamAgentConfig) -> DownstreamConnector:
        if agent.runtime == "mock":
            return _MockConnectorAdapter(agent)
        if agent.runtime == "codex-app-server":
            return CodexAppServerConnector(
                agent,
                headless_policy=self._headless_policy,
            )
        return AcpDownstreamConnector(
            agent,
            store=self._store,
            upstream_client_getter=lambda: self._upstream_client_connection,
            upstream_capabilities_getter=lambda: self._upstream_capabilities,
            headless_policy=self._headless_policy,
        )


def _exception_summary(exc: Exception) -> str:
    data = getattr(exc, "data", None)
    if isinstance(data, dict):
        details = data.get("details")
        if isinstance(details, str) and details.strip():
            return details.strip()
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    rendered = str(exc).strip()
    if rendered:
        return rendered
    return type(exc).__name__
