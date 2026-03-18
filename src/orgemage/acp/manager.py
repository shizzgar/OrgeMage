from __future__ import annotations

from typing import Callable

from ..downstream import MockDownstreamClient
from ..models import DownstreamAgentConfig, SessionSnapshot, PlanTask, WorkerResult
from .downstream_client import AcpDownstreamConnector, DownstreamConnector, DownstreamPromptResult

ConnectorFactory = Callable[[DownstreamAgentConfig], DownstreamConnector]


class _MockConnectorAdapter:
    def __init__(self, agent: DownstreamAgentConfig) -> None:
        self.agent = agent
        self.negotiated_state = None
        self._client = MockDownstreamClient()

    def execute_task(
        self,
        *,
        orchestrator_session_id: str,
        downstream_session_id: str | None,
        cwd: str,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
    ) -> DownstreamPromptResult:
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
    ) -> None:
        self._agents = {agent.agent_id: agent for agent in agents}
        self._connector_factory = connector_factory or self._build_connector
        self._connectors: dict[str, DownstreamConnector] = {}

    def get_connector(self, agent_id: str) -> DownstreamConnector:
        connector = self._connectors.get(agent_id)
        if connector is None:
            agent = self._agents[agent_id]
            connector = self._connector_factory(agent)
            self._connectors[agent_id] = connector
        return connector

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
        downstream_session_id = session.get_downstream_session_id(agent.agent_id)
        prompt_result = connector.execute_task(
            orchestrator_session_id=session.session_id,
            downstream_session_id=downstream_session_id,
            cwd=session.cwd,
            task=task,
            coordinator_prompt=coordinator_prompt,
            selected_model=selected_model,
        )
        session.set_downstream_session_mapping(agent.agent_id, prompt_result.downstream_session_id)
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
            },
        )

    def cancel_session(self, session: SessionSnapshot, agent_id: str | None = None) -> None:
        mappings = session.downstream_session_map()
        target_ids = [agent_id] if agent_id else list(mappings.keys())
        for current_agent_id in target_ids:
            downstream_session_id = mappings.get(current_agent_id)
            if downstream_session_id is None:
                continue
            self.get_connector(current_agent_id).cancel(downstream_session_id)

    def _build_connector(self, agent: DownstreamAgentConfig) -> DownstreamConnector:
        if agent.runtime == "mock":
            return _MockConnectorAdapter(agent)
        return AcpDownstreamConnector(agent)
