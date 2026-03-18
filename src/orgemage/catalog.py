from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

from .models import DownstreamAgentConfig, ModelOption


@dataclass(slots=True)
class ResolvedModel:
    composite_value: str
    agent: DownstreamAgentConfig
    option: ModelOption


@dataclass(slots=True)
class RuntimeCatalogState:
    agent_id: str
    discovered_config_options: list[dict[str, Any]] = field(default_factory=list)
    discovered_model_values: list[ModelOption] = field(default_factory=list)
    discovered_capabilities: dict[str, Any] = field(default_factory=dict)
    command_advertisements: list[str] = field(default_factory=list)
    last_refresh_started_at: float | None = None
    last_refresh_succeeded_at: float | None = None
    last_refresh_failed_at: float | None = None
    last_refresh_error: str | None = None
    discovery_reliable: bool = False
    refresh_required: bool = True

    def begin_refresh(self, timestamp: float | None = None) -> None:
        self.last_refresh_started_at = timestamp or time.time()
        self.last_refresh_error = None

    def record_success(
        self,
        *,
        config_options: list[dict[str, Any]],
        model_values: list[ModelOption],
        capabilities: dict[str, Any],
        command_advertisements: list[str],
        timestamp: float | None = None,
    ) -> None:
        now = timestamp or time.time()
        self.discovered_config_options = [dict(option) for option in config_options]
        self.discovered_model_values = list(model_values)
        self.discovered_capabilities = dict(capabilities)
        self.command_advertisements = list(command_advertisements)
        self.last_refresh_succeeded_at = now
        self.last_refresh_error = None
        self.discovery_reliable = True
        self.refresh_required = False

    def record_failure(self, error: str, timestamp: float | None = None) -> None:
        self.last_refresh_failed_at = timestamp or time.time()
        self.last_refresh_error = error
        self.discovery_reliable = False
        self.refresh_required = False

    def mark_refresh_required(self) -> None:
        self.refresh_required = True

    def effective_model_values(self, bootstrap_models: list[ModelOption]) -> list[ModelOption]:
        if self.discovered_model_values:
            return list(self.discovered_model_values)
        return list(bootstrap_models)


class FederatedModelCatalog:
    def __init__(self, agents: list[DownstreamAgentConfig]) -> None:
        self._agents = {agent.agent_id: agent for agent in agents}
        self._runtime_state = {
            agent.agent_id: RuntimeCatalogState(agent_id=agent.agent_id) for agent in agents
        }

    @property
    def agents(self) -> list[DownstreamAgentConfig]:
        return list(self._agents.values())

    def runtime_state(self, agent_id: str) -> RuntimeCatalogState:
        return self._runtime_state[agent_id]

    def record_discovery(
        self,
        *,
        agent_id: str,
        config_options: list[dict[str, Any]],
        capabilities: dict[str, Any],
        command_advertisements: list[str],
        timestamp: float | None = None,
    ) -> RuntimeCatalogState:
        agent = self._agents[agent_id]
        state = self._runtime_state[agent_id]
        state.begin_refresh(timestamp)
        discovered_models = _extract_discovered_model_options(config_options, agent)
        state.record_success(
            config_options=config_options,
            model_values=discovered_models,
            capabilities=capabilities,
            command_advertisements=command_advertisements,
            timestamp=timestamp,
        )
        return state

    def record_discovery_failure(
        self,
        agent_id: str,
        error: str,
        timestamp: float | None = None,
    ) -> RuntimeCatalogState:
        state = self._runtime_state[agent_id]
        state.begin_refresh(timestamp)
        state.record_failure(error, timestamp=timestamp)
        return state

    def mark_refresh_required(self, agent_id: str) -> None:
        self._runtime_state[agent_id].mark_refresh_required()

    def northbound_model_options(self) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        for agent in self._agents.values():
            state = self._runtime_state[agent.agent_id]
            discovered = state.effective_model_values(agent.models)
            source = "discovery" if state.discovered_model_values else "bootstrap"
            trusted = state.discovery_reliable if source == "discovery" else False
            for option in discovered:
                composite_value = f"{agent.agent_id}::{option.value}"
                label = f"{option.name} ({agent.name})"
                description = option.description or agent.description
                options.append(
                    {
                        "value": composite_value,
                        "name": label,
                        "description": description,
                        "category": "model",
                        "metadata": {
                            "agent_id": agent.agent_id,
                            "source": source,
                            "trusted": trusted,
                            "degraded": source == "bootstrap",
                            "last_refresh_started_at": state.last_refresh_started_at,
                            "last_refresh_succeeded_at": state.last_refresh_succeeded_at,
                            "last_refresh_failed_at": state.last_refresh_failed_at,
                            "last_refresh_error": state.last_refresh_error,
                        },
                    }
                )
        options.sort(key=lambda item: item["name"].lower())
        return options

    def resolve(self, composite_value: str) -> ResolvedModel:
        agent_id, _, raw_model = composite_value.partition("::")
        if not agent_id or not raw_model:
            raise KeyError(f"Invalid composite model value: {composite_value}")
        agent = self._agents.get(agent_id)
        if agent is None:
            raise KeyError(f"Unknown agent: {agent_id}")
        state = self._runtime_state[agent_id]
        for option in state.effective_model_values(agent.models):
            if option.value == raw_model:
                return ResolvedModel(composite_value=composite_value, agent=agent, option=option)
        raise KeyError(f"Unknown model '{raw_model}' for agent '{agent_id}'")


def _extract_discovered_model_options(
    config_options: list[dict[str, Any]],
    agent: DownstreamAgentConfig,
) -> list[ModelOption]:
    discovered: list[ModelOption] = []
    seen_values: set[str] = set()
    for option in config_options:
        if option.get("category") != "model":
            continue
        values = option.get("options") or option.get("values") or []
        for value_payload in values:
            value = str(value_payload.get("value", "")).strip()
            if not value or value in seen_values:
                continue
            discovered.append(
                ModelOption(
                    value=value,
                    name=str(value_payload.get("name") or value),
                    description=str(value_payload.get("description") or option.get("description") or agent.description),
                )
            )
            seen_values.add(value)
    return discovered
