from __future__ import annotations

from dataclasses import dataclass

from .models import DownstreamAgentConfig, ModelOption


@dataclass(slots=True)
class ResolvedModel:
    composite_value: str
    agent: DownstreamAgentConfig
    option: ModelOption


class FederatedModelCatalog:
    def __init__(self, agents: list[DownstreamAgentConfig]) -> None:
        self._agents = {agent.agent_id: agent for agent in agents}

    @property
    def agents(self) -> list[DownstreamAgentConfig]:
        return list(self._agents.values())

    def northbound_model_options(self) -> list[dict[str, str]]:
        options: list[dict[str, str]] = []
        for agent in self._agents.values():
            for composite_value, option in agent.composite_model_values().items():
                label = f"{option.name} ({agent.name})"
                description = option.description or agent.description
                options.append(
                    {
                        "value": composite_value,
                        "name": label,
                        "description": description,
                        "category": "model",
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
        for option in agent.models:
            if option.value == raw_model:
                return ResolvedModel(composite_value=composite_value, agent=agent, option=option)
        raise KeyError(f"Unknown model '{raw_model}' for agent '{agent_id}'")
