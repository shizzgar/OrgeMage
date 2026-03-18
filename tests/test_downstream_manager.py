from orgemage.acp.manager import DownstreamConnectorManager
from orgemage.catalog import FederatedModelCatalog
from orgemage.models import DownstreamAgentConfig, ModelOption, PlanTask, SessionSnapshot, TaskStatus
from orgemage.models import DownstreamNegotiatedState
from orgemage.acp.downstream_client import DownstreamPromptResult


class RecordingConnector:
    def __init__(self, agent: DownstreamAgentConfig) -> None:
        self.agent = agent
        self.negotiated_state = DownstreamNegotiatedState(
            agent_id=agent.agent_id,
            agent_info={"name": agent.name},
            agent_capabilities={"loadSession": True},
            auth_methods=["none"],
        )
        self.calls: list[dict[str, str | None | bool]] = []

    def discover_catalog(self, *, force: bool = False):
        self.calls.append({"discover": self.agent.agent_id, "force": force})
        return {
            "agent_id": self.agent.agent_id,
            "config_options": [
                {
                    "id": "model",
                    "category": "model",
                    "type": "select",
                    "options": [
                        {"value": "discovered-model", "name": "Discovered Model"},
                    ],
                }
            ],
            "capabilities": {"commands": ["read", "edit"]},
            "command_advertisements": ["read", "edit"],
        }

    def mark_catalog_refresh_required(self) -> None:
        self.calls.append({"mark_refresh": self.agent.agent_id})

    def execute_task(self, **kwargs):
        self.calls.append(
            {
                "orchestrator_session_id": kwargs["orchestrator_session_id"],
                "downstream_session_id": kwargs["downstream_session_id"],
            }
        )
        downstream_session_id = kwargs["downstream_session_id"] or f"downstream-{self.agent.agent_id}"
        self.negotiated_state.record_session(
            session_id=downstream_session_id,
            capabilities={"prompt": True},
            config_options=[{"id": "model", "type": "select"}],
        )
        return DownstreamPromptResult(
            downstream_session_id=downstream_session_id,
            status=TaskStatus.COMPLETED,
            summary=f"completed by {self.agent.agent_id}",
            raw_output="ok",
        )

    def cancel(self, downstream_session_id: str) -> None:
        self.calls.append({"cancel": downstream_session_id})


def test_connector_manager_caches_connectors_and_persists_session_mapping() -> None:
    agent = DownstreamAgentConfig(
        agent_id="codex",
        name="Codex",
        command="codex",
        models=[ModelOption(value="gpt-5-codex", name="GPT-5 Codex")],
    )
    created: list[RecordingConnector] = []

    def factory(current_agent: DownstreamAgentConfig):
        connector = RecordingConnector(current_agent)
        created.append(connector)
        return connector

    manager = DownstreamConnectorManager([agent], connector_factory=factory)
    session = SessionSnapshot(session_id="orch-1", cwd="/tmp/project")
    task = PlanTask(title="Task", details="Details", assignee="codex")

    first = manager.execute_task(
        session=session,
        task=task,
        coordinator_prompt="Plan",
        selected_model="gpt-5-codex",
        agent=agent,
    )
    second = manager.execute_task(
        session=session,
        task=task,
        coordinator_prompt="Plan",
        selected_model="gpt-5-codex",
        agent=agent,
    )

    assert len(created) == 1
    assert first.metadata["downstream_session_id"] == "downstream-codex"
    assert second.metadata["downstream_session_id"] == "downstream-codex"
    assert session.downstream_session_map()["codex"] == "downstream-codex"
    assert session.metadata["downstream_negotiated"]["codex"]["agent_info"] == {"name": "Codex"}
    assert created[0].calls[0]["downstream_session_id"] is None
    assert created[0].calls[1]["downstream_session_id"] == "downstream-codex"


def test_connector_manager_refreshes_catalog_from_discovery() -> None:
    agent = DownstreamAgentConfig(
        agent_id="codex",
        name="Codex",
        command="codex",
        models=[ModelOption(value="bootstrap-model", name="Bootstrap Model")],
    )
    manager = DownstreamConnectorManager([agent], connector_factory=RecordingConnector)
    catalog = FederatedModelCatalog([agent])

    manager.refresh_catalog(catalog)

    options = catalog.northbound_model_options()
    assert [item["value"] for item in options] == ["codex::discovered-model"]
    assert options[0]["metadata"]["source"] == "discovery"
