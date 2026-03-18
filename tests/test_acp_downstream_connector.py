from __future__ import annotations

import sys
import types

from orgemage.acp.downstream_client import AcpDownstreamConnector
from orgemage.acp.manager import DownstreamConnectorManager
from orgemage.models import DownstreamAgentConfig, ModelOption, PlanTask, SessionSnapshot, TaskStatus
from orgemage.orchestrator import Orchestrator
from orgemage.state import SQLiteSessionStore


class _FakePayload:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def model_dump(self, mode: str = "python", exclude_none: bool = True):
        return dict(self.__dict__)


class _FakeRequestError(Exception):
    @classmethod
    def method_not_found(cls, method: str) -> "_FakeRequestError":
        return cls(method)


class _FakeClient:
    pass


class _FakeClientCapabilities:
    pass


class _FakeImplementation(_FakePayload):
    def __init__(self, name: str, title: str, version: str) -> None:
        self.name = name
        self.title = title
        self.version = version


class _FakeConnection:
    def __init__(self, client: _FakeClient, state: dict[str, object]) -> None:
        self.client = client
        self.state = state

    async def initialize(self, protocol_version: int, client_capabilities, client_info):
        self.state.setdefault("initialize_calls", []).append(
            {
                "protocol_version": protocol_version,
                "client_info": client_info.model_dump(),
            }
        )
        return _FakePayload(
            protocol_version=protocol_version,
            agent_info={"name": "fake-downstream"},
            agent_capabilities={"loadSession": True},
            auth_methods=["none"],
        )

    async def new_session(self, cwd: str, mcp_servers: list[object]):
        self.state.setdefault("new_session_calls", []).append({"cwd": cwd, "mcp_servers": mcp_servers})
        return _FakePayload(
            session_id="downstream-session",
            config_options=[
                {
                    "id": "model",
                    "category": "model",
                    "type": "select",
                    "options": [{"value": "gpt-5-codex", "name": "GPT-5 Codex"}],
                }
            ],
            modes=["default"],
            commands=["read", "edit"],
        )

    async def load_session(self, cwd: str, mcp_servers: list[object], session_id: str):
        self.state.setdefault("load_session_calls", []).append(
            {"cwd": cwd, "mcp_servers": mcp_servers, "session_id": session_id}
        )
        return _FakePayload(
            session_id=session_id,
            config_options=[
                {
                    "id": "model",
                    "category": "model",
                    "type": "select",
                    "options": [{"value": "gpt-5-codex", "name": "GPT-5 Codex"}],
                }
            ],
            modes=["resumed"],
            commands=["read", "edit", "test"],
        )

    async def set_config_option(self, session_id: str, option_id: str, value: str):
        self.state.setdefault("set_config_option_calls", []).append(
            {"session_id": session_id, "option_id": option_id, "value": value}
        )

    async def prompt(self, session_id: str, prompt: list[object]):
        self.state.setdefault("prompt_calls", []).append({"session_id": session_id, "prompt": prompt})
        await self.client.session_update(
            session_id,
            {"message": {"content": [{"text": "stream update"}]}},
        )
        return _FakePayload(
            stop_reason="end_turn",
            message={"content": [{"text": "final response"}]},
        )

    async def cancel(self, session_id: str):
        self.state.setdefault("cancel_calls", []).append(session_id)


class _FakeSpawnContext:
    def __init__(self, client: _FakeClient, state: dict[str, object], command: str, args: tuple[str, ...]) -> None:
        self._client = client
        self._state = state
        self._command = command
        self._args = args

    async def __aenter__(self):
        self._state.setdefault("spawn_calls", []).append({"command": self._command, "args": list(self._args)})
        connection = _FakeConnection(self._client, self._state)
        self._state["connection"] = connection
        return connection, object()

    async def __aexit__(self, exc_type, exc, tb):
        self._state.setdefault("closed", 0)
        self._state["closed"] = int(self._state["closed"]) + 1
        return False


class _FakeTextBlock(_FakePayload):
    def __init__(self, text: str) -> None:
        self.text = text


def _install_fake_acp(monkeypatch):
    state: dict[str, object] = {}
    module = types.ModuleType("acp")
    module.PROTOCOL_VERSION = 1
    module.Client = _FakeClient
    module.RequestError = _FakeRequestError
    module.ClientCapabilities = _FakeClientCapabilities
    module.Implementation = _FakeImplementation
    module.text_block = lambda text: _FakeTextBlock(text)
    module.spawn_agent_process = lambda client, command, *args: _FakeSpawnContext(client, state, command, args)
    monkeypatch.setitem(sys.modules, "acp", module)
    return state


def test_acp_downstream_connector_runs_initialize_session_prompt_update_and_cancel(monkeypatch) -> None:
    state = _install_fake_acp(monkeypatch)
    agent = DownstreamAgentConfig(
        agent_id="codex",
        name="Codex",
        command="codex",
        args=["--acp"],
        models=[ModelOption(value="gpt-5-codex", name="GPT-5 Codex")],
        default_model="gpt-5-codex",
        runtime="acp",
    )
    connector = AcpDownstreamConnector(agent)
    task = PlanTask(title="Implement", details="Details", assignee="codex")

    catalog = connector.discover_catalog()
    first = connector.execute_task(
        orchestrator_session_id="orch-1",
        downstream_session_id=None,
        cwd="/tmp/project",
        task=task,
        coordinator_prompt="Plan this work",
        selected_model="codex::gpt-5-codex",
    )
    second = connector.execute_task(
        orchestrator_session_id="orch-1",
        downstream_session_id=first.downstream_session_id,
        cwd="/tmp/project",
        task=task,
        coordinator_prompt="Plan this work",
        selected_model="codex::gpt-5-codex",
    )
    connector.cancel(first.downstream_session_id)
    connector.close()

    assert catalog["config_options"][0]["category"] == "model"
    assert catalog["command_advertisements"] == ["read", "edit"]
    assert first.status is TaskStatus.COMPLETED
    assert first.downstream_session_id == "downstream-session"
    assert first.summary == "final response"
    assert first.updates == [{"message": {"content": [{"text": "stream update"}]}}]
    assert second.downstream_session_id == "downstream-session"
    assert state["spawn_calls"] == [{"command": "codex", "args": ["--acp"]}]
    assert len(state["initialize_calls"]) == 1
    assert len(state["new_session_calls"]) == 2
    assert len(state["load_session_calls"]) == 3
    assert state["set_config_option_calls"] == [
        {"session_id": "downstream-session", "option_id": "model", "value": "gpt-5-codex"},
        {"session_id": "downstream-session", "option_id": "model", "value": "gpt-5-codex"},
    ]
    assert state["cancel_calls"] == ["downstream-session"]
    assert connector.negotiated_state is not None
    assert connector.negotiated_state.agent_info == {"name": "fake-downstream"}
    assert connector.negotiated_state.agent_capabilities == {"loadSession": True}
    assert connector.negotiated_state.auth_methods == ["none"]
    assert connector.negotiated_state.session_capabilities["downstream-session"]["modes"] == ["resumed"]
    assert connector.negotiated_state.config_options["downstream-session"] == [
        {
            "id": "model",
            "category": "model",
            "type": "select",
            "options": [{"value": "gpt-5-codex", "name": "GPT-5 Codex"}],
        }
    ]


class _CancelRecordingConnector:
    def __init__(self, agent: DownstreamAgentConfig) -> None:
        self.agent = agent
        self.negotiated_state = None
        self.cancelled: list[str] = []

    def discover_catalog(self, *, force: bool = False):
        return {
            "agent_id": self.agent.agent_id,
            "config_options": [
                {
                    "id": "model",
                    "category": "model",
                    "type": "select",
                    "options": [{"value": "gpt-5-codex", "name": "GPT-5 Codex"}],
                }
            ],
            "capabilities": {},
            "command_advertisements": [],
        }

    def mark_catalog_refresh_required(self) -> None:
        return None

    def execute_task(self, **kwargs):  # pragma: no cover - not used in this test
        raise AssertionError("execute_task should not be called")

    def cancel(self, downstream_session_id: str) -> None:
        self.cancelled.append(downstream_session_id)


def test_orchestrator_cancel_propagates_to_connector_manager(tmp_path, monkeypatch) -> None:
    _install_fake_acp(monkeypatch)
    agent = DownstreamAgentConfig(
        agent_id="codex",
        name="Codex",
        command="codex",
        models=[ModelOption(value="gpt-5-codex", name="GPT-5 Codex")],
        runtime="acp",
    )
    created: list[_CancelRecordingConnector] = []

    def factory(current_agent: DownstreamAgentConfig):
        connector = _CancelRecordingConnector(current_agent)
        created.append(connector)
        return connector

    manager = DownstreamConnectorManager([agent], connector_factory=factory)
    orchestrator = Orchestrator([agent], SQLiteSessionStore(tmp_path / "state.db"), connector_manager=manager)
    session = orchestrator.create_session(tmp_path.as_posix(), "codex::gpt-5-codex")
    snapshot = orchestrator.store.load(session.session_id)
    assert snapshot is not None
    snapshot.metadata["downstream_sessions"] = {"codex": "downstream-session"}
    orchestrator.store.save(snapshot)

    orchestrator.cancel(session.session_id)

    assert len(created) == 1
    assert created[0].cancelled == ["downstream-session"]
