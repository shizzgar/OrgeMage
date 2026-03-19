from __future__ import annotations

import sys
import types
from pathlib import Path

from orgemage.acp.downstream_client import AcpDownstreamConnector
from orgemage.acp.manager import DownstreamConnectorManager
from orgemage.models import AgentCapabilities, DownstreamAgentConfig, ModelOption, PlanTask, SessionSnapshot, TaskStatus
from orgemage.orchestrator import Orchestrator
from orgemage.state import SQLiteSessionStore


class _FakePayload:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def model_dump(self, mode: str = "python", exclude_none: bool = True):
        return dict(self.__dict__)


class _FakeRequestError(Exception):
    def __init__(self, code: str, method: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.method = method
        self.message = message

    @classmethod
    def method_not_found(cls, method: str) -> "_FakeRequestError":
        return cls("method_not_found", method, method)

    @classmethod
    def invalid_params(cls, method: str, message: str) -> "_FakeRequestError":
        return cls("invalid_params", method, message)

    @classmethod
    def permission_denied(cls, method: str, message: str) -> "_FakeRequestError":
        return cls("permission_denied", method, message)

    @classmethod
    def forbidden(cls, method: str, message: str) -> "_FakeRequestError":
        return cls("forbidden", method, message)

    @classmethod
    def internal_error(cls, method: str, message: str) -> "_FakeRequestError":
        return cls("internal_error", method, message)


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
        prompt_hook = self.state.get("prompt_hook")
        if callable(prompt_hook):
            await prompt_hook(self.client, self.state, session_id)
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


class _FakeUpstreamClientConnection:
    def __init__(self) -> None:
        self.permission_requests: list[dict[str, object]] = []
        self.read_requests: list[str] = []
        self.write_requests: list[dict[str, str]] = []

    async def request_permission(self, **kwargs):
        self.permission_requests.append(dict(kwargs))
        return {"decision": "allow", "source": "upstream"}

    async def read_text_file(self, *, path: str):
        self.read_requests.append(path)
        return {"path": path, "content": Path(path).read_text(encoding="utf-8")}

    async def write_text_file(self, *, path: str, content: str):
        self.write_requests.append({"path": path, "content": content})
        Path(path).write_text(content, encoding="utf-8")
        return {"path": path, "bytes_written": len(content.encode("utf-8"))}


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


def _install_fake_acp_with_schema_capabilities(monkeypatch):
    state = _install_fake_acp(monkeypatch)
    module = sys.modules["acp"]
    schema = types.SimpleNamespace(
        ClientCapabilities=_FakeClientCapabilities,
        Implementation=_FakeImplementation,
    )
    delattr(module, "ClientCapabilities")
    delattr(module, "Implementation")
    module.schema = schema
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


def test_acp_downstream_connector_supports_schema_scoped_client_capabilities(monkeypatch) -> None:
    state = _install_fake_acp_with_schema_capabilities(monkeypatch)
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

    catalog = connector.discover_catalog()
    connector.close()

    assert catalog["config_options"][0]["category"] == "model"
    assert len(state["initialize_calls"]) == 1


def test_acp_downstream_connector_callback_layer_persists_permissions_filesystem_and_terminal(tmp_path, monkeypatch) -> None:
    state = _install_fake_acp(monkeypatch)
    test_file = tmp_path / "notes.txt"
    upstream = _FakeUpstreamClientConnection()
    agent = DownstreamAgentConfig(
        agent_id="codex",
        name="Codex",
        command="codex",
        args=["--acp"],
        models=[ModelOption(value="gpt-5-codex", name="GPT-5 Codex")],
        capabilities=AgentCapabilities(supports_terminal=True, supports_filesystem=True),
        runtime="acp",
    )
    store = SQLiteSessionStore(tmp_path / "state.db")
    store.save(SessionSnapshot(session_id="orch-1", cwd=tmp_path.as_posix()))

    async def prompt_hook(client, current_state, session_id: str) -> None:
        current_state["permission_response"] = await client.request_permission(
            session_id=session_id,
            request_id="perm-1",
            tool_name="write_file",
        )
        current_state["write_response"] = await client.write_text_file(
            session_id=session_id,
            path=test_file.as_posix(),
            content="hello from downstream\n",
        )
        current_state["read_response"] = await client.read_text_file(
            session_id=session_id,
            path=test_file.as_posix(),
        )
        terminal = await client.create_terminal(
            session_id=session_id,
            command=["python", "-c", "print('terminal-ok')"],
        )
        current_state["terminal"] = terminal
        terminal_id = terminal["terminal_id"]
        current_state["terminal_wait"] = await client.wait_for_terminal_exit(
            session_id=session_id,
            terminal_id=terminal_id,
            timeout_seconds=5,
        )
        current_state["terminal_output"] = await client.terminal_output(
            session_id=session_id,
            terminal_id=terminal_id,
        )

    state["prompt_hook"] = prompt_hook
    connector = AcpDownstreamConnector(
        agent,
        store=store,
        upstream_client_getter=lambda: upstream,
    )
    task = PlanTask(title="Implement", details="Details", assignee="codex")

    result = connector.execute_task(
        orchestrator_session_id="orch-1",
        downstream_session_id=None,
        cwd=tmp_path.as_posix(),
        task=task,
        coordinator_prompt="Plan this work",
        selected_model="codex::gpt-5-codex",
    )
    connector.close()

    assert result.status is TaskStatus.COMPLETED
    assert state["permission_response"]["decision"] == "allow"
    assert upstream.permission_requests == [{"session_id": "downstream-session", "request_id": "perm-1", "tool_name": "write_file"}]
    assert test_file.read_text(encoding="utf-8") == "hello from downstream\n"
    assert state["read_response"]["content"] == "hello from downstream\n"
    assert state["terminal_wait"]["status"] == "exited"
    assert "terminal-ok" in state["terminal_output"]["content"]

    loaded = store.load("orch-1")
    assert loaded is not None
    assert loaded.permission_requests[0].request_id == "perm-1"
    assert loaded.permission_requests[0].decision == "allow"
    assert loaded.permission_requests[0].metadata["decision_source"] == "upstream"
    assert loaded.terminal_mappings[0].owner_task_id == task.task_id
    assert loaded.terminal_mappings[0].owner_agent_id == "codex"
    assert loaded.terminal_mappings[0].status == "released"
    assert any(trace.metadata["method"] == "fs/write_text_file" for trace in loaded.trace_metadata)
    assert any(trace.metadata["method"] == "fs/read_text_file" for trace in loaded.trace_metadata)


def test_acp_downstream_connector_maps_callback_policy_failures_to_acp_errors(tmp_path, monkeypatch) -> None:
    state = _install_fake_acp(monkeypatch)
    agent = DownstreamAgentConfig(
        agent_id="codex",
        name="Codex",
        command="codex",
        args=["--acp"],
        models=[ModelOption(value="gpt-5-codex", name="GPT-5 Codex")],
        capabilities=AgentCapabilities(supports_terminal=True, supports_filesystem=True),
        runtime="acp",
    )
    store = SQLiteSessionStore(tmp_path / "state.db")
    store.save(SessionSnapshot(session_id="orch-1", cwd=tmp_path.as_posix()))

    async def prompt_hook(client, current_state, session_id: str) -> None:
        try:
            await client.read_text_file(
                session_id=session_id,
                path=(tmp_path.parent / "outside.txt").resolve().as_posix(),
            )
        except _FakeRequestError as exc:
            current_state["fs_error"] = exc
        terminal = await client.create_terminal(session_id=session_id, command=["python", "-c", "print('x')"])
        await client.wait_for_terminal_exit(session_id=session_id, terminal_id=terminal["terminal_id"], timeout_seconds=5)
        await client.release_terminal(session_id=session_id, terminal_id=terminal["terminal_id"])
        try:
            await client.kill_terminal(session_id=session_id, terminal_id=terminal["terminal_id"])
        except _FakeRequestError as exc:
            current_state["terminal_error"] = exc

    state["prompt_hook"] = prompt_hook
    connector = AcpDownstreamConnector(agent, store=store)
    task = PlanTask(title="Implement", details="Details", assignee="codex")

    connector.execute_task(
        orchestrator_session_id="orch-1",
        downstream_session_id=None,
        cwd=tmp_path.as_posix(),
        task=task,
        coordinator_prompt="Plan this work",
        selected_model="codex::gpt-5-codex",
    )
    connector.close()

    fs_error = state["fs_error"]
    terminal_error = state["terminal_error"]
    assert isinstance(fs_error, _FakeRequestError)
    assert fs_error.code == "permission_denied"
    assert fs_error.method == "fs/read_text_file"
    assert isinstance(terminal_error, _FakeRequestError)
    assert terminal_error.code == "forbidden"
    assert terminal_error.method == "terminal/kill"


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


def test_acp_downstream_connector_emits_structured_debug_logging(tmp_path, monkeypatch, caplog) -> None:
    import logging

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
    task = PlanTask(title="Implement", details="Details", assignee="codex", _meta={"turnId": "turn-1", "traceId": "trace-log"})

    caplog.set_level(logging.DEBUG)
    connector.discover_catalog()
    connector.execute_task(
        orchestrator_session_id="orch-1",
        downstream_session_id=None,
        cwd=tmp_path.as_posix(),
        task=task,
        coordinator_prompt="Plan this work",
        selected_model="codex::gpt-5-codex",
    )
    connector.cancel("downstream-session")
    connector.close()

    events = [record.message for record in caplog.records if record.name == "orgemage.acp.downstream_client"]
    assert any("connector.lifecycle.start" in message for message in events)
    assert any("connector.lifecycle.execute.complete" in message for message in events)
    assert any("connector.lifecycle.cancel" in message for message in events)
    assert state["cancel_calls"] == ["downstream-session"]
