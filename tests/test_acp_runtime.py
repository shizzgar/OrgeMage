import asyncio
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from orgemage.adapters.acp_sdk import AcpAgentRuntime
from orgemage.acp.downstream_client import DownstreamPromptResult
from orgemage.acp.manager import DownstreamConnectorManager
from orgemage.models import AgentCapabilities, DownstreamAgentConfig, ModelOption
from orgemage.orchestrator import Orchestrator
from orgemage.state import SQLiteSessionStore


class _FakeAcp:
    class Agent:
        pass

    class InitializeResponse:
        def __init__(self, protocol_version: int) -> None:
            self.protocol_version = protocol_version

    class NewSessionResponse:
        def __init__(self, session_id: str, config_options: list[object]) -> None:
            self.session_id = session_id
            self.config_options = config_options

    class LoadSessionResponse:
        def __init__(self, session_id: str, config_options: list[object]) -> None:
            self.session_id = session_id
            self.config_options = config_options

    class ListSessionsResponse:
        def __init__(self, sessions: list[dict[str, object]]) -> None:
            self.sessions = sessions

    class ConfigOption:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class ConfigOptionValue:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    class TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class Message:
        def __init__(self, content: list[object]) -> None:
            self.content = content

    class PromptResponse:
        def __init__(self, stop_reason: str, message: object) -> None:
            self.stop_reason = stop_reason
            self.message = message


class _FakeModernAcp:
    class Agent:
        pass

    class SessionListCapabilities:
        def __init__(self) -> None:
            pass

    class SessionResumeCapabilities:
        def __init__(self) -> None:
            pass

    class SessionCapabilities:
        def __init__(self, list=None, resume=None, **kwargs) -> None:
            self.list = list
            self.resume = resume
            self.extra = kwargs

    class McpCapabilities:
        def __init__(self, **kwargs) -> None:
            self.extra = kwargs

    class PromptCapabilities:
        def __init__(self, **kwargs) -> None:
            self.extra = kwargs

    class AgentCapabilities:
        def __init__(self, loadSession: bool, sessionCapabilities=None, mcpCapabilities=None, promptCapabilities=None, **kwargs) -> None:
            self.loadSession = loadSession
            self.sessionCapabilities = sessionCapabilities
            self.mcpCapabilities = mcpCapabilities
            self.promptCapabilities = promptCapabilities
            self.extra = kwargs

    class Implementation:
        def __init__(self, name: str, title: str | None = None, version: str = "") -> None:
            self.name = name
            self.title = title
            self.version = version

    class InitializeResponse:
        def __init__(self, protocolVersion: int, agentCapabilities=None, agentInfo=None, **kwargs) -> None:
            self.protocol_version = protocolVersion
            self.agent_capabilities = agentCapabilities
            self.agent_info = agentInfo
            self.extra = kwargs

    class SessionConfigOption:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> object:
            return SimpleNamespace(**payload)

    class ModelInfo:
        def __init__(self, id: str, name: str, description: str | None = None) -> None:
            self.id = id
            self.name = name
            self.description = description

    class SessionModelState:
        def __init__(self, currentModelId: str, availableModels: list[object], **kwargs) -> None:
            self.current_model_id = currentModelId
            self.available_models = availableModels
            self.extra = kwargs

    class SessionMode:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> object:
            return SimpleNamespace(**payload)

    class SessionModeState:
        def __init__(self, currentModeId: str, availableModes: list[object], **kwargs) -> None:
            self.current_mode_id = currentModeId
            self.available_modes = availableModes
            self.extra = kwargs

    class NewSessionResponse:
        def __init__(self, sessionId: str, configOptions: list[object] | None = None, models=None, modes=None, **kwargs) -> None:
            self.session_id = sessionId
            self.config_options = configOptions or []
            self.models = models
            self.modes = modes
            self.extra = kwargs

    class LoadSessionResponse:
        def __init__(self, configOptions: list[object] | None = None, models=None, modes=None, **kwargs) -> None:
            self.config_options = configOptions or []
            self.models = models
            self.modes = modes
            self.extra = kwargs

    class SessionInfo:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> object:
            return SimpleNamespace(**payload)

    class ListSessionsResponse:
        def __init__(self, sessions: list[object]) -> None:
            self.sessions = sessions

    class SessionInfoUpdate:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> object:
            return SimpleNamespace(**payload)

    class CurrentModeUpdate:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> object:
            return SimpleNamespace(**payload)

    class AvailableCommandsUpdate:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> object:
            return SimpleNamespace(**payload)

    class AgentPlanUpdate:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> object:
            return SimpleNamespace(**payload)

    class ToolCallStart:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> object:
            return SimpleNamespace(**payload)

    class ToolCallProgress:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> object:
            return SimpleNamespace(**payload)

    class AgentMessageChunk:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> object:
            return SimpleNamespace(**payload)

    class SetSessionConfigOptionResponse:
        def __init__(self, config_options: list[object]) -> None:
            self.config_options = config_options

    class ConfigOptionUpdate:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> object:
            return SimpleNamespace(**payload)

    class SetSessionModeResponse:
        def __init__(self) -> None:
            pass

    class SetSessionModelResponse:
        def __init__(self) -> None:
            pass

    class PromptResponse:
        def __init__(self, stopReason: str, **kwargs) -> None:
            self.stop_reason = stopReason
            self.extra = kwargs


class _RecordingConnection:
    def __init__(self) -> None:
        self.updates: list[tuple[str, dict[str, object]]] = []

    async def session_update(self, session_id: str, update: dict[str, object]) -> None:
        self.updates.append((session_id, update))


class _WireLevelBlockingClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def on_connect(self, conn) -> None:
        self.conn = conn

    async def request_permission(self, options, session_id: str, tool_call, **kwargs):
        del options, session_id, tool_call, kwargs
        return None

    async def session_update(self, session_id: str, update, **kwargs) -> None:
        del kwargs
        self.updates.append((session_id, update))
        self.started.set()
        await self.release.wait()

    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs):
        del content, path, session_id, kwargs
        return None

    async def read_text_file(self, path: str, session_id: str, limit: int | None = None, line: int | None = None, **kwargs):
        del path, session_id, limit, line, kwargs
        raise AssertionError('read_text_file should not be called in session/new handshake test')

    async def create_terminal(self, command: str, session_id: str, args=None, cwd: str | None = None, env=None, output_byte_limit: int | None = None, **kwargs):
        del command, session_id, args, cwd, env, output_byte_limit, kwargs
        raise AssertionError('create_terminal should not be called in session/new handshake test')

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs):
        del session_id, terminal_id, kwargs
        raise AssertionError('terminal_output should not be called in session/new handshake test')

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs):
        del session_id, terminal_id, kwargs
        return None

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs):
        del session_id, terminal_id, kwargs
        raise AssertionError('wait_for_terminal_exit should not be called in session/new handshake test')

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs):
        del session_id, terminal_id, kwargs
        return None

    async def ext_method(self, method: str, params: dict[str, object]) -> dict[str, object]:
        del method, params
        return {}

    async def ext_notification(self, method: str, params: dict[str, object]) -> None:
        del method, params
        return None


class _BlockingSessionUpdateConnection:
    def __init__(self) -> None:
        self.updates: list[tuple[str, dict[str, object]]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def session_update(self, session_id: str, update: dict[str, object]) -> None:
        self.updates.append((session_id, update))
        self.started.set()
        await self.release.wait()


def _agents() -> list[DownstreamAgentConfig]:
    return [
        DownstreamAgentConfig(
            agent_id="codex",
            name="Codex",
            command="codex",
            models=[ModelOption(value="gpt-5-codex", name="GPT-5 Codex")],
            capabilities=AgentCapabilities(
                supports_terminal=True,
                supports_filesystem=True,
                commands=["read", "edit", "test", "search"],
            ),
            runtime="mock",
        ),
        DownstreamAgentConfig(
            agent_id="qwen",
            name="Qwen",
            command="qwen",
            models=[ModelOption(value="qwen3-coder-plus", name="Qwen3 Coder Plus")],
            capabilities=AgentCapabilities(
                supports_terminal=True,
                supports_filesystem=True,
                commands=["edit", "test"],
            ),
            runtime="mock",
        ),
    ]


def test_orchestrator_emits_streaming_updates_and_lists_sessions(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    session = orchestrator.create_session(tmp_path.as_posix(), "codex::gpt-5-codex")
    updates: list[dict[str, object]] = []

    result = orchestrator.orchestrate_turn(
        session.session_id,
        "Build the ACP orchestrator and validate it.",
        emit_update=updates.append,
    )

    update_kinds = [update["sessionUpdate"] for update in updates]
    assert update_kinds[0] == "plan"
    assert update_kinds.count("tool_call") == 8
    assert update_kinds[-1] == "message"
    assert result["summary"] == "Completed 4/4 orchestrated tasks."

    history = orchestrator.list_sessions()
    assert len(history) == 1
    assert history[0].session_id == session.session_id
    assert history[0].selected_model == "codex::gpt-5-codex"


def test_acp_runtime_supports_session_loading_prompt_updates_and_cancel(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    runtime = AcpAgentRuntime(acp=_FakeAcp, orchestrator=orchestrator)
    connection = _RecordingConnection()

    async def scenario() -> None:
        initialize = await runtime.initialize(7, client_connection=connection)
        assert initialize.protocol_version == 7

        mcp_servers = [{"name": "filesystem", "transport": {"type": "stdio", "command": "fs-mcp"}}]
        created = await runtime.new_session(tmp_path.as_posix(), model="codex::gpt-5-codex", mcpServers=mcp_servers)
        assert created.session_id.startswith("orch-")
        assert created.config_options[0].id == "model"
        assert orchestrator.current_session_mode(created.session_id) == "auto"

        await runtime.set_config_option(created.session_id, "model", "qwen::qwen3-coder-plus")
        await runtime.set_mode(created.session_id, "read-only")
        loaded = await runtime.load_session(tmp_path.as_posix(), created.session_id, mcp_servers=mcp_servers)
        assert loaded.session_id == created.session_id

        listed = await runtime.list_sessions()
        assert listed.sessions[0]["session_id"] == created.session_id

        prompt_response = await runtime.prompt(
            created.session_id,
            [_FakeAcp.TextBlock("Inspect the repository and implement updates.")],
        )
        assert prompt_response.stop_reason == "end_turn"
        assert prompt_response.message.content[0].text == "Completed 4/4 orchestrated tasks."

        await runtime.cancel(created.session_id)

    asyncio.run(scenario())

    kinds = [update["sessionUpdate"] for _, update in connection.updates]
    assert "session_info" in kinds
    assert "plan" in kinds
    assert "tool_call" in kinds
    assert "message" in kinds
    assert "cancelled" in kinds
    assert kinds[-1] == "session_info"
    assert orchestrator.load_session(connection.updates[0][0]).metadata["cancelled"] is True
    assert orchestrator.load_session(connection.updates[0][0]).mcp_servers == [{"name": "filesystem", "transport": {"type": "stdio", "command": "fs-mcp"}}]
    assert orchestrator.current_session_mode(connection.updates[0][0]) == "read-only"



def test_acp_runtime_new_and_load_session_return_before_startup_update_delivery(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    runtime = AcpAgentRuntime(acp=_FakeAcp, orchestrator=orchestrator)
    connection = _BlockingSessionUpdateConnection()

    async def scenario() -> None:
        await runtime.initialize(7, client_connection=connection)

        create_task = asyncio.create_task(runtime.new_session(tmp_path.as_posix(), model="codex::gpt-5-codex"))
        await asyncio.sleep(0)
        assert create_task.done() is True
        created = create_task.result()
        assert created.session_id.startswith("orch-")

        await asyncio.wait_for(connection.started.wait(), timeout=1)
        assert connection.updates[0][0] == created.session_id
        assert connection.updates[0][1]["sessionUpdate"] in {"available_commands", "session_info"}
        assert {update["sessionUpdate"] for _, update in connection.updates} >= {
            "available_commands",
            "session_info",
        }

        loaded_connection = _BlockingSessionUpdateConnection()
        runtime.bind_client_connection(loaded_connection)
        load_task = asyncio.create_task(runtime.load_session(tmp_path.as_posix(), created.session_id))
        await asyncio.sleep(0)
        assert load_task.done() is True
        loaded = load_task.result()
        assert loaded.session_id == created.session_id

        await asyncio.wait_for(loaded_connection.started.wait(), timeout=1)
        assert loaded_connection.updates[0][0] == created.session_id
        assert loaded_connection.updates[0][1]["sessionUpdate"] in {"available_commands", "session_info"}
        assert {update["sessionUpdate"] for _, update in loaded_connection.updates} >= {
            "available_commands",
            "session_info",
        }

        connection.release.set()
        loaded_connection.release.set()
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_acp_runtime_new_session_uses_bootstrap_models_without_refreshing_downstreams(tmp_path: Path) -> None:
    class _ExplodingConnector:
        def __init__(self, agent: DownstreamAgentConfig) -> None:
            self.agent = agent
            self.negotiated_state = None

        def discover_catalog(self, *, force: bool = False) -> dict[str, object]:
            del force
            raise AssertionError("session/new should not refresh downstream catalog")

        def mark_catalog_refresh_required(self) -> None:
            return None

        def execute_task(self, **kwargs):  # pragma: no cover - not used here
            raise AssertionError("execute_task should not be called")

        def cancel(self, downstream_session_id: str) -> None:
            del downstream_session_id
            return None

    agents = _agents()
    store = SQLiteSessionStore(tmp_path / "state.db")
    manager = DownstreamConnectorManager(agents, connector_factory=_ExplodingConnector, store=store)
    orchestrator = Orchestrator(agents, store, connector_manager=manager)
    runtime = AcpAgentRuntime(acp=_FakeAcp, orchestrator=orchestrator)

    async def scenario() -> None:
        created = await runtime.new_session(tmp_path.as_posix(), model="codex::gpt-5-codex")
        assert created.session_id.startswith("orch-")
        assert created.config_options[0].options[0].value == "codex::gpt-5-codex"

    asyncio.run(scenario())

def test_acp_stdio_session_new_returns_before_blocked_startup_update(tmp_path: Path) -> None:
    acp = pytest.importorskip("acp")
    client = _WireLevelBlockingClient()
    db_path = tmp_path / "wire-runtime.db"
    repo_root = Path(__file__).resolve().parents[1]
    pythonpath_entries = [str(repo_root / "src")]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath_entries)}

    async def scenario() -> None:
        async with acp.spawn_agent_process(
            client,
            sys.executable,
            "-m",
            "orgemage.cli",
            "--db",
            str(db_path),
            "acp",
            "--stdio",
            cwd=repo_root,
            env=env,
        ) as (conn, process):
            initialize = await conn.initialize(
                acp.PROTOCOL_VERSION,
                client_info=acp.schema.Implementation(name="pytest-wire", version="0"),
            )
            assert initialize.protocolVersion == acp.PROTOCOL_VERSION

            new_session_task = asyncio.create_task(conn.new_session(cwd=tmp_path.as_posix()))
            await asyncio.wait_for(client.started.wait(), timeout=5)

            response = await asyncio.wait_for(new_session_task, timeout=5)
            session_id = getattr(response, "sessionId", getattr(response, "session_id", ""))
            assert isinstance(session_id, str) and session_id.startswith("orch-")

            assert client.updates
            assert getattr(client.updates[0][1], "sessionUpdate", None) in {
                "available_commands_update",
                "session_info_update",
            }
            assert {
                getattr(update, "sessionUpdate", None)
                for _, update in client.updates
            } >= {"available_commands_update", "session_info_update"}

            client.release.set()
            await asyncio.sleep(0)
            assert process.returncode is None

    asyncio.run(scenario())


def test_acp_runtime_prompt_returns_cancelled_stop_reason_for_active_turn(tmp_path: Path) -> None:
    from orgemage.models import TaskStatus

    planning_started = threading.Event()
    release_planning = threading.Event()

    class _BlockingConnector:
        def __init__(self, agent: DownstreamAgentConfig) -> None:
            self.agent = agent
            self.negotiated_state = None
            self.cancel_calls: list[str] = []

        def discover_catalog(self, *, force: bool = False) -> dict[str, object]:
            del force
            return {
                "agent_id": self.agent.agent_id,
                "config_options": [],
                "capabilities": {},
                "command_advertisements": [],
            }

        def mark_catalog_refresh_required(self) -> None:
            return None

        def execute_task(self, *, task, orchestrator_session_id, downstream_session_id, cwd, mcp_servers, coordinator_prompt, selected_model):
            del orchestrator_session_id, downstream_session_id, cwd, mcp_servers, coordinator_prompt, selected_model
            if task._meta.get("phase") == "planning":
                planning_started.set()
                release_planning.wait(timeout=5)
                return DownstreamPromptResult(
                    downstream_session_id="runtime-cancel-session",
                    status=TaskStatus.CANCELLED,
                    summary="Turn cancelled.",
                    response={"stop_reason": "cancelled", "message": {"content": [{"text": "Turn cancelled."}]}},
                )
            raise AssertionError("worker execution should not start after cancellation")

        def cancel(self, downstream_session_id: str) -> None:
            self.cancel_calls.append(downstream_session_id)
            release_planning.set()

    agents = _agents()
    store = SQLiteSessionStore(tmp_path / "state.db")
    manager = DownstreamConnectorManager(agents, connector_factory=_BlockingConnector, store=store)
    orchestrator = Orchestrator(agents, store, connector_manager=manager)
    runtime = AcpAgentRuntime(acp=_FakeAcp, orchestrator=orchestrator)
    connection = _RecordingConnection()

    async def scenario() -> None:
        await runtime.initialize(7, client_connection=connection)
        created = await runtime.new_session(tmp_path.as_posix(), model="codex::gpt-5-codex")

        prompt_task = asyncio.create_task(
            runtime.prompt(
                created.session_id,
                [_FakeAcp.TextBlock("Cancel this turn while planning is still running.")],
            )
        )
        deadline = asyncio.get_running_loop().time() + 5
        while not planning_started.is_set():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        await runtime.cancel(created.session_id)
        response = await prompt_task
        assert response.stop_reason == "cancelled"
        assert response.message.content[0].text == "Turn cancelled."

    asyncio.run(scenario())

    snapshot = orchestrator.load_session(connection.updates[0][0])
    assert snapshot.turns[-1].status.value == "cancelled"
    assert snapshot.turns[-1].stop_reason == "cancelled"
    assert any(update["sessionUpdate"] == "cancelled" for _, update in connection.updates)


def test_acp_runtime_includes_session_summary_in_history_and_session_info(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    runtime = AcpAgentRuntime(acp=_FakeAcp, orchestrator=orchestrator)
    connection = _RecordingConnection()

    class _MetaBlock(_FakeAcp.TextBlock):
        def __init__(self, text: str, metadata: dict[str, object]) -> None:
            super().__init__(text)
            self.metadata = metadata

    async def scenario() -> None:
        await runtime.initialize(7, client_connection=connection)
        created = await runtime.new_session(tmp_path.as_posix(), model="codex::gpt-5-codex")
        await runtime.prompt(
            created.session_id,
            [_MetaBlock("Make session history UX friendly.", {"traceId": "trace-runtime", "traceparent": "tp-runtime"})],
        )
        listed = await runtime.list_sessions()
        assert listed.sessions[0]["summary"].startswith("Completed")
        assert listed.sessions[0]["title"].startswith("OrgeMage: Make session history UX friendly")

    asyncio.run(scenario())

    session_info_updates = [update for _, update in connection.updates if update["sessionUpdate"] == "session_info"]
    assert session_info_updates[-1]["info"]["summary"].startswith("Completed")
    assert session_info_updates[-1]["info"]["history"]["summary"].startswith("Completed")


def test_acp_runtime_supports_modern_acp_schema_shapes(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    runtime = AcpAgentRuntime(acp=_FakeModernAcp, orchestrator=orchestrator)
    connection = _RecordingConnection()

    async def scenario() -> None:
        initialize = await runtime.initialize(1, conn=connection)
        assert initialize.protocol_version == 1
        assert initialize.agent_capabilities.loadSession is True
        assert initialize.agent_info.name == "orgemage"

        created = await runtime.new_session(tmp_path.as_posix(), model="codex::gpt-5-codex")
        assert created.session_id.startswith("orch-")
        assert created.config_options[0].currentValue == "codex::gpt-5-codex"
        assert created.models.current_model_id == "codex::gpt-5-codex"
        assert created.modes.current_mode_id == "auto"
        assert [mode.id for mode in created.modes.available_modes] == ["read-only", "auto", "full-access"]

        await runtime.set_session_model(created.session_id, "qwen::qwen3-coder-plus")
        config_response = await runtime.set_config_option(created.session_id, "model", "codex::gpt-5-codex")
        assert config_response.config_options[0].currentValue == "codex::gpt-5-codex"
        mode_response = await runtime.set_session_mode(created.session_id, "full-access")
        assert isinstance(mode_response, _FakeModernAcp.SetSessionModeResponse)

        loaded = await runtime.load_session(tmp_path.as_posix(), created.session_id)
        assert loaded.models.current_model_id == "codex::gpt-5-codex"
        assert loaded.modes.current_mode_id == "full-access"

        listed = await runtime.list_sessions()
        assert listed.sessions[0].sessionId == created.session_id

        prompt_response = await runtime.prompt(
            created.session_id,
            [_FakeAcp.TextBlock("Inspect the repository and implement updates.")],
        )
        assert prompt_response.stop_reason == "end_turn"

        await runtime.cancel(created.session_id)

    asyncio.run(scenario())

    kinds = [getattr(update, "sessionUpdate", None) for _, update in connection.updates]
    assert "available_commands_update" in kinds
    assert "session_info_update" in kinds
    assert "config_option_update" in kinds
    assert "current_mode_update" in kinds
    assert "plan" in kinds
    assert "tool_call" in kinds
    assert "agent_message_chunk" in kinds
    available_command_updates = [
        update for _, update in connection.updates if getattr(update, "sessionUpdate", None) == "available_commands_update"
    ]
    assert [command["name"] for command in available_command_updates[-1].availableCommands] == ["status", "models", "plan"]


def test_acp_runtime_accepts_model_from_meta_options_on_new_session(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    runtime = AcpAgentRuntime(acp=_FakeModernAcp, orchestrator=orchestrator)

    async def scenario() -> None:
        created = await runtime.new_session(
            tmp_path.as_posix(),
            _meta={"claudeCode": {"options": {"model": "qwen::qwen3-coder-plus"}}},
        )
        assert created.config_options[0].currentValue == "qwen::qwen3-coder-plus"
        assert created.models.current_model_id == "qwen::qwen3-coder-plus"

    asyncio.run(scenario())


def test_acp_runtime_accepts_model_from_flattened_meta_options_on_new_session(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    runtime = AcpAgentRuntime(acp=_FakeModernAcp, orchestrator=orchestrator)

    async def scenario() -> None:
        created = await runtime.new_session(
            tmp_path.as_posix(),
            claudeCode={"options": {"model": "qwen::qwen3-coder-plus"}},
        )
        assert created.config_options[0].currentValue == "qwen::qwen3-coder-plus"
        assert created.models.current_model_id == "qwen::qwen3-coder-plus"

    asyncio.run(scenario())


def test_acp_runtime_agent_exposes_session_method_aliases(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    runtime = AcpAgentRuntime(acp=_FakeModernAcp, orchestrator=orchestrator)

    async def scenario() -> None:
        created = await runtime.new_session(tmp_path.as_posix(), model="codex::gpt-5-codex")
        model_response = await runtime.agent.session_set_model(created.session_id, "qwen::qwen3-coder-plus")
        assert isinstance(model_response, _FakeModernAcp.SetSessionModelResponse)
        mode_response = await runtime.agent.session_set_mode(created.session_id, "full-access")
        assert isinstance(mode_response, _FakeModernAcp.SetSessionModeResponse)
        config_response = await runtime.agent.session_set_config_option(created.session_id, "model", "codex::gpt-5-codex")
        assert config_response.config_options[0].currentValue == "codex::gpt-5-codex"

    asyncio.run(scenario())


def test_acp_runtime_mode_round_trip_and_updates(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    runtime = AcpAgentRuntime(acp=_FakeModernAcp, orchestrator=orchestrator)
    connection = _RecordingConnection()

    async def scenario() -> None:
        await runtime.initialize(1, conn=connection)
        created = await runtime.new_session(tmp_path.as_posix(), model="codex::gpt-5-codex")
        assert created.modes.current_mode_id == "auto"

        response = await runtime.set_mode(created.session_id, "read-only")
        assert isinstance(response, _FakeModernAcp.SetSessionModeResponse)

        loaded = await runtime.load_session(tmp_path.as_posix(), created.session_id)
        assert loaded.modes.current_mode_id == "read-only"
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    kinds = [getattr(update, "sessionUpdate", None) for _, update in connection.updates]
    assert "available_commands_update" in kinds
    assert "current_mode_update" in kinds
    current_mode_updates = [update for _, update in connection.updates if getattr(update, "sessionUpdate", None) == "current_mode_update"]
    assert current_mode_updates[-1].currentModeId == "read-only"


def test_acp_runtime_slash_commands_emit_advertised_native_responses(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    runtime = AcpAgentRuntime(acp=_FakeModernAcp, orchestrator=orchestrator)
    connection = _RecordingConnection()

    async def scenario() -> str:
        await runtime.initialize(1, conn=connection)
        created = await runtime.new_session(tmp_path.as_posix(), model="codex::gpt-5-codex")
        response = await runtime.prompt(created.session_id, [_FakeAcp.TextBlock("/status")])
        assert response.stop_reason == "end_turn"
        await runtime.prompt(created.session_id, [_FakeAcp.TextBlock("/models")])
        await runtime.prompt(created.session_id, [_FakeAcp.TextBlock("/plan")])
        return created.session_id

    session_id = asyncio.run(scenario())

    chunks = [
        update.content["text"]
        for update_session_id, update in connection.updates
        if update_session_id == session_id and getattr(update, "sessionUpdate", None) == "agent_message_chunk"
    ]
    assert any("Session orch-" in chunk for chunk in chunks)
    assert any("Available coordinator models:" in chunk for chunk in chunks)
    assert any("No active orchestration plan is available" in chunk for chunk in chunks)
