import asyncio
from pathlib import Path

from orgemage.adapters.acp_sdk import AcpAgentRuntime
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


class _RecordingConnection:
    def __init__(self) -> None:
        self.updates: list[tuple[str, dict[str, object]]] = []

    async def session_update(self, session_id: str, update: dict[str, object]) -> None:
        self.updates.append((session_id, update))


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

        created = await runtime.new_session(tmp_path.as_posix(), model="codex::gpt-5-codex")
        assert created.session_id.startswith("orch-")
        assert created.config_options[0].id == "model"

        await runtime.set_config_option(created.session_id, "model", "qwen::qwen3-coder-plus")
        loaded = await runtime.load_session(tmp_path.as_posix(), created.session_id)
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
    assert kinds[-1] == "cancelled"
    assert orchestrator.load_session(connection.updates[0][0]).metadata["cancelled"] is True
