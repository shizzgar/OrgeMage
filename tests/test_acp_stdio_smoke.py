from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Any

import pytest


acp = pytest.importorskip("acp")


class _BridgeLikeWireClient:
    """Small ACP UI-style client harness for real stdio smoke tests."""

    def __init__(self) -> None:
        self.conn: Any | None = None
        self.updates: list[tuple[str, Any]] = []

    def on_connect(self, conn: Any) -> None:
        self.conn = conn

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        del kwargs
        self.updates.append((session_id, update))

    async def request_permission(
        self,
        options: Any,
        session_id: str,
        tool_call: Any,
        **kwargs: Any,
    ) -> None:
        del options, session_id, tool_call, kwargs
        return None

    async def write_text_file(
        self,
        content: str,
        path: str,
        session_id: str,
        **kwargs: Any,
    ) -> None:
        del content, path, session_id, kwargs
        return None

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del path, session_id, limit, line, kwargs
        return {"content": ""}

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del command, session_id, args, cwd, env, output_byte_limit, kwargs
        return {"terminal_id": "terminal-1"}

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> dict[str, Any]:
        del session_id, terminal_id, kwargs
        return {"chunks": [], "exit_code": 0}

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> None:
        del session_id, terminal_id, kwargs
        return None

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any) -> dict[str, Any]:
        del session_id, terminal_id, kwargs
        return {"exit_code": 0}

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> None:
        del session_id, terminal_id, kwargs
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        del method, params
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        del method, params
        return None

    def update_kinds(self, *, session_id: str | None = None) -> list[str]:
        return [
            getattr(update, "sessionUpdate", "")
            for update_session_id, update in self.updates
            if session_id is None or update_session_id == session_id
        ]

    def updates_for_kind(self, kind: str, *, session_id: str | None = None) -> list[Any]:
        return [
            update
            for update_session_id, update in self.updates
            if getattr(update, "sessionUpdate", "") == kind
            and (session_id is None or update_session_id == session_id)
        ]


def _stdio_env(repo_root: Path) -> dict[str, str]:
    pythonpath_entries = [str(repo_root / "src")]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath_entries)}


@asynccontextmanager
async def _spawn_orgemage_stdio(client: _BridgeLikeWireClient, db_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    async with acp.spawn_agent_process(
        client,
        sys.executable,
        "-m",
        "orgemage.cli",
        "--db",
        str(db_path),
        "--mock-downstream",
        "acp",
        "--stdio",
        cwd=repo_root,
        env=_stdio_env(repo_root),
    ) as (conn, process):
        initialize = await conn.initialize(
            acp.PROTOCOL_VERSION,
            client_info=acp.schema.Implementation(name="pytest-bridge", version="0"),
        )
        assert initialize.protocolVersion == acp.PROTOCOL_VERSION
        yield conn, process


async def _wait_for_updates(
    client: _BridgeLikeWireClient,
    expected: set[str],
    *,
    session_id: str,
    timeout: float = 5,
) -> None:
    async def _matches() -> None:
        while True:
            if expected.issubset(set(client.update_kinds(session_id=session_id))):
                return
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_matches(), timeout=timeout)


def test_acp_stdio_smoke_new_session_delivers_response_and_startup_updates(tmp_path: Path) -> None:
    db_path = tmp_path / "stdio-new-session.db"
    client = _BridgeLikeWireClient()

    async def scenario() -> None:
        async with _spawn_orgemage_stdio(client, db_path) as (conn, process):
            response = await asyncio.wait_for(
                conn.new_session(
                    cwd=tmp_path.as_posix(),
                    model="codex::gpt-5-codex",
                    mcp_servers=[
                        acp.schema.McpServerStdio(
                            name="filesystem",
                            command="fs-mcp",
                            args=[],
                            env=[],
                        )
                    ],
                ),
                timeout=5,
            )
            assert response.sessionId.startswith("orch-")
            assert response.models.currentModelId == "codex::gpt-5-codex"
            assert response.modes.currentModeId == "auto"

            await _wait_for_updates(
                client,
                {"available_commands_update", "session_info_update"},
                session_id=response.sessionId,
            )

            assert process.returncode is None

    asyncio.run(scenario())


def test_acp_stdio_smoke_prompt_streams_real_session_updates(tmp_path: Path) -> None:
    db_path = tmp_path / "stdio-prompt.db"
    client = _BridgeLikeWireClient()

    async def scenario() -> None:
        async with _spawn_orgemage_stdio(client, db_path) as (conn, process):
            created = await conn.new_session(
                cwd=tmp_path.as_posix(),
                model="codex::gpt-5-codex",
            )
            session_id = created.sessionId
            prompt_response = await asyncio.wait_for(
                conn.prompt(
                    [acp.schema.TextContentBlock(text="Inspect the repository and implement updates.", type="text")],
                    session_id=session_id,
                ),
                timeout=10,
            )
            assert prompt_response.stopReason == "end_turn"

            await _wait_for_updates(
                client,
                {"plan", "tool_call", "agent_message_chunk", "session_info_update"},
                session_id=session_id,
            )

            kinds = client.update_kinds(session_id=session_id)
            assert kinds.index("plan") < kinds.index("agent_message_chunk")
            assert kinds[-1] == "session_info_update"

            session_info = client.updates_for_kind("session_info_update", session_id=session_id)[-1]
            assert session_info.title.startswith("OrgeMage: Inspect the repository")
            assert session_info.field_meta["summary"].startswith("Completed 4/4 orchestrated tasks.")

            assert process.returncode is None

    asyncio.run(scenario())


def test_acp_stdio_smoke_load_session_restores_mode_and_history_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "stdio-load-session.db"
    first_client = _BridgeLikeWireClient()

    async def scenario() -> None:
        async with _spawn_orgemage_stdio(first_client, db_path) as (conn, process):
            created = await conn.new_session(
                cwd=tmp_path.as_posix(),
                model="codex::gpt-5-codex",
            )
            session_id = created.sessionId
            await conn.set_session_mode("full-access", session_id=session_id)
            await conn.prompt(
                [acp.schema.TextContentBlock(text="Make session history UX friendly.", type="text")],
                session_id=session_id,
            )
            await _wait_for_updates(
                first_client,
                {"current_mode_update", "session_info_update"},
                session_id=session_id,
            )
            assert process.returncode is None

        second_client = _BridgeLikeWireClient()
        async with _spawn_orgemage_stdio(second_client, db_path) as (conn, process):
            loaded = await asyncio.wait_for(
                conn.load_session(
                    cwd=tmp_path.as_posix(),
                    session_id=session_id,
                ),
                timeout=5,
            )
            assert loaded.modes.currentModeId == "full-access"
            assert loaded.models.currentModelId == "codex::gpt-5-codex"

            await _wait_for_updates(
                second_client,
                {"available_commands_update", "session_info_update"},
                session_id=session_id,
            )

            session_info = second_client.updates_for_kind("session_info_update", session_id=session_id)[-1]
            assert session_info.title.startswith("OrgeMage: Make session history UX friendly")
            assert session_info.field_meta["currentModeId"] == "full-access"
            assert session_info.field_meta["history"]["summary"].startswith("Completed 4/4 orchestrated tasks.")

            assert process.returncode is None

    asyncio.run(scenario())
