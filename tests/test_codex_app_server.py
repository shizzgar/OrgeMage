from __future__ import annotations

from orgemage.codex_app_server import CodexAppServerConnector
from orgemage.models import DownstreamAgentConfig, ModelOption, PlanTask, TaskStatus


class _FakeCodexTransport:
    def __init__(self, agent: DownstreamAgentConfig) -> None:
        self.agent = agent
        self.notification_handler = None
        self.request_handler = None
        self.requests: list[tuple[str, dict[str, object]]] = []

    def start(self, *, notification_handler, request_handler) -> None:
        self.notification_handler = notification_handler
        self.request_handler = request_handler

    def request(self, method: str, params: dict[str, object] | None = None, *, timeout: float | None = None):
        payload = dict(params or {})
        self.requests.append((method, payload))
        if method == "initialize":
            return {
                "userAgent": "orgemage/0.115.0 (Linux; x86_64)",
                "platformFamily": "unix",
                "platformOs": "linux",
            }
        if method == "model/list":
            return {
                "data": [
                    {
                        "id": "gpt-5.4",
                        "displayName": "gpt-5.4",
                        "description": "Latest frontier agentic coding model.",
                        "isDefault": True,
                    },
                    {
                        "id": "gpt-5.3-codex",
                        "displayName": "gpt-5.3-codex",
                        "description": "Codex model.",
                        "isDefault": False,
                    },
                ],
                "nextCursor": None,
            }
        if method == "thread/start":
            return {
                "thread": {"id": "thread-1", "cwd": payload.get("cwd")},
                "model": payload.get("model"),
                "approvalPolicy": payload.get("approvalPolicy"),
                "sandbox": {"type": "readOnly" if payload.get("sandbox") == "read-only" else "workspaceWrite"},
            }
        if method == "thread/resume":
            return {
                "thread": {"id": payload.get("threadId"), "cwd": payload.get("cwd")},
                "model": payload.get("model"),
                "approvalPolicy": payload.get("approvalPolicy"),
                "sandbox": {"type": "workspaceWrite"},
            }
        if method == "turn/start":
            assert payload.get("sandboxPolicy") is not None
            assert self.notification_handler is not None
            self.notification_handler(
                {
                    "method": "item/started",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "item": {
                            "type": "commandExecution",
                            "id": "cmd-1",
                            "command": "rg --files",
                            "status": "inProgress",
                        },
                    },
                }
            )
            self.notification_handler(
                {
                    "method": "item/commandExecution/outputDelta",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "itemId": "cmd-1",
                        "delta": "README.md\nsrc/orgemage/cli.py\n",
                    },
                }
            )
            self.notification_handler(
                {
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "itemId": "msg-1",
                        "delta": "Codex app-server response",
                    },
                }
            )
            self.notification_handler(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {"id": "turn-1", "status": "completed", "error": None, "items": []},
                    },
                }
            )
            return {"turn": {"id": "turn-1", "status": "inProgress", "error": None, "items": []}}
        if method == "turn/interrupt":
            return {}
        raise AssertionError(f"Unexpected request: {method}")

    def notify(self, method: str, params: dict[str, object] | None = None) -> None:
        return None

    def close(self) -> None:
        return None

    def stderr_tail(self) -> list[str]:
        return []


def test_codex_app_server_connector_discovers_models() -> None:
    agent = DownstreamAgentConfig(
        agent_id="codex",
        name="Codex",
        command="codex",
        args=["app-server"],
        models=[ModelOption(value="gpt-5.4", name="GPT-5.4")],
        default_model="gpt-5.4",
        runtime="codex-app-server",
    )
    connector = CodexAppServerConnector(agent, client_factory=_FakeCodexTransport)

    catalog = connector.discover_catalog()

    assert catalog["profile"]["id"] == "codex-app-server"
    assert catalog["config_options"][0]["options"][0]["value"] == "gpt-5.4"
    assert connector.negotiated_state is not None
    assert connector.negotiated_state.agent_info["name"] == "codex-app-server"
    connector.close()


def test_codex_app_server_connector_executes_turn_and_reuses_thread() -> None:
    agent = DownstreamAgentConfig(
        agent_id="codex",
        name="Codex",
        command="codex",
        args=["app-server"],
        models=[ModelOption(value="gpt-5.4", name="GPT-5.4")],
        default_model="gpt-5.4",
        runtime="codex-app-server",
    )
    connector = CodexAppServerConnector(agent, client_factory=_FakeCodexTransport)
    task = PlanTask(title="Respond directly to the user", details="Say who you are", assignee="codex")

    first = connector.execute_task(
        orchestrator_session_id="orch-1",
        downstream_session_id=None,
        cwd="/tmp/project",
        mcp_servers=[],
        task=task,
        coordinator_prompt="User request:\nТы кто?",
        selected_model="codex::gpt-5.4",
    )
    second = connector.execute_task(
        orchestrator_session_id="orch-1",
        downstream_session_id="thread-1",
        cwd="/tmp/project",
        mcp_servers=[],
        task=task,
        coordinator_prompt="User request:\nТы кто?",
        selected_model="codex::gpt-5.4",
    )

    transport = connector._client
    assert first.downstream_session_id == "thread-1"
    assert first.status == TaskStatus.COMPLETED
    assert first.summary == "Codex app-server response"
    assert any(update.get("sessionUpdate") == "agent_message_chunk" for update in first.updates)
    assert any(update.get("terminal") for update in first.updates)
    assert transport is not None
    assert ("thread/start", {"cwd": "/tmp/project", "model": "gpt-5.4", "approvalPolicy": "never", "sandbox": "read-only", "experimentalRawEvents": False, "persistExtendedHistory": False}) in transport.requests
    assert ("thread/resume", {"threadId": "thread-1", "cwd": "/tmp/project", "model": "gpt-5.4", "approvalPolicy": "never", "sandbox": "read-only", "persistExtendedHistory": False}) in transport.requests
    turn_start_request = next(params for method, params in transport.requests if method == "turn/start")
    assert turn_start_request["sandboxPolicy"]["type"] == "readOnly"
    assert second.summary == "Codex app-server response"
    connector.close()
