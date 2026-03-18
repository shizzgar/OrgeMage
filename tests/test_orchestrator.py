from pathlib import Path

from orgemage.models import AgentCapabilities, DownstreamAgentConfig, ModelOption
from orgemage.orchestrator import Orchestrator
from orgemage.state import SQLiteSessionStore


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


def test_orchestrator_selects_coordinator_and_runs_plan(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    session = orchestrator.create_session(tmp_path.as_posix(), "codex::gpt-5-codex")

    result = orchestrator.orchestrate_turn(session.session_id, "Build the ACP orchestrator and validate it.")

    assert result["coordinator"]["agent_id"] == "codex"
    assert len(result["plan"]) == 4
    assert result["planning"]["normalized_plan"]["_meta"]["source"] == "coordinator"
    assert result["planning"]["raw_coordinator_output"]
    assert all(task["status"] == "completed" for task in result["plan"])
    assert len(result["tool_events"]) == 8
    assert result["summary"] == "Completed 4/4 orchestrated tasks."
    assert result["session"]["downstream_session_mappings"][0]["downstream_session_id"].startswith("mock-codex-")
    assert result["session"]["turns"][0]["status"] == "completed"
    assert len(result["session"]["task_states"]) == 4
    assert result["session"]["metadata"]["planning"]["normalized_plan"]["_meta"]["source"] == "coordinator"


def test_orchestrator_defaults_to_first_model_if_none_selected(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    session = orchestrator.create_session(tmp_path.as_posix())

    result = orchestrator.orchestrate_turn(session.session_id, "Inspect the repository.")

    assert result["session"]["selected_model"] is not None
    assert result["coordinator"]["model"] in {"gpt-5-codex", "qwen3-coder-plus"}


class _StreamingConnector:
    def __init__(self, agent: DownstreamAgentConfig) -> None:
        self.agent = agent
        self.negotiated_state = None

    def discover_catalog(self, *, force: bool = False) -> dict[str, object]:
        del force
        return {
            "agent_id": self.agent.agent_id,
            "config_options": [
                {
                    "id": "model",
                    "name": "Model",
                    "category": "model",
                    "type": "select",
                    "options": [{"value": option.value, "name": option.name} for option in self.agent.models],
                }
            ],
            "capabilities": {},
            "command_advertisements": [],
        }

    def mark_catalog_refresh_required(self) -> None:
        return None

    def execute_task(self, *, task, orchestrator_session_id, downstream_session_id, cwd, coordinator_prompt, selected_model):
        del orchestrator_session_id, downstream_session_id, cwd, coordinator_prompt, selected_model
        from orgemage.acp.downstream_client import DownstreamPromptResult
        from orgemage.models import TaskStatus

        if task._meta.get("phase") == "planning":
            return DownstreamPromptResult(
                downstream_session_id="streaming-session",
                status=TaskStatus.COMPLETED,
                summary="Generated structured orchestration plan.",
                raw_output='{"tasks":[{"title":"Implement orchestrator changes","details":"Apply orchestration updates.","dependencies":[],"required_capabilities":{"needsFilesystem":true},"assignee_hints":["codex"],"acceptable_models":[],"priority":70,"_meta":{"planner":"test"}}],"_meta":{"planner":"test"}}',
            )
        return DownstreamPromptResult(
            downstream_session_id="streaming-session",
            status=TaskStatus.COMPLETED,
            summary="Worker completed implementation.",
            updates=[
                {
                    "tool_call": {"toolCallId": "down-tool-1"},
                    "message": {"content": [{"text": "worker is editing files"}]},
                    "locations": [{"path": "/workspace/OrgeMage/src/orgemage/orchestrator.py", "line": 42}],
                },
                {
                    "tool_call": {"toolCallId": "down-tool-2"},
                    "terminal": {"terminalId": "term-7", "content": "pytest -q\n1 passed"},
                    "message": {"content": [{"text": "terminal produced output"}]},
                },
            ],
            response={"message": {"content": [{"text": "Worker completed implementation."}]}},
        )

    def cancel(self, downstream_session_id: str) -> None:
        del downstream_session_id
        return None


def test_orchestrator_streaming_api_rebases_worker_updates_and_emits_file_follow_data(tmp_path: Path) -> None:
    from orgemage.acp.manager import DownstreamConnectorManager

    manager = DownstreamConnectorManager(_agents(), connector_factory=_StreamingConnector)
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"), connector_manager=manager)
    session = orchestrator.create_session(tmp_path.as_posix(), "codex::gpt-5-codex")
    updates: list[dict[str, object]] = []

    orchestrator.orchestrate(session.session_id, "Implement the orchestrator changes.", emit_update=updates.append)

    tool_updates = [update for update in updates if update["sessionUpdate"] == "tool_call"]
    assert any(update["toolCall"]["toolCallId"] == "orch-tool-task-"[:15] for update in tool_updates) is False
    assert any(update["toolCall"]["toolCallId"].startswith("orch-tool-") for update in tool_updates)
    rebased = [update for update in tool_updates if ":worker-" in update["toolCall"]["toolCallId"]]
    assert len(rebased) == 2
    assert rebased[0]["toolCall"]["locations"] == [{"path": "/workspace/OrgeMage/src/orgemage/orchestrator.py", "line": 42}]
    assert rebased[1]["toolCall"]["terminal"]["content"] == "pytest -q\n1 passed"
    assert updates[-1]["sessionUpdate"] == "message"
    assert updates[-1]["message"]["role"] == "assistant"
    assert orchestrator.load_session(session.session_id).terminal_mappings[0].upstream_terminal_id == "up-term-7"
