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

    result = orchestrator.orchestrate(session.session_id, "Build the ACP orchestrator and validate it.")

    assert result["coordinator"]["agent_id"] == "codex"
    assert len(result["plan"]) == 4
    assert all(task["status"] == "completed" for task in result["plan"])
    assert len(result["tool_events"]) == 8
    assert result["summary"] == "Completed 4/4 orchestrated tasks."
    assert result["session"]["downstream_session_mappings"][0]["downstream_session_id"].startswith("mock-codex-")
    assert result["session"]["turns"][0]["status"] == "completed"
    assert len(result["session"]["task_states"]) == 4


def test_orchestrator_defaults_to_first_model_if_none_selected(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    session = orchestrator.create_session(tmp_path.as_posix())

    result = orchestrator.orchestrate(session.session_id, "Inspect the repository.")

    assert result["session"]["selected_model"] is not None
    assert result["coordinator"]["model"] in {"gpt-5-codex", "qwen3-coder-plus"}
