from orgemage.models import AgentCapabilities, DownstreamAgentConfig, ModelOption
from orgemage.orchestrator import Orchestrator
from orgemage.planning import parse_coordinator_plan
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
        )
    ]


def test_parse_coordinator_plan_resolves_dependencies_and_metadata() -> None:
    raw_output = """
    {
      "tasks": [
        {
          "title": "Inspect repository",
          "details": "Read the impacted files.",
          "dependencies": [],
          "required_capabilities": {"needsFilesystem": true},
          "assignee_hints": ["codex"],
          "acceptable_models": ["codex::gpt-5-codex"],
          "priority": 80,
          "_meta": {"why": "Need context first"}
        },
        {
          "title": "Implement change",
          "details": "Modify the orchestrator.",
          "dependencies": ["Inspect repository"],
          "required_capabilities": {"needsFilesystem": true, "needsTerminal": true},
          "priority": 70,
          "_meta": {"why": "Apply the code change"}
        }
      ],
      "_meta": {"planner": "coordinator", "provenance": "unit-test"}
    }
    """

    result = parse_coordinator_plan(raw_output, coordinator_agent_id="codex")

    assert result.is_valid is True
    assert len(result.tasks) == 2
    assert result.tasks[0].assignee == "codex"
    assert result.tasks[1].dependency_ids == [result.tasks[0].task_id]
    assert result.normalized_plan["_meta"]["source"] == "coordinator"
    assert result.normalized_plan["tasks"][1]["dependencies"] == ["Inspect repository"]


def test_orchestrator_uses_local_fallback_when_planner_output_is_invalid(tmp_path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    session = orchestrator.create_session(tmp_path.as_posix(), "codex::gpt-5-codex")

    original_execute_task = orchestrator.connector_manager.execute_task

    def invalid_execute_task(**kwargs):
        task = kwargs["task"]
        result = original_execute_task(**kwargs)
        if task.title == "Generate structured orchestration plan":
            result.raw_output = "not json"
            result.summary = "planner failed"
        return result

    orchestrator.connector_manager.execute_task = invalid_execute_task  # type: ignore[method-assign]
    result = orchestrator.orchestrate(session.session_id, "Inspect and update the orchestrator.")

    assert result["planning"]["normalized_plan"]["_meta"]["source"] == "local_fallback"
    assert result["planning"]["normalized_plan"]["_meta"]["synthesized_locally"] is True
    assert result["planning"]["validation_errors"]
    assert result["session"]["metadata"]["planning"]["normalized_plan"]["_meta"]["source"] == "local_fallback"
    assert len(result["plan"]) == 4
