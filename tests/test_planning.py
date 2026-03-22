from orgemage.models import AgentCapabilities, DownstreamAgentConfig, ModelOption, OrchestrationTurnState, TaskStatus, TurnStatus, WorkerResult
from orgemage.orchestrator import Orchestrator
from orgemage.planning import optimize_coordinator_plan, parse_coordinator_plan, synthesize_local_fallback_plan
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


def test_parse_coordinator_plan_extracts_json_from_mixed_output() -> None:
    raw_output = """
    I will analyze the request and return the plan below.

    {
      "tasks": [
        {
          "title": "Inspect repository",
          "details": "Read the impacted files.",
          "dependencies": [],
          "required_capabilities": {"needsFilesystem": true, "commands": ["read"]},
          "assignee_hints": ["codex"],
          "acceptable_models": [],
          "priority": 80,
          "_meta": {"why": "Need context first"}
        }
      ],
      "_meta": {"planner": "coordinator", "provenance": "unit-test"}
    }
    """

    result = parse_coordinator_plan(raw_output, coordinator_agent_id="codex")

    assert result.is_valid is True
    assert len(result.tasks) == 1
    assert result.tasks[0].title == "Inspect repository"


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
    result = orchestrator.orchestrate_turn(session.session_id, "Inspect and update the orchestrator.")

    assert result["planning"]["normalized_plan"]["_meta"]["source"] == "local_fallback"
    assert result["planning"]["normalized_plan"]["_meta"]["synthesized_locally"] is True
    assert result["planning"]["validation_errors"]
    assert result["session"]["metadata"]["planning"]["normalized_plan"]["_meta"]["source"] == "local_fallback"
    assert len(result["plan"]) == 4


def test_local_fallback_uses_direct_response_for_conversational_prompt() -> None:
    result = synthesize_local_fallback_plan("Кто ты? Что ты можешь?", coordinator_agent_id="qwen")

    assert result.errors == []
    assert len(result.tasks) == 1
    assert result.tasks[0].assignee == "qwen"
    assert result.tasks[0].title == "Respond directly to the user"
    assert result.normalized_plan["_meta"]["fallback_mode"] == "direct_response"


def test_local_fallback_uses_read_only_response_for_read_only_analysis_prompt() -> None:
    result = synthesize_local_fallback_plan(
        "Без изменения файлов и без вмешательства в OS: проанализируй проект и назови ключевые модули.",
        coordinator_agent_id="qwen",
    )

    assert result.errors == []
    assert len(result.tasks) == 1
    assert result.tasks[0].title == "Respond directly to the user"
    assert result.tasks[0].required_capabilities == {"needsFilesystem": True, "commands": ["read", "search"]}
    assert result.normalized_plan["_meta"]["fallback_mode"] == "read_only_response"


def test_orchestrator_uses_local_fallback_when_planner_execution_fails(tmp_path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    session = orchestrator.create_session(tmp_path.as_posix(), "codex::gpt-5-codex")
    turn = OrchestrationTurnState(turn_id="turn-test", status=TurnStatus.RUNNING)

    def failed_execute_task(**kwargs):
        task = kwargs["task"]
        assert task.title == "Generate structured orchestration plan"
        return WorkerResult(
            task_id=task.task_id,
            agent_id="codex",
            status=TaskStatus.FAILED,
            summary="Connection error.",
            raw_output="Connection error.",
            metadata={"promptMetadata": dict(task._meta)},
        )

    orchestrator.connector_manager.execute_task = failed_execute_task  # type: ignore[method-assign]
    plan_parse_result, planner_record = orchestrator._generate_plan(  # type: ignore[attr-defined]
        snapshot=session,
        selected_model="gpt-5-codex",
        coordinator_agent=_agents()[0],
        user_prompt="Inspect and update the orchestrator.",
        turn=turn,
    )

    assert planner_record["planner_result"]["status"] == "failed"
    assert plan_parse_result.normalized_plan["_meta"]["source"] == "local_fallback"
    assert "Coordinator execution failed: Connection error." in planner_record["validation_errors"]


def test_optimize_coordinator_plan_collapses_single_agent_read_only_graph() -> None:
    raw_output = """
    {
      "tasks": [
        {
          "title": "Inspect repository",
          "details": "Read README and key files.",
          "dependencies": [],
          "required_capabilities": {"needsFilesystem": true, "commands": ["read", "list"]},
          "assignee_hints": ["qwen"],
          "priority": 90,
          "_meta": {"why": "Need context"}
        },
        {
          "title": "Search code patterns",
          "details": "Find important classes and functions.",
          "dependencies": ["Inspect repository"],
          "required_capabilities": {"needsFilesystem": true, "commands": ["grep", "glob"]},
          "assignee_hints": ["qwen"],
          "priority": 80,
          "_meta": {"why": "Need examples"}
        }
      ],
      "_meta": {"planner": "coordinator", "provenance": "unit-test"}
    }
    """

    parsed = parse_coordinator_plan(raw_output, coordinator_agent_id="qwen")
    optimized = optimize_coordinator_plan(
        parsed,
        coordinator_agent_id="qwen",
        user_prompt="Продемонстрируй возможности без вмешательства в OS.",
    )

    assert parsed.is_valid is True
    assert len(parsed.tasks) == 2
    assert len(optimized.tasks) == 1
    assert optimized.tasks[0].title == "Respond directly to the user"
    assert optimized.normalized_plan["_meta"]["optimized"] is True
    assert optimized.normalized_plan["_meta"]["original_task_count"] == 2
    assert optimized.tasks[0].required_capabilities["commands"] == ["glob", "grep", "list", "read"]


def test_optimize_coordinator_plan_collapses_read_only_user_request_even_with_write_task() -> None:
    raw_output = """
    {
      "tasks": [
        {
          "title": "Inspect repository",
          "details": "Read project files.",
          "dependencies": [],
          "required_capabilities": {"needsFilesystem": true, "commands": ["read"]},
          "assignee_hints": ["qwen"],
          "priority": 90,
          "_meta": {"why": "Need context"}
        },
        {
          "title": "Write report",
          "details": "Create a markdown report file.",
          "dependencies": ["Inspect repository"],
          "required_capabilities": {"needsFilesystem": true, "commands": ["read", "write"]},
          "assignee_hints": ["qwen"],
          "priority": 80,
          "_meta": {"why": "Need deliverable"}
        }
      ],
      "_meta": {"planner": "coordinator", "provenance": "unit-test"}
    }
    """

    parsed = parse_coordinator_plan(raw_output, coordinator_agent_id="qwen")
    optimized = optimize_coordinator_plan(
        parsed,
        coordinator_agent_id="qwen",
        user_prompt="Продемострируй свои возможности. Без вмешательства в работу OS.",
    )

    assert len(optimized.tasks) == 1
    assert optimized.tasks[0].required_capabilities["commands"] == ["read"]
    assert optimized.normalized_plan["_meta"]["optimized"] is True
