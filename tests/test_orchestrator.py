from pathlib import Path
import threading

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


def test_orchestrator_runs_ready_tasks_in_parallel_and_unblocks_dependencies(tmp_path: Path) -> None:
    import json
    import threading
    import time

    from orgemage.acp.downstream_client import DownstreamPromptResult
    from orgemage.acp.manager import DownstreamConnectorManager
    from orgemage.models import TaskStatus

    active_lock = threading.Lock()
    active_workers = 0
    max_parallelism = 0

    class _ParallelConnector:
        def __init__(self, agent: DownstreamAgentConfig) -> None:
            self.agent = agent
            self.negotiated_state = None
            self.started: dict[str, float] = {}
            self.finished: dict[str, float] = {}

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

        def execute_task(self, *, task, orchestrator_session_id, downstream_session_id, cwd, coordinator_prompt, selected_model):
            del orchestrator_session_id, downstream_session_id, cwd, coordinator_prompt, selected_model
            if task._meta.get("phase") == "planning":
                return DownstreamPromptResult(
                    downstream_session_id="parallel-session",
                    status=TaskStatus.COMPLETED,
                    summary="Generated structured orchestration plan.",
                    raw_output=json.dumps(
                        {
                            "tasks": [
                                {
                                    "title": "Task A",
                                    "details": "Run branch A.",
                                    "dependencies": [],
                                    "required_capabilities": {"needsTerminal": True},
                                    "assignee_hints": ["codex"],
                                    "acceptable_models": [],
                                    "priority": 90,
                                    "_meta": {},
                                },
                                {
                                    "title": "Task B",
                                    "details": "Run branch B.",
                                    "dependencies": [],
                                    "required_capabilities": {"needsTerminal": True},
                                    "assignee_hints": ["qwen"],
                                    "acceptable_models": [],
                                    "priority": 80,
                                    "_meta": {},
                                },
                                {
                                    "title": "Task C",
                                    "details": "Join both branches.",
                                    "dependencies": ["Task A", "Task B"],
                                    "required_capabilities": {"needsFilesystem": True},
                                    "assignee_hints": ["codex"],
                                    "acceptable_models": [],
                                    "priority": 70,
                                    "_meta": {},
                                },
                            ],
                            "_meta": {"planner": "test"},
                        }
                    ),
                )
            nonlocal active_workers, max_parallelism
            with active_lock:
                active_workers += 1
                max_parallelism = max(max_parallelism, active_workers)
                self.started[task.title] = time.monotonic()
            time.sleep(0.15)
            with active_lock:
                active_workers -= 1
                self.finished[task.title] = time.monotonic()
            return DownstreamPromptResult(
                downstream_session_id=f"parallel-{self.agent.agent_id}",
                status=TaskStatus.COMPLETED,
                summary=f"{task.title} complete",
            )

        def cancel(self, downstream_session_id: str) -> None:
            del downstream_session_id
            return None

    created: list[_ParallelConnector] = []

    def factory(agent: DownstreamAgentConfig) -> _ParallelConnector:
        connector = _ParallelConnector(agent)
        created.append(connector)
        return connector

    manager = DownstreamConnectorManager(_agents(), connector_factory=factory)
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"), connector_manager=manager)
    session = orchestrator.create_session(tmp_path.as_posix(), "codex::gpt-5-codex")
    updates: list[dict[str, object]] = []

    result = orchestrator.orchestrate_turn(session.session_id, "Run parallel execution.", emit_update=updates.append)

    assert result["summary"] == "Completed 3/3 orchestrated tasks."
    plan_by_title = {task["title"]: task for task in result["plan"]}
    assert plan_by_title["Task C"]["dependency_ids"] == [plan_by_title["Task A"]["task_id"], plan_by_title["Task B"]["task_id"]]
    plan_updates = [update for update in updates if update["sessionUpdate"] == "plan"]
    assert len(plan_updates) >= 4
    task_c_states = [
        next(task for task in update["plan"] if task["title"] == "Task C")
        for update in plan_updates
        if any(task["title"] == "Task C" for task in update["plan"])
    ]
    assert task_c_states[0]["status"] == "pending"
    assert task_c_states[0]["dependency_ids"]
    assert any(task["status"] == "in_progress" for task in task_c_states)
    starts = {title: timestamp for connector in created for title, timestamp in connector.started.items()}
    finishes = {title: timestamp for connector in created for title, timestamp in connector.finished.items()}
    assert max_parallelism >= 2
    assert starts["Task C"] >= finishes["Task A"]
    assert starts["Task C"] >= finishes["Task B"]



def test_orchestrator_applies_fail_fast_and_continue_failure_policies(tmp_path: Path) -> None:
    import json

    from orgemage.acp.downstream_client import DownstreamPromptResult
    from orgemage.acp.manager import DownstreamConnectorManager
    from orgemage.models import TaskStatus

    class _PolicyConnector:
        def __init__(self, agent: DownstreamAgentConfig) -> None:
            self.agent = agent
            self.negotiated_state = None

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

        def execute_task(self, *, task, orchestrator_session_id, downstream_session_id, cwd, coordinator_prompt, selected_model):
            del orchestrator_session_id, downstream_session_id, cwd, coordinator_prompt, selected_model
            if task._meta.get("phase") == "planning":
                return DownstreamPromptResult(
                    downstream_session_id="policy-session",
                    status=TaskStatus.COMPLETED,
                    summary="Generated structured orchestration plan.",
                    raw_output=json.dumps(
                        {
                            "tasks": [
                                {
                                    "title": "Critical task",
                                    "details": "This coordinator-critical task fails.",
                                    "dependencies": [],
                                    "required_capabilities": {"needsFilesystem": True},
                                    "assignee_hints": ["codex"],
                                    "acceptable_models": [],
                                    "priority": 90,
                                    "_meta": {"coordinator_critical": True, "execution_role": "coordinator"},
                                },
                                {
                                    "title": "Blocked dependent",
                                    "details": "Should never start after fail-fast.",
                                    "dependencies": ["Critical task"],
                                    "required_capabilities": {"needsFilesystem": True},
                                    "assignee_hints": ["qwen"],
                                    "acceptable_models": [],
                                    "priority": 80,
                                    "_meta": {},
                                },
                                {
                                    "title": "Worker failure",
                                    "details": "Fails but should not stop independent work.",
                                    "dependencies": [],
                                    "required_capabilities": {"needsTerminal": True},
                                    "assignee_hints": ["qwen"],
                                    "acceptable_models": [],
                                    "priority": 70,
                                    "_meta": {"failure_policy": "continue"},
                                },
                                {
                                    "title": "Independent success",
                                    "details": "Continues despite worker failure.",
                                    "dependencies": [],
                                    "required_capabilities": {"needsTerminal": True},
                                    "assignee_hints": ["codex"],
                                    "acceptable_models": [],
                                    "priority": 60,
                                    "_meta": {"failure_policy": "continue"},
                                },
                            ],
                            "_meta": {"planner": "test"},
                        }
                    ),
                )
            status_map = {
                "Critical task": TaskStatus.FAILED,
                "Worker failure": TaskStatus.FAILED,
                "Independent success": TaskStatus.COMPLETED,
            }
            return DownstreamPromptResult(
                downstream_session_id=f"policy-{self.agent.agent_id}",
                status=status_map[task.title],
                summary=f"{task.title} -> {status_map[task.title].value}",
            )

        def cancel(self, downstream_session_id: str) -> None:
            del downstream_session_id
            return None

    manager = DownstreamConnectorManager(_agents(), connector_factory=_PolicyConnector)
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"), connector_manager=manager)
    session = orchestrator.create_session(tmp_path.as_posix(), "codex::gpt-5-codex")

    result = orchestrator.orchestrate_turn(session.session_id, "Exercise failure policies.")

    plan_by_title = {task["title"]: task for task in result["plan"]}
    assert plan_by_title["Critical task"]["status"] == "failed"
    assert plan_by_title["Blocked dependent"]["status"] == "cancelled"
    assert plan_by_title["Worker failure"]["status"] == "failed"
    assert plan_by_title["Independent success"]["status"] == "completed"
    session_snapshot = orchestrator.load_session(session.session_id)
    states = {task.title: task for task in session_snapshot.task_states}
    assert states["Blocked dependent"].plan_metadata["status_reason"] in {"fail_fast", "dependency_failure"}
    assert states["Blocked dependent"].dependency_state == "blocked"


def test_orchestrator_marks_active_turn_and_tasks_cancelled_on_session_cancel(tmp_path: Path) -> None:
    import json
    import threading

    from orgemage.acp.downstream_client import DownstreamPromptResult
    from orgemage.acp.manager import DownstreamConnectorManager
    from orgemage.models import TaskStatus

    planning_started = threading.Event()
    worker_entered = threading.Event()
    worker_release = threading.Event()
    worker_started_titles: list[str] = []

    class _CancelableConnector:
        def __init__(self, agent: DownstreamAgentConfig) -> None:
            self.agent = agent
            self.negotiated_state = None
            self.cancelled: list[str] = []

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

        def execute_task(self, *, task, orchestrator_session_id, downstream_session_id, cwd, coordinator_prompt, selected_model):
            del orchestrator_session_id, downstream_session_id, cwd, coordinator_prompt, selected_model
            if task._meta.get("phase") == "planning":
                planning_started.set()
                return DownstreamPromptResult(
                    downstream_session_id="cancel-session",
                    status=TaskStatus.COMPLETED,
                    summary="Generated structured orchestration plan.",
                    raw_output=json.dumps(
                        {
                            "tasks": [
                                {
                                    "title": "Long task",
                                    "details": "Runs until cancelled.",
                                    "dependencies": [],
                                    "required_capabilities": {"needsTerminal": True},
                                    "assignee_hints": ["codex"],
                                    "acceptable_models": [],
                                    "priority": 90,
                                    "_meta": {},
                                },
                                {
                                    "title": "Blocked after cancel",
                                    "details": "Must never start.",
                                    "dependencies": ["Long task"],
                                    "required_capabilities": {"needsFilesystem": True},
                                    "assignee_hints": ["qwen"],
                                    "acceptable_models": [],
                                    "priority": 80,
                                    "_meta": {},
                                },
                            ],
                            "_meta": {"planner": "test"},
                        }
                    ),
                )
            worker_started_titles.append(task.title)
            worker_entered.set()
            worker_release.wait(timeout=5)
            return DownstreamPromptResult(
                downstream_session_id=f"cancel-{self.agent.agent_id}",
                status=TaskStatus.CANCELLED,
                summary="Task cancelled.",
                response={"stop_reason": "cancelled"},
            )

        def cancel(self, downstream_session_id: str) -> None:
            self.cancelled.append(downstream_session_id)
            worker_release.set()

    created: list[_CancelableConnector] = []

    def factory(agent: DownstreamAgentConfig) -> _CancelableConnector:
        connector = _CancelableConnector(agent)
        created.append(connector)
        return connector

    store = SQLiteSessionStore(tmp_path / "state.db")
    manager = DownstreamConnectorManager(_agents(), connector_factory=factory, store=store)
    orchestrator = Orchestrator(_agents(), store, connector_manager=manager)
    session = orchestrator.create_session(tmp_path.as_posix(), "codex::gpt-5-codex")
    result_holder: dict[str, object] = {}

    thread = threading.Thread(
        target=lambda: result_holder.setdefault("result", orchestrator.orchestrate_turn(session.session_id, "Cancel after launch.")),
        daemon=True,
    )
    thread.start()
    assert planning_started.wait(timeout=5)
    assert worker_entered.wait(timeout=5)

    orchestrator.cancel(session.session_id)
    thread.join(timeout=5)

    assert thread.is_alive() is False
    result = result_holder["result"]
    assert result["stop_reason"] == "cancelled"
    assert result["summary"] == "Turn cancelled."
    plan_by_title = {task["title"]: task for task in result["plan"]}
    assert plan_by_title["Long task"]["status"] == "cancelled"
    assert plan_by_title["Blocked after cancel"]["status"] == "cancelled"
    assert worker_started_titles == ["Long task"]
    snapshot = orchestrator.load_session(session.session_id)
    assert snapshot.turns[-1].status.value == "cancelled"
    assert snapshot.turns[-1].stop_reason == "cancelled"
    states = {task.title: task for task in snapshot.task_states}
    assert states["Long task"].status.value == "cancelled"
    assert states["Long task"].plan_metadata["status_reason"] == "turn_cancelled"
    assert states["Blocked after cancel"].status.value == "cancelled"
    assert states["Blocked after cancel"].plan_metadata["status_reason"] == "turn_cancelled"
    assert any(connector.cancelled for connector in created)


def test_orchestrator_propagates_prompt_metadata_into_plan_and_tool_updates(tmp_path: Path) -> None:
    orchestrator = Orchestrator(_agents(), SQLiteSessionStore(tmp_path / "state.db"))
    session = orchestrator.create_session(tmp_path.as_posix(), "codex::gpt-5-codex")
    updates: list[dict[str, object]] = []

    result = orchestrator.orchestrate_turn(
        session.session_id,
        "Investigate session metadata propagation.",
        emit_update=updates.append,
        prompt_metadata={
            "traceId": "trace-123",
            "traceparent": "00-abc-123-01",
            "policyAnnotations": {"mode": "safe"},
        },
    )

    plan_update = next(update for update in updates if update["sessionUpdate"] == "plan")
    first_plan_item = plan_update["globalPlan"][0]
    assert first_plan_item["_meta"]["traceId"] == "trace-123"
    assert first_plan_item["_meta"]["traceparent"] == "00-abc-123-01"
    assert first_plan_item["_meta"]["turnId"].startswith("turn-")
    assert "workerCorrelationId" in first_plan_item["_meta"]
    assert "planningProvenance" in first_plan_item["_meta"]
    assert first_plan_item["_meta"]["policyAnnotations"]["mode"] == "safe"

    tool_update = next(update for update in updates if update["sessionUpdate"] == "tool_call")
    assert tool_update["toolCall"]["_meta"]["traceId"] == "trace-123"
    assert tool_update["toolCall"]["_meta"]["traceparent"] == "00-abc-123-01"
    assert tool_update["toolCall"]["_meta"]["assignee"]["agentId"]
    assert result["session"]["metadata"]["session_summary"].startswith("Completed")


