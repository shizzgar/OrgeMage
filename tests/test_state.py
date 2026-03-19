from pathlib import Path
import sqlite3

from orgemage.models import (
    OrchestrationTurnState,
    PermissionRequestState,
    SessionSnapshot,
    TaskExecutionState,
    TaskStatus,
    TerminalMapping,
    TraceCorrelationState,
    TurnStatus,
)
from orgemage.state import SQLiteSessionStore


def test_sqlite_session_store_round_trip_with_runtime_state(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "orgemage.db")
    snapshot = SessionSnapshot(session_id="s1", cwd="/tmp/project", selected_model="codex::gpt-5-codex")
    snapshot.metadata = {"traceparent": "abc"}
    snapshot.set_downstream_session_mapping("codex", "downstream-1", metadata={"source": "test"})
    snapshot.turns.append(OrchestrationTurnState(turn_id="turn-1", status=TurnStatus.COMPLETED, stop_reason="end_turn"))
    snapshot.task_states.append(
        TaskExecutionState(
            task_id="t1",
            title="Implement",
            details="Change the store",
            parent_turn_id="turn-1",
            assignee="codex",
            status=TaskStatus.COMPLETED,
            output="done",
            plan_metadata={"step": 1},
        )
    )
    snapshot.terminal_mappings.append(
        TerminalMapping(
            upstream_terminal_id="up-1",
            downstream_terminal_id="down-1",
            owner_task_id="t1",
            refcount=2,
        )
    )
    snapshot.permission_requests.append(
        PermissionRequestState(
            request_id="perm-1",
            owner_task_id="t1",
            status="decided",
            decision="allow",
            payload={"scope": "filesystem"},
        )
    )
    snapshot.trace_metadata.append(
        TraceCorrelationState(trace_key="trace-1", turn_id="turn-1", task_id="t1", metadata={"span": "abc"})
    )

    store.save(snapshot)
    loaded = store.load("s1")

    assert loaded is not None
    assert loaded.selected_model == "codex::gpt-5-codex"
    assert loaded.metadata["traceparent"] == "abc"
    assert loaded.downstream_session_map() == {"codex": "downstream-1"}
    assert loaded.turns[0].turn_id == "turn-1"
    assert loaded.task_graph == [{
        "title": "Implement",
        "details": "Change the store",
        "required_capabilities": {},
        "acceptable_models": [],
        "dependency_ids": [],
        "assignee_hints": [],
        "_meta": {},
        "assignee": "codex",
        "task_id": "t1",
        "priority": 0,
        "status": "completed",
        "output": "done",
    }]
    assert loaded.terminal_mappings[0].downstream_terminal_id == "down-1"
    assert loaded.permission_requests[0].decision == "allow"
    assert loaded.trace_metadata[0].metadata == {"span": "abc"}


def test_store_incremental_runtime_methods_and_legacy_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "orgemage.db"
    store = SQLiteSessionStore(db_path)
    store.save(SessionSnapshot(session_id="legacy", cwd="/tmp/project"))

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE sessions SET task_graph_json = ?, metadata_json = ? WHERE session_id = ?
            """,
            (
                '[{"task_id":"legacy-task","title":"Legacy","details":"Imported","status":"completed","assignee":"codex"}]',
                '{"downstream_sessions":{"codex":"downstream-legacy"},"trace_metadata":{"root":{"traceparent":"abc"}}}',
                "legacy",
            ),
        )

    migrated = store.load("legacy")
    assert migrated is not None
    assert migrated.downstream_session_map()["codex"] == "downstream-legacy"
    assert migrated.task_states[0].task_id == "legacy-task"
    assert migrated.trace_metadata[0].trace_key == "root"

    turn = store.create_or_update_turn_state("legacy", OrchestrationTurnState(turn_id="turn-2", status=TurnStatus.RUNNING))
    task = store.persist_task_update(
        "legacy",
        TaskExecutionState(
            task_id="task-2",
            title="Task 2",
            details="Incremental",
            parent_turn_id=turn.turn_id,
            assignee="codex",
            status=TaskStatus.IN_PROGRESS,
            plan_metadata={"phase": "run"},
        ),
    )
    permission = store.persist_permission_event(
        "legacy",
        PermissionRequestState(
            request_id="perm-2",
            owner_task_id=task.task_id,
            status="decided",
            decision="allow",
            payload={"command": "pytest"},
        ),
    )
    terminal = store.persist_terminal_event(
        "legacy",
        TerminalMapping(
            upstream_terminal_id="up-2",
            downstream_terminal_id="down-2",
            owner_task_id=task.task_id,
            status="released",
        ),
    )
    trace = store.persist_trace_metadata(
        "legacy",
        TraceCorrelationState(trace_key="task-2-trace", turn_id=turn.turn_id, task_id=task.task_id, metadata={"span": "def"}),
    )
    mapping = store.save_downstream_session_mapping("legacy", "qwen", "downstream-qwen")

    reloaded = store.load("legacy")
    assert reloaded is not None
    assert any(current.turn_id == turn.turn_id for current in reloaded.turns)
    assert any(current.task_id == task.task_id for current in reloaded.task_states)
    assert any(current.request_id == permission.request_id for current in reloaded.permission_requests)
    assert any(current.upstream_terminal_id == terminal.upstream_terminal_id for current in reloaded.terminal_mappings)
    assert any(current.trace_key == trace.trace_key for current in reloaded.trace_metadata)
    assert reloaded.downstream_session_map()[mapping.agent_id] == mapping.downstream_session_id


def test_store_can_cancel_outstanding_permissions_and_terminals(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "orgemage.db")
    snapshot = SessionSnapshot(session_id="cancelled", cwd="/tmp/project")
    store.save(snapshot)

    store.persist_permission_event(
        "cancelled",
        PermissionRequestState(
            request_id="perm-requested",
            owner_task_id="task-1",
            status="requested",
            payload={"scope": "terminal"},
        ),
    )
    store.persist_permission_event(
        "cancelled",
        PermissionRequestState(
            request_id="perm-decided",
            owner_task_id="task-2",
            status="decided",
            decision="allow",
            payload={"scope": "filesystem"},
        ),
    )
    store.persist_terminal_event(
        "cancelled",
        TerminalMapping(
            upstream_terminal_id="up-active",
            downstream_terminal_id="down-active",
            owner_task_id="task-1",
            owner_agent_id="codex",
            status="active",
        ),
    )

    cancelled_permissions = store.cancel_permission_requests(
        "cancelled",
        owner_task_ids={"task-1"},
        metadata={"cancel_reason": "test"},
    )
    cancelled_terminals = store.mark_terminal_mappings_cancelled(
        "cancelled",
        owner_task_ids={"task-1"},
        metadata={"cleanup_reason": "test"},
    )

    loaded = store.load("cancelled")
    assert loaded is not None
    assert cancelled_permissions[0].status == "cancelled"
    assert cancelled_permissions[0].decision == "cancelled"
    assert loaded.permission_requests[0].metadata["cancel_reason"] == "test"
    assert loaded.permission_requests[1].status == "decided"
    assert cancelled_terminals[0].status == "cancelled"
    assert loaded.terminal_mappings[0].metadata["cleanup_reason"] == "test"
