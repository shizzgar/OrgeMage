from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .models import (
    DownstreamSessionMapping,
    OrchestrationTurnState,
    PermissionRequestState,
    SessionSnapshot,
    TaskExecutionState,
    TaskStatus,
    TerminalMapping,
    TraceCorrelationState,
)


class SQLiteSessionStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(":memory:")
                self._memory_connection.row_factory = sqlite3.Row
            return self._memory_connection
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    cwd TEXT NOT NULL,
                    selected_model TEXT,
                    coordinator_agent_id TEXT,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    task_graph_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS downstream_session_mappings (
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    downstream_session_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (session_id, agent_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orchestration_turns (
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stop_reason TEXT,
                    started_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (session_id, turn_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_execution_state (
                    session_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    parent_turn_id TEXT,
                    assignee TEXT,
                    dependency_state TEXT NOT NULL,
                    title TEXT NOT NULL,
                    details TEXT NOT NULL,
                    required_capabilities_json TEXT NOT NULL DEFAULT '{}',
                    acceptable_models_json TEXT NOT NULL DEFAULT '[]',
                    dependency_ids_json TEXT NOT NULL DEFAULT '[]',
                    priority INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    output TEXT NOT NULL DEFAULT '',
                    plan_metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, task_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS terminal_mappings (
                    session_id TEXT NOT NULL,
                    upstream_terminal_id TEXT NOT NULL,
                    downstream_terminal_id TEXT NOT NULL,
                    owner_task_id TEXT,
                    owner_agent_id TEXT,
                    refcount INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, upstream_terminal_id, downstream_terminal_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS permission_requests (
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    owner_task_id TEXT,
                    status TEXT NOT NULL,
                    decision TEXT,
                    requested_at REAL NOT NULL,
                    decided_at REAL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (session_id, request_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trace_correlation_metadata (
                    session_id TEXT NOT NULL,
                    trace_key TEXT NOT NULL,
                    turn_id TEXT,
                    task_id TEXT,
                    parent_trace_key TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, trace_key),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )

    def save(self, snapshot: SessionSnapshot) -> None:
        snapshot.updated_at = time.time()
        with self._connect() as conn:
            self._save_session_header(conn, snapshot)
            self._replace_runtime_state(conn, snapshot)

    def _save_session_header(self, conn: sqlite3.Connection, snapshot: SessionSnapshot) -> None:
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, cwd, selected_model, coordinator_agent_id, title,
                created_at, updated_at, task_graph_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                cwd=excluded.cwd,
                selected_model=excluded.selected_model,
                coordinator_agent_id=excluded.coordinator_agent_id,
                title=excluded.title,
                updated_at=excluded.updated_at,
                task_graph_json=excluded.task_graph_json,
                metadata_json=excluded.metadata_json
            """,
            (
                snapshot.session_id,
                snapshot.cwd,
                snapshot.selected_model,
                snapshot.coordinator_agent_id,
                snapshot.title,
                snapshot.created_at,
                snapshot.updated_at,
                json.dumps(snapshot.task_graph),
                json.dumps(snapshot.metadata),
            ),
        )

    def _replace_runtime_state(self, conn: sqlite3.Connection, snapshot: SessionSnapshot) -> None:
        for table in (
            "downstream_session_mappings",
            "orchestration_turns",
            "task_execution_state",
            "terminal_mappings",
            "permission_requests",
            "trace_correlation_metadata",
        ):
            conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (snapshot.session_id,))
        for mapping in snapshot.downstream_session_mappings:
            self._persist_downstream_session_mapping(conn, snapshot.session_id, mapping)
        for turn in snapshot.turns:
            self._persist_turn_state(conn, snapshot.session_id, turn)
        for task_state in snapshot.task_states:
            self._persist_task_state(conn, snapshot.session_id, task_state)
        for mapping in snapshot.terminal_mappings:
            self._persist_terminal_mapping(conn, snapshot.session_id, mapping)
        for request in snapshot.permission_requests:
            self._persist_permission_request(conn, snapshot.session_id, request)
        for trace in snapshot.trace_metadata:
            self._persist_trace_metadata(conn, snapshot.session_id, trace)

    def load(self, session_id: str) -> SessionSnapshot | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, cwd, selected_model, coordinator_agent_id, title,
                       created_at, updated_at, task_graph_json, metadata_json
                FROM sessions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            self._migrate_legacy_session_state(conn, row)
            return self._hydrate_snapshot(conn, row)

    def list_sessions(self) -> list[SessionSnapshot]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, cwd, selected_model, coordinator_agent_id, title,
                       created_at, updated_at, task_graph_json, metadata_json
                FROM sessions ORDER BY updated_at DESC
                """
            ).fetchall()
            snapshots: list[SessionSnapshot] = []
            for row in rows:
                self._migrate_legacy_session_state(conn, row)
                snapshots.append(self._hydrate_snapshot(conn, row))
            return snapshots

    def load_downstream_session_mapping(self, session_id: str, agent_id: str) -> DownstreamSessionMapping | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT agent_id, downstream_session_id, created_at, updated_at, metadata_json
                FROM downstream_session_mappings
                WHERE session_id = ? AND agent_id = ?
                """,
                (session_id, agent_id),
            ).fetchone()
        if row is None:
            return None
        return DownstreamSessionMapping(
            agent_id=row["agent_id"],
            downstream_session_id=row["downstream_session_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads_dict(row["metadata_json"]),
        )

    def save_downstream_session_mapping(
        self,
        session_id: str,
        agent_id: str,
        downstream_session_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> DownstreamSessionMapping:
        now = time.time()
        mapping = DownstreamSessionMapping(
            agent_id=agent_id,
            downstream_session_id=downstream_session_id,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        with self._connect() as conn:
            self._persist_downstream_session_mapping(conn, session_id, mapping)
            self._touch_session(conn, session_id, now)
        return self.load_downstream_session_mapping(session_id, agent_id) or mapping

    def create_or_update_turn_state(self, session_id: str, turn: OrchestrationTurnState) -> OrchestrationTurnState:
        turn.updated_at = time.time()
        with self._connect() as conn:
            self._persist_turn_state(conn, session_id, turn)
            self._touch_session(conn, session_id, turn.updated_at)
        return turn

    def persist_task_update(self, session_id: str, task_state: TaskExecutionState) -> TaskExecutionState:
        task_state.updated_at = time.time()
        with self._connect() as conn:
            existing_row = conn.execute(
                "SELECT created_at FROM task_execution_state WHERE session_id = ? AND task_id = ?",
                (session_id, task_state.task_id),
            ).fetchone()
            if existing_row is not None:
                task_state.created_at = existing_row["created_at"]
            self._persist_task_state(conn, session_id, task_state)
            self._update_legacy_task_graph(conn, session_id)
            self._touch_session(conn, session_id, task_state.updated_at)
        return task_state

    def persist_permission_event(self, session_id: str, request: PermissionRequestState) -> PermissionRequestState:
        if request.decision and request.decided_at is None:
            request.decided_at = time.time()
        with self._connect() as conn:
            self._persist_permission_request(conn, session_id, request)
            self._touch_session(conn, session_id, request.decided_at or request.requested_at)
        return request

    def persist_terminal_event(self, session_id: str, mapping: TerminalMapping) -> TerminalMapping:
        mapping.updated_at = time.time()
        with self._connect() as conn:
            existing_row = conn.execute(
                """
                SELECT created_at FROM terminal_mappings
                WHERE session_id = ? AND upstream_terminal_id = ? AND downstream_terminal_id = ?
                """,
                (session_id, mapping.upstream_terminal_id, mapping.downstream_terminal_id),
            ).fetchone()
            if existing_row is not None:
                mapping.created_at = existing_row["created_at"]
            self._persist_terminal_mapping(conn, session_id, mapping)
            self._touch_session(conn, session_id, mapping.updated_at)
        return mapping

    def persist_trace_metadata(self, session_id: str, trace: TraceCorrelationState) -> TraceCorrelationState:
        trace.updated_at = time.time()
        with self._connect() as conn:
            existing_row = conn.execute(
                "SELECT created_at FROM trace_correlation_metadata WHERE session_id = ? AND trace_key = ?",
                (session_id, trace.trace_key),
            ).fetchone()
            if existing_row is not None:
                trace.created_at = existing_row["created_at"]
            self._persist_trace_metadata(conn, session_id, trace)
            self._touch_session(conn, session_id, trace.updated_at)
        return trace

    def _hydrate_snapshot(self, conn: sqlite3.Connection, row: sqlite3.Row) -> SessionSnapshot:
        snapshot = SessionSnapshot(
            session_id=row["session_id"],
            cwd=row["cwd"],
            selected_model=row["selected_model"],
            coordinator_agent_id=row["coordinator_agent_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads_dict(row["metadata_json"]),
            downstream_session_mappings=self._load_downstream_session_mappings(conn, row["session_id"]),
            turns=self._load_turns(conn, row["session_id"]),
            task_states=self._load_task_states(conn, row["session_id"]),
            terminal_mappings=self._load_terminal_mappings(conn, row["session_id"]),
            permission_requests=self._load_permission_requests(conn, row["session_id"]),
            trace_metadata=self._load_trace_metadata(conn, row["session_id"]),
        )
        return snapshot

    def _migrate_legacy_session_state(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        session_id = row["session_id"]
        has_runtime = any(
            conn.execute(f"SELECT 1 FROM {table} WHERE session_id = ? LIMIT 1", (session_id,)).fetchone()
            is not None
            for table in (
                "downstream_session_mappings",
                "orchestration_turns",
                "task_execution_state",
                "terminal_mappings",
                "permission_requests",
                "trace_correlation_metadata",
            )
        )
        if has_runtime:
            return
        task_graph = _loads_list(row["task_graph_json"])
        metadata = _loads_dict(row["metadata_json"])
        if not task_graph and not metadata:
            return
        now = time.time()
        for agent_id, downstream_session_id in metadata.get("downstream_sessions", {}).items():
            self._persist_downstream_session_mapping(
                conn,
                session_id,
                DownstreamSessionMapping(
                    agent_id=agent_id,
                    downstream_session_id=str(downstream_session_id),
                    created_at=now,
                    updated_at=now,
                    metadata={},
                ),
            )
        for index, payload in enumerate(task_graph):
            status_value = payload.get("status", TaskStatus.PENDING.value)
            try:
                status = TaskStatus(status_value)
            except ValueError:
                status = TaskStatus.PENDING
            known_keys = {
                "task_id",
                "title",
                "details",
                "parent_turn_id",
                "assignee",
                "dependency_state",
                "required_capabilities",
                "acceptable_models",
                "dependency_ids",
                "priority",
                "status",
                "output",
                "plan_metadata",
            }
            task_state = TaskExecutionState(
                task_id=str(payload.get("task_id", f"legacy-task-{index}")),
                title=str(payload.get("title", payload.get("task_id", f"Task {index + 1}"))),
                details=str(payload.get("details", "")),
                parent_turn_id=payload.get("parent_turn_id"),
                assignee=payload.get("assignee"),
                dependency_state=str(payload.get("dependency_state", "blocked" if payload.get("dependency_ids") else "ready")),
                plan_metadata=dict(payload.get("plan_metadata", {k: v for k, v in payload.items() if k not in known_keys})),
                required_capabilities=dict(payload.get("required_capabilities", {})),
                acceptable_models=list(payload.get("acceptable_models", [])),
                dependency_ids=list(payload.get("dependency_ids", [])),
                priority=int(payload.get("priority", 0)),
                status=status,
                output=str(payload.get("output", "")),
                created_at=now,
                updated_at=now,
            )
            self._persist_task_state(conn, session_id, task_state)
        for agent_id, negotiated in metadata.get("downstream_negotiated", {}).items():
            self._persist_trace_metadata(
                conn,
                session_id,
                TraceCorrelationState(
                    trace_key=f"negotiated:{agent_id}",
                    metadata={"agent_id": agent_id, "negotiated": negotiated},
                    created_at=now,
                    updated_at=now,
                ),
            )
        for trace_key, trace_payload in metadata.get("trace_metadata", {}).items():
            self._persist_trace_metadata(
                conn,
                session_id,
                TraceCorrelationState(
                    trace_key=str(trace_key),
                    metadata=dict(trace_payload if isinstance(trace_payload, dict) else {"value": trace_payload}),
                    created_at=now,
                    updated_at=now,
                ),
            )

    def _load_downstream_session_mappings(self, conn: sqlite3.Connection, session_id: str) -> list[DownstreamSessionMapping]:
        rows = conn.execute(
            """
            SELECT agent_id, downstream_session_id, created_at, updated_at, metadata_json
            FROM downstream_session_mappings
            WHERE session_id = ?
            ORDER BY updated_at ASC, agent_id ASC
            """,
            (session_id,),
        ).fetchall()
        return [
            DownstreamSessionMapping(
                agent_id=row["agent_id"],
                downstream_session_id=row["downstream_session_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=_loads_dict(row["metadata_json"]),
            )
            for row in rows
        ]

    def _load_turns(self, conn: sqlite3.Connection, session_id: str) -> list[OrchestrationTurnState]:
        rows = conn.execute(
            """
            SELECT turn_id, status, stop_reason, started_at, updated_at, metadata_json
            FROM orchestration_turns
            WHERE session_id = ?
            ORDER BY started_at ASC, turn_id ASC
            """,
            (session_id,),
        ).fetchall()
        return [
            OrchestrationTurnState(
                turn_id=row["turn_id"],
                status=row["status"],
                stop_reason=row["stop_reason"],
                started_at=row["started_at"],
                updated_at=row["updated_at"],
                metadata=_loads_dict(row["metadata_json"]),
            )
            for row in rows
        ]

    def _load_task_states(self, conn: sqlite3.Connection, session_id: str) -> list[TaskExecutionState]:
        rows = conn.execute(
            """
            SELECT task_id, parent_turn_id, assignee, dependency_state, title, details,
                   required_capabilities_json, acceptable_models_json, dependency_ids_json,
                   priority, status, output, plan_metadata_json, created_at, updated_at
            FROM task_execution_state
            WHERE session_id = ?
            ORDER BY created_at ASC, task_id ASC
            """,
            (session_id,),
        ).fetchall()
        task_states: list[TaskExecutionState] = []
        for row in rows:
            status_value = row["status"]
            try:
                status = TaskStatus(status_value)
            except ValueError:
                status = TaskStatus.PENDING
            task_states.append(
                TaskExecutionState(
                    task_id=row["task_id"],
                    parent_turn_id=row["parent_turn_id"],
                    assignee=row["assignee"],
                    dependency_state=row["dependency_state"],
                    title=row["title"],
                    details=row["details"],
                    required_capabilities=_loads_dict(row["required_capabilities_json"]),
                    acceptable_models=_loads_list(row["acceptable_models_json"]),
                    dependency_ids=_loads_list(row["dependency_ids_json"]),
                    priority=row["priority"],
                    status=status,
                    output=row["output"],
                    plan_metadata=_loads_dict(row["plan_metadata_json"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return task_states

    def _load_terminal_mappings(self, conn: sqlite3.Connection, session_id: str) -> list[TerminalMapping]:
        rows = conn.execute(
            """
            SELECT upstream_terminal_id, downstream_terminal_id, owner_task_id, owner_agent_id,
                   refcount, status, metadata_json, created_at, updated_at
            FROM terminal_mappings
            WHERE session_id = ?
            ORDER BY created_at ASC, upstream_terminal_id ASC, downstream_terminal_id ASC
            """,
            (session_id,),
        ).fetchall()
        return [
            TerminalMapping(
                upstream_terminal_id=row["upstream_terminal_id"],
                downstream_terminal_id=row["downstream_terminal_id"],
                owner_task_id=row["owner_task_id"],
                owner_agent_id=row["owner_agent_id"],
                refcount=row["refcount"],
                status=row["status"],
                metadata=_loads_dict(row["metadata_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def _load_permission_requests(self, conn: sqlite3.Connection, session_id: str) -> list[PermissionRequestState]:
        rows = conn.execute(
            """
            SELECT request_id, owner_task_id, status, decision, requested_at, decided_at, payload_json, metadata_json
            FROM permission_requests
            WHERE session_id = ?
            ORDER BY requested_at ASC, request_id ASC
            """,
            (session_id,),
        ).fetchall()
        return [
            PermissionRequestState(
                request_id=row["request_id"],
                owner_task_id=row["owner_task_id"],
                status=row["status"],
                decision=row["decision"],
                requested_at=row["requested_at"],
                decided_at=row["decided_at"],
                payload=_loads_dict(row["payload_json"]),
                metadata=_loads_dict(row["metadata_json"]),
            )
            for row in rows
        ]

    def _load_trace_metadata(self, conn: sqlite3.Connection, session_id: str) -> list[TraceCorrelationState]:
        rows = conn.execute(
            """
            SELECT trace_key, turn_id, task_id, parent_trace_key, metadata_json, created_at, updated_at
            FROM trace_correlation_metadata
            WHERE session_id = ?
            ORDER BY created_at ASC, trace_key ASC
            """,
            (session_id,),
        ).fetchall()
        return [
            TraceCorrelationState(
                trace_key=row["trace_key"],
                turn_id=row["turn_id"],
                task_id=row["task_id"],
                parent_trace_key=row["parent_trace_key"],
                metadata=_loads_dict(row["metadata_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def _persist_downstream_session_mapping(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        mapping: DownstreamSessionMapping,
    ) -> None:
        existing = conn.execute(
            "SELECT created_at FROM downstream_session_mappings WHERE session_id = ? AND agent_id = ?",
            (session_id, mapping.agent_id),
        ).fetchone()
        if existing is not None:
            mapping.created_at = existing["created_at"]
        conn.execute(
            """
            INSERT INTO downstream_session_mappings (
                session_id, agent_id, downstream_session_id, created_at, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, agent_id) DO UPDATE SET
                downstream_session_id=excluded.downstream_session_id,
                updated_at=excluded.updated_at,
                metadata_json=excluded.metadata_json
            """,
            (
                session_id,
                mapping.agent_id,
                mapping.downstream_session_id,
                mapping.created_at,
                mapping.updated_at,
                json.dumps(mapping.metadata),
            ),
        )

    def _persist_turn_state(self, conn: sqlite3.Connection, session_id: str, turn: OrchestrationTurnState) -> None:
        existing = conn.execute(
            "SELECT started_at FROM orchestration_turns WHERE session_id = ? AND turn_id = ?",
            (session_id, turn.turn_id),
        ).fetchone()
        if existing is not None:
            turn.started_at = existing["started_at"]
        conn.execute(
            """
            INSERT INTO orchestration_turns (
                session_id, turn_id, status, stop_reason, started_at, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, turn_id) DO UPDATE SET
                status=excluded.status,
                stop_reason=excluded.stop_reason,
                updated_at=excluded.updated_at,
                metadata_json=excluded.metadata_json
            """,
            (
                session_id,
                turn.turn_id,
                turn.status,
                turn.stop_reason,
                turn.started_at,
                turn.updated_at,
                json.dumps(turn.metadata),
            ),
        )

    def _persist_task_state(self, conn: sqlite3.Connection, session_id: str, task_state: TaskExecutionState) -> None:
        existing = conn.execute(
            "SELECT created_at FROM task_execution_state WHERE session_id = ? AND task_id = ?",
            (session_id, task_state.task_id),
        ).fetchone()
        if existing is not None:
            task_state.created_at = existing["created_at"]
        conn.execute(
            """
            INSERT INTO task_execution_state (
                session_id, task_id, parent_turn_id, assignee, dependency_state, title, details,
                required_capabilities_json, acceptable_models_json, dependency_ids_json,
                priority, status, output, plan_metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, task_id) DO UPDATE SET
                parent_turn_id=excluded.parent_turn_id,
                assignee=excluded.assignee,
                dependency_state=excluded.dependency_state,
                title=excluded.title,
                details=excluded.details,
                required_capabilities_json=excluded.required_capabilities_json,
                acceptable_models_json=excluded.acceptable_models_json,
                dependency_ids_json=excluded.dependency_ids_json,
                priority=excluded.priority,
                status=excluded.status,
                output=excluded.output,
                plan_metadata_json=excluded.plan_metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                session_id,
                task_state.task_id,
                task_state.parent_turn_id,
                task_state.assignee,
                task_state.dependency_state,
                task_state.title,
                task_state.details,
                json.dumps(task_state.required_capabilities),
                json.dumps(task_state.acceptable_models),
                json.dumps(task_state.dependency_ids),
                task_state.priority,
                task_state.status.value,
                task_state.output,
                json.dumps(task_state.plan_metadata),
                task_state.created_at,
                task_state.updated_at,
            ),
        )

    def _persist_terminal_mapping(self, conn: sqlite3.Connection, session_id: str, mapping: TerminalMapping) -> None:
        existing = conn.execute(
            """
            SELECT created_at FROM terminal_mappings
            WHERE session_id = ? AND upstream_terminal_id = ? AND downstream_terminal_id = ?
            """,
            (session_id, mapping.upstream_terminal_id, mapping.downstream_terminal_id),
        ).fetchone()
        if existing is not None:
            mapping.created_at = existing["created_at"]
        conn.execute(
            """
            INSERT INTO terminal_mappings (
                session_id, upstream_terminal_id, downstream_terminal_id, owner_task_id, owner_agent_id,
                refcount, status, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, upstream_terminal_id, downstream_terminal_id) DO UPDATE SET
                owner_task_id=excluded.owner_task_id,
                owner_agent_id=excluded.owner_agent_id,
                refcount=excluded.refcount,
                status=excluded.status,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                session_id,
                mapping.upstream_terminal_id,
                mapping.downstream_terminal_id,
                mapping.owner_task_id,
                mapping.owner_agent_id,
                mapping.refcount,
                mapping.status,
                json.dumps(mapping.metadata),
                mapping.created_at,
                mapping.updated_at,
            ),
        )

    def _persist_permission_request(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        request: PermissionRequestState,
    ) -> None:
        conn.execute(
            """
            INSERT INTO permission_requests (
                session_id, request_id, owner_task_id, status, decision, requested_at, decided_at,
                payload_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, request_id) DO UPDATE SET
                owner_task_id=excluded.owner_task_id,
                status=excluded.status,
                decision=excluded.decision,
                decided_at=excluded.decided_at,
                payload_json=excluded.payload_json,
                metadata_json=excluded.metadata_json
            """,
            (
                session_id,
                request.request_id,
                request.owner_task_id,
                request.status,
                request.decision,
                request.requested_at,
                request.decided_at,
                json.dumps(request.payload),
                json.dumps(request.metadata),
            ),
        )

    def _persist_trace_metadata(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        trace: TraceCorrelationState,
    ) -> None:
        existing = conn.execute(
            "SELECT created_at FROM trace_correlation_metadata WHERE session_id = ? AND trace_key = ?",
            (session_id, trace.trace_key),
        ).fetchone()
        if existing is not None:
            trace.created_at = existing["created_at"]
        conn.execute(
            """
            INSERT INTO trace_correlation_metadata (
                session_id, trace_key, turn_id, task_id, parent_trace_key, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, trace_key) DO UPDATE SET
                turn_id=excluded.turn_id,
                task_id=excluded.task_id,
                parent_trace_key=excluded.parent_trace_key,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                session_id,
                trace.trace_key,
                trace.turn_id,
                trace.task_id,
                trace.parent_trace_key,
                json.dumps(trace.metadata),
                trace.created_at,
                trace.updated_at,
            ),
        )

    def _touch_session(self, conn: sqlite3.Connection, session_id: str, timestamp: float) -> None:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (timestamp, session_id),
        )

    def _update_legacy_task_graph(self, conn: sqlite3.Connection, session_id: str) -> None:
        task_rows = self._load_task_states(conn, session_id)
        conn.execute(
            "UPDATE sessions SET task_graph_json = ? WHERE session_id = ?",
            (json.dumps([task.apply_to_plan_task().to_dict() for task in task_rows]), session_id),
        )


def _loads_dict(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    loaded = json.loads(payload)
    if isinstance(loaded, dict):
        return loaded
    return {"value": loaded}


def _loads_list(payload: str | None) -> list[Any]:
    if not payload:
        return []
    loaded = json.loads(payload)
    if isinstance(loaded, list):
        return loaded
    return [loaded]
