from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .models import SessionSnapshot


class SQLiteSessionStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(":memory:")
            return self._memory_connection
        return sqlite3.connect(self.db_path)

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
                    task_graph_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def save(self, snapshot: SessionSnapshot) -> None:
        snapshot.updated_at = time.time()
        with self._connect() as conn:
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
        return SessionSnapshot(
            session_id=row[0],
            cwd=row[1],
            selected_model=row[2],
            coordinator_agent_id=row[3],
            title=row[4],
            created_at=row[5],
            updated_at=row[6],
            task_graph=json.loads(row[7]),
            metadata=json.loads(row[8]),
        )

    def list_sessions(self) -> list[SessionSnapshot]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, cwd, selected_model, coordinator_agent_id, title,
                       created_at, updated_at, task_graph_json, metadata_json
                FROM sessions ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            SessionSnapshot(
                session_id=row[0],
                cwd=row[1],
                selected_model=row[2],
                coordinator_agent_id=row[3],
                title=row[4],
                created_at=row[5],
                updated_at=row[6],
                task_graph=json.loads(row[7]),
                metadata=json.loads(row[8]),
            )
            for row in rows
        ]
