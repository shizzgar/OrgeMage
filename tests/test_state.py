from pathlib import Path

from orgemage.models import SessionSnapshot
from orgemage.state import SQLiteSessionStore


def test_sqlite_session_store_round_trip(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "orgemage.db")
    snapshot = SessionSnapshot(session_id="s1", cwd="/tmp/project", selected_model="codex::gpt-5-codex")
    snapshot.task_graph = [{"task_id": "t1", "status": "completed"}]
    snapshot.metadata = {"traceparent": "abc"}

    store.save(snapshot)
    loaded = store.load("s1")

    assert loaded is not None
    assert loaded.selected_model == "codex::gpt-5-codex"
    assert loaded.task_graph == [{"task_id": "t1", "status": "completed"}]
    assert loaded.metadata["traceparent"] == "abc"
