from __future__ import annotations

from typing import Any
import re

_DEFAULT_HISTORY_TITLE = "OrgeMage Session"


def extract_prompt_metadata(prompt: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for candidate in (kwargs.get("_meta"), kwargs.get("metadata"), kwargs.get("meta")):
        if isinstance(candidate, dict):
            merged.update(candidate)
    for block in prompt or []:
        metadata = _extract_block_metadata(block)
        if metadata:
            merged.update(metadata)
    return merged


def _extract_block_metadata(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        for key in ("_meta", "metadata", "meta"):
            value = block.get(key)
            if isinstance(value, dict):
                return dict(value)
        return {}
    for key in ("_meta", "metadata", "meta"):
        value = getattr(block, key, None)
        if isinstance(value, dict):
            return dict(value)
    return {}


def build_turn_metadata(*, session_id: str, turn_id: str, prompt_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    prompt_metadata = dict(prompt_metadata or {})
    metadata = dict(prompt_metadata)
    metadata.setdefault("traceId", _first_str(prompt_metadata, "traceId", "trace_id"))
    metadata.setdefault("traceparent", _first_str(prompt_metadata, "traceparent"))
    metadata["turnId"] = turn_id
    metadata.setdefault("workerCorrelationId", f"{session_id}:{turn_id}:coordinator")
    metadata.setdefault("assignee", {"agentId": None, "role": "coordinator"})
    metadata.setdefault("planningProvenance", {})
    metadata.setdefault("policyAnnotations", {})
    return _drop_none(metadata)


def propagate_task_metadata(
    task_meta: dict[str, Any] | None,
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    assignee: str | None,
    assignee_hints: list[str],
    planning_provenance: dict[str, Any] | None,
    prompt_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = dict(prompt_metadata or {})
    if task_meta:
        base.update(task_meta)
    base.setdefault("traceId", _first_str(base, "traceId", "trace_id"))
    base.setdefault("traceparent", _first_str(base, "traceparent"))
    base["turnId"] = turn_id
    base.setdefault("workerCorrelationId", f"{session_id}:{turn_id}:{task_id}:{assignee or 'unassigned'}")
    base["assignee"] = {
        **(dict(base.get("assignee")) if isinstance(base.get("assignee"), dict) else {}),
        "agentId": assignee,
        "hints": list(assignee_hints),
        "taskId": task_id,
    }
    if planning_provenance:
        base["planningProvenance"] = {
            **(dict(base.get("planningProvenance")) if isinstance(base.get("planningProvenance"), dict) else {}),
            **dict(planning_provenance),
        }
    base["policyAnnotations"] = _policy_annotations(base)
    return _drop_none(base)


def event_metadata(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    task_meta: dict[str, Any] | None,
    assignee: str | None,
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(task_meta or {})
    payload.setdefault("turnId", turn_id)
    payload.setdefault("workerCorrelationId", f"{session_id}:{turn_id}:{task_id}:{assignee or 'unassigned'}")
    payload["assignee"] = {
        **(dict(payload.get("assignee")) if isinstance(payload.get("assignee"), dict) else {}),
        "agentId": assignee,
        "taskId": task_id,
    }
    payload["policyAnnotations"] = _policy_annotations(payload)
    payload["source"] = source
    if extra:
        payload.update(extra)
    return _drop_none(payload)


def summarize_session(user_prompt: str, final_summary: str | None = None) -> dict[str, str]:
    normalized = " ".join(user_prompt.strip().split())
    if not normalized:
        normalized = "Orchestration request"
    compact = normalized[:72].rstrip(" ,.;:")
    title = compact or "Orchestration request"
    summary_source = final_summary or normalized
    summary = summary_source[:160].rstrip()
    if not summary:
        summary = title
    return {"title": title, "summary": summary}


def session_title(snapshot_title: str, derived_title: str) -> str:
    if not snapshot_title or snapshot_title == _DEFAULT_HISTORY_TITLE or snapshot_title.startswith("OrgeMage: "):
        return f"OrgeMage: {derived_title}"
    return snapshot_title


def _policy_annotations(payload: dict[str, Any]) -> dict[str, Any]:
    annotations = dict(payload.get("policyAnnotations")) if isinstance(payload.get("policyAnnotations"), dict) else {}
    for source_key, target_key in (
        ("failure_policy", "failurePolicy"),
        ("failurePolicy", "failurePolicy"),
        ("dependency_failure_policy", "dependencyFailurePolicy"),
        ("dependencyFailurePolicy", "dependencyFailurePolicy"),
        ("permissions", "permissions"),
    ):
        value = payload.get(source_key)
        if value is not None:
            annotations[target_key] = value
    return annotations


def _first_str(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
