from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .models import PlanTask


@dataclass(slots=True)
class PlanParseResult:
    tasks: list[PlanTask]
    normalized_plan: dict[str, Any]
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return bool(self.tasks) and not self.errors


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def parse_coordinator_plan(
    raw_output: str,
    *,
    coordinator_agent_id: str,
) -> PlanParseResult:
    errors: list[str] = []
    payload = _extract_json_payload(raw_output)
    if not isinstance(payload, dict):
        return PlanParseResult(
            tasks=[],
            normalized_plan=_invalid_plan_payload(raw_output, coordinator_agent_id, ["Coordinator output did not contain a JSON object."]),
            errors=["Coordinator output did not contain a JSON object."],
        )

    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        errors.append("Structured plan must contain a non-empty 'tasks' list.")
        return PlanParseResult(
            tasks=[],
            normalized_plan=_invalid_plan_payload(raw_output, coordinator_agent_id, errors, payload=payload),
            errors=errors,
        )

    title_to_id: dict[str, str] = {}
    task_records: list[tuple[PlanTask, dict[str, Any], list[str]]] = []
    plan_meta = payload.get("_meta")
    if plan_meta is None:
        plan_meta = {}
    if not isinstance(plan_meta, dict):
        errors.append("Plan-level _meta must be an object if provided.")
        plan_meta = {"invalid_meta": plan_meta}

    for index, item in enumerate(raw_tasks):
        task, raw_dependencies, task_errors = _parse_task(item, index=index, coordinator_agent_id=coordinator_agent_id, plan_meta=plan_meta)
        if task_errors:
            errors.extend(task_errors)
            continue
        assert task is not None
        if task.title in title_to_id:
            errors.append(f"Duplicate task title in structured plan: {task.title!r}.")
            continue
        title_to_id[task.title] = task.task_id
        task_records.append((task, item, raw_dependencies))

    if errors:
        return PlanParseResult(
            tasks=[],
            normalized_plan=_invalid_plan_payload(raw_output, coordinator_agent_id, errors, payload=payload),
            errors=errors,
        )

    normalized_tasks: list[dict[str, Any]] = []
    tasks: list[PlanTask] = []
    for task, item, raw_dependencies in task_records:
        dependency_ids: list[str] = []
        for dependency in raw_dependencies:
            dependency_id = title_to_id.get(dependency)
            if dependency_id is None:
                errors.append(f"Unknown dependency reference {dependency!r} for task {task.title!r}.")
                continue
            dependency_ids.append(dependency_id)
        task.dependency_ids = dependency_ids
        normalized_tasks.append(
            {
                "task_id": task.task_id,
                "title": task.title,
                "details": task.details,
                "dependencies": list(raw_dependencies),
                "dependency_ids": list(dependency_ids),
                "required_capabilities": dict(task.required_capabilities),
                "assignee_hints": list(task.assignee_hints),
                "acceptable_models": list(task.acceptable_models),
                "priority": task.priority,
                "_meta": dict(task._meta),
                "raw": item,
            }
        )
        tasks.append(task)

    if errors:
        return PlanParseResult(
            tasks=[],
            normalized_plan=_invalid_plan_payload(raw_output, coordinator_agent_id, errors, payload=payload),
            errors=errors,
        )

    normalized_plan = {
        "tasks": normalized_tasks,
        "_meta": {
            **dict(plan_meta),
            "source": "coordinator",
            "coordinator_agent_id": coordinator_agent_id,
            "task_count": len(normalized_tasks),
        },
    }
    return PlanParseResult(tasks=tasks, normalized_plan=normalized_plan, errors=[])


def synthesize_local_fallback_plan(
    user_prompt: str,
    *,
    coordinator_agent_id: str,
) -> PlanParseResult:
    normalized = user_prompt.strip()
    title = normalized.splitlines()[0][:72] if normalized else "Orchestration request"
    tasks = [
        PlanTask(
            title="Analyze request and produce execution plan",
            details=f"Coordinator decomposes the request: {title}",
            required_capabilities={"planner": True, "needsPermissions": True},
            assignee=coordinator_agent_id,
            priority=100,
            _meta={
                "source": "local_fallback",
                "synthesized_locally": True,
                "fallback_reason": "coordinator_plan_invalid",
            },
        ),
        PlanTask(
            title="Inspect repository and gather context",
            details="Read project files, identify impacted modules, and capture constraints.",
            required_capabilities={"needsFilesystem": True, "commands": ["read", "search"]},
            priority=80,
            _meta={"source": "local_fallback", "synthesized_locally": True},
        ),
        PlanTask(
            title="Implement orchestrator changes",
            details="Edit project files, add code paths, and wire core orchestrator behavior.",
            required_capabilities={"needsFilesystem": True, "needsTerminal": True, "commands": ["edit", "test"]},
            priority=70,
            _meta={"source": "local_fallback", "synthesized_locally": True},
        ),
        PlanTask(
            title="Validate behavior",
            details="Run tests or dry-run validations and summarize execution results.",
            required_capabilities={"needsTerminal": True, "commands": ["test"]},
            priority=60,
            _meta={"source": "local_fallback", "synthesized_locally": True},
        ),
    ]
    task_dependencies = {
        tasks[0].task_id: [],
        tasks[1].task_id: [tasks[0].task_id],
        tasks[2].task_id: [tasks[1].task_id],
        tasks[3].task_id: [tasks[2].task_id],
    }
    title_dependencies = {
        tasks[0].task_id: [],
        tasks[1].task_id: [tasks[0].title],
        tasks[2].task_id: [tasks[1].title],
        tasks[3].task_id: [tasks[2].title],
    }
    for task in tasks:
        task.dependency_ids = list(task_dependencies[task.task_id])

    normalized_plan = {
        "tasks": [
            {
                "task_id": task.task_id,
                "title": task.title,
                "details": task.details,
                "dependencies": list(title_dependencies[task.task_id]),
                "dependency_ids": list(task.dependency_ids),
                "required_capabilities": dict(task.required_capabilities),
                "assignee_hints": list(task.assignee_hints),
                "acceptable_models": list(task.acceptable_models),
                "priority": task.priority,
                "_meta": dict(task._meta),
            }
            for task in tasks
        ],
        "_meta": {
            "source": "local_fallback",
            "coordinator_agent_id": coordinator_agent_id,
            "synthesized_locally": True,
            "task_count": len(tasks),
        },
    }
    return PlanParseResult(tasks=tasks, normalized_plan=normalized_plan, errors=[])


def _parse_task(
    item: Any,
    *,
    index: int,
    coordinator_agent_id: str,
    plan_meta: dict[str, Any],
) -> tuple[PlanTask | None, list[str], list[str]]:
    errors: list[str] = []
    if not isinstance(item, dict):
        return None, [], [f"Task at index {index} must be an object."]

    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append(f"Task at index {index} is missing a non-empty 'title'.")
    details = item.get("details")
    if not isinstance(details, str) or not details.strip():
        errors.append(f"Task {title or index!r} is missing a non-empty 'details'.")
    dependencies = item.get("dependencies", [])
    if dependencies is None:
        dependencies = []
    if not isinstance(dependencies, list) or not all(isinstance(value, str) and value.strip() for value in dependencies):
        errors.append(f"Task {title or index!r} must provide 'dependencies' as a list of non-empty strings.")
        dependencies = []
    required_capabilities = item.get("required_capabilities", {})
    if required_capabilities is None:
        required_capabilities = {}
    if not isinstance(required_capabilities, dict):
        errors.append(f"Task {title or index!r} must provide 'required_capabilities' as an object.")
        required_capabilities = {}
    assignee_hints = item.get("assignee_hints", [])
    if assignee_hints is None:
        assignee_hints = []
    if isinstance(assignee_hints, str):
        assignee_hints = [assignee_hints]
    if not isinstance(assignee_hints, list) or not all(isinstance(value, str) and value.strip() for value in assignee_hints):
        errors.append(f"Task {title or index!r} must provide 'assignee_hints' as a list of non-empty strings if present.")
        assignee_hints = []
    acceptable_models = item.get("acceptable_models", [])
    if acceptable_models is None:
        acceptable_models = []
    if isinstance(acceptable_models, str):
        acceptable_models = [acceptable_models]
    if not isinstance(acceptable_models, list) or not all(isinstance(value, str) and value.strip() for value in acceptable_models):
        errors.append(f"Task {title or index!r} must provide 'acceptable_models' as a list of non-empty strings if present.")
        acceptable_models = []
    priority = item.get("priority", 0)
    try:
        priority_value = int(priority)
    except (TypeError, ValueError):
        errors.append(f"Task {title or index!r} must provide an integer 'priority'.")
        priority_value = 0
    task_meta = item.get("_meta", {})
    if task_meta is None:
        task_meta = {}
    if not isinstance(task_meta, dict):
        errors.append(f"Task {title or index!r} must provide '_meta' as an object if present.")
        task_meta = {"invalid_meta": task_meta}

    if errors:
        return None, [], errors

    task = PlanTask(
        title=title.strip(),
        details=details.strip(),
        required_capabilities=dict(required_capabilities),
        acceptable_models=[value.strip() for value in acceptable_models],
        assignee_hints=[value.strip() for value in assignee_hints],
        assignee=_resolve_assignee_hint(assignee_hints),
        priority=priority_value,
        _meta={
            **dict(task_meta),
            "source": "coordinator",
            "coordinator_agent_id": coordinator_agent_id,
            "coordinator_task_index": index,
            "plan_meta": dict(plan_meta),
        },
    )
    return task, [value.strip() for value in dependencies], []


def _resolve_assignee_hint(assignee_hints: list[str]) -> str | None:
    for hint in assignee_hints:
        value = hint.strip()
        if value and "::" not in value and " " not in value:
            return value
    return None


def _extract_json_payload(raw_output: str) -> Any:
    stripped = raw_output.strip()
    candidates = [match.group(1) for match in _JSON_BLOCK_RE.finditer(stripped)]
    candidates.append(stripped)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _invalid_plan_payload(
    raw_output: str,
    coordinator_agent_id: str,
    errors: list[str],
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "tasks": [],
        "_meta": {
            "source": "coordinator",
            "coordinator_agent_id": coordinator_agent_id,
            "valid": False,
            "errors": list(errors),
        },
        "raw_output": raw_output,
        "payload": payload or {},
    }
