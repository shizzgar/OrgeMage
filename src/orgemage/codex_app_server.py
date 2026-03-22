from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any, Callable, Protocol

from . import __version__
from .debug import debug_event, get_logger
from .models import (
    DownstreamAgentConfig,
    DownstreamNegotiatedState,
    PlanTask,
    TaskStatus,
)
from .acp.downstream_client import DownstreamConnectorError, DownstreamPromptResult

_LOG = get_logger(__name__)


class _CodexTransport(Protocol):
    def start(
        self,
        *,
        notification_handler: Callable[[dict[str, Any]], None],
        request_handler: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        ...

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        ...

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        ...

    def close(self) -> None:
        ...

    def stderr_tail(self) -> list[str]:
        ...


class _JsonRpcTransportError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class _StdIoCodexTransport:
    def __init__(self, agent: DownstreamAgentConfig) -> None:
        self.agent = agent
        self._proc: subprocess.Popen[str] | None = None
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._send_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._request_id = 0
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=100)
        self._notification_handler: Callable[[dict[str, Any]], None] | None = None
        self._request_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def start(
        self,
        *,
        notification_handler: Callable[[dict[str, Any]], None],
        request_handler: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        if self._proc is not None:
            return
        self._notification_handler = notification_handler
        self._request_handler = request_handler
        env = os.environ.copy()
        self._proc = subprocess.Popen(
            [self.agent.command, *self.agent.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._reader = threading.Thread(
            target=self._reader_loop,
            name=f"orgemage-codex-app-server-{self.agent.agent_id}",
            daemon=True,
        )
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._stderr_loop,
            name=f"orgemage-codex-app-server-stderr-{self.agent.agent_id}",
            daemon=True,
        )
        self._stderr_reader.start()

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        request_id = self._next_request_id()
        waiter: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = waiter
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        try:
            payload = waiter.get(timeout=timeout)
        except queue.Empty as exc:
            raise DownstreamConnectorError(
                f"Timed out waiting for Codex app-server response to {method!r}."
            ) from exc
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)
        error = payload.get("error")
        if isinstance(error, dict):
            raise _JsonRpcTransportError(
                str(error.get("message") or f"Codex app-server request failed: {method}"),
                code=error.get("code"),
                data=error.get("data"),
            )
        result = payload.get("result")
        if isinstance(result, dict):
            return result
        return {}

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def stderr_tail(self) -> list[str]:
        return list(self._stderr_tail)

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        if self._reader is not None:
            self._reader.join(timeout=1)
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=1)

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise DownstreamConnectorError("Codex app-server is not running.")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._send_lock:
            try:
                self._proc.stdin.write(encoded + "\n")
                self._proc.stdin.flush()
            except BrokenPipeError as exc:
                raise DownstreamConnectorError("Codex app-server closed its stdin pipe.") from exc

    def _reader_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self._stderr_tail.append(f"invalid-json:{line[:200]}")
                continue
            if "method" in payload and "id" in payload:
                self._handle_server_request(payload)
                continue
            if "id" in payload:
                waiter = None
                with self._pending_lock:
                    waiter = self._pending.get(int(payload["id"]))
                if waiter is not None:
                    waiter.put(payload)
                continue
            if "method" in payload and self._notification_handler is not None:
                self._notification_handler(payload)
        failure = {
            "error": {
                "message": "Codex app-server closed unexpectedly.",
                "data": {"stderr_tail": self.stderr_tail()},
            }
        }
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for waiter in pending:
            waiter.put(failure)

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr_tail.append(line.rstrip())

    def _handle_server_request(self, payload: dict[str, Any]) -> None:
        request_id = payload.get("id")
        if request_id is None:
            return
        if self._request_handler is None:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "No request handler configured."},
                }
            )
            return
        try:
            result = self._request_handler(payload)
        except Exception as exc:  # pragma: no cover - defensive
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(exc)},
                }
            )
            return
        self._send({"jsonrpc": "2.0", "id": request_id, "result": result})


@dataclass(slots=True)
class _ActiveTurn:
    thread_id: str
    turn_id: str | None = None
    updates: list[dict[str, Any]] = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)
    status: TaskStatus = TaskStatus.IN_PROGRESS
    turn_payload: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    agent_message_item_ids_with_deltas: set[str] = field(default_factory=set)


class CodexAppServerConnector:
    def __init__(
        self,
        agent: DownstreamAgentConfig,
        *,
        client_factory: Callable[[DownstreamAgentConfig], _CodexTransport] | None = None,
        headless_policy: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> None:
        self.agent = agent
        self.negotiated_state: DownstreamNegotiatedState | None = None
        self._client_factory = client_factory or _StdIoCodexTransport
        self._client: _CodexTransport | None = None
        self._lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._catalog_state: dict[str, Any] | None = None
        self._catalog_refresh_required = True
        self._headless_policy = headless_policy or (lambda method, payload: "allow")
        self._active_turn: _ActiveTurn | None = None
        self._request_timeout = float(os.environ.get("ORGEMAGE_CODEX_APP_SERVER_REQUEST_TIMEOUT", "30"))
        self._turn_timeout = float(os.environ.get("ORGEMAGE_CODEX_APP_SERVER_TURN_TIMEOUT", "300"))

    def discover_catalog(self, *, force: bool = False) -> dict[str, Any]:
        self._ensure_started()
        if not force and self._catalog_state is not None and not self._catalog_refresh_required:
            return dict(self._catalog_state)
        return self._refresh_catalog()

    def mark_catalog_refresh_required(self) -> None:
        self._catalog_refresh_required = True

    def execute_task(
        self,
        *,
        orchestrator_session_id: str,
        downstream_session_id: str | None,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | list[Any] | None,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
    ) -> DownstreamPromptResult:
        del mcp_servers
        self._ensure_started()
        assert self._client is not None
        with self._operation_lock:
            thread_id, thread_payload = self._ensure_thread(
                cwd=cwd,
                downstream_session_id=downstream_session_id,
                selected_model=selected_model,
                task=task,
            )
            self._record_thread_state(thread_id, thread_payload)
            active_turn = _ActiveTurn(thread_id=thread_id)
            self._active_turn = active_turn
            prompt_text = self._build_prompt(
                orchestrator_session_id=orchestrator_session_id,
                task=task,
                coordinator_prompt=coordinator_prompt,
                selected_model=selected_model,
            )
            try:
                turn_payload = self._client.request(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [
                            {
                                "type": "text",
                                "text": prompt_text,
                                "text_elements": [],
                            }
                        ],
                        "cwd": cwd,
                        "model": self._resolve_model(selected_model),
                        "approvalPolicy": "never",
                        "sandboxPolicy": self._sandbox_policy_for_task(task=task, cwd=cwd),
                    },
                    timeout=self._request_timeout,
                )
                response_turn = turn_payload.get("turn") or {}
                if isinstance(response_turn, dict):
                    active_turn.turn_id = str(response_turn.get("id") or "") or active_turn.turn_id
                    if str(response_turn.get("status") or "") in {"completed", "interrupted", "failed"}:
                        self._ingest_turn_completion(response_turn, active_turn)
                if not active_turn.done.wait(timeout=self._turn_timeout):
                    raise DownstreamConnectorError(
                        f"Timed out waiting for Codex app-server turn completion for thread {thread_id!r}."
                    )
                summary = _extract_summary_from_updates(active_turn.updates) or active_turn.error_message
                if not summary:
                    summary = f"Downstream agent {self.agent.name} completed {task.title!r}."
                return DownstreamPromptResult(
                    downstream_session_id=thread_id,
                    status=active_turn.status,
                    summary=summary,
                    raw_output=summary,
                    updates=list(active_turn.updates),
                    response={"thread": thread_payload, "turn": dict(active_turn.turn_payload)},
                )
            finally:
                self._active_turn = None

    def cancel(self, downstream_session_id: str) -> None:
        self._ensure_started()
        active_turn = self._active_turn
        if self._client is None or active_turn is None:
            return
        if active_turn.thread_id != downstream_session_id or not active_turn.turn_id:
            return
        self._client.request(
            "turn/interrupt",
            {"threadId": active_turn.thread_id, "turnId": active_turn.turn_id},
            timeout=self._request_timeout,
        )

    def close(self) -> None:
        if self._client is None:
            return
        self._client.close()
        self._client = None

    def _ensure_started(self) -> None:
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:
                return
            client = self._client_factory(self.agent)
            client.start(
                notification_handler=self._handle_notification,
                request_handler=self._handle_server_request,
            )
            try:
                initialize = client.request(
                    "initialize",
                    {
                        "clientInfo": {"name": "orgemage", "version": __version__},
                        "capabilities": {},
                    },
                    timeout=self._request_timeout,
                )
                client.notify("initialized", {})
            except Exception:
                client.close()
                raise
            self._client = client
            user_agent = str(initialize.get("userAgent") or "")
            self.negotiated_state = DownstreamNegotiatedState(
                agent_id=self.agent.agent_id,
                agent_info={
                    "name": "codex-app-server",
                    "title": "Codex App Server",
                    "version": _extract_codex_version(user_agent),
                    "userAgent": user_agent,
                },
                agent_capabilities={
                    "loadSession": True,
                    "sessionCapabilities": {"list": {}, "resume": {}},
                },
                protocol_version=None,
                profile={
                    "id": "codex-app-server",
                    "title": "Codex App Server",
                    "command": Path(self.agent.command).name,
                    "args": list(self.agent.args),
                    "supported_mcp_transports": ["stdio"],
                    "known_quirks": [
                        "codex app-server is a JSON-RPC thread/turn protocol, not ACP.",
                        "Model discovery comes from model/list and is synthesized into OrgeMage config options.",
                    ],
                },
            )
            debug_event(
                _LOG,
                "connector.codex_app_server.started",
                agent_id=self.agent.agent_id,
                agent_info=self.negotiated_state.agent_info,
            )
            self._refresh_catalog()

    def _refresh_catalog(self) -> dict[str, Any]:
        assert self._client is not None
        assert self.negotiated_state is not None
        discovered_models = self._list_models()
        current_model = self._current_model_from_discovery(discovered_models)
        config_options = [
            {
                "id": "model",
                "name": "Model",
                "category": "model",
                "type": "select",
                "currentValue": current_model,
                "options": [
                    {
                        "value": item["id"],
                        "name": item["name"],
                        "description": item["description"],
                    }
                    for item in discovered_models
                ],
            }
        ]
        capabilities = {
            "loadSession": True,
            "supportsTerminal": self.agent.capabilities.supports_terminal,
            "supportsFilesystem": self.agent.capabilities.supports_filesystem,
            "supportsPermissions": self.agent.capabilities.supports_permissions,
            "supportsMcp": self.agent.capabilities.supports_mcp,
            "commands": list(self.agent.capabilities.commands),
        }
        self._catalog_state = {
            "agent_id": self.agent.agent_id,
            "config_options": config_options,
            "capabilities": capabilities,
            "command_advertisements": list(self.agent.capabilities.commands),
            "profile": dict(self.negotiated_state.profile),
            "diagnostics": list(self.negotiated_state.diagnostics),
            "refreshed_at": time.time(),
        }
        self._catalog_refresh_required = False
        return dict(self._catalog_state)

    def _list_models(self) -> list[dict[str, str]]:
        assert self._client is not None
        discovered: list[dict[str, str]] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            payload = self._client.request("model/list", params, timeout=self._request_timeout)
            for item in payload.get("data") or []:
                if not isinstance(item, dict):
                    continue
                model_id = str(item.get("id") or item.get("model") or "").strip()
                if not model_id:
                    continue
                discovered.append(
                    {
                        "id": model_id,
                        "name": str(item.get("displayName") or item.get("model") or model_id),
                        "description": str(item.get("description") or ""),
                        "is_default": "true" if item.get("isDefault") else "",
                    }
                )
            cursor = payload.get("nextCursor")
            if not isinstance(cursor, str) or not cursor:
                break
        return discovered

    def _current_model_from_discovery(self, discovered_models: list[dict[str, str]]) -> str:
        for item in discovered_models:
            if item.get("is_default"):
                return item["id"]
        raw_selected = self.agent.default_model or ""
        return self._resolve_model(f"{self.agent.agent_id}::{raw_selected}") if raw_selected else ""

    def _ensure_thread(
        self,
        *,
        cwd: str,
        downstream_session_id: str | None,
        selected_model: str,
        task: PlanTask,
    ) -> tuple[str, dict[str, Any]]:
        assert self._client is not None
        model = self._resolve_model(selected_model)
        if downstream_session_id:
            payload = self._client.request(
                "thread/resume",
                {
                    "threadId": downstream_session_id,
                    "cwd": cwd,
                    "model": model,
                    "approvalPolicy": "never",
                    "sandbox": self._sandbox_mode_for_task(task),
                    "persistExtendedHistory": False,
                },
                timeout=self._request_timeout,
            )
            return downstream_session_id, payload
        payload = self._client.request(
            "thread/start",
            {
                "cwd": cwd,
                "model": model,
                "approvalPolicy": "never",
                "sandbox": self._sandbox_mode_for_task(task),
                "experimentalRawEvents": False,
                "persistExtendedHistory": False,
            },
            timeout=self._request_timeout,
        )
        thread = payload.get("thread") or {}
        thread_id = str(thread.get("id") or "")
        if not thread_id:
            raise DownstreamConnectorError("Codex app-server returned no thread id.")
        return thread_id, payload

    def _record_thread_state(self, thread_id: str, payload: dict[str, Any]) -> None:
        if self.negotiated_state is None:
            return
        discovered_models = self._list_models()
        current_model = str(payload.get("model") or self._current_model_from_discovery(discovered_models))
        current_mode = _sandbox_type_to_mode_id(payload.get("sandbox"))
        config_options = [
            {
                "id": "model",
                "name": "Model",
                "category": "model",
                "type": "select",
                "currentValue": current_model,
                "options": [
                    {
                        "value": item["id"],
                        "name": item["name"],
                        "description": item["description"],
                    }
                    for item in discovered_models
                ],
            },
            {
                "id": "mode",
                "name": "Mode",
                "category": "mode",
                "type": "select",
                "currentValue": current_mode,
                "options": [
                    {"value": "read-only", "name": "Read Only", "description": "Read-only sandbox"},
                    {
                        "value": "workspace-write",
                        "name": "Workspace Write",
                        "description": "Workspace-write sandbox",
                    },
                    {
                        "value": "danger-full-access",
                        "name": "Danger Full Access",
                        "description": "Unrestricted sandbox",
                    },
                ],
            },
        ]
        self.negotiated_state.record_session(
            session_id=thread_id,
            capabilities={
                "thread": payload.get("thread") or {},
                "cwd": payload.get("cwd"),
                "approvalPolicy": payload.get("approvalPolicy"),
                "sandbox": payload.get("sandbox"),
            },
            config_options=config_options,
            models={
                "availableModels": [
                    {
                        "modelId": item["id"],
                        "name": item["name"],
                        "description": item["description"],
                    }
                    for item in discovered_models
                ],
                "currentModelId": current_model,
            },
            modes={
                "availableModes": [
                    {"id": "read-only", "name": "Read Only", "description": "Read-only sandbox"},
                    {
                        "id": "workspace-write",
                        "name": "Workspace Write",
                        "description": "Workspace-write sandbox",
                    },
                    {
                        "id": "danger-full-access",
                        "name": "Danger Full Access",
                        "description": "Unrestricted sandbox",
                    },
                ],
                "currentModeId": current_mode,
            },
            available_commands=[],
        )

    def _resolve_model(self, selected_model: str) -> str:
        requested = self.agent.resolve_model(selected_model) or self.agent.default_model or selected_model
        discovered = []
        if self._catalog_state is not None:
            for option in self._catalog_state.get("config_options", []):
                if option.get("id") != "model":
                    continue
                for item in option.get("options") or []:
                    if isinstance(item, dict) and item.get("value"):
                        discovered.append(str(item["value"]))
        if not discovered:
            return requested
        if requested in discovered:
            return requested
        matches = [candidate for candidate in discovered if requested in _model_aliases(candidate)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return matches[0]
        if requested == "gpt-5-codex":
            codex_candidates = [candidate for candidate in discovered if candidate.endswith("-codex")]
            if codex_candidates:
                return codex_candidates[0]
        for candidate in discovered:
            if candidate == "gpt-5.4":
                return candidate
        return discovered[0]

    def _sandbox_mode_for_task(self, task: PlanTask) -> str:
        commands = {str(command).lower() for command in task.required_capabilities.get("commands", [])}
        needs_terminal = bool(task.required_capabilities.get("needsTerminal"))
        write_markers = {"edit", "write", "test", "terminal", "exec", "command"}
        if not needs_terminal and not commands.intersection(write_markers):
            return "read-only"
        return "workspace-write"

    def _sandbox_policy_for_task(self, *, task: PlanTask, cwd: str) -> dict[str, Any]:
        mode = self._sandbox_mode_for_task(task)
        if mode == "read-only":
            return {
                "type": "readOnly",
                "access": {"type": "fullAccess"},
                "networkAccess": False,
            }
        writable_roots = [cwd, "/tmp"]
        memories_root = str(Path.home() / ".codex" / "memories")
        if memories_root not in writable_roots:
            writable_roots.append(memories_root)
        return {
            "type": "workspaceWrite",
            "writableRoots": writable_roots,
            "readOnlyAccess": {"type": "fullAccess"},
            "networkAccess": False,
            "excludeTmpdirEnvVar": False,
            "excludeSlashTmp": False,
        }

    def _handle_notification(self, payload: dict[str, Any]) -> None:
        active_turn = self._active_turn
        if active_turn is None:
            return
        method = str(payload.get("method") or "")
        params = payload.get("params")
        if not isinstance(params, dict):
            return
        thread_id = str(params.get("threadId") or "")
        if thread_id and thread_id != active_turn.thread_id:
            return
        if method == "turn/started":
            turn = params.get("turn")
            if isinstance(turn, dict):
                active_turn.turn_id = str(turn.get("id") or active_turn.turn_id or "")
            return
        if method == "turn/completed":
            turn = params.get("turn")
            if isinstance(turn, dict):
                self._ingest_turn_completion(turn, active_turn)
            return
        translated = self._translate_notification(method, params, active_turn)
        if translated is not None:
            active_turn.updates.append(translated)

    def _ingest_turn_completion(self, turn: dict[str, Any], active_turn: _ActiveTurn) -> None:
        status = str(turn.get("status") or "")
        active_turn.turn_payload = dict(turn)
        active_turn.turn_id = str(turn.get("id") or active_turn.turn_id or "")
        if status == "completed":
            active_turn.status = TaskStatus.COMPLETED
        elif status == "interrupted":
            active_turn.status = TaskStatus.CANCELLED
        else:
            active_turn.status = TaskStatus.FAILED
        error = turn.get("error")
        if isinstance(error, dict):
            active_turn.error_message = str(error.get("message") or error.get("additionalDetails") or "")
        active_turn.done.set()

    def _translate_notification(
        self,
        method: str,
        params: dict[str, Any],
        active_turn: _ActiveTurn,
    ) -> dict[str, Any] | None:
        if method == "item/agentMessage/delta":
            item_id = str(params.get("itemId") or "")
            if item_id:
                active_turn.agent_message_item_ids_with_deltas.add(item_id)
            delta = str(params.get("delta") or "")
            if not delta:
                return None
            return {
                "sessionUpdate": "agent_message_chunk",
                "id": item_id,
                "content": {"type": "text", "text": delta},
            }
        if method == "item/commandExecution/outputDelta":
            item_id = str(params.get("itemId") or "")
            return {
                "sessionUpdate": "tool_call_update",
                "id": item_id,
                "status": "in_progress",
                "terminal": {
                    "terminalId": item_id or f"cmd-{active_turn.thread_id}",
                    "content": str(params.get("delta") or ""),
                },
            }
        if method == "item/fileChange/outputDelta":
            item_id = str(params.get("itemId") or "")
            delta = str(params.get("delta") or "")
            if not delta:
                return None
            return {
                "sessionUpdate": "tool_call_update",
                "id": item_id,
                "status": "in_progress",
                "text": delta,
            }
        if method == "turn/plan/updated":
            explanation = str(params.get("explanation") or "").strip()
            if not explanation:
                explanation = _render_plan_steps(params.get("plan") or [])
            if not explanation:
                return None
            return {
                "sessionUpdate": "tool_call_update",
                "id": f"{active_turn.thread_id}:plan",
                "status": "in_progress",
                "text": explanation,
            }
        if method == "item/started":
            return _translate_item_update(params.get("item"), status="in_progress")
        if method == "item/completed":
            item = params.get("item")
            translated = _translate_item_update(item, status=None)
            if translated is not None:
                return translated
            if isinstance(item, dict) and item.get("type") == "agentMessage":
                item_id = str(item.get("id") or "")
                if item_id not in active_turn.agent_message_item_ids_with_deltas:
                    text = str(item.get("text") or "")
                    if text:
                        return {
                            "sessionUpdate": "agent_message_chunk",
                            "id": item_id,
                            "content": {"type": "text", "text": text},
                        }
        return None

    def _handle_server_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        method = str(payload.get("method") or "")
        params = payload.get("params")
        if not isinstance(params, dict):
            params = {}
        decision = self._headless_policy(method, params)
        allow = decision != "deny"
        if method in {"execCommandApproval", "applyPatchApproval"}:
            return {"decision": "approved" if allow else "denied"}
        if method == "item/commandExecution/requestApproval":
            return {"decision": "accept" if allow else "decline"}
        if method == "item/fileChange/requestApproval":
            return {"decision": "accept" if allow else "decline"}
        if method == "item/permissions/requestApproval":
            return {"permissions": {}, "scope": "turn"}
        if method == "item/tool/call":
            return {"contentItems": [], "success": False}
        if method == "item/tool/requestUserInput":
            return {"answers": {}}
        if method == "mcpServer/elicitation/request":
            return {"action": "cancel", "content": None, "_meta": None}
        return {}

    def _build_prompt(
        self,
        *,
        orchestrator_session_id: str,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
    ) -> str:
        if task._meta.get("phase") == "planning":
            return (
                f"OrgeMage orchestrator session: {orchestrator_session_id}\n"
                f"Delegation metadata: {task._meta}\n"
                f"Target downstream agent: {self.agent.agent_id}\n"
                f"Selected coordinator model: {selected_model}\n\n"
                f"Coordinator directive:\n{coordinator_prompt}\n\n"
                f"Delegated task title: {task.title}\n"
                f"Delegated task details: {task.details}\n"
                f"Required capabilities: {task.required_capabilities}\n"
                f"Dependency IDs: {task.dependency_ids}\n"
            )
        user_request = _extract_user_request_from_coordinator_prompt(coordinator_prompt)
        return (
            f"OrgeMage orchestrator session: {orchestrator_session_id}\n"
            f"Delegation metadata: {task._meta}\n"
            f"Target downstream agent: {self.agent.agent_id}\n"
            f"Selected coordinator model: {selected_model}\n\n"
            f"Original user request:\n{user_request}\n\n"
            "You are executing a delegated task, not planning.\n"
            "Do not return a JSON task graph or orchestration plan unless the delegated task explicitly asks for one.\n"
            "Complete this task directly and return the actual result.\n\n"
            f"Delegated task title: {task.title}\n"
            f"Delegated task details: {task.details}\n"
            f"Required capabilities: {task.required_capabilities}\n"
            f"Dependency IDs: {task.dependency_ids}\n"
        )


def _extract_user_request_from_coordinator_prompt(prompt: str) -> str:
    marker = "User request:\n"
    if marker not in prompt:
        return prompt.strip()
    return prompt.split(marker, 1)[1].strip()


def _extract_summary_from_updates(updates: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for update in updates:
        if str(update.get("sessionUpdate") or "") != "agent_message_chunk":
            continue
        content = update.get("content")
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
    return "".join(chunks).strip()


def _render_plan_steps(plan: list[Any]) -> str:
    parts: list[str] = []
    for item in plan:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("step") or "").strip()
        if title:
            parts.append(title)
    return "; ".join(parts)


def _sandbox_type_to_mode_id(payload: Any) -> str:
    if isinstance(payload, dict):
        sandbox_type = str(payload.get("type") or "")
        if sandbox_type == "readOnly":
            return "read-only"
        if sandbox_type == "dangerFullAccess":
            return "danger-full-access"
        if sandbox_type == "workspaceWrite":
            return "workspace-write"
    if isinstance(payload, str):
        return payload
    return "workspace-write"


def _translate_item_update(item: Any, *, status: str | None) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    item_type = str(item.get("type") or "")
    item_id = str(item.get("id") or "")
    if not item_type or not item_id:
        return None
    resolved_status = status or _item_status(item)
    if item_type == "agentMessage":
        return None
    if item_type == "reasoning":
        return None
    text = _item_title(item)
    if not text:
        return None
    payload: dict[str, Any] = {
        "sessionUpdate": "tool_call" if resolved_status == "in_progress" else "tool_call_update",
        "id": item_id,
        "status": resolved_status,
        "text": text,
    }
    locations = _item_locations(item)
    if locations:
        payload["locations"] = locations
    return payload


def _item_title(item: dict[str, Any]) -> str:
    item_type = str(item.get("type") or "")
    if item_type == "commandExecution":
        return str(item.get("command") or "commandExecution")
    if item_type == "fileChange":
        paths = ", ".join(change["path"] for change in _item_locations(item) if change.get("path"))
        return f"fileChange: {paths}" if paths else "fileChange"
    if item_type == "mcpToolCall":
        server = str(item.get("server") or "")
        tool = str(item.get("tool") or "")
        return "/".join(part for part in (server, tool) if part) or "mcpToolCall"
    if item_type == "dynamicToolCall":
        return str(item.get("tool") or "dynamicToolCall")
    if item_type == "webSearch":
        return str(item.get("query") or "webSearch")
    if item_type == "collabAgentToolCall":
        return str(item.get("tool") or "collabAgentToolCall")
    if item_type == "plan":
        return str(item.get("text") or "plan")
    return item_type


def _item_locations(item: dict[str, Any]) -> list[dict[str, Any]]:
    if str(item.get("type") or "") != "fileChange":
        return []
    locations: list[dict[str, Any]] = []
    for change in item.get("changes") or []:
        if not isinstance(change, dict):
            continue
        path = change.get("path")
        if isinstance(path, str) and path:
            locations.append({"path": path})
    return locations


def _item_status(item: dict[str, Any]) -> str:
    raw_status = str(item.get("status") or "").lower()
    if raw_status in {"completed", "applied", "success", "succeeded"}:
        return "completed"
    if raw_status in {"failed", "error"}:
        return "failed"
    if raw_status in {"cancelled", "canceled"}:
        return "cancelled"
    return "completed"


def _extract_codex_version(user_agent: str) -> str:
    if "/" not in user_agent:
        return ""
    prefix = user_agent.split(" ", 1)[0]
    _, _, version = prefix.partition("/")
    return version


def _model_aliases(value: str) -> set[str]:
    aliases = {value}
    for separator in ("/", "::"):
        if separator in value:
            _, _, suffix = value.rpartition(separator)
            if suffix:
                aliases.add(suffix)
    return aliases
