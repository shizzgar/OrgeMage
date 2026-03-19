from __future__ import annotations

import asyncio
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Callable, Protocol
import uuid

from ..debug import debug_event, get_logger
from ..models import (
    DownstreamAgentConfig,
    DownstreamNegotiatedState,
    PermissionRequestState,
    PlanTask,
    TaskStatus,
    TerminalMapping,
    TraceCorrelationState,
)
from ..state import SQLiteSessionStore

_LOG = get_logger(__name__)


class DownstreamConnectorError(RuntimeError):
    """Raised when a downstream ACP connection cannot be established or used."""


def _acp_attr(acp: Any, name: str) -> Any:
    value = getattr(acp, name, None)
    if value is not None:
        return value
    schema = getattr(acp, "schema", None)
    if schema is not None:
        return getattr(schema, name, None)
    return None


@dataclass(slots=True)
class DownstreamPromptResult:
    downstream_session_id: str
    status: TaskStatus
    summary: str
    raw_output: str = ""
    updates: list[dict[str, Any]] = field(default_factory=list)
    response: dict[str, Any] = field(default_factory=dict)


class DownstreamConnector(Protocol):
    agent: DownstreamAgentConfig
    negotiated_state: DownstreamNegotiatedState | None

    def discover_catalog(self, *, force: bool = False) -> dict[str, Any]:
        ...

    def mark_catalog_refresh_required(self) -> None:
        ...

    def execute_task(
        self,
        *,
        orchestrator_session_id: str,
        downstream_session_id: str | None,
        cwd: str,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
    ) -> DownstreamPromptResult:
        ...

    def cancel(self, downstream_session_id: str) -> None:
        ...


@dataclass(slots=True)
class _SdkBindings:
    module: Any
    protocol_version: int
    client_base: type[Any]
    request_error: type[Exception]
    client_capabilities: type[Any]
    implementation: type[Any]
    spawn_agent_process: Callable[..., Any]
    text_block: Callable[[str], Any]


@dataclass(slots=True)
class _ExecutionContext:
    orchestrator_session_id: str
    downstream_session_id: str
    cwd: str
    task: PlanTask
    turn_id: str | None = None
    prompt_metadata: dict[str, Any] = field(default_factory=dict)


class _BufferedTerminalProcess:
    def __init__(self, terminal_id: str, command: list[str], *, cwd: str, env: dict[str, str]) -> None:
        self.terminal_id = terminal_id
        self.command = list(command)
        self.cwd = cwd
        self.env = dict(env)
        self._lock = threading.Lock()
        self._buffer: list[str] = []
        self._released = False
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_output, name=f"orgemage-terminal-{terminal_id}", daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        for chunk in self.process.stdout:
            with self._lock:
                self._buffer.append(chunk)
        self.process.stdout.close()

    def output(self) -> str:
        with self._lock:
            return "".join(self._buffer)

    def wait(self, timeout: float | None = None) -> int | None:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> int | None:
        if self.process.poll() is None:
            self.process.kill()
        return self.wait(timeout=5)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.process.poll() is None:
            self.process.kill()
            self.wait(timeout=5)
        self._reader.join(timeout=1)


class _DownstreamCallbackLayer:
    def __init__(
        self,
        *,
        agent: DownstreamAgentConfig,
        request_error: type[Exception],
        store: SQLiteSessionStore | None,
        upstream_client_getter: Callable[[], Any | None] | None = None,
        upstream_capabilities_getter: Callable[[], Any | None] | None = None,
        headless_policy: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> None:
        self.agent = agent
        self.request_error = request_error
        self.store = store
        self._upstream_client_getter = upstream_client_getter or (lambda: None)
        self._upstream_capabilities_getter = upstream_capabilities_getter or (lambda: None)
        self._headless_policy = headless_policy or (lambda method, payload: "allow")
        self._contexts: dict[str, _ExecutionContext] = {}
        self._terminals: dict[str, _BufferedTerminalProcess] = {}
        self._terminal_to_session: dict[str, str] = {}
        self._pending_permissions: dict[str, set[str]] = defaultdict(set)
        self._cancelled_sessions: set[str] = set()
        self._lock = threading.Lock()

    def bind_execution(self, context: _ExecutionContext) -> None:
        with self._lock:
            self._contexts[context.downstream_session_id] = context
        debug_event(_LOG, "connector.callback.bind_execution", downstream_session_id=context.downstream_session_id, orchestrator_session_id=context.orchestrator_session_id, task_id=context.task.task_id, turn_id=context.turn_id)

    def cleanup_session(self, downstream_session_id: str, *, reason: str) -> None:
        with self._lock:
            context = self._contexts.pop(downstream_session_id, None)
            terminal_ids = [
                terminal_id
                for terminal_id, current_session_id in self._terminal_to_session.items()
                if current_session_id == downstream_session_id
            ]
        if context is None:
            return
        debug_event(_LOG, "connector.callback.cleanup", downstream_session_id=downstream_session_id, task_id=context.task.task_id, reason=reason)
        for terminal_id in terminal_ids:
            self._release_terminal(context, terminal_id=terminal_id, reason=reason, allow_missing=True)

    async def cancel_session(self, downstream_session_id: str, *, reason: str) -> None:
        with self._lock:
            self._cancelled_sessions.add(downstream_session_id)
            context = self._contexts.get(downstream_session_id)
            pending_permission_ids = list(self._pending_permissions.get(downstream_session_id, set()))
        if context is None:
            return
        debug_event(_LOG, "connector.callback.cancel", downstream_session_id=downstream_session_id, task_id=context.task.task_id, reason=reason, pending_permissions=pending_permission_ids)
        for request_id in pending_permission_ids:
            self._persist_permission(
                context,
                PermissionRequestState(
                    request_id=request_id,
                    owner_task_id=context.task.task_id,
                    status="cancelled",
                    decision="cancelled",
                    metadata={"agent_id": self.agent.agent_id, "decision_source": "cancel", "cancel_reason": reason},
                ),
            )
        await self._release_all_terminals(context, reason=reason)
        self.cleanup_session(downstream_session_id, reason=reason)

    async def request_permission(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        if not self.agent.capabilities.supports_permissions:
            raise self._request_error(
                "session/request_permission",
                "Permission brokering is not enabled for this downstream agent",
                kind="forbidden",
            )
        request_id = str(kwargs.get("request_id") or kwargs.get("requestId") or uuid.uuid4().hex)
        payload = _to_plain_data(kwargs)
        request_state = PermissionRequestState(
            request_id=request_id,
            owner_task_id=context.task.task_id,
            status="requested",
            payload=payload,
            metadata={"agent_id": self.agent.agent_id, "decision_source": "pending", "turnId": context.turn_id, "promptMetadata": dict(context.prompt_metadata)},
        )
        self._persist_permission(context, request_state)
        with self._lock:
            self._pending_permissions[context.downstream_session_id].add(request_id)
        upstream_client = self._upstream_client_getter()
        try:
            if upstream_client is not None and hasattr(upstream_client, "request_permission"):
                response = await upstream_client.request_permission(**kwargs)
                if self._is_cancelled(context.downstream_session_id):
                    request_state.status = "cancelled"
                    request_state.decision = "cancelled"
                    request_state.metadata["decision_source"] = "cancel"
                    request_state.metadata["audit"] = {"request": payload, "decision": "cancelled"}
                    self._persist_permission(context, request_state)
                    return {"request_id": request_id, "decision": "cancelled"}
                decision = self._extract_permission_decision(response)
                request_state.status = "decided"
                request_state.decision = decision
                request_state.metadata["decision_source"] = "upstream"
                request_state.metadata["audit"] = {
                    "request": payload,
                    "decision": decision,
                    "response": _to_plain_data(response),
                }
                self._persist_permission(context, request_state)
                debug_event(_LOG, "connector.permission.decision", agent_id=self.agent.agent_id, task_id=context.task.task_id, request_id=request_id, decision=decision, source="upstream")
                return response
            decision = self._headless_policy("session/request_permission", payload)
            if self._is_cancelled(context.downstream_session_id):
                decision = "cancelled"
                request_state.status = "cancelled"
                request_state.metadata["decision_source"] = "cancel"
            else:
                request_state.status = "decided"
                request_state.metadata["decision_source"] = "headless_policy"
            request_state.decision = decision
            request_state.metadata["audit"] = {
                "request": payload,
                "decision": decision,
            }
            self._persist_permission(context, request_state)
            debug_event(_LOG, "connector.permission.decision", agent_id=self.agent.agent_id, task_id=context.task.task_id, request_id=request_id, decision=decision, source=request_state.metadata.get("decision_source"))
            return {"request_id": request_id, "decision": decision}
        finally:
            with self._lock:
                self._pending_permissions.get(context.downstream_session_id, set()).discard(request_id)

    async def read_text_file(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        if not self.agent.capabilities.supports_filesystem:
            raise self._request_error("fs/read_text_file", "Filesystem access is not enabled for this downstream agent", kind="forbidden")
        requested_path = self._extract_path(kwargs)
        resolved_path = self._validate_filesystem_access(
            context=context,
            method="fs/read_text_file",
            path=requested_path,
            upstream_method_name="read_text_file",
        )
        upstream_client = self._upstream_client_getter()
        self._persist_trace(
            context,
            trace_key=f"fs:{uuid.uuid4().hex}",
            metadata={"method": "fs/read_text_file", "path": str(resolved_path), "decision": "allow"},
        )
        if upstream_client is not None and hasattr(upstream_client, "read_text_file"):
            return await upstream_client.read_text_file(path=str(resolved_path))
        return {"path": str(resolved_path), "content": Path(resolved_path).read_text(encoding="utf-8")}

    async def write_text_file(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        if not self.agent.capabilities.supports_filesystem:
            raise self._request_error("fs/write_text_file", "Filesystem access is not enabled for this downstream agent", kind="forbidden")
        requested_path = self._extract_path(kwargs)
        resolved_path = self._validate_filesystem_access(
            context=context,
            method="fs/write_text_file",
            path=requested_path,
            upstream_method_name="write_text_file",
        )
        content = kwargs.get("content")
        if not isinstance(content, str):
            raise self._request_error("fs/write_text_file", "Expected string 'content' payload", kind="invalid_params")
        upstream_client = self._upstream_client_getter()
        self._persist_trace(
            context,
            trace_key=f"fs:{uuid.uuid4().hex}",
            metadata={"method": "fs/write_text_file", "path": str(resolved_path), "decision": "allow"},
        )
        if upstream_client is not None and hasattr(upstream_client, "write_text_file"):
            return await upstream_client.write_text_file(path=str(resolved_path), content=content)
        Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
        Path(resolved_path).write_text(content, encoding="utf-8")
        return {"path": str(resolved_path), "bytes_written": len(content.encode("utf-8"))}

    async def create_terminal(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        if not self.agent.capabilities.supports_terminal:
            raise self._request_error("terminal/create", "Terminal access is not enabled for this downstream agent", kind="forbidden")
        command = self._normalize_terminal_command(kwargs)
        terminal_id = str(kwargs.get("terminal_id") or kwargs.get("terminalId") or f"term-{uuid.uuid4().hex[:12]}")
        env = os.environ.copy()
        upstream_client = self._upstream_client_getter()
        if upstream_client is not None and hasattr(upstream_client, "create_terminal"):
            response = await upstream_client.create_terminal(command=command, cwd=context.cwd)
            upstream_terminal_id = str(
                _to_plain_data(response).get("terminal_id")
                or _to_plain_data(response).get("terminalId")
                or terminal_id
            )
        else:
            process = await asyncio.to_thread(
                _BufferedTerminalProcess,
                terminal_id,
                command,
                cwd=context.cwd,
                env=env,
            )
            with self._lock:
                self._terminals[terminal_id] = process
                self._terminal_to_session[terminal_id] = context.downstream_session_id
            upstream_terminal_id = terminal_id
        self._persist_terminal(
            context,
            TerminalMapping(
                upstream_terminal_id=upstream_terminal_id,
                downstream_terminal_id=terminal_id,
                owner_task_id=context.task.task_id,
                owner_agent_id=self.agent.agent_id,
                refcount=1,
                status="active",
                metadata={"command": command},
            ),
        )
        return {"terminal_id": terminal_id}

    async def terminal_output(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        terminal_id = self._extract_terminal_id(kwargs)
        mapping = self._require_terminal_mapping(context, terminal_id, method="terminal/output")
        upstream_client = self._upstream_client_getter()
        if upstream_client is not None and hasattr(upstream_client, "terminal_output"):
            return await upstream_client.terminal_output(terminal_id=mapping.upstream_terminal_id)
        process = self._require_local_terminal(terminal_id, method="terminal/output")
        return {"terminal_id": terminal_id, "content": process.output()}

    async def wait_for_terminal_exit(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        terminal_id = self._extract_terminal_id(kwargs)
        mapping = self._require_terminal_mapping(context, terminal_id, method="terminal/wait_for_exit")
        timeout = kwargs.get("timeout_seconds") or kwargs.get("timeoutSeconds")
        timeout_value = float(timeout) if isinstance(timeout, (int, float)) else None
        upstream_client = self._upstream_client_getter()
        if upstream_client is not None and hasattr(upstream_client, "wait_for_terminal_exit"):
            response = await upstream_client.wait_for_terminal_exit(
                terminal_id=mapping.upstream_terminal_id,
                timeout_seconds=timeout_value,
            )
            self._mark_terminal_status(context, mapping, status="exited")
            return response
        process = self._require_local_terminal(terminal_id, method="terminal/wait_for_exit")
        exit_code = await asyncio.to_thread(process.wait, timeout_value)
        if exit_code is None:
            return {"terminal_id": terminal_id, "status": "running"}
        self._mark_terminal_status(context, mapping, status="exited")
        return {"terminal_id": terminal_id, "exit_code": exit_code, "status": "exited"}

    async def kill_terminal(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        terminal_id = self._extract_terminal_id(kwargs)
        mapping = self._require_terminal_mapping(context, terminal_id, method="terminal/kill")
        if mapping.status in {"released", "completed"}:
            raise self._request_error("terminal/kill", f"Terminal {terminal_id!r} is not active", kind="forbidden")
        upstream_client = self._upstream_client_getter()
        if upstream_client is not None and hasattr(upstream_client, "kill_terminal"):
            response = await upstream_client.kill_terminal(terminal_id=mapping.upstream_terminal_id)
        else:
            process = self._require_local_terminal(terminal_id, method="terminal/kill")
            exit_code = await asyncio.to_thread(process.kill)
            response = {"terminal_id": terminal_id, "exit_code": exit_code, "status": "killed"}
        self._mark_terminal_status(context, mapping, status="killed")
        return response

    async def release_terminal(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        terminal_id = self._extract_terminal_id(kwargs)
        self._release_terminal(context, terminal_id=terminal_id, reason="release", allow_missing=False)
        return {"terminal_id": terminal_id, "status": "released"}

    def _release_terminal(
        self,
        context: _ExecutionContext,
        *,
        terminal_id: str,
        reason: str,
        allow_missing: bool,
    ) -> None:
        try:
            mapping = self._require_terminal_mapping(context, terminal_id, method="terminal/release")
        except Exception:
            if allow_missing:
                return
            raise
        process = None
        with self._lock:
            process = self._terminals.pop(terminal_id, None)
            self._terminal_to_session.pop(terminal_id, None)
        if process is not None:
            process.release()
        self._mark_terminal_status(context, mapping, status="released", extra_metadata={"cleanup_reason": reason})

    async def _release_all_terminals(self, context: _ExecutionContext, *, reason: str) -> None:
        if self.store is None:
            terminal_ids = [
                terminal_id
                for terminal_id, session_id in self._terminal_to_session.items()
                if session_id == context.downstream_session_id
            ]
        else:
            snapshot = self.store.load(context.orchestrator_session_id)
            terminal_ids = []
            if snapshot is not None:
                terminal_ids = [
                    mapping.downstream_terminal_id
                    for mapping in snapshot.terminal_mappings
                    if mapping.owner_task_id == context.task.task_id and mapping.owner_agent_id == self.agent.agent_id
                ]
        upstream_client = self._upstream_client_getter()
        for terminal_id in terminal_ids:
            try:
                mapping = self._require_terminal_mapping(context, terminal_id, method="terminal/release")
            except Exception:
                continue
            if upstream_client is not None and hasattr(upstream_client, "release_terminal"):
                try:
                    await upstream_client.release_terminal(terminal_id=mapping.upstream_terminal_id)
                except Exception:
                    pass
            self._release_terminal(context, terminal_id=terminal_id, reason=reason, allow_missing=True)

    def _mark_terminal_status(
        self,
        context: _ExecutionContext,
        mapping: TerminalMapping,
        *,
        status: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        mapping.status = status
        mapping.updated_at = time.time()
        if extra_metadata:
            mapping.metadata.update(extra_metadata)
        self._persist_terminal(context, mapping)

    def _require_local_terminal(self, terminal_id: str, *, method: str) -> _BufferedTerminalProcess:
        with self._lock:
            process = self._terminals.get(terminal_id)
        if process is None:
            raise self._request_error(method, f"Terminal {terminal_id!r} is not available", kind="forbidden")
        return process

    def _require_terminal_mapping(
        self,
        context: _ExecutionContext,
        terminal_id: str,
        *,
        method: str,
    ) -> TerminalMapping:
        if self.store is None:
            return TerminalMapping(
                upstream_terminal_id=terminal_id,
                downstream_terminal_id=terminal_id,
                owner_task_id=context.task.task_id,
                owner_agent_id=self.agent.agent_id,
            )
        snapshot = self.store.load(context.orchestrator_session_id)
        if snapshot is None:
            raise self._request_error(method, "Orchestrator session is unavailable", kind="internal_error")
        for mapping in snapshot.terminal_mappings:
            if (
                mapping.downstream_terminal_id == terminal_id
                and mapping.owner_task_id == context.task.task_id
                and mapping.owner_agent_id == self.agent.agent_id
            ):
                return mapping
        raise self._request_error(method, f"Terminal {terminal_id!r} is not mapped to this task", kind="forbidden")

    def _validate_filesystem_access(
        self,
        *,
        context: _ExecutionContext,
        method: str,
        path: str,
        upstream_method_name: str,
    ) -> Path:
        raw_path = Path(path)
        if not raw_path.is_absolute():
            self._persist_trace(context, trace_key=f"fs:{uuid.uuid4().hex}", metadata={"method": method, "path": path, "decision": "deny", "reason": "path_not_absolute"})
            raise self._request_error(method, f"Path must be absolute: {path!r}", kind="invalid_params")
        resolved = raw_path.resolve()
        cwd = Path(context.cwd).resolve()
        try:
            resolved.relative_to(cwd)
        except ValueError as exc:
            self._persist_trace(context, trace_key=f"fs:{uuid.uuid4().hex}", metadata={"method": method, "path": str(resolved), "decision": "deny", "reason": "path_out_of_scope"})
            raise self._request_error(method, f"Path {resolved} is outside session cwd {cwd}", kind="permission_denied") from exc
        upstream_client = self._upstream_client_getter()
        if upstream_client is not None and not hasattr(upstream_client, upstream_method_name):
            self._persist_trace(context, trace_key=f"fs:{uuid.uuid4().hex}", metadata={"method": method, "path": str(resolved), "decision": "deny", "reason": "upstream_capability_gated"})
            raise self._request_error(method, f"Upstream client does not permit {method}", kind="forbidden")
        upstream_capabilities = _to_plain_data(self._upstream_capabilities_getter())
        if upstream_client is not None and isinstance(upstream_capabilities, dict):
            filesystem_capability = upstream_capabilities.get("filesystem") or upstream_capabilities.get("fs")
            if filesystem_capability is False:
                self._persist_trace(context, trace_key=f"fs:{uuid.uuid4().hex}", metadata={"method": method, "path": str(resolved), "decision": "deny", "reason": "upstream_capability_disabled"})
                raise self._request_error(method, f"Upstream capability gating denied {method}", kind="forbidden")
        return resolved

    def _normalize_terminal_command(self, payload: dict[str, Any]) -> list[str]:
        command = payload.get("command")
        if isinstance(command, str) and command:
            return ["/bin/sh", "-lc", command]
        if isinstance(command, list) and all(isinstance(item, str) and item for item in command):
            return list(command)
        raise self._request_error("terminal/create", "Expected non-empty 'command' string or string list", kind="invalid_params")

    def _extract_path(self, payload: dict[str, Any]) -> str:
        path = payload.get("path")
        if isinstance(path, str) and path:
            return path
        raise self._request_error("fs/path", "Expected non-empty 'path' string", kind="invalid_params")

    def _extract_terminal_id(self, payload: dict[str, Any]) -> str:
        terminal_id = payload.get("terminal_id") or payload.get("terminalId")
        if isinstance(terminal_id, str) and terminal_id:
            return terminal_id
        raise self._request_error("terminal/id", "Expected non-empty 'terminal_id' string", kind="invalid_params")

    def _extract_permission_decision(self, response: Any) -> str:
        payload = _to_plain_data(response)
        if isinstance(payload, dict):
            decision = payload.get("decision") or payload.get("result")
            if isinstance(decision, str) and decision:
                return decision
        return "allow"

    def _persist_permission(self, context: _ExecutionContext, request: PermissionRequestState) -> None:
        if self.store is None:
            return
        self.store.persist_permission_event(context.orchestrator_session_id, request)

    def _is_cancelled(self, downstream_session_id: str) -> bool:
        with self._lock:
            return downstream_session_id in self._cancelled_sessions

    def _persist_terminal(self, context: _ExecutionContext, mapping: TerminalMapping) -> None:
        if self.store is None:
            return
        self.store.persist_terminal_event(context.orchestrator_session_id, mapping)

    def _persist_trace(self, context: _ExecutionContext, *, trace_key: str, metadata: dict[str, Any]) -> None:
        if self.store is None:
            return
        self.store.persist_trace_metadata(
            context.orchestrator_session_id,
            TraceCorrelationState(
                trace_key=trace_key,
                task_id=context.task.task_id,
                metadata={
                    "agent_id": self.agent.agent_id,
                    "turnId": context.turn_id,
                    "promptMetadata": dict(context.prompt_metadata),
                    **metadata,
                },
            ),
        )

    def _require_context(self, payload: dict[str, Any]) -> _ExecutionContext:
        session_id = payload.get("session_id") or payload.get("sessionId")
        if isinstance(session_id, str):
            with self._lock:
                context = self._contexts.get(session_id)
            if context is not None:
                return context
        with self._lock:
            if len(self._contexts) == 1:
                return next(iter(self._contexts.values()))
        raise self._request_error("session/context", "Downstream callback invoked without an active execution context", kind="internal_error")

    def _request_error(self, method: str, message: str, *, kind: str) -> Exception:
        builder = getattr(self.request_error, kind, None)
        if callable(builder):
            return builder(method, message)
        message_with_method = f"{method}: {message}"
        try:
            return self.request_error(message_with_method)
        except Exception:
            return RuntimeError(message_with_method)


class _LoopThread:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="orgemage-acp-loop", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coroutine: Any) -> Any:
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1)


class _SessionUpdateCollector:
    def __init__(self) -> None:
        self._updates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._updates[session_id] = []

    def append(self, session_id: str, update: Any) -> None:
        payload = _to_plain_data(update)
        with self._lock:
            self._updates[session_id].append(payload)

    def get(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._updates.get(session_id, []))


class AcpDownstreamConnector:
    def __init__(
        self,
        agent: DownstreamAgentConfig,
        *,
        store: SQLiteSessionStore | None = None,
        upstream_client_getter: Callable[[], Any | None] | None = None,
        upstream_capabilities_getter: Callable[[], Any | None] | None = None,
        headless_policy: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> None:
        self.agent = agent
        self.negotiated_state: DownstreamNegotiatedState | None = None
        self._loop_thread: _LoopThread | None = None
        self._collector = _SessionUpdateCollector()
        self._context_manager: Any = None
        self._conn: Any = None
        self._process: Any = None
        self._sdk: _SdkBindings | None = None
        self._lock = threading.Lock()
        self._catalog_state: dict[str, Any] | None = None
        self._catalog_refresh_required = True
        self._store = store
        self._upstream_client_getter = upstream_client_getter or (lambda: None)
        self._upstream_capabilities_getter = upstream_capabilities_getter or (lambda: None)
        self._headless_policy = headless_policy
        self._callback_layer: _DownstreamCallbackLayer | None = None

    def discover_catalog(self, *, force: bool = False) -> dict[str, Any]:
        self._ensure_started()
        assert self._loop_thread is not None
        return self._loop_thread.run(self._discover_catalog_async(force=force))

    def mark_catalog_refresh_required(self) -> None:
        self._catalog_refresh_required = True

    def execute_task(
        self,
        *,
        orchestrator_session_id: str,
        downstream_session_id: str | None,
        cwd: str,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
    ) -> DownstreamPromptResult:
        self._ensure_started()
        assert self._loop_thread is not None
        return self._loop_thread.run(
            self._execute_task_async(
                orchestrator_session_id=orchestrator_session_id,
                downstream_session_id=downstream_session_id,
                cwd=cwd,
                task=task,
                coordinator_prompt=coordinator_prompt,
                selected_model=selected_model,
            )
        )

    def cancel(self, downstream_session_id: str) -> None:
        self._ensure_started()
        assert self._loop_thread is not None
        self._loop_thread.run(self._cancel_async(downstream_session_id))

    def close(self) -> None:
        if self._loop_thread is None:
            return
        self._loop_thread.run(self._close_async())
        self._loop_thread.stop()
        self._loop_thread = None

    def _ensure_started(self) -> None:
        if self._loop_thread is not None and self._conn is not None:
            return
        with self._lock:
            if self._loop_thread is None:
                self._loop_thread = _LoopThread()
            if self._conn is None:
                assert self._loop_thread is not None
                self._loop_thread.run(self._start_async())

    async def _start_async(self) -> None:
        debug_event(_LOG, "connector.lifecycle.start", agent_id=self.agent.agent_id, command=self.agent.command, args=self.agent.args)
        sdk = _load_sdk()
        self._sdk = sdk
        self._callback_layer = _DownstreamCallbackLayer(
            agent=self.agent,
            request_error=sdk.request_error,
            store=self._store,
            upstream_client_getter=self._upstream_client_getter,
            upstream_capabilities_getter=self._upstream_capabilities_getter,
            headless_policy=self._headless_policy,
        )
        client_impl = _build_runtime_client(sdk, self._collector, self._callback_layer)
        self._context_manager = sdk.spawn_agent_process(client_impl, self.agent.command, *self.agent.args)
        try:
            self._conn, self._process = await self._context_manager.__aenter__()
        except Exception as exc:  # pragma: no cover - depends on external runtime
            raise DownstreamConnectorError(
                f"Unable to start downstream agent '{self.agent.agent_id}' using {self.agent.command!r}: {exc}"
            ) from exc

        initialize_response = await self._conn.initialize(
            protocol_version=sdk.protocol_version,
            client_capabilities=sdk.client_capabilities(),
            client_info=sdk.implementation(name="orgemage", title="OrgeMage", version="0.1.0"),
        )
        payload = _to_plain_data(initialize_response)
        self.negotiated_state = DownstreamNegotiatedState(
            agent_id=self.agent.agent_id,
            agent_info=payload.get("agent_info") or payload.get("agentInfo") or {},
            agent_capabilities=payload.get("agent_capabilities") or payload.get("agentCapabilities") or {},
            auth_methods=payload.get("auth_methods") or payload.get("authMethods") or [],
            protocol_version=payload.get("protocol_version") or payload.get("protocolVersion"),
        )
        await self._refresh_catalog_async(force=True)
        debug_event(_LOG, "connector.lifecycle.started", agent_id=self.agent.agent_id, negotiated_state=self.negotiated_state.to_dict() if self.negotiated_state is not None else None)

    async def _close_async(self) -> None:
        debug_event(_LOG, "connector.lifecycle.close", agent_id=self.agent.agent_id)
        if self._callback_layer is not None:
            for session_id in list(self._callback_layer._contexts.keys()):
                self._callback_layer.cleanup_session(session_id, reason="connector_close")
        if self._context_manager is not None:
            await self._context_manager.__aexit__(None, None, None)
        self._context_manager = None
        self._conn = None
        self._process = None
        self._sdk = None
        self._catalog_state = None
        self._catalog_refresh_required = True

    async def _cancel_async(self, downstream_session_id: str) -> None:
        debug_event(_LOG, "connector.lifecycle.cancel", agent_id=self.agent.agent_id, downstream_session_id=downstream_session_id)
        if self._conn is None:
            return
        try:
            await self._conn.cancel(session_id=downstream_session_id)
        finally:
            if self._callback_layer is not None:
                await self._callback_layer.cancel_session(downstream_session_id, reason="cancel")

    async def _discover_catalog_async(self, *, force: bool = False) -> dict[str, Any]:
        if self._conn is None:
            raise DownstreamConnectorError(f"Downstream agent '{self.agent.agent_id}' is not connected")
        debug_event(_LOG, "connector.lifecycle.discover_catalog", agent_id=self.agent.agent_id, force=force)
        if not force and self._catalog_state is not None and not self._catalog_refresh_required:
            return dict(self._catalog_state)
        return await self._refresh_catalog_async(force=force)

    async def _execute_task_async(
        self,
        *,
        orchestrator_session_id: str,
        downstream_session_id: str | None,
        cwd: str,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
    ) -> DownstreamPromptResult:
        if self._conn is None or self._sdk is None:
            raise DownstreamConnectorError(f"Downstream agent '{self.agent.agent_id}' is not connected")

        debug_event(_LOG, "connector.lifecycle.execute.start", agent_id=self.agent.agent_id, orchestrator_session_id=orchestrator_session_id, task_id=task.task_id, downstream_session_id=downstream_session_id)
        session_id, session_payload = await self._ensure_session_async(cwd, downstream_session_id)
        self._record_session_state(session_id, session_payload)
        await self._apply_model_selection_async(session_id, selected_model, cwd=cwd)
        if self._callback_layer is not None:
            self._callback_layer.bind_execution(
                _ExecutionContext(
                    orchestrator_session_id=orchestrator_session_id,
                    downstream_session_id=session_id,
                    cwd=cwd,
                    task=task,
                    turn_id=str(task._meta.get("turnId")) if task._meta.get("turnId") is not None else None,
                    prompt_metadata=dict(task._meta),
                )
            )

        prompt_text = self._build_prompt(
            orchestrator_session_id=orchestrator_session_id,
            task=task,
            coordinator_prompt=coordinator_prompt,
            selected_model=selected_model,
        )
        self._collector.reset(session_id)
        cleanup_reason = "completion"
        try:
            response = await self._conn.prompt(
                session_id=session_id,
                prompt=[self._sdk.text_block(prompt_text)],
            )
            updates = self._collector.get(session_id)
            summary = _extract_summary(_to_plain_data(response), updates)
            if not summary:
                summary = f"Downstream agent {self.agent.name} completed {task.title!r}."
            stop_reason = _extract_stop_reason(_to_plain_data(response))
            result = DownstreamPromptResult(
                downstream_session_id=session_id,
                status=TaskStatus.CANCELLED if stop_reason == "cancelled" else TaskStatus.COMPLETED,
                summary=summary,
                raw_output=summary,
                updates=updates,
                response=_to_plain_data(response),
            )
            debug_event(_LOG, "connector.lifecycle.execute.complete", agent_id=self.agent.agent_id, task_id=task.task_id, downstream_session_id=session_id, status=result.status.value, update_count=len(updates))
            return result
        except Exception:
            cleanup_reason = "failure"
            raise
        finally:
            if self._callback_layer is not None:
                self._callback_layer.cleanup_session(session_id, reason=cleanup_reason)

    async def _ensure_session_async(self, cwd: str, downstream_session_id: str | None) -> tuple[str, Any]:
        if downstream_session_id and self.negotiated_state and self.negotiated_state.load_session_supported:
            loaded = await self._conn.load_session(
                cwd=cwd,
                mcp_servers=[],
                session_id=downstream_session_id,
            )
            if loaded is not None:
                return downstream_session_id, loaded
        created = await self._conn.new_session(cwd=cwd, mcp_servers=[])
        payload = _to_plain_data(created)
        created_session_id = payload.get("session_id") or payload.get("sessionId")
        if not created_session_id:
            raise DownstreamConnectorError(
                f"Downstream agent '{self.agent.agent_id}' returned no session id from session/new"
            )
        return str(created_session_id), created

    async def _apply_model_selection_async(self, session_id: str, selected_model: str, *, cwd: str) -> None:
        if self._conn is None:
            return
        raw_model = self.agent.resolve_model(selected_model) or self.agent.default_model
        if raw_model is None:
            return
        config_options = []
        if self.negotiated_state is not None:
            config_options = self.negotiated_state.config_options.get(session_id, [])
        if not any(option.get("id") == "model" for option in config_options):
            return
        await self._conn.set_config_option(session_id=session_id, option_id="model", value=raw_model)
        await self._refresh_catalog_async(force=True, cwd=cwd, downstream_session_id=session_id)

    def _record_session_state(self, session_id: str, session_payload: Any) -> None:
        if self.negotiated_state is None:
            return
        payload = _to_plain_data(session_payload)
        config_options = payload.get("config_options") or payload.get("configOptions") or []
        session_capabilities = {
            key: value
            for key, value in payload.items()
            if key not in {"config_options", "configOptions", "session_id", "sessionId"}
        }
        self.negotiated_state.record_session(
            session_id=session_id,
            capabilities=session_capabilities,
            config_options=config_options,
        )

    async def _refresh_catalog_async(
        self,
        *,
        force: bool = False,
        cwd: str | None = None,
        downstream_session_id: str | None = None,
    ) -> dict[str, Any]:
        if self._conn is None:
            raise DownstreamConnectorError(f"Downstream agent '{self.agent.agent_id}' is not connected")
        if not force and self._catalog_state is not None and not self._catalog_refresh_required:
            return dict(self._catalog_state)
        normalized_cwd = cwd or "."
        if downstream_session_id and self.negotiated_state and self.negotiated_state.load_session_supported:
            session_payload = await self._conn.load_session(
                cwd=normalized_cwd,
                mcp_servers=[],
                session_id=downstream_session_id,
            )
            if session_payload is None:
                session_payload = await self._conn.new_session(cwd=normalized_cwd, mcp_servers=[])
        else:
            session_payload = await self._conn.new_session(cwd=normalized_cwd, mcp_servers=[])
        payload = _to_plain_data(session_payload)
        session_id = str(payload.get("session_id") or payload.get("sessionId") or downstream_session_id or "")
        if session_id:
            self._record_session_state(session_id, session_payload)
        config_options = payload.get("config_options") or payload.get("configOptions") or []
        capabilities = {
            key: value
            for key, value in payload.items()
            if key not in {"config_options", "configOptions", "session_id", "sessionId"}
        }
        if self.negotiated_state is not None:
            capabilities = {**self.negotiated_state.agent_capabilities, **capabilities}
        self._catalog_state = {
            "agent_id": self.agent.agent_id,
            "config_options": config_options,
            "capabilities": capabilities,
            "command_advertisements": _extract_command_advertisements(capabilities),
            "refreshed_at": time.time(),
        }
        self._catalog_refresh_required = False
        return dict(self._catalog_state)

    def _build_prompt(
        self,
        *,
        orchestrator_session_id: str,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
    ) -> str:
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


def _load_sdk() -> _SdkBindings:
    try:
        acp = importlib.import_module("acp")
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise DownstreamConnectorError(
            "The optional dependency 'agent-client-protocol' is required for real downstream "
            "connectors. Install with `pip install orgemage[acp]` or enable the mock runtime explicitly."
        ) from exc
    try:
        bindings = _SdkBindings(
            module=acp,
            protocol_version=acp.PROTOCOL_VERSION,
            client_base=_acp_attr(acp, "Client"),
            request_error=_acp_attr(acp, "RequestError"),
            client_capabilities=_acp_attr(acp, "ClientCapabilities"),
            implementation=_acp_attr(acp, "Implementation"),
            spawn_agent_process=acp.spawn_agent_process,
            text_block=acp.text_block,
        )
    except AttributeError as exc:  # pragma: no cover - depends on external runtime
        raise DownstreamConnectorError(f"Installed ACP SDK is missing required API surface: {exc}") from exc
    required_fields = {
        "Client": bindings.client_base,
        "RequestError": bindings.request_error,
        "ClientCapabilities": bindings.client_capabilities,
        "Implementation": bindings.implementation,
        "spawn_agent_process": bindings.spawn_agent_process,
        "text_block": bindings.text_block,
    }
    missing = [name for name, value in required_fields.items() if value is None]
    if missing:  # pragma: no cover - depends on external runtime
        raise DownstreamConnectorError(
            "Installed ACP SDK is missing required API surface: "
            + ", ".join(f"module acp has no attribute {name}" for name in missing)
        )
    return bindings


def _build_runtime_client(
    sdk: _SdkBindings,
    collector: _SessionUpdateCollector,
    callback_layer: _DownstreamCallbackLayer,
) -> Any:
    request_error = sdk.request_error

    class RuntimeClient(sdk.client_base):
        async def request_permission(self, **kwargs: Any) -> Any:
            return await callback_layer.request_permission(**kwargs)

        async def write_text_file(self, **kwargs: Any) -> Any:
            return await callback_layer.write_text_file(**kwargs)

        async def read_text_file(self, **kwargs: Any) -> Any:
            return await callback_layer.read_text_file(**kwargs)

        async def create_terminal(self, **kwargs: Any) -> Any:
            return await callback_layer.create_terminal(**kwargs)

        async def terminal_output(self, **kwargs: Any) -> Any:
            return await callback_layer.terminal_output(**kwargs)

        async def release_terminal(self, **kwargs: Any) -> Any:
            return await callback_layer.release_terminal(**kwargs)

        async def wait_for_terminal_exit(self, **kwargs: Any) -> Any:
            return await callback_layer.wait_for_terminal_exit(**kwargs)

        async def kill_terminal(self, **kwargs: Any) -> Any:
            return await callback_layer.kill_terminal(**kwargs)

        async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
            collector.append(session_id, update)

        async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            raise request_error.method_not_found(method)

        async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
            raise request_error.method_not_found(method)

    return RuntimeClient()


def _extract_summary(response: dict[str, Any], updates: list[dict[str, Any]]) -> str:
    summary = _extract_text(response)
    if summary:
        return summary
    for update in reversed(updates):
        summary = _extract_text(update)
        if summary:
            return summary
    return ""


def _extract_stop_reason(response: dict[str, Any]) -> str | None:
    stop_reason = response.get("stop_reason") or response.get("stopReason")
    if isinstance(stop_reason, str) and stop_reason:
        return stop_reason
    return None


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str) and text:
            return text
        for key in ("content", "message"):
            extracted = _extract_text(payload.get(key))
            if extracted:
                return extracted
        for value in payload.values():
            extracted = _extract_text(value)
            if extracted:
                return extracted
        return ""
    if isinstance(payload, list):
        parts = [_extract_text(item) for item in payload]
        return "\n".join(part for part in parts if part)
    return ""


def _to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_plain_data(value.model_dump(mode="python", exclude_none=True))
    if hasattr(value, "dict"):
        return _to_plain_data(value.dict(exclude_none=True))
    if hasattr(value, "__dict__"):
        return {key: _to_plain_data(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def _extract_command_advertisements(capabilities: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("commands", "command_ads", "commandAdvertisements", "advertised_commands", "advertisedCommands"):
        value = capabilities.get(key)
        if isinstance(value, list):
            candidates.extend(str(item) for item in value if item)
    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
