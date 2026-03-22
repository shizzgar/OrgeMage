from __future__ import annotations

import asyncio
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
import importlib
import inspect
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any, Callable, Protocol
import uuid

from .. import __version__
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
        mcp_servers: list[dict[str, Any]] | list[Any] | None,
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
    file_system_capability: type[Any]
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


@dataclass(frozen=True, slots=True)
class _DownstreamProfileSpec:
    profile_id: str
    title: str
    command: str
    required_args: tuple[str, ...] = ()
    optional_args: tuple[str, ...] = ()
    required_initialize_fields: tuple[str, ...] = ()
    optional_initialize_fields: tuple[str, ...] = ()
    required_session_fields: tuple[str, ...] = ()
    optional_session_fields: tuple[str, ...] = ()
    supported_mcp_transports: tuple[str, ...] = ()
    known_quirks: tuple[str, ...] = ()


_DOWNSTREAM_PROFILE_SPECS: dict[str, _DownstreamProfileSpec] = {
    "codex-acp": _DownstreamProfileSpec(
        profile_id="codex-acp",
        title="Codex ACP",
        command="codex-acp",
        required_initialize_fields=("agentCapabilities.loadSession",),
        optional_initialize_fields=("authMethods", "agentCapabilities.mcpCapabilities"),
        required_session_fields=("models.availableModels", "modes.availableModes", "configOptions"),
        optional_session_fields=("available_commands_update",),
        supported_mcp_transports=("stdio",),
        known_quirks=(
            "codex-acp exposes ACP over a dedicated bridge binary rather than the raw codex CLI.",
            "Its stdio entrypoint can fail inside restrictive sandboxes; treat transport startup errors as environment issues before assuming protocol breakage.",
        ),
    ),
    "gemini": _DownstreamProfileSpec(
        profile_id="gemini",
        title="Gemini CLI",
        command="gemini",
        required_args=("--acp",),
        optional_args=("--experimental-acp",),
        required_initialize_fields=("agentCapabilities.loadSession", "authMethods"),
        optional_initialize_fields=("agentCapabilities.mcpCapabilities",),
        required_session_fields=("models.availableModels", "modes.availableModes"),
        optional_session_fields=("available_commands_update",),
        supported_mcp_transports=("stdio", "sse", "http"),
        known_quirks=(
            "Gemini CLI returns modes/models on session creation but not configOptions in its ACP responses.",
            "Startup slash commands are emitted asynchronously after session/new or session/load.",
        ),
    ),
    "qwen": _DownstreamProfileSpec(
        profile_id="qwen",
        title="Qwen Code",
        command="qwen",
        required_args=("--acp",),
        optional_args=("--experimental-acp",),
        required_initialize_fields=("agentCapabilities.loadSession", "authMethods"),
        optional_initialize_fields=("agentCapabilities.sessionCapabilities.resume",),
        required_session_fields=("models.availableModels", "modes.availableModes", "configOptions"),
        optional_session_fields=("available_commands_update",),
        supported_mcp_transports=("stdio",),
        known_quirks=(
            "Qwen ACP models are auth-scoped and may be encoded as authType/modelId composite values.",
            "Non-stdio MCP servers are ignored by the CLI, so OrgeMage records a transport diagnostic instead of silently pretending they were applied.",
        ),
    ),
}


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
        request_id = str(self._get_param(kwargs, "request_id", "requestId") or uuid.uuid4().hex)
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
                    return {
                        "request_id": request_id, 
                        "decision": "cancelled",
                        "outcome": "success",
                        "content": [{"type": "text", "text": "Permission cancelled"}]
                    }
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
                
                # Wrap response if needed
                res_payload = _to_plain_data(response)
                if isinstance(res_payload, dict):
                    # Ensure minimal required fields for permission response
                    if "decision" not in res_payload and "result" in res_payload:
                        res_payload["decision"] = res_payload["result"]
                return res_payload

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
            
            # Pure ACP permission response
            return {
                "request_id": request_id, 
                "decision": decision
            }
        finally:
            with self._lock:
                self._pending_permissions.get(context.downstream_session_id, set()).discard(request_id)

    async def read_text_file(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        requested_path = self._extract_path(kwargs)
        session_id = kwargs.get("session_id") or kwargs.get("sessionId")
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
        
        try:
            content = Path(resolved_path).read_text(encoding="utf-8")
            return {
                "path": str(resolved_path), 
                "content": content,
                "outcome": "success",
                "success": True,
                "sessionId": session_id,
                "session_id": session_id
            }
        except FileNotFoundError:
            return {
                "path": str(resolved_path),
                "error": "File not found",
                "outcome": "failure",
                "success": False,
                "sessionId": session_id,
                "session_id": session_id
            }
        except Exception as e:
            return {
                "path": str(resolved_path),
                "error": str(e),
                "outcome": "failure",
                "success": False,
                "sessionId": session_id,
                "session_id": session_id
            }

    async def write_text_file(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        if not self.agent.capabilities.supports_filesystem:
            raise self._request_error("fs/write_text_file", "Filesystem access is not enabled for this downstream agent", kind="forbidden")
        requested_path = self._extract_path(kwargs)
        session_id = kwargs.get("session_id") or kwargs.get("sessionId")
        resolved_path = self._validate_filesystem_access(
            context=context,
            method="fs/write_text_file",
            path=requested_path,
            upstream_method_name="write_text_file",
        )
        content = self._get_param(kwargs, "content")
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
        msg = f"Successfully wrote {len(content.encode('utf-8'))} bytes to {resolved_path}"
        return {
            "path": str(resolved_path), 
            "bytes_written": len(content.encode("utf-8")),
            "outcome": "success",
            "success": True,
            "content": [{"type": "text", "text": msg}],
            "sessionId": session_id,
            "session_id": session_id
        }

    async def list_directory(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        if not self.agent.capabilities.supports_filesystem:
            raise self._request_error("fs/list_directory", "Filesystem access is not enabled for this downstream agent", kind="forbidden")
        
        dir_path = kwargs.get("dir_path") or kwargs.get("path") or "."
        session_id = kwargs.get("session_id") or kwargs.get("sessionId")
        # Validate path
        raw_path = Path(dir_path)
        if not raw_path.is_absolute():
            resolved = (Path(context.cwd) / raw_path).resolve()
        else:
            resolved = raw_path.resolve()
            
        cwd = Path(context.cwd).resolve()
        try:
            resolved.relative_to(cwd)
        except ValueError:
            raise self._request_error("fs/list_directory", f"Path {resolved} is outside session cwd {cwd}", kind="permission_denied")

        import os
        entries = []
        for entry in os.scandir(resolved):
            entries.append({
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "is_file": entry.is_file()
            })
            
        return {
            "entries": entries,
            "outcome": "success",
            "content": [{"type": "text", "text": f"Directory listed: {len(entries)} items"}],
            "sessionId": session_id,
            "session_id": session_id
        }

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
            upstream_terminal_id = terminal_id
        with self._lock:
            self._terminal_to_session[terminal_id] = context.downstream_session_id
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
        return {
            "terminal_id": terminal_id,
            "outcome": "success",
            "content": [{"type": "text", "text": f"Terminal {terminal_id} created"}]
        }

    async def terminal_output(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        terminal_id = self._extract_terminal_id(kwargs)
        mapping = self._require_terminal_mapping(context, terminal_id, method="terminal/output")
        upstream_client = self._upstream_client_getter()
        if upstream_client is not None and hasattr(upstream_client, "terminal_output"):
            return await upstream_client.terminal_output(terminal_id=mapping.upstream_terminal_id)
        process = self._require_local_terminal(terminal_id, method="terminal/output")
        output = process.output()
        return {
            "terminal_id": terminal_id, 
            "content": output,
            "outcome": "success",
            "content_list": [{"type": "text", "text": output}] # Some agents use content as list
        }

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
        return {
            "terminal_id": terminal_id, 
            "exit_code": exit_code, 
            "status": "exited",
            "outcome": "success" if exit_code == 0 else "failure"
        }

    async def run_shell_command(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        if not self.agent.capabilities.supports_terminal:
            raise self._request_error("shell/run", "Terminal access is not enabled for this downstream agent", kind="forbidden")
        
        command = self._get_param(kwargs, "command")
        if not command:
            raise self._request_error("shell/run", "Expected 'command' payload", kind="invalid_params")
        
        # Normalize command to list if it is a string
        if isinstance(command, str):
            cmd_args = ["/bin/sh", "-c", command]
        else:
            cmd_args = list(command)

        debug_event(_LOG, "connector.callback.run_shell_command", agent_id=self.agent.agent_id, command=command)
        
        result = await asyncio.to_thread(
            subprocess.run,
            cmd_args,
            cwd=context.cwd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = f"Exit code: {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}"
        session_id = kwargs.get("session_id") or kwargs.get("sessionId")
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "outcome": "success" if result.returncode == 0 else "failure",
            "content": [{"type": "text", "text": output}],
            "sessionId": session_id,
            "session_id": session_id
        }

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
            response = {
                "terminal_id": terminal_id, 
                "exit_code": exit_code, 
                "status": "killed",
                "outcome": "success",
                "content": [{"type": "text", "text": f"Terminal {terminal_id} killed"}]
            }
        self._mark_terminal_status(context, mapping, status="killed")
        return response

    async def release_terminal(self, **kwargs: Any) -> Any:
        context = self._require_context(kwargs)
        terminal_id = self._extract_terminal_id(kwargs)
        self._release_terminal(context, terminal_id=terminal_id, reason="release", allow_missing=False)
        return {
            "terminal_id": terminal_id, 
            "status": "released",
            "outcome": "success",
            "content": [{"type": "text", "text": f"Terminal {terminal_id} released"}]
        }

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
            # Automatically resolve relative paths against the context CWD
            resolved = (Path(context.cwd) / raw_path).resolve()
        else:
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

    def _get_param(self, payload: dict[str, Any], *names: str) -> Any | None:
        """Robustly extract a parameter from payload, checking root and nested tool_call."""
        for name in names:
            # Check root
            val = payload.get(name)
            if val is not None:
                return val
            
            # Check tool_call
            tool_call = payload.get("tool_call") or payload.get("toolCall")
            if isinstance(tool_call, dict):
                # Check tool_call.params
                params = tool_call.get("params")
                if isinstance(params, dict):
                    val = params.get(name)
                    if val is not None:
                        return val
                # Check tool_call directly (some agents put params there)
                val = tool_call.get(name)
                if val is not None:
                    return val
        return None

    def _normalize_terminal_command(self, payload: dict[str, Any]) -> list[str]:
        command = self._get_param(payload, "command")
        if isinstance(command, str) and command:
            return ["/bin/sh", "-lc", command]
        if isinstance(command, list) and all(isinstance(item, str) and item for item in command):
            return list(command)
        raise self._request_error("terminal/create", "Expected non-empty 'command' string or string list", kind="invalid_params")

    def _extract_path(self, payload: dict[str, Any]) -> str:
        path = self._get_param(payload, "path")
        if isinstance(path, str) and path:
            return path
        raise self._request_error("fs/path", "Expected non-empty 'path' string", kind="invalid_params")

    def _extract_terminal_id(self, payload: dict[str, Any]) -> str:
        terminal_id = self._get_param(payload, "terminal_id", "terminalId")
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
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: queue.Queue[tuple[Any, Future[Any]] | None] = queue.Queue()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._run, name="orgemage-acp-loop", daemon=True)
        self._thread.start()
        self._started.wait(timeout=1)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._started.set()
        while True:
            item = self._queue.get()
            if item is None:
                break
            coroutine, future = item
            if future.cancelled():
                continue
            try:
                result = self._loop.run_until_complete(coroutine)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
        self._loop.close()

    def run(self, coroutine: Any) -> Any:
        self._started.wait(timeout=1)
        future = Future()
        self._queue.put((coroutine, future))
        return future.result()

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=1)


class _SessionUpdateCollector:
    def __init__(self, *, on_update: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self._updates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._on_update = on_update

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._updates[session_id] = []

    def append(self, session_id: str, update: Any) -> None:
        payload = _to_plain_data(update)
        with self._lock:
            self._updates[session_id].append(payload)
        if self._on_update is not None:
            self._on_update(session_id, payload)

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
        self._profile_spec = _resolve_profile_spec(agent)
        self._loop_thread: _LoopThread | None = None
        self._collector = _SessionUpdateCollector(on_update=self._handle_session_update)
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
        mcp_servers: list[dict[str, Any]] | list[Any] | None,
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
                mcp_servers=mcp_servers,
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

        # Create enabled capabilities
        fs_caps = sdk.file_system_capability(read_text_file=True, write_text_file=True)
        client_caps = sdk.client_capabilities(fs=fs_caps, terminal=True)

        initialize_response = await self._conn.initialize(
            protocol_version=sdk.protocol_version,
            protocolVersion=sdk.protocol_version,
            client_capabilities=client_caps,
            clientCapabilities=client_caps,
            client_info=sdk.implementation(name="orgemage", title="OrgeMage", version=__version__),
            clientInfo=sdk.implementation(name="orgemage", title="OrgeMage", version=__version__),
        )
        payload = _to_plain_data(initialize_response)
        profile = _profile_metadata(self.agent, self._profile_spec)
        diagnostics = _profile_initialize_diagnostics(self.agent, self._profile_spec, payload)
        self.negotiated_state = DownstreamNegotiatedState(
            agent_id=self.agent.agent_id,
            agent_info=payload.get("agent_info") or payload.get("agentInfo") or {},
            agent_capabilities=_normalize_agent_capabilities(payload.get("agent_capabilities") or payload.get("agentCapabilities") or {}),
            auth_methods=_normalize_auth_methods(payload.get("auth_methods") or payload.get("authMethods") or []),
            protocol_version=payload.get("protocol_version") or payload.get("protocolVersion"),
            profile=profile,
        )
        for diagnostic in diagnostics:
            self.negotiated_state.add_diagnostic(
                kind=diagnostic["kind"],
                message=diagnostic["message"],
                metadata=diagnostic.get("metadata"),
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
        mcp_servers: list[dict[str, Any]] | list[Any] | None,
        task: PlanTask,
        coordinator_prompt: str,
        selected_model: str,
    ) -> DownstreamPromptResult:
        if self._conn is None or self._sdk is None:
            raise DownstreamConnectorError(f"Downstream agent '{self.agent.agent_id}' is not connected")

        debug_event(_LOG, "connector.lifecycle.execute.start", agent_id=self.agent.agent_id, orchestrator_session_id=orchestrator_session_id, task_id=task.task_id, downstream_session_id=downstream_session_id)
        session_id, session_payload = await self._ensure_session_async(
            cwd,
            downstream_session_id,
            mcp_servers=mcp_servers,
        )
        self._record_session_state(session_id, session_payload)
        await self._apply_model_selection_async(
            session_id,
            selected_model,
            cwd=cwd,
            mcp_servers=mcp_servers,
        )
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
            # SDK signature: prompt(self, prompt: list[...], session_id: str, **kwargs: Any)
            response = await self._conn.prompt(
                [self._sdk.text_block(prompt_text)],
                session_id,
                sessionId=session_id
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

    async def _ensure_session_async(
        self,
        cwd: str,
        downstream_session_id: str | None,
        *,
        mcp_servers: list[dict[str, Any]] | list[Any] | None,
    ) -> tuple[str, Any]:
        normalized_mcp_servers = _normalize_mcp_servers(mcp_servers)
        self._record_mcp_diagnostic(normalized_mcp_servers)
        
        # For Gemini, we prefer a fresh session if the previous one is not found on disk,
        # as Gemini CLI state is often volatile across process restarts.
        if downstream_session_id and self.negotiated_state and self.negotiated_state.load_session_supported:
            try:
                loaded = await self._conn.load_session(
                    cwd=cwd,
                    mcp_servers=normalized_mcp_servers,
                    mcpServers=normalized_mcp_servers,
                    session_id=downstream_session_id,
                    sessionId=downstream_session_id,
                )
                if loaded is not None:
                    return downstream_session_id, loaded
            except Exception as exc:
                self._record_load_session_fallback(
                    downstream_session_id=downstream_session_id,
                    error=exc,
                    phase="ensure_session",
                )
        
        created = await self._conn.new_session(
            cwd=cwd, 
            mcp_servers=normalized_mcp_servers,
            mcpServers=normalized_mcp_servers,
        )
        
        payload = _to_plain_data(created)
        created_session_id = payload.get("sessionId") or payload.get("session_id")
        
        if not created_session_id:
            raise DownstreamConnectorError(
                f"Downstream agent '{self.agent.agent_id}' returned no session id from session/new"
            )
            
        # Critical stabilization for Gemini
        if self.agent.agent_id == "gemini":
            # Force persistence by sending a dummy message that does nothing but triggers a save
            try:
                debug_event(_LOG, "connector.lifecycle.gemini_warmup.start", session_id=created_session_id)
                await self._conn.prompt(
                    [self._sdk.text_block("warmup")],
                    str(created_session_id),
                    sessionId=str(created_session_id)
                )
                await asyncio.sleep(1.0) # Give it time to actually write the file
            except Exception as e:
                debug_event(_LOG, "connector.lifecycle.gemini_warmup.failed", error=str(e))
            
        return str(created_session_id), created

    async def _apply_model_selection_async(
        self,
        session_id: str,
        selected_model: str,
        *,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | list[Any] | None,
    ) -> None:
        if self._conn is None:
            return

        raw_model = self.agent.resolve_model(selected_model) or self.agent.default_model
        if raw_model is None:
            return
        config_options: list[dict[str, Any]] = []
        models: dict[str, Any] = {}
        if self.negotiated_state is not None:
            config_options = self.negotiated_state.config_options.get(session_id, [])
            models = self.negotiated_state.models.get(session_id, {})
        resolved_model = self._resolve_downstream_model_value(
            raw_model,
            config_options=config_options,
            models=models,
        )
        current_model = self._current_downstream_model_value(
            config_options=config_options,
            models=models,
        )
        if current_model and self._model_values_equivalent(current_model, resolved_model):
            pass # Already on right model, but might still need mode switch
        else:
            if any(option.get("id") == "model" for option in config_options):
                await self._set_config_option_async(session_id=session_id, option_id="model", value=resolved_model)
            elif models.get("availableModels"):
                await self._set_session_model_async(session_id=session_id, model_id=resolved_model)
        
        # Try to enable YOLO or Auto-Edit mode to bypass permission issues
        current_modes = self.negotiated_state.modes.get(session_id, {}) if self.negotiated_state else {}
        available_modes = [m.get("id") or m.get("value") for m in current_modes.get("availableModes") or current_modes.get("available_modes") or []]
        for mode_to_try in ["yolo", "auto-edit", "autoEdit"]:
            if mode_to_try in available_modes:
                debug_event(_LOG, "connector.lifecycle.set_mode.auto", agent_id=self.agent.agent_id, mode=mode_to_try)
                await self._set_session_mode_async(session_id=session_id, mode_id=mode_to_try)
                break

        await self._refresh_catalog_async(
            force=True,
            cwd=cwd,
            downstream_session_id=session_id,
            mcp_servers=mcp_servers,
        )

    def _resolve_downstream_model_value(
        self,
        requested_model: str,
        *,
        config_options: list[dict[str, Any]],
        models: dict[str, Any],
    ) -> str:
        candidates = self._candidate_model_values(config_options=config_options, models=models)
        if not candidates:
            return requested_model
        exact_matches = [candidate for candidate in candidates if candidate == requested_model]
        if exact_matches:
            return exact_matches[0]
        alias_matches = [candidate for candidate in candidates if requested_model in _model_aliases(candidate)]
        if len(alias_matches) == 1:
            return alias_matches[0]
        if alias_matches:
            current_value = self._current_downstream_model_value(config_options=config_options, models=models)
            if current_value:
                current_auth_scope = _auth_scope(current_value)
                for candidate in alias_matches:
                    if _auth_scope(candidate) == current_auth_scope:
                        return candidate
            return alias_matches[0]
        if len(candidates) == 1:
            sole_candidate = candidates[0]
            if self._candidate_requires_opaque_mapping(sole_candidate):
                return sole_candidate
        return requested_model

    def _candidate_requires_opaque_mapping(self, candidate: str) -> bool:
        if _auth_scope(candidate):
            return True
        return "(" in candidate and ")" in candidate

    def _candidate_model_values(
        self,
        *,
        config_options: list[dict[str, Any]],
        models: dict[str, Any],
    ) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for option in config_options:
            if option.get("id") != "model":
                continue
            for item in option.get("options") or []:
                if not isinstance(item, dict):
                    continue
                value = str(item.get("value") or "").strip()
                if value and value not in seen:
                    values.append(value)
                    seen.add(value)
        for item in models.get("availableModels") or []:
            if not isinstance(item, dict):
                continue
            value = str(item.get("modelId") or item.get("value") or "").strip()
            if value and value not in seen:
                values.append(value)
                seen.add(value)
        return values

    def _current_downstream_model_value(
        self,
        *,
        config_options: list[dict[str, Any]],
        models: dict[str, Any],
    ) -> str | None:
        for option in config_options:
            if option.get("id") != "model":
                continue
            value = option.get("currentValue") or option.get("current_value")
            if isinstance(value, str) and value:
                return value
        value = models.get("currentModelId") or models.get("current_model_id")
        if isinstance(value, str) and value:
            return value
        return None

    def _should_fallback_from_load_session_error(self, error: Exception) -> bool:
        code = str(getattr(error, "code", "") or "").lower()
        method = str(getattr(error, "method", "") or "").lower()
        message = str(getattr(error, "message", "") or str(error)).lower()
        if "load" not in method and "session/load" not in method and "session" not in message:
            return False
        fallback_markers = (
            "no previous sessions found",
            "previous sessions found for this project",
            "session not found",
            "unknown session",
            "resume",
        )
        if any(marker in message for marker in fallback_markers):
            return True
        return code in {"method_not_found", "invalid_params", "internal_error"}

    def _record_load_session_fallback(
        self,
        *,
        downstream_session_id: str,
        error: Exception,
        phase: str,
    ) -> None:
        if self.negotiated_state is None:
            return
        self.negotiated_state.add_diagnostic(
            kind="load_session_fallback",
            message=(
                f"Downstream agent '{self.agent.agent_id}' failed to resume session "
                f"{downstream_session_id!r}; falling back to session/new."
            ),
            metadata={
                "downstream_session_id": downstream_session_id,
                "phase": phase,
                "error_code": getattr(error, "code", None),
                "error_method": getattr(error, "method", None),
                "error_message": getattr(error, "message", str(error)),
            },
        )

    def _model_values_equivalent(self, left: str, right: str) -> bool:
        if left == right:
            return True
        left_aliases = _model_aliases(left)
        right_aliases = _model_aliases(right)
        return bool(left_aliases.intersection(right_aliases))

    async def _set_config_option_async(self, *, session_id: str, option_id: str, value: str) -> Any:
        if self._conn is None:
            return None
        setter = getattr(self._conn, "set_config_option", None)
        if setter is None:
            return None
        try:
            parameters = inspect.signature(setter).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "config_id" in parameters:
            return await setter(config_id=option_id, session_id=session_id, value=value, configId=option_id, sessionId=session_id)
        if "option_id" in parameters:
            return await setter(session_id=session_id, option_id=option_id, value=value, sessionId=session_id, optionId=option_id)
        try:
            return await setter(config_id=option_id, session_id=session_id, value=value, configId=option_id, sessionId=session_id)
        except TypeError:
            try:
                return await setter(session_id=session_id, option_id=option_id, value=value, sessionId=session_id, optionId=option_id)
            except TypeError:
                return await setter(option_id, session_id, value)

    async def _set_session_mode_async(self, *, session_id: str, mode_id: str) -> Any:
        if self._conn is None:
            return None
        candidate_names = (
            "set_session_mode",
            "unstable_set_session_mode",
            "set_mode",
            "unstable_set_mode",
        )
        setter = None
        for name in candidate_names:
            setter = getattr(self._conn, name, None)
            if setter is not None:
                break
        if setter is None:
            return None
        try:
            parameters = inspect.signature(setter).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "mode_id" in parameters:
            return await setter(session_id=session_id, mode_id=mode_id, sessionId=session_id, modeId=mode_id)
        try:
            return await setter(session_id=session_id, modeId=mode_id, sessionId=session_id)
        except TypeError:
            try:
                return await setter(mode_id=mode_id, session_id=session_id, modeId=mode_id, sessionId=session_id)
            except TypeError:
                return await setter(session_id, mode_id)

    async def _set_session_model_async(self, *, session_id: str, model_id: str) -> Any:
        if self._conn is None:
            return None
        candidate_names = (
            "set_session_model",
            "unstable_set_session_model",
            "set_model",
            "unstable_set_model",
        )
        setter = None
        for name in candidate_names:
            setter = getattr(self._conn, name, None)
            if setter is not None:
                break
        if setter is None:
            if self.negotiated_state is not None:
                self.negotiated_state.add_diagnostic(
                    kind="model_selection_unsupported",
                    message=f"Downstream agent '{self.agent.agent_id}' advertised models but exposed no model-selection RPC.",
                    metadata={"session_id": session_id, "model_id": model_id},
                )
            return None
        try:
            parameters = inspect.signature(setter).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "model_id" in parameters:
            return await setter(session_id=session_id, model_id=model_id, sessionId=session_id, modelId=model_id)
        try:
            return await setter(session_id=session_id, modelId=model_id, sessionId=session_id)
        except TypeError:
            try:
                return await setter(model_id=model_id, session_id=session_id, modelId=model_id, sessionId=session_id)
            except TypeError:
                return await setter(session_id, model_id)

    def _record_session_state(self, session_id: str, session_payload: Any) -> None:
        if self.negotiated_state is None:
            return
        payload = _to_plain_data(session_payload)
        config_options = _normalize_config_options(payload.get("config_options") or payload.get("configOptions") or [])
        models = _normalize_models(payload)
        modes = _normalize_modes(payload)
        available_commands = _normalize_available_commands(payload)
        session_capabilities = {
            key: value
            for key, value in payload.items()
            if key not in {"config_options", "configOptions", "session_id", "sessionId"}
        }
        self.negotiated_state.record_session(
            session_id=session_id,
            capabilities=session_capabilities,
            config_options=config_options,
            models=models,
            modes=modes,
            available_commands=available_commands,
        )
        diagnostic_payload = dict(payload)
        if "models" not in diagnostic_payload and session_id in self.negotiated_state.models:
            diagnostic_payload["models"] = self.negotiated_state.models[session_id]
        if "modes" not in diagnostic_payload and session_id in self.negotiated_state.modes:
            diagnostic_payload["modes"] = self.negotiated_state.modes[session_id]
        if "configOptions" not in diagnostic_payload and session_id in self.negotiated_state.config_options:
            diagnostic_payload["configOptions"] = self.negotiated_state.config_options[session_id]
        if session_id in self.negotiated_state.available_commands:
            diagnostic_payload["available_commands_update"] = self.negotiated_state.available_commands[session_id]
        for diagnostic in _profile_session_diagnostics(self.agent, self._profile_spec, diagnostic_payload):
            self.negotiated_state.add_diagnostic(
                kind=diagnostic["kind"],
                message=diagnostic["message"],
                metadata=diagnostic.get("metadata"),
            )

    async def _refresh_catalog_async(
        self,
        *,
        force: bool = False,
        cwd: str | None = None,
        downstream_session_id: str | None = None,
        mcp_servers: list[dict[str, Any]] | list[Any] | None = None,
    ) -> dict[str, Any]:
        if self._conn is None:
            raise DownstreamConnectorError(f"Downstream agent '{self.agent.agent_id}' is not connected")
        if not force and self._catalog_state is not None and not self._catalog_refresh_required:
            return dict(self._catalog_state)
        normalized_cwd = cwd or "."
        normalized_mcp_servers = _normalize_mcp_servers(mcp_servers)
        self._record_mcp_diagnostic(normalized_mcp_servers)
        if downstream_session_id and self.negotiated_state and self.negotiated_state.load_session_supported:
            try:
                session_payload = await self._conn.load_session(
                    cwd=normalized_cwd,
                    mcp_servers=normalized_mcp_servers,
                    mcpServers=normalized_mcp_servers,
                    session_id=downstream_session_id,
                    sessionId=downstream_session_id,
                )
            except Exception as exc:
                if not self._should_fallback_from_load_session_error(exc):
                    raise
                self._record_load_session_fallback(
                    downstream_session_id=downstream_session_id,
                    error=exc,
                    phase="refresh_catalog",
                )
                session_payload = await self._conn.new_session(
                    cwd=normalized_cwd, 
                    mcp_servers=normalized_mcp_servers,
                    mcpServers=normalized_mcp_servers,
                )
            else:
                if session_payload is None:
                    session_payload = await self._conn.new_session(
                        cwd=normalized_cwd, 
                        mcp_servers=normalized_mcp_servers,
                        mcpServers=normalized_mcp_servers,
                    )
        else:
            session_payload = await self._conn.new_session(
                cwd=normalized_cwd, 
                mcp_servers=normalized_mcp_servers,
                mcpServers=normalized_mcp_servers,
            )
        payload = _to_plain_data(session_payload)
        session_id = str(payload.get("session_id") or payload.get("sessionId") or downstream_session_id or "")
        if session_id:
            self._record_session_state(session_id, session_payload)
        config_options = _normalize_config_options(payload.get("config_options") or payload.get("configOptions") or [])
        catalog_config_options = config_options or _synthesize_model_config_options(_normalize_models(payload))
        capabilities = {
            key: value
            for key, value in payload.items()
            if key not in {"config_options", "configOptions", "session_id", "sessionId"}
        }
        if self.negotiated_state is not None:
            capabilities = {**self.negotiated_state.agent_capabilities, **capabilities}
        command_advertisements = _extract_command_advertisements(capabilities)
        if self.negotiated_state is not None and session_id:
            command_advertisements = _command_names(self.negotiated_state.available_commands.get(session_id, [])) or command_advertisements
        self._catalog_state = {
            "agent_id": self.agent.agent_id,
            "config_options": catalog_config_options,
            "capabilities": capabilities,
            "command_advertisements": command_advertisements,
            "profile": dict(self.negotiated_state.profile) if self.negotiated_state is not None else _profile_metadata(self.agent, self._profile_spec),
            "diagnostics": list(self.negotiated_state.diagnostics) if self.negotiated_state is not None else [],
            "refreshed_at": time.time(),
        }
        self._catalog_refresh_required = False
        return dict(self._catalog_state)

    def _record_mcp_diagnostic(self, mcp_servers: list[dict[str, Any]]) -> None:
        if self.negotiated_state is None or not mcp_servers:
            return
        if self._profile_spec is not None:
            supported = set(self._profile_spec.supported_mcp_transports)
            for server in mcp_servers:
                transport = _infer_mcp_transport(server)
                if transport is None:
                    continue
                if supported and transport not in supported:
                    self.negotiated_state.add_diagnostic(
                        kind="mcp_transport_unsupported",
                        message=(
                            f"Downstream agent '{self.agent.agent_id}' profile {self._profile_spec.profile_id!r} "
                            f"does not advertise MCP transport {transport!r}."
                        ),
                        metadata={"mcp_server": dict(server), "transport": transport, "supported_transports": sorted(supported)},
                    )
        capabilities = self.negotiated_state.agent_capabilities
        supports_mcp = capabilities.get("supportsMcp")
        if supports_mcp is None:
            supports_mcp = capabilities.get("supports_mcp")
        if supports_mcp is None:
            supports_mcp = capabilities.get("mcp")
        if supports_mcp is None:
            mcp_capabilities = capabilities.get("mcpCapabilities") or capabilities.get("mcp_capabilities")
            if isinstance(mcp_capabilities, dict):
                supports_mcp = any(bool(value) for value in mcp_capabilities.values())
        if supports_mcp is False:
            self.negotiated_state.add_diagnostic(
                kind="mcp_capability_mismatch",
                message=f"Downstream agent '{self.agent.agent_id}' received MCP servers but advertised no MCP support.",
                metadata={"mcp_servers": list(mcp_servers)},
            )

    def _handle_session_update(self, session_id: str, update: dict[str, Any]) -> None:
        if self.negotiated_state is None:
            return
        session_update = update.get("sessionUpdate") or update.get("session_update")
        if session_update == "available_commands_update":
            self.negotiated_state.record_available_commands(session_id, _normalize_available_commands(update))
            return
        if session_update == "current_mode_update":
            current_mode_id = update.get("currentModeId") or update.get("current_mode_id")
            if isinstance(current_mode_id, str) and current_mode_id:
                self.negotiated_state.update_current_mode(session_id, current_mode_id)
            return
        if session_update == "config_option_update":
            config_options = update.get("configOptions") or update.get("config_options") or []
            if isinstance(config_options, list):
                self.negotiated_state.update_config_options(session_id, _normalize_config_options(config_options))

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
            file_system_capability=_acp_attr(acp, "FileSystemCapability"),
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

        async def list_directory(self, **kwargs: Any) -> Any:
            return await callback_layer.list_directory(**kwargs)

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

        async def run_shell_command(self, **kwargs: Any) -> Any:
            return await callback_layer.run_shell_command(**kwargs)

        async def kill_terminal(self, **kwargs: Any) -> Any:
            return await callback_layer.kill_terminal(**kwargs)

        async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
            collector.append(session_id, update)

        async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            # Use callback_layer.agent since agent is not in scope
            debug_event(_LOG, "connector.callback.ext_method", agent_id=callback_layer.agent.agent_id, method=method, params=params)
            raise request_error.method_not_found(method)

        async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
            raise request_error.method_not_found(method)

    return RuntimeClient()


def _extract_summary(response: dict[str, Any], updates: list[dict[str, Any]]) -> str:
    summary = _extract_message_text(response)
    if summary:
        return summary
    streamed_summary = _collect_streamed_message_text(updates)
    if streamed_summary:
        return streamed_summary
    for update in reversed(updates):
        if _extract_update_kind(update) == "agent_thought_chunk":
            continue
        summary = _extract_message_text(update)
        if summary:
            return summary
    return ""


def _extract_stop_reason(response: dict[str, Any]) -> str | None:
    stop_reason = response.get("stop_reason") or response.get("stopReason")
    if isinstance(stop_reason, str) and stop_reason:
        return stop_reason
    return None


def _extract_update_kind(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    kind = payload.get("session_update") or payload.get("sessionUpdate")
    return kind if isinstance(kind, str) else ""


def _collect_streamed_message_text(updates: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for update in updates:
        if _extract_update_kind(update) != "agent_message_chunk":
            continue
        text = _extract_message_text(update)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def _extract_message_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str) and text:
            return text
        for key in ("content", "message"):
            extracted = _extract_message_text(payload.get(key))
            if extracted:
                return extracted
        return ""
    if isinstance(payload, list):
        parts = [_extract_message_text(item) for item in payload]
        return "".join(part for part in parts if part)
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


def _normalize_mcp_servers(value: list[dict[str, Any]] | list[Any] | Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        plain_item = _to_plain_data(item)
        if isinstance(plain_item, dict):
            normalized.append({str(key): item_value for key, item_value in plain_item.items()})
        else:
            normalized.append({"id": f"mcp-{index}", "value": plain_item})
    return normalized


def _resolve_profile_spec(agent: DownstreamAgentConfig) -> _DownstreamProfileSpec | None:
    command_name = Path(agent.command).name
    args = set(agent.args)
    if command_name == "codex-acp":
        return _DOWNSTREAM_PROFILE_SPECS["codex-acp"]
    if agent.agent_id == "gemini" or command_name == "gemini":
        return _DOWNSTREAM_PROFILE_SPECS["gemini"]
    if agent.agent_id == "qwen" or command_name == "qwen":
        return _DOWNSTREAM_PROFILE_SPECS["qwen"]
    if command_name == "codex" and "app-server" in args:
        return None
    return None


def _profile_metadata(agent: DownstreamAgentConfig, spec: _DownstreamProfileSpec | None) -> dict[str, Any]:
    if spec is None:
        return {
            "id": "generic-acp",
            "title": "Generic ACP",
            "command": Path(agent.command).name,
            "args": list(agent.args),
            "known_quirks": [],
        }
    return {
        "id": spec.profile_id,
        "title": spec.title,
        "command": spec.command,
        "args": list(agent.args),
        "supported_mcp_transports": list(spec.supported_mcp_transports),
        "known_quirks": list(spec.known_quirks),
    }


def _profile_initialize_diagnostics(
    agent: DownstreamAgentConfig,
    spec: _DownstreamProfileSpec | None,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    if spec is None:
        return diagnostics
    command_name = Path(agent.command).name
    if command_name != spec.command:
        diagnostics.append(
            {
                "kind": "profile_command_mismatch",
                "message": (
                    f"Downstream agent '{agent.agent_id}' was mapped to profile {spec.profile_id!r} "
                    f"but uses command {command_name!r}."
                ),
                "metadata": {"expected_command": spec.command, "actual_command": command_name},
            }
        )
    missing_args = [arg for arg in spec.required_args if arg not in agent.args]
    if missing_args:
        diagnostics.append(
            {
                "kind": "profile_arg_mismatch",
                "message": (
                    f"Downstream agent '{agent.agent_id}' is missing required ACP profile args for {spec.profile_id!r}."
                ),
                "metadata": {"required_args": list(spec.required_args), "actual_args": list(agent.args), "missing_args": missing_args},
            }
        )
    for field in spec.required_initialize_fields:
        if _dotted_get(payload, field) is None:
            diagnostics.append(
                {
                    "kind": "initialize_field_missing",
                    "message": f"Downstream agent '{agent.agent_id}' did not return expected initialize field {field!r}.",
                    "metadata": {"profile": spec.profile_id, "field": field},
                }
            )
    return diagnostics


def _profile_session_diagnostics(
    agent: DownstreamAgentConfig,
    spec: _DownstreamProfileSpec | None,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    if spec is None:
        return []
    diagnostics: list[dict[str, Any]] = []
    for field in spec.required_session_fields:
        if _dotted_get(payload, field) is None:
            diagnostics.append(
                {
                    "kind": "session_field_missing",
                    "message": f"Downstream agent '{agent.agent_id}' did not return expected session field {field!r}.",
                    "metadata": {"profile": spec.profile_id, "field": field},
                }
            )
    return diagnostics


def _dotted_get(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for segment in path.split("."):
        if not isinstance(current, dict):
            return None
        current = _get_with_aliases(current, segment)
    return current


def _get_with_aliases(payload: dict[str, Any], segment: str) -> Any:
    for key in _segment_aliases(segment):
        if key in payload:
            return payload[key]
    return None


def _segment_aliases(segment: str) -> list[str]:
    aliases = [segment]
    snake = _camel_to_snake(segment)
    camel = _snake_to_camel(segment)
    for candidate in (snake, camel):
        if candidate not in aliases:
            aliases.append(candidate)
    return aliases


def _camel_to_snake(value: str) -> str:
    result: list[str] = []
    for index, char in enumerate(value):
        if char.isupper() and index > 0 and value[index - 1] != "_":
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _snake_to_camel(value: str) -> str:
    parts = value.split("_")
    if not parts:
        return value
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _normalize_agent_capabilities(value: Any) -> dict[str, Any]:
    payload = _to_plain_data(value)
    if isinstance(payload, dict):
        return {str(key): item for key, item in payload.items()}
    return {}


def _normalize_auth_methods(value: Any) -> list[Any]:
    payload = _to_plain_data(value)
    if not isinstance(payload, list):
        return []
    result: list[Any] = []
    for item in payload:
        if isinstance(item, dict):
            result.append({str(key): val for key, val in item.items()})
        elif isinstance(item, str) and item:
            result.append(item)
    return result


def _normalize_models(payload: dict[str, Any]) -> dict[str, Any]:
    models = payload.get("models")
    if isinstance(models, dict):
        normalized = {str(key): value for key, value in models.items()}
        available = normalized.get("availableModels") or normalized.get("available_models")
        if isinstance(available, list):
            normalized["availableModels"] = [
                _normalize_model_item(item) for item in available
            ]
        return normalized
    available_models = payload.get("availableModels") or payload.get("available_models")
    current_model_id = payload.get("currentModelId") or payload.get("current_model_id")
    if isinstance(available_models, list) or isinstance(current_model_id, str):
        normalized: dict[str, Any] = {}
        if isinstance(available_models, list):
            normalized["availableModels"] = [
                _normalize_model_item(item) for item in available_models
            ]
        if isinstance(current_model_id, str):
            normalized["currentModelId"] = current_model_id
        return normalized
    return {}


def _normalize_model_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"modelId": str(item)}
    normalized = {str(key): value for key, value in item.items()}
    model_id = normalized.get("modelId") or normalized.get("model_id") or normalized.get("value")
    if model_id:
        normalized.setdefault("modelId", str(model_id))
    return normalized


def _normalize_modes(payload: dict[str, Any]) -> dict[str, Any]:
    modes = payload.get("modes")
    if isinstance(modes, dict):
        normalized = {str(key): value for key, value in modes.items()}
        available = normalized.get("availableModes") or normalized.get("available_modes")
        if isinstance(available, list):
            normalized["availableModes"] = [
                _normalize_mode_item(item) for item in available
            ]
        return normalized
    if isinstance(modes, list):
        return {
            "availableModes": [
                _normalize_mode_item(item) for item in modes
            ]
        }
    available_modes = payload.get("availableModes") or payload.get("available_modes")
    current_mode_id = payload.get("currentModeId") or payload.get("current_mode_id")
    if isinstance(available_modes, list) or isinstance(current_mode_id, str):
        normalized: dict[str, Any] = {}
        if isinstance(available_modes, list):
            normalized["availableModes"] = [
                _normalize_mode_item(item) for item in available_modes
            ]
        if isinstance(current_mode_id, str):
            normalized["currentModeId"] = current_mode_id
        return normalized
    return {}


def _normalize_mode_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"id": str(item), "name": str(item)}
    normalized = {str(key): value for key, value in item.items()}
    mode_id = normalized.get("id") or normalized.get("mode_id") or normalized.get("modeId")
    if mode_id:
        normalized.setdefault("id", str(mode_id))
    return normalized


def _normalize_config_options(value: Any) -> list[dict[str, Any]]:
    payload = _to_plain_data(value)
    if not isinstance(payload, list):
        return []
    result: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            normalized = {str(key): val for key, val in item.items()}
            options = normalized.get("options")
            if isinstance(options, list):
                normalized["options"] = [
                    {str(key): value for key, value in option.items()} if isinstance(option, dict) else {"value": str(option), "name": str(option)}
                    for option in options
                ]
            result.append(normalized)
    return result


def _normalize_available_commands(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = (
        payload.get("availableCommands")
        or payload.get("available_commands")
        or payload.get("commands")
        or payload.get("commandAdvertisements")
        or payload.get("command_advertisements")
    )
    if not isinstance(candidates, list):
        return []
    result: list[dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, dict):
            name = item.get("name") or item.get("command") or item.get("id")
            if isinstance(name, str) and name:
                normalized = {str(key): value for key, value in item.items()}
                normalized.setdefault("name", name)
                result.append(normalized)
        elif isinstance(item, str) and item:
            result.append({"name": item})
    return result


def _synthesize_model_config_options(models: dict[str, Any]) -> list[dict[str, Any]]:
    available_models = models.get("availableModels")
    if not isinstance(available_models, list) or not available_models:
        return []
    return [
        {
            "id": "model",
            "name": "Model",
            "description": "Synthesized from ACP models payload",
            "category": "model",
            "type": "select",
            "currentValue": models.get("currentModelId"),
            "options": [
                {
                    "value": model.get("modelId") or model.get("value") or "",
                    "name": model.get("name") or model.get("modelId") or model.get("value") or "",
                    "description": model.get("description") or "",
                }
                for model in available_models
                if isinstance(model, dict) and (model.get("modelId") or model.get("value"))
            ],
        }
    ]


def _command_names(commands: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for command in commands:
        name = command.get("name")
        if isinstance(name, str) and name and name not in seen:
            result.append(name)
            seen.add(name)
    return result


def _model_aliases(value: str) -> set[str]:
    aliases = {value}
    for separator in ("/", "::"):
        if separator in value:
            _, _, suffix = value.rpartition(separator)
            if suffix:
                aliases.add(suffix)
    return aliases


def _auth_scope(value: str) -> str | None:
    for separator in ("/", "::"):
        if separator in value:
            prefix, _, _ = value.partition(separator)
            if prefix:
                return prefix
    return None


def _infer_mcp_transport(server: dict[str, Any]) -> str | None:
    transport = server.get("transport")
    if isinstance(transport, dict):
        transport_type = transport.get("type")
        if isinstance(transport_type, str) and transport_type:
            return transport_type
    server_type = server.get("type")
    if isinstance(server_type, str) and server_type:
        return server_type
    if isinstance(server.get("command"), str):
        return "stdio"
    url = server.get("url")
    if isinstance(url, str):
        if url.startswith("http://") or url.startswith("https://"):
            return "http"
        if url.startswith("sse://"):
            return "sse"
    return None


def _extract_command_advertisements(capabilities: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("commands", "command_ads", "commandAdvertisements", "advertised_commands", "advertisedCommands"):
        value = capabilities.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("command") or item.get("id")
                    if isinstance(name, str) and name:
                        candidates.append(name)
                elif item:
                    candidates.append(str(item))
    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
