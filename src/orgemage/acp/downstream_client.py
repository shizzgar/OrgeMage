from __future__ import annotations

import asyncio
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
import importlib
import threading
from typing import Any, Callable, Protocol

from ..models import DownstreamAgentConfig, DownstreamNegotiatedState, PlanTask, TaskStatus


class DownstreamConnectorError(RuntimeError):
    """Raised when a downstream ACP connection cannot be established or used."""


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
        with self._lock:
            self._updates[session_id].append(_to_plain_data(update))

    def get(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._updates.get(session_id, []))


class AcpDownstreamConnector:
    def __init__(self, agent: DownstreamAgentConfig) -> None:
        self.agent = agent
        self.negotiated_state: DownstreamNegotiatedState | None = None
        self._loop_thread: _LoopThread | None = None
        self._collector = _SessionUpdateCollector()
        self._context_manager: Any = None
        self._conn: Any = None
        self._process: Any = None
        self._sdk: _SdkBindings | None = None
        self._lock = threading.Lock()

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
        sdk = _load_sdk()
        self._sdk = sdk
        client_impl = _build_runtime_client(sdk, self._collector)
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

    async def _close_async(self) -> None:
        if self._context_manager is not None:
            await self._context_manager.__aexit__(None, None, None)
        self._context_manager = None
        self._conn = None
        self._process = None
        self._sdk = None

    async def _cancel_async(self, downstream_session_id: str) -> None:
        if self._conn is None:
            return
        await self._conn.cancel(session_id=downstream_session_id)

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

        session_id, session_payload = await self._ensure_session_async(cwd, downstream_session_id)
        self._record_session_state(session_id, session_payload)
        await self._apply_model_selection_async(session_id, selected_model)

        prompt_text = self._build_prompt(
            orchestrator_session_id=orchestrator_session_id,
            task=task,
            coordinator_prompt=coordinator_prompt,
            selected_model=selected_model,
        )
        self._collector.reset(session_id)
        response = await self._conn.prompt(
            session_id=session_id,
            prompt=[self._sdk.text_block(prompt_text)],
        )
        updates = self._collector.get(session_id)
        summary = _extract_summary(_to_plain_data(response), updates)
        if not summary:
            summary = f"Downstream agent {self.agent.name} completed {task.title!r}."
        return DownstreamPromptResult(
            downstream_session_id=session_id,
            status=TaskStatus.COMPLETED,
            summary=summary,
            raw_output=summary,
            updates=updates,
            response=_to_plain_data(response),
        )

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

    async def _apply_model_selection_async(self, session_id: str, selected_model: str) -> None:
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
        return _SdkBindings(
            module=acp,
            protocol_version=acp.PROTOCOL_VERSION,
            client_base=acp.Client,
            request_error=acp.RequestError,
            client_capabilities=acp.ClientCapabilities,
            implementation=acp.Implementation,
            spawn_agent_process=acp.spawn_agent_process,
            text_block=acp.text_block,
        )
    except AttributeError as exc:  # pragma: no cover - depends on external runtime
        raise DownstreamConnectorError(f"Installed ACP SDK is missing required API surface: {exc}") from exc


def _build_runtime_client(sdk: _SdkBindings, collector: _SessionUpdateCollector) -> Any:
    request_error = sdk.request_error

    class RuntimeClient(sdk.client_base):
        async def request_permission(self, **kwargs: Any) -> Any:
            raise request_error.method_not_found("session/request_permission")

        async def write_text_file(self, **kwargs: Any) -> Any:
            raise request_error.method_not_found("fs/write_text_file")

        async def read_text_file(self, **kwargs: Any) -> Any:
            raise request_error.method_not_found("fs/read_text_file")

        async def create_terminal(self, **kwargs: Any) -> Any:
            raise request_error.method_not_found("terminal/create")

        async def terminal_output(self, **kwargs: Any) -> Any:
            raise request_error.method_not_found("terminal/output")

        async def release_terminal(self, **kwargs: Any) -> Any:
            raise request_error.method_not_found("terminal/release")

        async def wait_for_terminal_exit(self, **kwargs: Any) -> Any:
            raise request_error.method_not_found("terminal/wait_for_exit")

        async def kill_terminal(self, **kwargs: Any) -> Any:
            raise request_error.method_not_found("terminal/kill")

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
