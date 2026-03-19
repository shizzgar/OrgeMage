from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import importlib.util
import logging
from typing import Any

from ..metadata import extract_prompt_metadata
from ..orchestrator import Orchestrator


_LOG = logging.getLogger(__name__)


class AcpSdkUnavailableError(RuntimeError):
    pass


def _acp_attr(acp: Any, name: str) -> Any:
    value = getattr(acp, name, None)
    if value is not None:
        return value
    schema = getattr(acp, "schema", None)
    if schema is not None:
        return getattr(schema, name, None)
    return None


class AcpSdkBridge:
    """Lazy ACP SDK bridge.

    The repository can be tested without the external SDK installed, while production
    deployments can enable the official ACP Python SDK through the optional `acp` extra.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator

    @staticmethod
    def sdk_available() -> bool:
        return importlib.util.find_spec("acp") is not None

    def create_runtime(self) -> "AcpAgentRuntime":
        if not self.sdk_available():
            raise AcpSdkUnavailableError(
                "The optional dependency 'agent-client-protocol' is not installed. "
                "Install with `pip install orgemage[acp]`."
            )
        import acp  # type: ignore[import-not-found]

        return AcpAgentRuntime(acp=acp, orchestrator=self.orchestrator)

    def create_agent(self) -> Any:
        return self.create_runtime().agent


class AcpAgentRuntime:
    def __init__(self, *, acp: Any, orchestrator: Orchestrator) -> None:
        self.acp = acp
        self.orchestrator = orchestrator
        self.client_connection: Any = None
        self.client_capabilities: Any = None
        self._prompt_tasks: dict[str, asyncio.Task[Any]] = {}
        self._tool_call_seen: dict[str, set[str]] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self.agent = _build_agent(acp, self)

    def bind_client_connection(self, connection: Any) -> None:
        self.client_connection = connection
        self.orchestrator.connector_manager.bind_upstream_client_connection(
            connection,
            capabilities=getattr(self, "client_capabilities", None),
        )

    async def initialize(self, protocol_version: int, **kwargs: Any) -> Any:
        client = kwargs.get("client") or kwargs.get("client_connection") or kwargs.get("conn")
        self.client_capabilities = kwargs.get("client_capabilities")
        if client is not None:
            self.bind_client_connection(client)
        return _build_initialize_response(self.acp, protocol_version)

    async def new_session(self, cwd: str, **kwargs: Any) -> Any:
        session = self.orchestrator.create_session(cwd, kwargs.get("model"))
        response = _build_new_session_response(self.acp, session.session_id, self.orchestrator)
        self._schedule_session_info_update(session.session_id)
        return response

    async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> Any:
        del cwd
        snapshot = self.orchestrator.load_session(session_id, selected_model=kwargs.get("model"))
        response = _build_load_session_response(self.acp, snapshot.session_id, self.orchestrator)
        self._schedule_session_info_update(snapshot.session_id)
        return response

    async def list_sessions(self, **kwargs: Any) -> Any:
        del kwargs
        history = [_session_info_record(self.acp, entry.to_dict()) for entry in self.orchestrator.list_sessions()]
        response_type = getattr(self.acp, "ListSessionsResponse", None)
        if response_type is None:
            response_type = _acp_attr(self.acp, "ListSessionsResponse")
        if response_type is not None:
            return response_type(sessions=history)
        return {"sessions": history}

    async def set_config_option(self, session_id: str, option_id: str, value: str, **kwargs: Any) -> Any:
        del kwargs
        if option_id == "model":
            self.orchestrator.set_selected_model(session_id, value)
            await self._send_session_info_update(session_id)
        response_type = _acp_attr(self.acp, "SetSessionConfigOptionResponse")
        if response_type is not None:
            return response_type(config_options=_model_config_options(self.acp, self.orchestrator, session_id=session_id))
        return None

    async def set_session_model(self, session_id: str, model_id: str, **kwargs: Any) -> Any:
        del kwargs
        self.orchestrator.set_selected_model(session_id, model_id)
        await self._send_session_info_update(session_id)
        response_type = _acp_attr(self.acp, "SetSessionModelResponse")
        return response_type() if response_type is not None else None

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
        prompt_metadata = extract_prompt_metadata(prompt, **kwargs)
        text = "\n".join(getattr(block, "text", str(block)) for block in prompt)
        loop = asyncio.get_running_loop()

        def emit_update(update: dict[str, Any]) -> None:
            future = asyncio.run_coroutine_threadsafe(self._send_session_update(session_id, update), loop)
            future.result()

        task = asyncio.current_task()
        if task is not None:
            self._prompt_tasks[session_id] = task
        try:
            result = await asyncio.to_thread(
                self.orchestrator.orchestrate_turn,
                session_id,
                text,
                emit_update=emit_update,
                prompt_metadata=prompt_metadata,
            )
        finally:
            self._prompt_tasks.pop(session_id, None)
        await self._send_session_info_update(session_id)
        return self._build_prompt_response(result["summary"], stop_reason=str(result.get("stop_reason") or "end_turn"))

    async def cancel(self, session_id: str, **kwargs: Any) -> Any:
        agent_id = kwargs.get("agent_id")
        self.orchestrator.cancel(session_id, agent_id=agent_id)
        await self._send_cancel_update(session_id, agent_id=agent_id)
        await self._send_session_info_update(session_id)
        return None

    async def _send_session_update(self, session_id: str, update: dict[str, Any]) -> None:
        if self.client_connection is None:
            return
        session_update = getattr(self.client_connection, "session_update", None)
        if session_update is None:
            return
        for notification in _session_update_notifications(
            self.acp,
            update,
            session_id=session_id,
            tool_calls_seen=self._tool_call_seen.setdefault(session_id, set()),
        ):
            await session_update(session_id=session_id, update=notification)

    async def _send_session_info_update(self, session_id: str) -> None:
        await self._send_session_update(
            session_id,
            {
                "sessionUpdate": "session_info",
                "session_id": session_id,
                "info": self.orchestrator.session_info(session_id),
            },
        )

    async def _send_cancel_update(self, session_id: str, *, agent_id: str | None) -> None:
        await self._send_session_update(
            session_id,
            {
                "sessionUpdate": "cancelled",
                "session_id": session_id,
                "agent_id": agent_id,
            },
        )

    def _schedule_session_info_update(self, session_id: str) -> None:
        if self.client_connection is None:
            return
        loop = asyncio.get_running_loop()
        # Mirror the codex-acp startup pattern: send post-response session updates
        # asynchronously so they do not race the NewSessionResponse/LoadSessionResponse path.
        loop.call_soon(self._start_session_info_update_task, session_id)

    def _start_session_info_update_task(self, session_id: str) -> None:
        self._start_background_task(self._send_session_info_update(session_id))

    def _start_background_task(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._finalize_background_task)

    def _finalize_background_task(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            _LOG.exception("Background ACP session update task failed")

    def _build_prompt_response(self, summary: str, *, stop_reason: str) -> Any:
        response_type = _acp_attr(self.acp, "PromptResponse")
        try:
            return response_type(stopReason=stop_reason)
        except TypeError:
            return response_type(
                stop_reason=stop_reason,
                message=self.acp.Message(content=[self.acp.TextBlock(text=summary)]),
            )


def _build_agent(acp: Any, runtime: AcpAgentRuntime) -> Any:
    base = getattr(acp, "Agent", object)

    class OrgeMageAcpAgent(base):
        def __init__(self, runtime: AcpAgentRuntime) -> None:
            self.runtime = runtime

        def on_connect(self, conn: Any) -> None:
            self.runtime.bind_client_connection(conn)

        async def initialize(self, protocol_version: int, **kwargs: Any) -> Any:
            return await self.runtime.initialize(protocol_version, **kwargs)

        async def new_session(self, cwd: str, **kwargs: Any) -> Any:
            return await self.runtime.new_session(cwd, **kwargs)

        async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> Any:
            return await self.runtime.load_session(cwd, session_id, **kwargs)

        async def list_sessions(self, **kwargs: Any) -> Any:
            return await self.runtime.list_sessions(**kwargs)

        async def set_config_option(self, config_id: str, session_id: str, value: str, **kwargs: Any) -> Any:
            return await self.runtime.set_config_option(session_id, config_id, value, **kwargs)

        async def set_session_model(self, model_id: str, session_id: str, **kwargs: Any) -> Any:
            return await self.runtime.set_session_model(session_id, model_id, **kwargs)

        async def prompt(self, prompt: list[Any], session_id: str, **kwargs: Any) -> Any:
            return await self.runtime.prompt(session_id, prompt, **kwargs)

        async def cancel(self, session_id: str, **kwargs: Any) -> Any:
            return await self.runtime.cancel(session_id, **kwargs)

    return OrgeMageAcpAgent(runtime=runtime)


def _build_initialize_response(acp: Any, protocol_version: int) -> Any:
    response_type = _acp_attr(acp, "InitializeResponse")
    capabilities_type = _acp_attr(acp, "AgentCapabilities")
    session_capabilities_type = _acp_attr(acp, "SessionCapabilities")
    session_list_type = _acp_attr(acp, "SessionListCapabilities")
    session_resume_type = _acp_attr(acp, "SessionResumeCapabilities")
    mcp_capabilities_type = _acp_attr(acp, "McpCapabilities")
    prompt_capabilities_type = _acp_attr(acp, "PromptCapabilities")
    implementation_type = _acp_attr(acp, "Implementation")
    if capabilities_type is None:
        return response_type(protocol_version=protocol_version)
    return response_type(
        protocolVersion=protocol_version,
        agentCapabilities=capabilities_type(
            loadSession=True,
            sessionCapabilities=(
                session_capabilities_type(
                    list=session_list_type() if session_list_type is not None else None,
                    resume=session_resume_type() if session_resume_type is not None else None,
                )
                if session_capabilities_type is not None
                else None
            ),
            mcpCapabilities=mcp_capabilities_type() if mcp_capabilities_type is not None else None,
            promptCapabilities=prompt_capabilities_type() if prompt_capabilities_type is not None else None,
        ),
        agentInfo=implementation_type(name="orgemage", title="OrgeMage", version="0.1.0")
        if implementation_type is not None
        else None,
    )


def _build_new_session_response(acp: Any, session_id: str, orchestrator: Orchestrator) -> Any:
    response_type = _acp_attr(acp, "NewSessionResponse")
    kwargs = {
        "config_options": _model_config_options(acp, orchestrator, session_id=session_id),
        "models": _model_state(acp, orchestrator, session_id=session_id),
    }
    try:
        return response_type(sessionId=session_id, configOptions=kwargs["config_options"], models=kwargs["models"])
    except TypeError:
        return response_type(session_id=session_id, config_options=kwargs["config_options"])


def _build_load_session_response(acp: Any, session_id: str, orchestrator: Orchestrator) -> Any:
    response_type = _acp_attr(acp, "LoadSessionResponse")
    kwargs = {
        "config_options": _model_config_options(acp, orchestrator, session_id=session_id),
        "models": _model_state(acp, orchestrator, session_id=session_id),
    }
    try:
        return response_type(configOptions=kwargs["config_options"], models=kwargs["models"])
    except TypeError:
        return response_type(session_id=session_id, config_options=kwargs["config_options"])


def _model_config_options(acp: Any, orchestrator: Orchestrator, *, session_id: str | None = None) -> list[Any]:
    options = orchestrator.list_model_options()
    current_value = _current_model_value(orchestrator, session_id=session_id, options=options)
    if hasattr(acp, "ConfigOption"):
        return [
            acp.ConfigOption(
                id="model",
                name="Coordinator model",
                category="model",
                type="select",
                options=[
                    acp.ConfigOptionValue(
                        value=option["value"],
                        name=option["name"],
                        description=option["description"],
                    )
                    for option in options
                ],
            )
        ]
    session_config_option = _acp_attr(acp, "SessionConfigOption")
    if session_config_option is None:
        return []
    return [
        session_config_option.model_validate(
            {
                "id": "model",
                "name": "Coordinator model",
                "category": "model",
                "type": "select",
                "currentValue": current_value,
                "options": [
                    {
                        "value": option["value"],
                        "name": option["name"],
                        "description": option["description"],
                    }
                    for option in options
                ],
            }
        )
    ]


def _model_state(acp: Any, orchestrator: Orchestrator, *, session_id: str | None = None) -> Any:
    session_model_state = _acp_attr(acp, "SessionModelState")
    model_info = _acp_attr(acp, "ModelInfo")
    if session_model_state is None or model_info is None:
        return None
    options = orchestrator.list_model_options()
    current_value = _current_model_value(orchestrator, session_id=session_id, options=options)
    return session_model_state(
        currentModelId=current_value,
        availableModels=[
            _model_info_record(model_info, option)
            for option in options
        ],
    )


def _current_model_value(orchestrator: Orchestrator, *, session_id: str | None, options: list[dict[str, Any]]) -> str:
    if session_id is not None:
        selected_model = orchestrator.session_info(session_id).get("selected_model")
        if isinstance(selected_model, str) and selected_model:
            return selected_model
    return str(options[0]["value"]) if options else ""


def _model_info_record(model_info: Any, option: dict[str, Any]) -> Any:
    try:
        return model_info(
            modelId=option["value"],
            name=option["name"],
            description=option["description"],
        )
    except TypeError:
        return model_info(
            id=option["value"],
            name=option["name"],
            description=option["description"],
        )


def _session_info_record(acp: Any, payload: dict[str, Any]) -> Any:
    response_type = _acp_attr(acp, "SessionInfo")
    if response_type is None:
        return payload
    return response_type.model_validate(
        {
            "sessionId": payload["session_id"],
            "cwd": payload["cwd"],
            "title": payload.get("title"),
            "updatedAt": _timestamp_to_iso(payload.get("updated_at")),
            "_meta": {
                "summary": payload.get("summary"),
                "history": payload,
            },
        }
    )


def _session_update_notifications(acp: Any, update: dict[str, Any], *, session_id: str, tool_calls_seen: set[str]) -> list[Any]:
    del session_id
    session_update = str(update.get("sessionUpdate") or "")
    if hasattr(acp, "ConfigOption"):
        return [update]
    notifications: list[Any] = []
    if session_update == "plan":
        plan_update_type = _acp_attr(acp, "AgentPlanUpdate")
        if plan_update_type is not None:
            notifications.append(
                plan_update_type.model_validate(
                    {
                        "sessionUpdate": "plan",
                        "entries": update.get("globalPlan") or update.get("plan") or [],
                        "_meta": update.get("_meta"),
                    }
                )
            )
    elif session_update == "tool_call":
        tool_call = dict(update.get("toolCall") or update.get("tool_call") or {})
        tool_call_id = str(tool_call.get("toolCallId") or tool_call.get("tool_call_id") or "")
        if tool_call_id:
            kind = "tool_call" if tool_call_id not in tool_calls_seen else "tool_call_update"
            tool_calls_seen.add(tool_call_id)
            notification_type = _acp_attr(acp, "ToolCallStart" if kind == "tool_call" else "ToolCallProgress")
            if notification_type is not None:
                notifications.append(
                    notification_type.model_validate(
                        {
                            **tool_call,
                            "sessionUpdate": kind,
                            "_meta": update.get("_meta"),
                        }
                    )
                )
    elif session_update == "message":
        chunk_type = _acp_attr(acp, "AgentMessageChunk")
        if chunk_type is not None:
            for block in (update.get("message") or {}).get("content", []):
                if block.get("type") != "text":
                    continue
                notifications.append(
                    chunk_type.model_validate(
                        {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": block.get("text", "")},
                            "_meta": update.get("_meta"),
                        }
                    )
                )
    elif session_update == "session_info":
        info = dict(update.get("info") or {})
        info_type = _acp_attr(acp, "SessionInfoUpdate")
        if info_type is not None:
            notifications.append(
                info_type.model_validate(
                    {
                        "sessionUpdate": "session_info_update",
                        "title": info.get("title"),
                        "updatedAt": _timestamp_to_iso(info.get("updated_at")),
                        "_meta": {
                            "summary": info.get("summary"),
                            "history": info.get("history"),
                            "sessionId": info.get("session_id"),
                            "selectedModel": info.get("selected_model"),
                            "activeTurnId": info.get("active_turn_id"),
                            "activeTurnStatus": info.get("active_turn_status"),
                            "taskCount": info.get("task_count"),
                            "coordinatorAgentId": info.get("coordinator_agent_id"),
                            "createdAt": info.get("created_at"),
                            "cwd": info.get("cwd"),
                        },
                    }
                )
            )
    elif session_update == "cancelled":
        chunk_type = _acp_attr(acp, "AgentMessageChunk")
        if chunk_type is not None:
            notifications.append(
                chunk_type.model_validate(
                    {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "Turn cancelled."},
                        "_meta": {"agentId": update.get("agent_id")},
                    }
                )
            )
    return notifications or [update]


def _timestamp_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)
