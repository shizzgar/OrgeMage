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


def _extract_mcp_servers(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = kwargs.get("mcp_servers")
    if candidate is None:
        candidate = kwargs.get("mcpServers")
    if not isinstance(candidate, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, server in enumerate(candidate):
        if isinstance(server, dict):
            normalized.append({str(key): value for key, value in server.items()})
        else:
            normalized.append({"id": f"mcp-{index}", "value": server})
    return normalized


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
        session = self.orchestrator.create_session(
            cwd,
            kwargs.get("model"),
            mcp_servers=_extract_mcp_servers(kwargs),
        )
        response = _build_new_session_response(self.acp, session.session_id, self.orchestrator)
        self._schedule_startup_updates(session.session_id)
        return response

    async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> Any:
        del cwd
        snapshot = self.orchestrator.load_session(
            session_id,
            selected_model=kwargs.get("model"),
            mcp_servers=_extract_mcp_servers(kwargs),
        )
        response = _build_load_session_response(self.acp, snapshot.session_id, self.orchestrator)
        self._schedule_startup_updates(snapshot.session_id)
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
            await self._send_config_option_update(session_id)
            await self._send_session_info_update(session_id)
        response_type = _acp_attr(self.acp, "SetSessionConfigOptionResponse")
        if response_type is not None:
            return response_type(config_options=_model_config_options(self.acp, self.orchestrator, session_id=session_id))
        return None

    async def set_session_model(self, session_id: str, model_id: str, **kwargs: Any) -> Any:
        del kwargs
        self.orchestrator.set_selected_model(session_id, model_id)
        await self._send_config_option_update(session_id)
        await self._send_session_info_update(session_id)
        response_type = _acp_attr(self.acp, "SetSessionModelResponse")
        return response_type() if response_type is not None else None

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> Any:
        del kwargs
        self.orchestrator.set_session_mode(session_id, mode_id)
        await self._send_current_mode_update(session_id)
        await self._send_session_info_update(session_id)
        response_type = _acp_attr(self.acp, "SetSessionModeResponse")
        return response_type() if response_type is not None else None

    async def set_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> Any:
        return await self.set_session_mode(session_id, mode_id, **kwargs)

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
        prompt_metadata = extract_prompt_metadata(prompt, **kwargs)
        text = "\n".join(getattr(block, "text", str(block)) for block in prompt)
        slash_command_response = await self._maybe_handle_slash_command(session_id, text)
        if slash_command_response is not None:
            return slash_command_response
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

    async def _send_current_mode_update(self, session_id: str) -> None:
        await self._send_session_update(
            session_id,
            {
                "sessionUpdate": "current_mode",
                "session_id": session_id,
                "modes": _mode_state(self.acp, self.orchestrator, session_id=session_id),
            },
        )

    async def _send_config_option_update(self, session_id: str) -> None:
        await self._send_session_update(
            session_id,
            {
                "sessionUpdate": "config_options",
                "session_id": session_id,
                "config_options": _model_config_options(self.acp, self.orchestrator, session_id=session_id),
            },
        )

    async def _send_available_commands_update(self, session_id: str) -> None:
        await self._send_session_update(
            session_id,
            {
                "sessionUpdate": "available_commands",
                "session_id": session_id,
                "available_commands": _available_commands(self.orchestrator, session_id=session_id),
            },
        )

    def _schedule_startup_updates(self, session_id: str) -> None:
        if self.client_connection is None:
            return
        loop = asyncio.get_running_loop()
        # Mirror the codex-acp startup pattern: send post-response session updates
        # asynchronously so they do not race the NewSessionResponse/LoadSessionResponse path.
        loop.call_soon(self._start_startup_update_tasks, session_id)

    def _start_startup_update_tasks(self, session_id: str) -> None:
        self._start_background_task(self._send_available_commands_update(session_id))
        self._start_background_task(self._send_session_info_update(session_id))

    async def _maybe_handle_slash_command(self, session_id: str, text: str) -> Any | None:
        command_name, command_input = _parse_slash_command(text)
        if command_name is None:
            return None
        handler = _SLASH_COMMAND_HANDLERS.get(command_name)
        if handler is None:
            await self._send_agent_message_chunk(
                session_id,
                f"Unknown command '/{command_name}'. Available commands: {', '.join(_slash_command_labels())}.",
            )
            return self._build_prompt_response("", stop_reason="end_turn")
        await self._send_agent_message_chunk(
            session_id,
            handler(self.orchestrator, session_id=session_id, command_input=command_input),
        )
        await self._send_session_info_update(session_id)
        return self._build_prompt_response("", stop_reason="end_turn")

    async def _send_agent_message_chunk(self, session_id: str, text: str) -> None:
        await self._send_session_update(
            session_id,
            {
                "sessionUpdate": "message",
                "session_id": session_id,
                "message": {"content": [{"type": "text", "text": text}]},
            },
        )

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

        async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> Any:
            return await self.runtime.set_session_mode(session_id, mode_id, **kwargs)

        async def set_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> Any:
            return await self.runtime.set_mode(session_id, mode_id, **kwargs)

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
        "modes": _mode_state(acp, orchestrator, session_id=session_id),
    }
    try:
        return response_type(
            sessionId=session_id,
            configOptions=kwargs["config_options"],
            models=kwargs["models"],
            modes=kwargs["modes"],
        )
    except TypeError:
        return response_type(session_id=session_id, config_options=kwargs["config_options"])


def _build_load_session_response(acp: Any, session_id: str, orchestrator: Orchestrator) -> Any:
    response_type = _acp_attr(acp, "LoadSessionResponse")
    kwargs = {
        "config_options": _model_config_options(acp, orchestrator, session_id=session_id),
        "models": _model_state(acp, orchestrator, session_id=session_id),
        "modes": _mode_state(acp, orchestrator, session_id=session_id),
    }
    try:
        return response_type(configOptions=kwargs["config_options"], models=kwargs["models"], modes=kwargs["modes"])
    except TypeError:
        return response_type(session_id=session_id, config_options=kwargs["config_options"])


def _model_config_options(acp: Any, orchestrator: Orchestrator, *, session_id: str | None = None) -> list[Any]:
    options = orchestrator.list_model_options(refresh=False)
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
    options = orchestrator.list_model_options(refresh=False)
    current_value = _current_model_value(orchestrator, session_id=session_id, options=options)
    return session_model_state(
        currentModelId=current_value,
        availableModels=[
            _model_info_record(model_info, option)
            for option in options
        ],
    )


def _mode_state(acp: Any, orchestrator: Orchestrator, *, session_id: str | None = None) -> Any:
    session_mode_state = _acp_attr(acp, "SessionModeState")
    session_mode = _acp_attr(acp, "SessionMode")
    if session_mode_state is None or session_mode is None:
        return None
    available_modes = orchestrator.available_session_modes()
    current_mode_id = (
        orchestrator.current_session_mode(session_id)
        if session_id is not None
        else orchestrator.DEFAULT_SESSION_MODE
    )
    return session_mode_state(
        currentModeId=current_mode_id,
        availableModes=[
            _session_mode_record(session_mode, mode)
            for mode in available_modes
        ],
    )


def _available_commands(orchestrator: Orchestrator, *, session_id: str | None = None) -> list[dict[str, Any]]:
    del orchestrator, session_id
    return [dict(command) for command in _NORTHBOUND_COMMANDS]


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


def _session_mode_record(session_mode: Any, mode: dict[str, Any]) -> Any:
    validate = getattr(session_mode, "model_validate", None)
    if callable(validate):
        return validate(mode)
    try:
        return session_mode(
            id=mode["id"],
            name=mode["name"],
            description=mode["description"],
        )
    except TypeError:
        return session_mode(**mode)


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
                            "currentModeId": info.get("current_mode_id"),
                            "availableModes": info.get("available_modes"),
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
    elif session_update == "current_mode":
        mode_state = update.get("modes")
        current_mode_type = _acp_attr(acp, "CurrentModeUpdate")
        if current_mode_type is not None and mode_state is not None:
            notifications.append(
                current_mode_type.model_validate(
                    {
                        "sessionUpdate": "current_mode_update",
                        "currentModeId": getattr(mode_state, "current_mode_id", None)
                        or getattr(mode_state, "currentModeId", None),
                        "_meta": {"sessionId": update.get("session_id")},
                    }
                )
            )
    elif session_update == "config_options":
        config_options = update.get("config_options") or []
        config_type = _acp_attr(acp, "ConfigOptionUpdate")
        if config_type is not None:
            notifications.append(
                config_type.model_validate(
                    {
                        "sessionUpdate": "config_option_update",
                        "configOptions": [_config_option_payload(option) for option in config_options],
                        "_meta": {"sessionId": update.get("session_id")},
                    }
                )
            )
    elif session_update == "available_commands":
        commands = update.get("available_commands") or []
        commands_type = _acp_attr(acp, "AvailableCommandsUpdate")
        if commands_type is not None:
            notifications.append(
                commands_type.model_validate(
                    {
                        "sessionUpdate": "available_commands_update",
                        "availableCommands": [_available_command_payload(command) for command in commands],
                        "_meta": {"sessionId": update.get("session_id")},
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


def _config_option_payload(option: Any) -> dict[str, Any]:
    if isinstance(option, dict):
        return dict(option)
    payload = dict(getattr(option, "__dict__", {}))
    if "current_value" in payload and "currentValue" not in payload:
        payload["currentValue"] = payload.pop("current_value")
    return payload


def _available_command_payload(command: Any) -> dict[str, Any]:
    payload = dict(command) if isinstance(command, dict) else dict(getattr(command, "__dict__", {}))
    input_payload = payload.get("input")
    if input_payload is None:
        payload.pop("input", None)
    elif not isinstance(input_payload, dict):
        payload["input"] = dict(getattr(input_payload, "__dict__", {}))
    return payload


def _parse_slash_command(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None, ""
    name, _, remainder = stripped[1:].partition(" ")
    normalized_name = name.strip().lower()
    if not normalized_name:
        return None, ""
    return normalized_name, remainder.strip()


def _format_status_command(orchestrator: Orchestrator, *, session_id: str, command_input: str) -> str:
    del command_input
    info = orchestrator.session_info(session_id)
    return (
        f"Session {info['session_id']}\n"
        f"Title: {info.get('title') or 'Untitled'}\n"
        f"Mode: {info.get('current_mode_id') or orchestrator.DEFAULT_SESSION_MODE}\n"
        f"Model: {info.get('selected_model') or 'Not selected'}\n"
        f"Tasks: {info.get('task_count', 0)}\n"
        f"Summary: {info.get('summary') or 'No completed turn yet.'}"
    )


def _format_models_command(orchestrator: Orchestrator, *, session_id: str, command_input: str) -> str:
    del session_id, command_input
    options = orchestrator.list_model_options(refresh=False)
    if not options:
        return "No coordinator models are currently available."
    lines = ["Available coordinator models:"]
    for option in options:
        lines.append(f"- {option['value']}: {option['name']}")
    return "\n".join(lines)


def _format_plan_command(orchestrator: Orchestrator, *, session_id: str, command_input: str) -> str:
    del command_input
    snapshot = orchestrator.load_session(session_id)
    plan = snapshot.task_graph
    if not plan:
        return "No active orchestration plan is available for this session yet."
    lines = ["Current orchestration plan:"]
    for index, task in enumerate(plan, start=1):
        task_id = task.get("task_id") or f"task-{index}"
        title = task.get("title") or task_id
        status = task.get("status") or "pending"
        assignee = task.get("assignee")
        line = f"{index}. [{status}] {title}"
        if assignee:
            line += f" (assignee: {assignee})"
        lines.append(line)
    return "\n".join(lines)


def _slash_command_labels() -> list[str]:
    return [f"/{command['name']}" for command in _NORTHBOUND_COMMANDS]


_NORTHBOUND_COMMANDS: tuple[dict[str, Any], ...] = (
    {
        "name": "status",
        "description": "Show the current OrgeMage session status, model, mode, and summary.",
    },
    {
        "name": "models",
        "description": "List the coordinator models currently exposed by OrgeMage.",
    },
    {
        "name": "plan",
        "description": "Show the current orchestration plan for this session.",
    },
)

_SLASH_COMMAND_HANDLERS = {
    "status": _format_status_command,
    "models": _format_models_command,
    "plan": _format_plan_command,
}
