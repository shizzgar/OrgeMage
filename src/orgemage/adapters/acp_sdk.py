from __future__ import annotations

import asyncio
import importlib.util
from typing import Any

from ..metadata import extract_prompt_metadata
from ..orchestrator import Orchestrator


class AcpSdkUnavailableError(RuntimeError):
    pass


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
        self.agent = _build_agent(acp, self)

    def bind_client_connection(self, connection: Any) -> None:
        self.client_connection = connection
        self.orchestrator.connector_manager.bind_upstream_client_connection(
            connection,
            capabilities=getattr(self, "client_capabilities", None),
        )

    async def initialize(self, protocol_version: int, **kwargs: Any) -> Any:
        client = kwargs.get("client") or kwargs.get("client_connection")
        self.client_capabilities = kwargs.get("client_capabilities")
        if client is not None:
            self.bind_client_connection(client)
        return _build_initialize_response(self.acp, protocol_version)

    async def new_session(self, cwd: str, **kwargs: Any) -> Any:
        session = self.orchestrator.create_session(cwd, kwargs.get("model"))
        await self._send_session_info_update(session.session_id)
        return _build_new_session_response(self.acp, session.session_id, self.orchestrator)

    async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> Any:
        del cwd
        snapshot = self.orchestrator.load_session(session_id, selected_model=kwargs.get("model"))
        await self._send_session_info_update(snapshot.session_id)
        return _build_load_session_response(self.acp, snapshot.session_id, self.orchestrator)

    async def list_sessions(self, **kwargs: Any) -> Any:
        del kwargs
        history = [entry.to_dict() for entry in self.orchestrator.list_sessions()]
        response_type = getattr(self.acp, "ListSessionsResponse", None)
        if response_type is not None:
            return response_type(sessions=history)
        return {"sessions": history}

    async def set_config_option(self, session_id: str, option_id: str, value: str, **kwargs: Any) -> Any:
        del kwargs
        if option_id == "model":
            self.orchestrator.set_selected_model(session_id, value)
            await self._send_session_info_update(session_id)
        return None

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
        await self._send_session_update(
            session_id,
            {
                "sessionUpdate": "cancelled",
                "session_id": session_id,
                "agent_id": agent_id,
            },
        )
        await self._send_session_info_update(session_id)
        return None

    async def _send_session_update(self, session_id: str, update: dict[str, Any]) -> None:
        if self.client_connection is None:
            return
        session_update = getattr(self.client_connection, "session_update", None)
        if session_update is None:
            return
        await session_update(session_id=session_id, update=update)

    async def _send_session_info_update(self, session_id: str) -> None:
        await self._send_session_update(
            session_id,
            {
                "sessionUpdate": "session_info",
                "session_id": session_id,
                "info": self.orchestrator.session_info(session_id),
            },
        )

    def _build_prompt_response(self, summary: str, *, stop_reason: str) -> Any:
        return self.acp.PromptResponse(
            stop_reason=stop_reason,
            message=self.acp.Message(content=[self.acp.TextBlock(text=summary)]),
        )


def _build_agent(acp: Any, runtime: AcpAgentRuntime) -> Any:
    base = getattr(acp, "Agent", object)

    class OrgeMageAcpAgent(base):
        def __init__(self, runtime: AcpAgentRuntime) -> None:
            self.runtime = runtime

        async def initialize(self, protocol_version: int, **kwargs: Any) -> Any:
            return await self.runtime.initialize(protocol_version, **kwargs)

        async def new_session(self, cwd: str, **kwargs: Any) -> Any:
            return await self.runtime.new_session(cwd, **kwargs)

        async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> Any:
            return await self.runtime.load_session(cwd, session_id, **kwargs)

        async def list_sessions(self, **kwargs: Any) -> Any:
            return await self.runtime.list_sessions(**kwargs)

        async def set_config_option(self, session_id: str, option_id: str, value: str, **kwargs: Any) -> Any:
            return await self.runtime.set_config_option(session_id, option_id, value, **kwargs)

        async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
            return await self.runtime.prompt(session_id, prompt, **kwargs)

        async def cancel(self, session_id: str, **kwargs: Any) -> Any:
            return await self.runtime.cancel(session_id, **kwargs)

    return OrgeMageAcpAgent(runtime=runtime)


def _build_initialize_response(acp: Any, protocol_version: int) -> Any:
    response_type = getattr(acp, "InitializeResponse")
    return response_type(protocol_version=protocol_version)


def _build_new_session_response(acp: Any, session_id: str, orchestrator: Orchestrator) -> Any:
    response_type = getattr(acp, "NewSessionResponse")
    return response_type(
        session_id=session_id,
        config_options=_model_config_options(acp, orchestrator),
    )


def _build_load_session_response(acp: Any, session_id: str, orchestrator: Orchestrator) -> Any:
    response_type = getattr(acp, "LoadSessionResponse")
    return response_type(
        session_id=session_id,
        config_options=_model_config_options(acp, orchestrator),
    )


def _model_config_options(acp: Any, orchestrator: Orchestrator) -> list[Any]:
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
                for option in orchestrator.list_model_options()
            ],
        )
    ]
