from __future__ import annotations

import importlib.util
from typing import Any

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

    def create_agent(self) -> Any:
        if not self.sdk_available():
            raise AcpSdkUnavailableError(
                "The optional dependency 'agent-client-protocol' is not installed. "
                "Install with `pip install orgemage[acp]`."
            )
        import acp  # type: ignore[import-not-found]

        orchestrator = self.orchestrator

        class OrgeMageAcpAgent(acp.Agent):
            async def initialize(self, protocol_version: int, **kwargs: Any) -> Any:
                return acp.InitializeResponse(protocol_version=protocol_version)

            async def new_session(self, cwd: str, **kwargs: Any) -> Any:
                session = orchestrator.create_session(cwd)
                return acp.NewSessionResponse(
                    session_id=session.session_id,
                    config_options=_model_config_options(acp, orchestrator),
                )

            async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> Any:
                snapshot = orchestrator.load_session(session_id, selected_model=kwargs.get("model"))
                return acp.LoadSessionResponse(
                    session_id=snapshot.session_id,
                    config_options=_model_config_options(acp, orchestrator),
                )

            async def set_config_option(self, session_id: str, option_id: str, value: str, **kwargs: Any) -> Any:
                if option_id == "model":
                    orchestrator.set_selected_model(session_id, value)
                return None

            async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
                text = "\n".join(getattr(block, "text", str(block)) for block in prompt)
                result = orchestrator.orchestrate(session_id, text)
                return acp.PromptResponse(
                    stop_reason="end_turn",
                    message=acp.Message(content=[acp.TextBlock(text=result["summary"])]),
                )

            async def cancel(self, session_id: str, **kwargs: Any) -> Any:
                orchestrator.cancel(session_id)
                return None

        return OrgeMageAcpAgent()


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
