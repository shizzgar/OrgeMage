from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters.acp_sdk import AcpSdkBridge, AcpSdkUnavailableError
from .models import AgentCapabilities, DownstreamAgentConfig, ModelOption
from .orchestrator import Orchestrator
from .state import SQLiteSessionStore


def _default_agents(mock_downstream: bool = False) -> list[DownstreamAgentConfig]:
    runtime = "mock" if mock_downstream else "acp"
    return [
        DownstreamAgentConfig(
            agent_id="codex",
            name="Codex",
            command="codex",
            models=[
                ModelOption("gpt-5-codex", "GPT-5 Codex", "Strong coordinator and coding model."),
            ],
            capabilities=AgentCapabilities(
                supports_terminal=True,
                supports_filesystem=True,
                supports_permissions=True,
                supports_mcp=True,
                commands=["read", "edit", "test", "search"],
            ),
            description="OpenAI coding agent",
            default_model="gpt-5-codex",
            runtime=runtime,
        ),
        DownstreamAgentConfig(
            agent_id="gemini",
            name="Gemini CLI",
            command="gemini",
            args=["--experimental-acp"],
            models=[
                ModelOption("gemini-2.5-pro", "Gemini 2.5 Pro", "Strong planner and research model."),
            ],
            capabilities=AgentCapabilities(
                supports_terminal=True,
                supports_filesystem=True,
                supports_permissions=True,
                commands=["read", "search", "fetch"],
            ),
            description="Google Gemini CLI",
            default_model="gemini-2.5-pro",
            runtime=runtime,
        ),
        DownstreamAgentConfig(
            agent_id="qwen",
            name="Qwen Code",
            command="qwen",
            args=["--acp"],
            models=[
                ModelOption("qwen3-coder-plus", "Qwen3 Coder Plus", "Efficient worker model for implementation."),
            ],
            capabilities=AgentCapabilities(
                supports_terminal=True,
                supports_filesystem=True,
                supports_permissions=True,
                commands=["edit", "test"],
            ),
            description="Alibaba Qwen Code",
            default_model="qwen3-coder-plus",
            runtime=runtime,
        ),
    ]


def _load_config(path: str | None, mock_downstream: bool = False) -> list[DownstreamAgentConfig]:
    if path is None:
        return _default_agents(mock_downstream=mock_downstream)
    payload = json.loads(Path(path).read_text())
    agents: list[DownstreamAgentConfig] = []
    for item in payload["agents"]:
        capabilities = AgentCapabilities(**item.get("capabilities", {}))
        models = [ModelOption(**model) for model in item.get("models", [])]
        runtime = "mock" if mock_downstream else item.get("runtime", "acp")
        agents.append(
            DownstreamAgentConfig(
                agent_id=item["agent_id"],
                name=item["name"],
                command=item["command"],
                args=item.get("args", []),
                models=models,
                capabilities=capabilities,
                description=item.get("description", ""),
                default_model=item.get("default_model"),
                runtime=runtime,
                metadata=item.get("metadata", {}),
            )
        )
    return agents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OrgeMage ACP orchestrator")
    parser.add_argument("--config", help="JSON file describing downstream agents")
    parser.add_argument("--db", default=":memory:", help="SQLite database path")
    parser.add_argument(
        "--mock-downstream",
        action="store_true",
        help="Use MockDownstreamClient for downstream agents instead of real ACP subprocesses",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("models", help="List federated coordinator models")

    session_parser = subparsers.add_parser("session", help="Create a session")
    session_parser.add_argument("--cwd", default=".")
    session_parser.add_argument("--model")

    run_parser = subparsers.add_parser("run", help="Run one orchestration turn")
    run_parser.add_argument("--cwd", default=".")
    run_parser.add_argument("--model")
    run_parser.add_argument("prompt")

    acp_parser = subparsers.add_parser("acp", help="Create an ACP agent instance")
    acp_parser.add_argument("--check", action="store_true", help="Only verify that ACP SDK support is available")
    return parser


def _make_orchestrator(args: argparse.Namespace) -> Orchestrator:
    agents = _load_config(args.config, mock_downstream=args.mock_downstream)
    return Orchestrator(agents=agents, store=SQLiteSessionStore(args.db))


def _emit(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    orchestrator = _make_orchestrator(args)

    if args.command == "models":
        _emit({"models": orchestrator.list_model_options()})
        return

    if args.command == "session":
        session = orchestrator.create_session(args.cwd, args.model)
        _emit({"session_id": session.session_id, "selected_model": session.selected_model})
        return

    if args.command == "run":
        session = orchestrator.create_session(args.cwd, args.model)
        result = orchestrator.orchestrate(session.session_id, args.prompt)
        _emit(result)
        return

    if args.command == "acp":
        bridge = AcpSdkBridge(orchestrator)
        if args.check:
            _emit({"acp_sdk_available": bridge.sdk_available()})
            return
        try:
            runtime = bridge.create_runtime()
        except AcpSdkUnavailableError as exc:
            parser.error(str(exc))
        _emit({"agent": runtime.agent.__class__.__name__, "runtime": runtime.__class__.__name__, "acp_sdk_available": True})
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
