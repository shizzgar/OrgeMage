import sys
from types import SimpleNamespace

import orgemage.cli as cli


def test_build_parser_accepts_acp_stdio_flag() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["acp", "--stdio"])

    assert args.command == "acp"
    assert args.stdio is True
    assert args.check is False


def test_main_runs_acp_stdio_agent(monkeypatch) -> None:
    calls: list[object] = []

    class _FakeBridge:
        def __init__(self, orchestrator) -> None:
            self.orchestrator = orchestrator

        def create_runtime(self):
            return SimpleNamespace(agent="fake-agent")

    async def _fake_run_agent(agent):
        calls.append(agent)

    monkeypatch.setattr(cli, "_make_orchestrator", lambda args: object())
    monkeypatch.setattr(cli, "AcpSdkBridge", _FakeBridge)
    monkeypatch.setitem(sys.modules, "acp", SimpleNamespace(run_agent=_fake_run_agent))
    monkeypatch.setattr(sys, "argv", ["orgemage", "acp", "--stdio"])

    cli.main()

    assert calls == ["fake-agent"]


def test_default_agents_use_codex_acp(monkeypatch) -> None:
    monkeypatch.delenv("ORGEMAGE_CODEX_ACP_COMMAND", raising=False)

    agents = cli._default_agents()

    codex = next(agent for agent in agents if agent.agent_id == "codex")
    assert codex.command == "codex-acp"


def test_default_agents_allow_codex_acp_env_override(monkeypatch) -> None:
    monkeypatch.setenv("ORGEMAGE_CODEX_ACP_COMMAND", "/tmp/custom-codex-acp")

    agents = cli._default_agents()

    codex = next(agent for agent in agents if agent.agent_id == "codex")
    assert codex.command == "/tmp/custom-codex-acp"


def test_load_config_rejects_raw_codex_for_acp_runtime(tmp_path) -> None:
    config_path = tmp_path / "agents.json"
    config_path.write_text(
        """
{
  "agents": [
    {
      "agent_id": "codex",
      "name": "Codex",
      "command": "codex",
      "runtime": "acp"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    try:
        cli._load_config(str(config_path))
    except ValueError as exc:
        assert "codex-acp" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ACP validation to reject raw codex")


def test_load_config_allows_custom_acp_wrapper_with_metadata(tmp_path) -> None:
    config_path = tmp_path / "agents.json"
    config_path.write_text(
        """
{
  "agents": [
    {
      "agent_id": "codex",
      "name": "Codex",
      "command": "/opt/bin/custom-codex-wrapper",
      "runtime": "acp",
      "metadata": {"acp_entrypoint": true}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    agents = cli._load_config(str(config_path))

    assert agents[0].command == "/opt/bin/custom-codex-wrapper"
