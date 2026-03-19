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
