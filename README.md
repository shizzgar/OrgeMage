# OrgeMage

OrgeMage is an ACP-native orchestrator that exposes a single upstream ACP agent while coordinating multiple downstream ACP agents behind a unified model selector.

## What is implemented

This repository now contains a working Python implementation that provides:

- a federated downstream model catalog
- coordinator selection based on ACP-style `model` config values
- session persistence backed by SQLite
- a scheduler that assigns subtasks to the best downstream worker
- deterministic local orchestration runs through a mock downstream client
- an optional bridge to the official ACP Python SDK for northbound ACP agent hosting
- a CLI for listing models, creating sessions, running orchestration turns, and checking ACP SDK availability

## Project layout

- `src/orgemage/` — orchestrator package
- `src/orgemage/adapters/acp_sdk.py` — optional official ACP SDK bridge
- `tests/` — unit tests for catalog, state persistence, and orchestration behavior
- `docs/acp-orchestrator-architecture.md` — architecture reference
- `docs/python-sdk-recommendation.md` — SDK rationale

## Quickstart

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2. List the unified model selector values

```bash
orgemage models
```

### 3. Run a local orchestration turn

```bash
orgemage run --model codex::gpt-5-codex "Implement the ACP orchestrator"
```

### 4. Check ACP SDK support

```bash
pip install -e .[acp]
orgemage acp --check
```

## Configuration

By default the CLI bootstraps three downstream agents from the design documents:

- Codex
- Gemini CLI
- Qwen Code

You can also pass `--config path/to/agents.json` with this shape:

```json
{
  "agents": [
    {
      "agent_id": "codex",
      "name": "Codex",
      "command": "codex",
      "args": ["--some-flag"],
      "description": "OpenAI coding agent",
      "default_model": "gpt-5-codex",
      "models": [
        {
          "value": "gpt-5-codex",
          "name": "GPT-5 Codex",
          "description": "Primary coding model"
        }
      ],
      "capabilities": {
        "supports_terminal": true,
        "supports_filesystem": true,
        "supports_permissions": true,
        "commands": ["read", "edit", "test"]
      }
    }
  ]
}
```

## Notes on ACP integration

The orchestrator core is intentionally dependency-light so it can be tested in isolation. For real ACP transport hosting, install the optional `acp` extra and use the `AcpSdkBridge`, which creates a northbound ACP agent on top of the official ACP Python SDK.
