# OrgeMage

OrgeMage is an ACP-native orchestrator that exposes a single upstream ACP agent while coordinating multiple downstream ACP agents behind a unified model selector.

## What is implemented

This repository now contains a working Python implementation that provides:

- a federated downstream model catalog
- coordinator selection based on ACP-style `model` config values
- session persistence backed by SQLite
- a scheduler that assigns subtasks to the best downstream worker
- real downstream ACP connectors that spawn agent subprocesses via the official Python SDK
- negotiated downstream state tracking and a lazy connector manager for live sessions
- deterministic local orchestration runs through an explicitly enabled mock downstream client
- an optional bridge to the official ACP Python SDK for northbound ACP agent hosting
- a CLI for listing models, creating sessions, running orchestration turns, and checking ACP SDK availability

## Project layout

- `src/orgemage/` — orchestrator package
- `src/orgemage/adapters/acp_sdk.py` — optional official ACP SDK bridge
- `tests/` — unit tests for catalog, state persistence, and orchestration behavior
- `docs/acp-orchestrator-architecture.md` — architecture reference
- `docs/python-sdk-recommendation.md` — SDK rationale

## Quickstart

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
```

If you want to use real ACP transports instead of the deterministic mock runtime, install the optional ACP dependency as well:

```bash
python -m pip install -e .[acp,dev]
```

### 2. Verify that the CLI is available

```bash
orgemage --help
```

This prints the top-level commands:

- `models` — show the unified coordinator model selector
- `session` — create a stored orchestration session
- `run` — create a session and execute one orchestration turn
- `acp` — expose OrgeMage as an ACP agent

### 3. Inspect the federated model catalog

By default, OrgeMage bootstraps three downstream agents:

- `codex` (spawned through `codex-acp` in ACP mode)
- `gemini`
- `qwen`

To see the northbound model values that an upstream client can select:

```bash
orgemage models
```

You should see composite model IDs such as:

- `codex::gpt-5-codex`
- `gemini::gemini-2.5-pro`
- `qwen::qwen3-coder-plus`

### 4. Run a local orchestration turn from the CLI

#### Fast deterministic mode for local development

Use the mock runtime when you want predictable behavior without launching external ACP agents:

```bash
orgemage \
  --mock-downstream \
  --db .orgemage/dev.db \
  run \
  --cwd . \
  --model codex::gpt-5-codex \
  "Design the task, delegate implementation, and summarize the result."
```

This mode is best for:

- trying the CLI quickly
- debugging the orchestration flow
- running tests locally without external agent dependencies

#### Real ACP downstream execution

Use the real ACP runtime when you want OrgeMage to spawn configured downstream ACP agents such as Codex, Gemini CLI, and Qwen Code:

```bash
orgemage \
  --db .orgemage/runtime.db \
  run \
  --cwd . \
  --model codex::gpt-5-codex \
  "Implement the ACP orchestrator and validate it."
```

Before running this mode, make sure the downstream commands referenced by your config are installed and available on `PATH`. For Codex, the ACP entrypoint is `codex-acp`, not the raw `codex` CLI.

### 5. Create and reuse sessions explicitly

If you want session persistence across multiple invocations, point OrgeMage at a file-backed SQLite database and create a session first:

```bash
orgemage --db .orgemage/runtime.db session --cwd . --model codex::gpt-5-codex
```

Then run future turns against the same database file so the stored session catalog and runtime state are preserved.

### 6. Check ACP SDK support

```bash
orgemage acp --check
```

If this returns `"acp_sdk_available": true`, the current environment can host OrgeMage as an upstream ACP agent.

### 7. Use OrgeMage from ACP UI

[ACP UI](https://github.com/formulahendry/acp-ui) can launch OrgeMage as a local ACP agent over stdio. ACP UI stores agent definitions in:

- Windows: `%APPDATA%\\acp-ui\\agents.json`
- macOS: `~/Library/Application Support/acp-ui/agents.json`
- Linux: `~/.config/acp-ui/agents.json`

Add an entry like this to `agents.json`:

```json
{
  "agents": {
    "OrgeMage": {
      "command": "/absolute/path/to/OrgeMage/.venv/bin/orgemage",
      "args": [
        "--db",
        "/absolute/path/to/OrgeMage/.orgemage/runtime.db",
        "acp",
        "--stdio"
      ],
      "env": {}
    }
  }
}
```

How to use this configuration:

1. Replace `/absolute/path/to/OrgeMage` with the path to your local checkout.
2. Make sure the virtual environment contains `orgemage` and the optional ACP dependency.
3. Ensure the downstream agent commands you want OrgeMage to launch are installed on the same machine and resolvable from ACP UI's environment.
4. In ACP UI, choose the `OrgeMage` agent, select your project folder, then create a new session.

If you want to override the default downstream agents, create a dedicated OrgeMage config file and pass it through ACP UI as well:

```json
{
  "agents": {
    "OrgeMage": {
      "command": "/absolute/path/to/OrgeMage/.venv/bin/orgemage",
      "args": [
        "--config",
        "/absolute/path/to/OrgeMage/orgemage-agents.json",
        "--db",
        "/absolute/path/to/OrgeMage/.orgemage/runtime.db",
        "acp",
        "--stdio"
      ],
      "env": {}
    }
  }
}
```

Example `orgemage-agents.json`:

```json
{
  "agents": [
    {
      "agent_id": "codex",
      "name": "Codex",
      "command": "/absolute/path/to/codex-acp",
      "args": [],
      "description": "OpenAI coding agent",
      "default_model": "gpt-5-codex",
      "runtime": "acp",
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
        "commands": ["read", "edit", "test", "search"]
      }
    }
  ]
}
```

## Configuration

By default the CLI bootstraps three downstream agents from the design documents:

- Codex via `codex-acp`
- Gemini CLI via `gemini --experimental-acp`
- Qwen Code via `qwen --acp`

You can override the default Codex bridge path with `ORGEMAGE_CODEX_ACP_COMMAND=/custom/path/to/codex-acp`.

> Warning: `codex-acp` and `codex` are different roles. `codex-acp` is the ACP bridge that OrgeMage can launch as a downstream ACP peer; raw `codex` is not guaranteed to expose ACP transport on stdio.

When `runtime` is `"acp"`, OrgeMage validates that the configured command looks like an ACP-compatible entrypoint. Known-good defaults are `codex-acp`, `gemini --experimental-acp`, and `qwen --acp`. For custom wrappers, set `"metadata": {"acp_entrypoint": true}` to opt in explicitly.

You can also pass `--config path/to/agents.json` with this shape:

```json
{
  "agents": [
    {
      "agent_id": "codex",
      "name": "Codex",
      "command": "codex-acp",
      "args": [],
      "description": "OpenAI coding agent",
      "default_model": "gpt-5-codex",
      "runtime": "acp",
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

The orchestrator core is intentionally dependency-light so it can be tested in isolation. For real ACP transport hosting, install the optional `acp` extra and use the `acp` CLI command:

```bash
orgemage acp --check
orgemage --db .orgemage/runtime.db acp --stdio
```

`orgemage acp --stdio` runs a northbound ACP agent on stdin/stdout via the official ACP Python SDK, which makes it suitable for ACP-compatible desktop clients such as ACP UI.

For startup handshakes, OrgeMage intentionally returns `session/new` and `session/load` before sending the initial `session/update`. This is the same compatibility pattern used by `codex-acp` in `src/agent/core.rs`, where post-creation updates are dispatched asynchronously so they do not race the `NewSessionResponse` delivery path.

After `session/new` and `session/load`, OrgeMage also advertises a small set of northbound ACP slash commands via `available_commands_update`: `/status`, `/models`, and `/plan`. These are OrgeMage-native commands handled directly by the northbound ACP runtime rather than being silently proxied to downstream CLIs.


## Runtime behavior

By default, OrgeMage now treats downstream agents as real ACP peers and lazily spawns them through the official Python SDK when they are first selected. The mock downstream client is retained only as an explicit fallback via `--mock-downstream` or a per-agent config entry with `"runtime": "mock"`.
