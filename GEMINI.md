# OrgeMage: ACP-Native Multi-Agent Orchestrator

OrgeMage is a specialized orchestrator designed to coordinate multiple downstream ACP (Agent Client Protocol) agents behind a unified model selector. It functions as both a Northbound ACP agent for upstream clients (like IDEs or CLIs) and a Southbound ACP client for downstream agents (e.g., Codex, Gemini, Qwen).

## Project Overview

- **Core Goal:** Present a single model selector to the user, where the selected model determines the "coordinator" agent responsible for planning and delegating tasks to specialized "worker" agents.
- **Key Technologies:**
  - **Language:** Python 3.10+
  - **Protocol:** [Agent Client Protocol (ACP)](https://github.com/formulahendry/agent-client-protocol)
  - **Persistence:** SQLite (via `src/orgemage/state.py`)
  - **Downstream Agents:** Codex (via `codex app-server`), Gemini CLI, Qwen Code.

## Architecture

1.  **Northbound ACP Agent:** Implements the ACP server surface (`initialize`, `session/new`, `session/prompt`, etc.) for upstream clients.
2.  **Southbound ACP Connectors:** Acts as an ACP client to downstream agents, managing session lifecycles and propagating updates.
3.  **Federated Model Catalog:** Aggregates models from all configured downstream agents into a unified `agent_id::model_id` format (e.g., `gemini::gemini-2.5-pro`).
4.  **Coordinator Engine:** The agent selected by the user prompt's model config becomes the coordinator, generating a structured plan.
5.  **Scheduler & Execution Graph:** Executes the coordinator's plan, assigning subtasks to agents based on required capabilities and managing dependencies.
6.  **Normalizer:** Rebases downstream agent updates into a consistent upstream stream, handling tool calls, terminal outputs, and progress updates.

## Key Files & Directories

- `src/orgemage/orchestrator.py`: The central hub managing session flow, coordination, and turn execution.
- `src/orgemage/models.py`: Efficient `dataclass` (with `slots=True`) definitions for sessions, tasks, turns, and agent states.
- `src/orgemage/catalog.py`: Logic for federating and selecting models from downstream agents.
- `src/orgemage/scheduler.py`: Logic for task assignment and parallel execution management.
- `src/orgemage/state.py`: SQLite-backed persistence for sessions, tasks, and terminal mappings.
- `src/orgemage/acp/`: Downstream connector management and ACP client implementations.
- `src/orgemage/adapters/`: Specific adapters for non-standard runtimes (e.g., `codex app-server`).
- `orgemage-agents.json`: Default configuration for downstream agents.

## Development & Operations

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
# Install with dev and optional acp dependencies
pip install -e .[acp,dev]
```

### Building & Running

- **List Models:** `orgemage models`
- **Create Session:** `orgemage session --cwd . --model codex::gpt-5.4`
- **Run Turn:** `orgemage run --cwd . --model codex::gpt-5.4 "Your prompt here"`
- **ACP Mode (stdio):** `orgemage acp --stdio` (used by ACP UI)
- **Mock Mode:** Use `--mock-downstream` for deterministic local development without spawning real subprocesses.

### Testing

Tests are located in the `tests/` directory and use `pytest`.

```bash
pytest
```

The test suite covers:
- Session and state persistence.
- Federated catalog discovery.
- Orchestration logic with mock agents.
- Parallel task execution and dependency handling.
- Cancellation and error recovery.

## Coding Conventions

- **Type Safety:** Strict use of type hints (`from __future__ import annotations`).
- **Data Models:** Prefer `dataclasses` with `slots=True` for performance and clarity.
- **Asynchronous Updates:** Northbound updates are streamed via callbacks; downstream agents are often managed in separate threads/processes.
- **Metadata Propagation:** Extensive use of `_meta` fields to carry trace IDs, turn IDs, and planning provenance across the ACP boundary.

## Configuration

Default agents are bootstrapped in `src/orgemage/cli.py`. Overrides can be provided via a JSON config file passed to `--config`.
Environment variables like `ORGEMAGE_CODEX_APP_SERVER_COMMAND` can override default paths for specific runtimes.
