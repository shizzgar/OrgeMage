# OrgeMage

OrgeMage is a design repository for an Agent Client Protocol (ACP) orchestrator that lets a user pick a single model from a unified catalog while the orchestrator coordinates multiple downstream ACP agents.

## Repository contents

- `docs/acp-orchestrator-architecture.md` — reference architecture for the orchestrator.
- `docs/python-sdk-recommendation.md` — evaluation of Python implementation options and the recommended SDK.

## Current direction

The current recommended implementation approach is:

1. Build the orchestrator as a full ACP peer.
2. Expose one northbound ACP agent surface to IDEs and CLIs.
3. Connect southbound to multiple ACP agents such as Codex, Gemini, and Qwen.
4. Federate downstream model options into one northbound ACP `model` config option.
5. Let the selected model choose the downstream coordinator agent.
6. Implement the system in Python on top of the official ACP Python SDK.

See the documents in `docs/` for the detailed design and rationale.
