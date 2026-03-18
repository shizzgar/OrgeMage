# Architecture for an ACP Agent Orchestrator with a User-Selected Coordinator Model

## Overview

The orchestrator should be implemented as a full ACP peer that plays two roles at once:

- **Northbound ACP agent** for upstream IDEs and CLIs.
- **Southbound ACP client** for downstream ACP agents such as Codex, Gemini, and Qwen.

This keeps the entire system ACP-native instead of wrapping agents with custom request formats. The orchestrator owns one upstream session, creates multiple downstream sessions, and preserves ACP semantics for plans, tool-call updates, permissions, filesystem access, terminal access, and cancellation.

## Core goals

The design targets five goals:

1. Present a **single model selector** to the user through ACP session config options.
2. Build that selector from the union of all downstream agents' `model` options.
3. Treat the selected model as the **coordinator selection**, not only a display choice.
4. Let the chosen coordinator generate the plan and task breakdown.
5. Execute subtasks across worker agents while streaming one unified ACP experience back to the client.

## High-level architecture

The orchestrator should contain these major components.

### 1. Northbound ACP agent server

This layer implements the ACP agent surface for the upstream client, including:

- `initialize`
- `session/new`
- `session/load` when supported
- `session/prompt`
- `session/cancel`
- streaming `session/update` notifications

Its job is to make the orchestrator look like a normal ACP agent from the client's perspective.

### 2. Southbound ACP connectors

Each downstream agent connector is an ACP client responsible for:

- starting or connecting to a downstream ACP agent
- running `initialize`
- creating and resuming downstream sessions
- sending prompts
- consuming `session/update` streams
- propagating cancellations

Each connector should store the downstream agent's capabilities, config options, auth state, and session mappings.

### 3. Virtual client surface

When a downstream agent wants client-side ACP services, the orchestrator must act like a real ACP client. That includes brokering:

- permission requests
- filesystem operations
- terminal lifecycle and streaming

This is the layer that lets downstream coding agents safely use the upstream IDE's resources.

### 4. Federated capability and model catalog

The orchestrator should maintain a live catalog of:

- downstream agent identities
- initialization capabilities
- session capabilities
- config options
- model values exposed through ACP config options
- optional command or MCP-related capability signals

This catalog is the source of truth for building the northbound model selector and for routing work.

### 5. Coordinator engine

The coordinator engine maps the user's selected model to a specific downstream agent session. That session becomes the planning authority for the turn and is responsible for producing the task plan.

### 6. Scheduler and execution graph

The scheduler executes the coordinator's plan by:

- spawning worker tasks
- assigning subtasks to matching agents
- honoring dependencies between tasks
- running independent work in parallel
- collecting updates from workers
- publishing unified plan and tool-call updates upstream

### 7. State store

Persisted state should include:

- orchestrator session metadata
- northbound-to-southbound session mappings
- model selection state
- task graph state
- permission decisions where policy allows caching
- terminal mappings
- trace identifiers

A lightweight SQLite store is a reasonable starting point.

## Unified model selection

ACP session config options should be the only user-facing model-selection mechanism.

### Discovery

Because ACP models are typically exposed as config options with category `model`, the orchestrator should:

1. initialize each downstream agent
2. create a lightweight downstream session
3. inspect returned config options
4. extract all selectable values from the downstream `model` option

### Northbound selector

The orchestrator should expose one northbound select option with category `model`. Each option value should encode both the downstream agent and the downstream model, for example:

- `codex::gpt-5-codex`
- `gemini::gemini-2.5-pro`
- `qwen::qwen3-coder-plus`

The user sees one selector, but the orchestrator can deterministically map each choice back to a downstream connector.

### Coordinator election

The chosen option determines the coordinator agent. The election rule is simple:

> the downstream agent session configured with the selected model becomes the coordinator for the current orchestration session

If the user switches the option, the orchestrator can either reuse the existing downstream session when possible or create a fresh coordinator session.

## Planning and delegation

## Coordinator-generated plans

The selected coordinator should receive the user's request plus orchestrator-specific instructions to emit a structured plan. The orchestrator should mirror that plan upstream through ACP plan updates.

Each plan entry can carry orchestration metadata in `_meta`, such as:

- `assignee`
- `requiredCapabilities`
- `acceptableModels`
- `dependencyIds`
- `priority`

This keeps orchestration metadata ACP-compatible without introducing custom top-level protocol fields.

## Scheduling rules

The scheduler should route work using ACP-native signals first:

- negotiated capabilities from initialization
- downstream config options
- available commands exposed during a session
- MCP support when relevant
- prompt modality support when richer inputs are needed

Heuristics can fill gaps, but ACP signals should remain the primary routing inputs.

## Unified progress streaming

The upstream client should see one consistent execution story even though multiple workers are active.

### Plan updates

Use ACP plan updates to represent the global execution graph and current task states.

### Tool-call updates

Represent each delegated subtask as an orchestrator-level tool call so the upstream client can render:

- in-progress activity
- final status
- logs or summarized worker output
- file locations
- terminal output when appropriate

### Worker output handling

There are two reasonable normalization strategies:

1. **Nest worker output** inside orchestrator tool-call content.
2. **Rebase worker output** into orchestrator-owned tool-call identifiers.

Rebasing usually provides the most stable upstream UX because it decouples the UI from downstream agent-specific event shapes.

## Permission, filesystem, and terminal brokering

## Permission brokering

Downstream agents may ask for permission before taking sensitive actions. The orchestrator should broker those requests to the upstream client or apply policy when running in a trusted headless mode.

Permission flow should preserve:

- the original request context
- the affected subtask identity
- cancellation semantics
- any auditable rationale attached to the decision

## Filesystem safety

The orchestrator should enforce path boundaries based on the upstream session's `cwd`. If downstream agents are allowed to call filesystem methods, the orchestrator should:

- require absolute paths
- reject out-of-scope access by default
- honor upstream capability gating
- log file access for traceability

## Terminal lifecycle

Terminal handling should be explicit and reference-counted. The orchestrator should map each worker subtask to terminal resources and guarantee cleanup on:

- normal completion
- failure
- cancellation
- coordinator replacement

## Cancellation semantics

When the upstream client cancels a turn, the orchestrator should:

1. mark the orchestration turn as cancelling
2. propagate cancellation to the coordinator and active workers
3. stop scheduling new dependent work
4. resolve outstanding permission prompts as cancelled
5. emit final upstream updates with cancelled statuses where appropriate

Cancellation should be modeled as normal ACP behavior, not as an exceptional crash path.

## Extensibility and observability

The orchestrator should use ACP's `_meta` field for cross-cutting orchestration data such as:

- trace identifiers
- worker correlation ids
- assignment metadata
- plan provenance
- policy-decision annotations

That same metadata should let the orchestrator propagate trace context across the northbound prompt, coordinator turn, worker turns, and proxied tool calls.

## Recommended implementation stack

A practical implementation stack is:

- **Language:** Python
- **Protocol foundation:** official ACP Python SDK
- **Persistence:** SQLite initially
- **Transport baseline:** stdio for downstream agent processes
- **Tracing:** `_meta` propagation with optional OpenTelemetry integration

## Initial implementation milestones

1. Build a minimal northbound ACP agent that supports session creation and prompt streaming.
2. Add one downstream connector and prove end-to-end prompt delegation.
3. Implement model catalog federation and expose one northbound `model` selector.
4. Add coordinator-driven planning and upstream plan streaming.
5. Add worker scheduling with tool-call rebasing.
6. Broker permissions, filesystem calls, and terminal calls safely.
7. Persist sessions and task state.
8. Add tracing, metrics, and registry-based agent discovery.

## Design recommendation

The orchestrator should be treated as a specialized ACP conductor rather than a custom wrapper. That approach best preserves ACP UX semantics, keeps the system extensible, and lets a user-selected model cleanly determine the downstream coordinator agent.
