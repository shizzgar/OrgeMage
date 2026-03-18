# Choosing a Python SDK for an ACP Orchestrator

## Recommendation

The best Python foundation for this orchestrator is the **official ACP Python SDK**.

## Why this is the best fit

The orchestrator must do three things at once:

1. act as a northbound ACP agent
2. act as a southbound ACP client to multiple downstream agents
3. implement the virtual client surface needed by downstream agents, especially permission, filesystem, and terminal brokering

The official ACP Python SDK is the strongest fit because it is designed for both ACP agents and ACP clients, is aligned to the ACP schema, and supports the async streaming model an orchestrator needs.

## Evaluation criteria

The SDK choice should be judged against these requirements:

- support for building an ACP agent server
- support for building ACP clients and managing downstream sessions
- strong schema support for ACP updates, config options, plans, and tool calls
- stdio-first transport support for spawning ACP agent subprocesses
- asyncio-friendly concurrency for multi-session orchestration
- enough extensibility to layer policy, persistence, tracing, and catalog federation on top

## Recommended path: official ACP Python SDK

### Strengths

- Supports both agent and client implementations.
- Matches the ACP protocol shape directly instead of forcing a custom abstraction.
- Fits a stdio-first orchestrator that spawns downstream ACP agents as subprocesses.
- Works well with asyncio for concurrent downstream sessions.
- Provides a clean base for adding permission brokers, schedulers, and model federation logic.

### Likely responsibilities left to the application

Even with the official SDK, the orchestrator must still implement:

- session mapping and persistence
- model catalog federation
- coordinator selection logic
- plan interpretation
- worker scheduling
- permission policy
- filesystem boundary enforcement
- terminal cleanup and rebroadcasting
- trace propagation

That is expected. The SDK should provide protocol correctness and transport plumbing, while the application provides orchestration behavior.

## Alternatives

### Deep Agents ACP wrapper

If the main goal is only to expose an existing Python agent framework over ACP quickly, a higher-level ACP wrapper could be convenient. It is less attractive for this project because the orchestrator is not just an ACP server. It also needs robust southbound ACP client connectors and proxy behavior.

### Roll your own JSON-RPC layer

A custom implementation can work, but it increases the risk of protocol drift and adds maintenance cost. It should only be chosen if there is a strong requirement that the official SDK cannot satisfy, such as an immediate need for a transport not yet supported by the SDK.

## Implementation guidance

A sensible Python implementation plan is:

1. build the northbound ACP agent with the official SDK
2. represent each downstream connector as an ACP client session manager
3. persist orchestrator session state in SQLite
4. normalize worker activity into orchestrator-owned plan and tool-call updates
5. propagate trace metadata through `_meta`
6. add registry-based discovery later as a separate integration module

## Final recommendation

Use the official ACP Python SDK as the protocol layer for the orchestrator. It is the most direct match for a Python system that must simultaneously behave as an ACP agent, an ACP client, and a policy-aware proxy for downstream agent capabilities.
