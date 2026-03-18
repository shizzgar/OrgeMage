from __future__ import annotations

from collections import defaultdict

from .models import DownstreamAgentConfig, PlanTask


class Scheduler:
    def assign_tasks(
        self,
        tasks: list[PlanTask],
        agents: list[DownstreamAgentConfig],
        coordinator_agent_id: str,
    ) -> list[PlanTask]:
        assignments = defaultdict(int)
        by_id = {agent.agent_id: agent for agent in agents}
        coordinator = by_id[coordinator_agent_id]

        for task in tasks:
            if task.assignee and task.assignee in by_id:
                assignments[task.assignee] += 1
                continue
            if task.assignee and task.assignee not in by_id:
                task.assignee = None
            candidates = self._candidate_agents(task, agents, coordinator)
            candidates.sort(
                key=lambda agent: (
                    self._score(agent, task, coordinator),
                    -assignments[agent.agent_id],
                    agent.name.lower(),
                ),
                reverse=True,
            )
            chosen = candidates[0]
            task.assignee = chosen.agent_id
            assignments[chosen.agent_id] += 1
        return tasks

    def _candidate_agents(
        self,
        task: PlanTask,
        agents: list[DownstreamAgentConfig],
        coordinator: DownstreamAgentConfig,
    ) -> list[DownstreamAgentConfig]:
        if task.acceptable_models:
            allowed = set(task.acceptable_models)
            filtered = [
                agent
                for agent in agents
                if agent.agent_id in allowed or any(model.value in allowed or f"{agent.agent_id}::{model.value}" in allowed for model in agent.models)
            ]
            if filtered:
                return filtered
        if task.assignee_hints:
            hinted_ids = {hint for hint in task.assignee_hints if hint in {agent.agent_id for agent in agents}}
            hinted = [agent for agent in agents if agent.agent_id in hinted_ids]
            if hinted:
                return hinted
        needs_complex_reasoning = task.required_capabilities.get("planner")
        if needs_complex_reasoning:
            return [coordinator]
        return agents

    def _score(
        self,
        agent: DownstreamAgentConfig,
        task: PlanTask,
        coordinator: DownstreamAgentConfig,
    ) -> int:
        score = agent.capabilities.score_for_task(task)
        if agent.agent_id == coordinator.agent_id:
            score += 1
        if any(command in task.details.lower() for command in agent.capabilities.commands):
            score += 1
        return score
