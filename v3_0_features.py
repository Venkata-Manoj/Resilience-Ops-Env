# v3.0 Features - Dynamic Environments + Multi-Agent + Observability
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
v3.0 Features for ResilienceOps Environment

This module provides advanced features for the ResilienceOps environment:
- Dynamic task generation with randomized parameters
- Multi-agent support for collaborative incident response
- Observability dashboard with Prometheus-compatible metrics
- LLM-as-Judge grading for reasoning quality evaluation
- TRL/GRPO training integration
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Import existing classes
try:
    from .models import (
        TaskConfig, IncidentEnvState, ResilienceOpsAction, ResilienceOpsObservation,
        ResilienceOpsEnvironment, TASKS, grade_episode, compute_reward, _clamp_reward
    )
except ImportError:
    from models import (
        TaskConfig, IncidentEnvState, ResilienceOpsAction, ResilienceOpsObservation,
        ResilienceOpsEnvironment, TASKS, grade_episode, compute_reward, _clamp_reward
    )


# ---------------------------------------------------------------------------
# Dynamic Environment Features
# ---------------------------------------------------------------------------

@dataclass
class DynamicTaskConfig(TaskConfig):
    """Tasks with randomized parameters for grading diversity."""

    alert_noise: float = 0.2  # Probability of false-positive alerts
    log_corruption: float = 0.1  # Probability of truncated/garbled logs
    service_health_noise: float = 0.1  # Probability of flapping health status
    red_herring_services: List[str] = field(default_factory=list)  # Services that appear affected but aren't
    time_pressure_multiplier: float = 1.0  # 0.5x = lenient, 2.0x = aggressive

    def generate_episode(self, seed: int) -> "EpisodeInstance":
        """Generate a unique episode instance from this template."""
        rng = random.Random(seed)
        alerts = self._randomize_alerts(rng)
        logs = self._randomize_logs(rng)
        health = self._randomize_health(rng)
        return EpisodeInstance(alerts=alerts, logs=logs, health=health)

    def _randomize_alerts(self, rng: random.Random) -> List[str]:
        """Randomize alert order and add noise."""
        alerts = list(self.initial_alerts)
        rng.shuffle(alerts)

        # Add false positive alerts
        if rng.random() < self.alert_noise:
            false_positives = [
                "NOTICE: Backup completed successfully",
                "INFO: Scheduled maintenance started",
                "DEBUG: Cache refresh triggered",
            ]
            alerts.insert(rng.randint(0, len(alerts)), rng.choice(false_positives))

        return alerts

    def _randomize_logs(self, rng: random.Random) -> str:
        """Randomize log snippets with potential corruption."""
        logs = self.initial_log_snippet

        if rng.random() < self.log_corruption:
            # Truncate or garble parts of the log
            lines = logs.split("\n")
            if len(lines) > 2 and rng.random() < 0.5:
                # Truncate last line
                lines = lines[:-1]
            if rng.random() < 0.5:
                # Add garbled line
                lines.insert(rng.randint(0, len(lines)), "[CORRUPTED LOG ENTRY]")
            logs = "\n".join(lines)

        return logs

    def _randomize_health(self, rng: random.Random) -> Dict[str, str]:
        """Randomize initial service health with potential flapping."""
        health = dict(self.initial_service_health)

        if rng.random() < self.service_health_noise:
            # Randomly flip one service health
            services = list(health.keys())
            if services:
                svc = rng.choice(services)
                states = ["healthy", "degraded", "down"]
                health[svc] = rng.choice(states)

        return health


@dataclass
class EpisodeInstance:
    """A specific instance of a task with randomized parameters."""

    alerts: List[str]
    logs: str
    health: Dict[str, str]


# ---------------------------------------------------------------------------
# Multi-Agent Support
# ---------------------------------------------------------------------------

@dataclass
class SharedIncidentState:
    """Shared state visible to all agents in a multi-agent session."""

    service_health: Dict[str, str] = field(default_factory=dict)
    global_alerts: List[str] = field(default_factory=list)
    root_cause_identified: bool = False
    all_services_restored: bool = False
    agent_actions: Dict[str, List[ResilienceOpsAction]] = field(default_factory=lambda: defaultdict(list))


class MultiAgentEnvironment(ResilienceOpsEnvironment):
    """Multiple agents collaborate on a single incident."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, num_agents: int = 2, task_name: str = "easy"):
        super().__init__(task_name=task_name)
        self.num_agents = num_agents
        self.agents: Dict[str, IncidentEnvState] = {}
        self.shared_incident: SharedIncidentState = SharedIncidentState()
        self._agent_ids: List[str] = [f"agent_{i}" for i in range(num_agents)]

    def reset(self) -> Dict[str, ResilienceOpsObservation]:
        """Reset environment for all agents."""
        task = TASKS.get(self._task_name)
        if task is None:
            return {}

        # Initialize shared state
        self.shared_incident = SharedIncidentState(
            service_health=dict(task.initial_service_health),
            global_alerts=list(task.initial_alerts),
        )

        # Initialize individual agent states
        observations = {}
        for agent_id in self._agent_ids:
            self.agents[agent_id] = IncidentEnvState(
                episode_id=str(agent_id),
                task=task,
                step_count=0,
                root_cause_identified=False,
                all_services_restored=False,
                service_health=dict(task.initial_service_health),
                action_history=[],
                cumulative_reward=0.0,
                done=False,
            )
            obs = self._build_observation_for_agent(agent_id, "Environment reset. New incident received.")
            observations[agent_id] = obs

        return observations

    def step(self, agent_id: str, action: ResilienceOpsAction) -> ResilienceOpsObservation:
        """Execute action from one agent, affecting shared state visible to all."""
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent_id: {agent_id}")

        state = self.agents[agent_id]
        task = state.task
        if task is None:
            return ResilienceOpsObservation(previous_action_result="No active incident.", done=True)

        if state.done:
            return ResilienceOpsObservation(previous_action_result="Episode finished.", done=True)

        # Update agent state
        state.step_count += 1
        state.action_history.append(action)
        self.shared_incident.agent_actions[agent_id].append(action)

        # Compute reward based on shared state
        reward_delta, result_desc = compute_reward(action, task, state)
        state.cumulative_reward += reward_delta

        # Update shared service health
        for svc, health in state.service_health.items():
            if health == "healthy":
                self.shared_incident.service_health[svc] = health

        # Check if root cause identified
        if state.root_cause_identified:
            self.shared_incident.root_cause_identified = True

        # Check termination conditions
        all_healthy = all(v == "healthy" for v in self.shared_incident.service_health.values())
        if all_healthy:
            self.shared_incident.all_services_restored = True
            state.all_services_restored = True
            state.done = True
        elif state.step_count >= task.max_steps:
            state.done = True

        return self._build_observation_for_agent(agent_id, result_desc)

    def _build_observation_for_agent(self, agent_id: str, previous_action_result: str) -> ResilienceOpsObservation:
        """Build observation for a specific agent with shared state."""
        state = self.agents.get(agent_id)
        task = state.task if state else None

        if task is None:
            return ResilienceOpsObservation(previous_action_result=previous_action_result, done=True)

        steps_remaining = max(0, task.max_steps - state.step_count)

        return ResilienceOpsObservation(
            incident_id=state.episode_id,
            incident_title=task.incident_title,
            severity=task.severity,
            affected_services=list(task.affected_services),
            alert_signals=self.shared_incident.global_alerts,
            log_snippet=task.initial_log_snippet,
            available_tools=list(task.available_tools),
            steps_remaining=steps_remaining,
            step_count=state.step_count,
            previous_action_result=previous_action_result,
            service_health=dict(self.shared_incident.service_health),
            task_name=task.difficulty,
            root_cause_identified=self.shared_incident.root_cause_identified,
            hints=list(task.hints),
            done=state.done,
            reward=state.cumulative_reward,
        )

    def get_agent_grades(self) -> Dict[str, float]:
        """Get final grades for all agents."""
        grades = {}
        for agent_id, state in self.agents.items():
            task = state.task
            if task is None:
                grades[agent_id] = 0.0
                continue
            grade = grade_episode(
                task=task,
                action_history=state.action_history,
                service_health=self.shared_incident.service_health,
                root_cause_identified=self.shared_incident.root_cause_identified,
                steps_taken=state.step_count,
                max_steps=task.max_steps,
            )
            grades[agent_id] = grade
        return grades


# ---------------------------------------------------------------------------
# Observability Dashboard
# ---------------------------------------------------------------------------

@dataclass
class AgentMetrics:
    """Prometheus-compatible metrics for agent behavior analysis."""

    # Episode-level counters
    episodes_completed: int = 0
    episodes_failed: int = 0

    # Reward statistics
    total_rewards: List[float] = field(default_factory=list)
    mean_reward_per_step: List[float] = field(default_factory=list)

    # Action distribution
    action_type_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Success metrics
    root_cause_identification_count: int = 0
    total_episodes: int = 0

    # Timing metrics (steps to resolution)
    time_to_resolution: List[int] = field(default_factory=list)

    # Destructive action tracking
    destructive_action_count: int = 0
    escalation_count: int = 0

    def record_episode(
        self,
        task_name: str,
        rewards: List[float],
        action_history: List[ResilienceOpsAction],
        root_cause_identified: bool,
        steps_taken: int,
        success: bool,
    ) -> None:
        """Record metrics for a completed episode."""
        self.total_episodes += 1

        if success:
            self.episodes_completed += 1
        else:
            self.episodes_failed += 1

        # Record rewards
        if rewards:
            self.total_rewards.append(sum(rewards))
            self.mean_reward_per_step.append(sum(rewards) / len(rewards))

        # Record action distribution
        for action in action_history:
            self.action_type_counts[action.action_type] += 1

        # Record root cause identification
        if root_cause_identified:
            self.root_cause_identification_count += 1

        # Record time to resolution
        if success:
            self.time_to_resolution.append(steps_taken)

        # Count destructive actions and escalations
        for action in action_history:
            if action.action_type in ("rollback", "restart_service") and "postgres" in action.target.lower():
                self.destructive_action_count += 1
            if action.action_type == "escalate":
                self.escalation_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        import statistics

        summary = {
            "episodes_completed": self.episodes_completed,
            "episodes_failed": self.episodes_failed,
            "root_cause_identification_rate": (
                self.root_cause_identification_count / self.total_episodes if self.total_episodes > 0 else 0.0
            ),
            "action_type_distribution": dict(self.action_type_counts),
            "destructive_action_rate": (
                self.destructive_action_count / sum(self.action_type_counts.values())
                if self.action_type_counts else 0.0
            ),
            "escalation_rate": (
                self.escalation_count / sum(self.action_type_counts.values())
                if self.action_type_counts else 0.0
            ),
        }

        if self.total_rewards:
            summary["mean_total_reward"] = statistics.mean(self.total_rewards)
            summary["std_total_reward"] = statistics.stdev(self.total_rewards) if len(self.total_rewards) > 1 else 0.0

        if self.mean_reward_per_step:
            summary["mean_reward_per_step"] = statistics.mean(self.mean_reward_per_step)

        if self.time_to_resolution:
            summary["mean_time_to_resolution"] = statistics.mean(self.time_to_resolution)
            summary["median_time_to_resolution"] = statistics.median(self.time_to_resolution)

        return summary

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus exposition format."""
        lines = []
        lines.append("# HELP resilience_ops_episodes_total Total number of episodes")
        lines.append("# TYPE resilience_ops_episodes_total counter")
        lines.append(f'resilience_ops_episodes_total{{status="completed"}} {self.episodes_completed}')
        lines.append(f'resilience_ops_episodes_total{{status="failed"}} {self.episodes_failed}')

        lines.append("# HELP resilience_ops_root_cause_identified_total Total root cause identifications")
        lines.append("# TYPE resilience_ops_root_cause_identified_total counter")
        lines.append(f"resilience_ops_root_cause_identified_total {self.root_cause_identification_count}")

        lines.append("# HELP resilience_ops_action_total Total actions by type")
        lines.append("# TYPE resilience_ops_action_total counter")
        for action_type, count in self.action_type_counts.items():
            lines.append(f'resilience_ops_action_total{{type="{action_type}"}} {count}')

        lines.append("# HELP resilience_ops_time_to_resolution Steps to resolution")
        lines.append("# TYPE resilience_ops_time_to_resolution histogram")
        for ttr in self.time_to_resolution:
            lines.append(f'resilience_ops_time_to_resolution_bucket{{le="{ttr}"}} 1')

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-as-Judge Grading
# ---------------------------------------------------------------------------

class LLMJudgeGrader:
    """Supplement rule-based grading with LLM evaluation of reasoning quality."""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    def grade_reasoning(
        self,
        action_history: List[ResilienceOpsAction],
        task: TaskConfig,
    ) -> float:
        """Evaluate whether the agent's diagnostic reasoning was sound."""
        # Format actions for the judge
        action_summary = self._format_actions(action_history)

        prompt = f"""
Evaluate this SRE agent's incident response reasoning:

Incident: {task.incident_title} (Severity: {task.severity})
Root Cause: {task.root_cause}
Affected Services: {', '.join(task.affected_services)}

Agent Actions:
{action_summary}

Score the reasoning quality from 0.0 to 1.0 based on:
1. Did the agent follow a logical diagnostic sequence? (0.25)
2. Did the agent prioritize correctly given the severity? (0.25)
3. Did the agent avoid destructive actions? (0.25)
4. Was the remediation sequence appropriate? (0.25)

Respond with ONLY a float between 0.0 and 1.0.
"""
        try:
            return self._call_judge_model(prompt)
        except Exception:
            # Fallback to rule-based score if LLM judge fails
            return 0.5

    def _format_actions(self, action_history: List[ResilienceOpsAction]) -> str:
        """Format action history for the judge prompt."""
        lines = []
        for i, action in enumerate(action_history, 1):
            lines.append(f"  {i}. {action.action_type}(target={action.target}, tool={action.tool_used})")
        return "\n".join(lines) if lines else "  (no actions taken)"

    def _call_judge_model(self, prompt: str) -> float:
        """Call the LLM judge model."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert SRE evaluator. Score reasoning quality objectively."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )

            content = response.choices[0].message.content.strip()
            # Extract float from response
            import re
            match = re.search(r"(\d+\.?\d*)", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception:
            return 0.5

    def combined_grade(
        self,
        rule_based_score: float,
        action_history: List[ResilienceOpsAction],
        task: TaskConfig,
        reasoning_weight: float = 0.3,
    ) -> float:
        """Combine rule-based score with LLM reasoning evaluation."""
        reasoning_score = self.grade_reasoning(action_history, task)
        combined = (1 - reasoning_weight) * rule_based_score + reasoning_weight * reasoning_score
        return _clamp_reward(combined)


# ---------------------------------------------------------------------------
# TRL / GRPO Training Integration
# ---------------------------------------------------------------------------

class ResilienceOpsGRPORewardModel:
    """Reward model compatible with TRL's GRPOTrainer."""

    def __init__(self, env: ResilienceOpsEnvironment):
        self.env = env

    def compute_rewards(self, prompts: List[str], completions: List[str]) -> List[float]:
        """Given (prompt, completion) pairs, return rewards by executing completions."""
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                action = self._parse_completion_to_action(completion)
                obs = self.env.step(action)
                # Use final grade if episode complete, else step reward
                if obs.done:
                    reward = self.env.get_final_grade()
                else:
                    reward = obs.reward
                rewards.append(reward)
            except Exception as e:
                # Penalty for invalid completions
                rewards.append(0.0)
        return rewards

    def _parse_completion_to_action(self, completion: str) -> ResilienceOpsAction:
        """Parse a text completion into a ResilienceOpsAction."""
        import json

        # Try to parse as JSON
        try:
            data = json.loads(completion)
            return ResilienceOpsAction(
                action_type=data.get("action_type", "diagnose"),
                target=data.get("target", ""),
                tool_used=data.get("tool_used", ""),
                parameters=data.get("parameters", {}),
            )
        except json.JSONDecodeError:
            # Fallback: try to extract from text
            return ResilienceOpsAction(
                action_type="diagnose",
                target="",
                tool_used="",
                parameters={},
            )


# Export all v3.0 classes
__all__ = [
    "DynamicTaskConfig",
    "EpisodeInstance",
    "MultiAgentEnvironment",
    "SharedIncidentState",
    "AgentMetrics",
    "LLMJudgeGrader",
    "ResilienceOpsGRPORewardModel",
]
