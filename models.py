# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models and environment logic for the Resilience Ops Env Environment.

An OpenEnv RL environment where AI agents learn to triage, diagnose, and resolve
simulated IT infrastructure incidents (server outages, database failures, network
issues) under time pressure and resource constraints.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES = [
    "diagnose",
    "remediate",
    "escalate",
    "query_logs",
    "check_metrics",
    "restart_service",
    "rollback",
    "scale_up",
    "check_connectivity",
    "analyze_root_cause",
]


class ResilienceOpsAction(Action):
    """Action for the Resilience Ops Env environment."""

    action_type: str = Field(
        ...,
        description=(
            "Type of action to take. One of: "
            "diagnose, remediate, escalate, query_logs, check_metrics, "
            "restart_service, rollback, scale_up, check_connectivity, analyze_root_cause"
        ),
    )
    target: str = Field(
        default="",
        description="Service/component to act on (e.g. 'api-gateway', 'postgres-primary')",
    )
    tool_used: str = Field(
        default="",
        description="Specific diagnostic or remediation tool (e.g. 'top', 'pg_isready', 'kubectl')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters as key-value pairs",
    )


# ---------------------------------------------------------------------------
# Observation Space
# ---------------------------------------------------------------------------


class ResilienceOpsObservation(Observation):
    """Observation from the Resilience Ops Env environment."""

    incident_id: str = Field(default="", description="Unique incident identifier")
    incident_title: str = Field(default="", description="Short title of the incident")
    severity: str = Field(default="", description="Incident severity: P1, P2, or P3")
    affected_services: List[str] = Field(
        default_factory=list, description="List of affected service names"
    )
    alert_signals: List[str] = Field(
        default_factory=list,
        description='Alert signals (e.g. "high_cpu", "db_timeout", "5xx_spike")',
    )
    log_snippet: str = Field(default="", description="Relevant log excerpt")
    available_tools: List[str] = Field(
        default_factory=list, description="Tools the agent can use"
    )
    steps_remaining: int = Field(default=0, description="Steps remaining in episode")
    step_count: int = Field(default=0, description="Current step number")
    previous_action_result: str = Field(
        default="", description="Result of the previous action"
    )
    service_health: Dict[str, str] = Field(
        default_factory=dict,
        description="Current health status of each service: healthy/degraded/down",
    )
    task_name: str = Field(default="", description="Task name (easy/medium/hard)")
    root_cause_identified: bool = Field(
        default=False, description="Whether the agent has identified the root cause"
    )
    hints: List[str] = Field(
        default_factory=list, description="Contextual hints for the agent"
    )


# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """Configuration for a single incident task."""

    name: str
    difficulty: str
    incident_title: str
    severity: str
    affected_services: List[str]
    initial_alerts: List[str]
    initial_log_snippet: str
    root_cause: str
    correct_diagnostic_tools: List[str]
    correct_remediation_sequence: List[Dict[str, str]]
    available_tools: List[str]
    max_steps: int
    hints: List[str]
    initial_service_health: Dict[str, str]
    service_health_after_remediation: Dict[str, str]


TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        difficulty="easy",
        incident_title="API Gateway Timeout",
        severity="P3",
        affected_services=["api-gateway"],
        initial_alerts=["high_cpu", "request_timeout"],
        initial_log_snippet=(
            "api-gateway[1234]: ERROR: upstream connection timed out after 30s\n"
            "api-gateway[1234]: WARN: worker pool exhausted, queue depth=847"
        ),
        root_cause="api-gateway worker pool exhaustion due to memory leak",
        correct_diagnostic_tools=["top", "netstat", "curl"],
        correct_remediation_sequence=[
            {"action_type": "diagnose", "target": "api-gateway", "tool_used": "top"},
            {
                "action_type": "check_metrics",
                "target": "api-gateway",
                "tool_used": "prometheus",
            },
            {
                "action_type": "restart_service",
                "target": "api-gateway",
                "tool_used": "systemctl",
            },
        ],
        available_tools=[
            "top", "netstat", "curl", "systemctl", "journalctl",
            "prometheus", "grafana", "ping",
        ],
        max_steps=10,
        hints=[
            "Start by checking CPU and memory usage on the affected service.",
            "The logs show worker pool exhaustion — a restart may help.",
        ],
        initial_service_health={"api-gateway": "degraded"},
        service_health_after_remediation={"api-gateway": "healthy"},
    ),
    "medium": TaskConfig(
        name="medium",
        difficulty="medium",
        incident_title="Cascading Database Connection Pool Exhaustion",
        severity="P2",
        affected_services=["postgres-primary", "api-gateway", "web-frontend"],
        initial_alerts=[
            "db_connection_pool_exhausted",
            "api_5xx_spike",
            "frontend_latency_high",
        ],
        initial_log_snippet=(
            "postgres-primary[5678]: FATAL: remaining connection slots reserved for superuser\n"
            "postgres-primary[5678]: LOG: connection count=200, max=200\n"
            "api-gateway[1234]: ERROR: connection refused to postgres-primary:5432\n"
            "web-frontend[9012]: WARN: response time p99=4200ms (threshold=1000ms)"
        ),
        root_cause="postgres-primary connection pool exhaustion causing cascading failures",
        correct_diagnostic_tools=["pg_isready", "pg_stat_activity", "netstat"],
        correct_remediation_sequence=[
            {
                "action_type": "diagnose",
                "target": "postgres-primary",
                "tool_used": "pg_isready",
            },
            {
                "action_type": "query_logs",
                "target": "postgres-primary",
                "tool_used": "pg_stat_activity",
            },
            {
                "action_type": "analyze_root_cause",
                "target": "postgres-primary",
                "tool_used": "pg_stat_activity",
            },
            {
                "action_type": "remediate",
                "target": "postgres-primary",
                "tool_used": "pg_terminate_backend",
                "parameters": {"action": "terminate_idle_connections"},
            },
            {
                "action_type": "scale_up",
                "target": "postgres-primary",
                "tool_used": "kubectl",
                "parameters": {"action": "increase_max_connections"},
            },
        ],
        available_tools=[
            "pg_isready", "pg_stat_activity", "netstat", "kubectl",
            "systemctl", "journalctl", "prometheus", "grafana",
            "pg_terminate_backend", "curl", "ping",
        ],
        max_steps=15,
        hints=[
            "Multiple services are affected — look for a common dependency.",
            "The database logs show connection pool at maximum capacity.",
            "Identify the root cause before treating symptoms.",
        ],
        initial_service_health={
            "postgres-primary": "down",
            "api-gateway": "degraded",
            "web-frontend": "degraded",
        },
        service_health_after_remediation={
            "postgres-primary": "healthy",
            "api-gateway": "healthy",
            "web-frontend": "healthy",
        },
    ),
    "hard": TaskConfig(
        name="hard",
        difficulty="hard",
        incident_title="Multi-Region Network Partition",
        severity="P1",
        affected_services=[
            "us-east-api",
            "eu-west-api",
            "global-load-balancer",
            "shared-postgres",
        ],
        initial_alerts=[
            "cross_region_latency_spike",
            "split_brain_detected",
            "load_balancer_health_check_fail",
            "data_replication_lag_critical",
        ],
        initial_log_snippet=(
            "global-load-balancer[3456]: ERROR: health check failed for eu-west-api (timeout=10s)\n"
            "shared-postgres[7890]: WARN: replication lag=45s (threshold=5s)\n"
            "shared-postgres[7890]: ERROR: split-brain detected — both regions accepting writes\n"
            "us-east-api[1111]: INFO: serving traffic normally\n"
            "eu-west-api[2222]: ERROR: cannot reach us-east-api, partition detected"
        ),
        root_cause="network partition between us-east and eu-west causing split-brain in shared database",
        correct_diagnostic_tools=[
            "traceroute", "ping", "dig", "pg_replication_status",
        ],
        correct_remediation_sequence=[
            {
                "action_type": "check_connectivity",
                "target": "global-load-balancer",
                "tool_used": "traceroute",
            },
            {
                "action_type": "diagnose",
                "target": "shared-postgres",
                "tool_used": "pg_replication_status",
            },
            {
                "action_type": "analyze_root_cause",
                "target": "shared-postgres",
                "tool_used": "pg_replication_status",
            },
            {
                "action_type": "remediate",
                "target": "shared-postgres",
                "tool_used": "pg_set_read_only",
                "parameters": {"action": "set_eu_west_read_only"},
            },
            {
                "action_type": "remediate",
                "target": "global-load-balancer",
                "tool_used": "kubectl",
                "parameters": {"action": "drain_eu_west_traffic"},
            },
            {
                "action_type": "remediate",
                "target": "shared-postgres",
                "tool_used": "pg_rewind",
                "parameters": {"action": "resync_replication"},
            },
            {
                "action_type": "check_metrics",
                "target": "global-load-balancer",
                "tool_used": "prometheus",
                "parameters": {"action": "verify_health"},
            },
        ],
        available_tools=[
            "traceroute", "ping", "dig", "pg_replication_status",
            "kubectl", "prometheus", "grafana", "pg_set_read_only",
            "pg_rewind", "systemctl", "journalctl", "netstat",
        ],
        max_steps=20,
        hints=[
            "Cross-region alerts suggest a network-level issue.",
            "Split-brain is dangerous — avoid destructive writes.",
            "Safe remediation: make one region read-only first.",
            "Verify recovery after remediation steps.",
        ],
        initial_service_health={
            "us-east-api": "healthy",
            "eu-west-api": "down",
            "global-load-balancer": "degraded",
            "shared-postgres": "degraded",
        },
        service_health_after_remediation={
            "us-east-api": "healthy",
            "eu-west-api": "healthy",
            "global-load-balancer": "healthy",
            "shared-postgres": "healthy",
        },
    ),
}


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

REWARD_TABLE: Dict[str, float] = {
    "correct_severity_classification": 0.15,
    "correct_root_cause_identification": 0.25,
    "successful_remediation": 0.35,
    "correct_tool_selection": 0.10,
    "wasted_step": -0.08,
    "unnecessary_escalation": -0.10,
    "failed_critical_escalation": -0.30,
    "time_penalty_per_step": -0.02,
    "episode_completion_bonus": 0.15,
    "partial_resolution": 0.10,
    "correct_diagnostic_sequence": 0.10,
    "safe_action_on_p1": 0.05,
    "destructive_action_penalty": -0.20,
    "verification_step": 0.05,
}


def _clamp_reward(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_reward(
    action: ResilienceOpsAction,
    task: TaskConfig,
    state: "IncidentEnvState",
) -> Tuple[float, str]:
    """Compute reward for a single step.

    Returns (reward_delta, result_description).
    """
    reward = 0.0
    result_parts: List[str] = []

    # Time penalty every step
    reward += REWARD_TABLE["time_penalty_per_step"]

    action_type = action.action_type.lower()
    target = action.target.lower()
    tool = action.tool_used.lower()

    # --- Correct tool selection for diagnosis ---
    if action_type in ("diagnose", "check_metrics", "check_connectivity", "query_logs"):
        if tool in [t.lower() for t in task.correct_diagnostic_tools]:
            reward += REWARD_TABLE["correct_tool_selection"]
            result_parts.append(f"Correct diagnostic tool '{tool}' selected")
        elif tool:
            reward += REWARD_TABLE["wasted_step"]
            result_parts.append(f"Suboptimal tool '{tool}' for diagnosis")

    # --- Root cause identification ---
    if action_type == "analyze_root_cause" and not state.root_cause_identified:
        if target in [s.lower() for s in task.affected_services]:
            state.root_cause_identified = True
            reward += REWARD_TABLE["correct_root_cause_identification"]
            result_parts.append("Root cause correctly identified")
        else:
            reward += REWARD_TABLE["wasted_step"]
            result_parts.append("Incorrect root cause target")

    # --- Remediation ---
    if action_type == "remediate":
        seq = task.correct_remediation_sequence
        # Check if this remediation matches an expected step
        matched = False
        for expected in seq:
            if (
                expected.get("action_type", "").lower() == action_type
                and expected.get("target", "").lower() == target
            ):
                matched = True
                break
        if matched:
            reward += REWARD_TABLE["successful_remediation"] * 0.5
            result_parts.append(f"Valid remediation on '{target}'")
            if target in state.service_health:
                state.service_health[target] = "healthy"
            # Cascading recovery: fix all services that depend on this root cause
            for svc in state.service_health:
                if state.service_health[svc] != "healthy":
                    state.service_health[svc] = "healthy"
                    result_parts.append(f"Cascading recovery: '{svc}' restored")
        else:
            reward += REWARD_TABLE["wasted_step"]
            result_parts.append(f"Remediation on '{target}' not in expected sequence")

    # --- Restart service ---
    if action_type == "restart_service":
        if task.severity == "P1" and "postgres" in target:
            reward += REWARD_TABLE["destructive_action_penalty"]
            result_parts.append(
                "DANGEROUS: Restarting database during P1 incident — data loss risk"
            )
        elif target in [s.lower() for s in task.affected_services]:
            reward += REWARD_TABLE["successful_remediation"] * 0.3
            result_parts.append(f"Service '{target}' restarted")
            if target in state.service_health:
                state.service_health[target] = "healthy"
        else:
            reward += REWARD_TABLE["wasted_step"]
            result_parts.append(f"Restart target '{target}' not in affected services")

    # --- Scale up ---
    if action_type == "scale_up":
        if target in [s.lower() for s in task.affected_services]:
            reward += REWARD_TABLE["successful_remediation"] * 0.3
            result_parts.append(f"Scale-up initiated for '{target}'")
            if target in state.service_health:
                state.service_health[target] = "healthy"
            # Cascading recovery
            for svc in state.service_health:
                if state.service_health[svc] != "healthy":
                    state.service_health[svc] = "healthy"
                    result_parts.append(f"Cascading recovery: '{svc}' restored")
        else:
            reward += REWARD_TABLE["wasted_step"]
            result_parts.append(f"Scale-up target '{target}' not applicable")

    # --- Escalate ---
    if action_type == "escalate":
        if task.severity in ("P2", "P1"):
            reward += REWARD_TABLE["unnecessary_escalation"]
            result_parts.append("Unnecessary escalation — agent should attempt remediation first")
        else:
            reward += REWARD_TABLE["failed_critical_escalation"]
            result_parts.append("Failed to escalate critical incident")

    # --- Rollback ---
    if action_type == "rollback":
        if task.severity == "P1" and "postgres" in target:
            reward += REWARD_TABLE["destructive_action_penalty"]
            result_parts.append(
                "DANGEROUS: Rollback on database during P1 — potential data loss"
            )
        else:
            reward += REWARD_TABLE["partial_resolution"]
            result_parts.append(f"Rollback on '{target}'")

    # --- Verification step ---
    if action_type == "check_metrics" and state.root_cause_identified:
        reward += REWARD_TABLE["verification_step"]
        result_parts.append("Verification step after root cause identification")

    # --- Safe action on P1 ---
    if task.severity == "P1" and action_type in (
        "check_connectivity", "diagnose", "analyze_root_cause", "check_metrics"
    ):
        reward += REWARD_TABLE["safe_action_on_p1"]
        result_parts.append("Safe diagnostic action on P1 incident")

    # --- Check if all services are healthy ---
    all_healthy = all(
        v == "healthy" for v in state.service_health.values()
    ) if state.service_health else False

    if all_healthy and not state.all_services_restored:
        state.all_services_restored = True
        reward += REWARD_TABLE["episode_completion_bonus"]
        result_parts.append("All services restored to healthy state")

    result_str = "; ".join(result_parts) if result_parts else "No significant outcome"
    return _clamp_reward(max(0.0, reward)), result_str


# ---------------------------------------------------------------------------
# Grader Logic
# ---------------------------------------------------------------------------


def grade_episode(
    task: TaskConfig,
    action_history: List[ResilienceOpsAction],
    service_health: Dict[str, str],
    root_cause_identified: bool,
    steps_taken: int,
    max_steps: int,
) -> float:
    """Grade a completed episode on a 0.0–1.0 scale.

    Grading criteria (weighted):
    - Severity awareness (implicit in correct actions): 10%
    - Root cause identification: 25%
    - Correct tool usage: 20%
    - Remediation success (services restored): 30%
    - Efficiency (fewer steps = better): 15%
    """
    score = 0.0

    # Root cause identification (25%)
    if root_cause_identified:
        score += 0.25

    # Correct tool usage (20%)
    correct_tools = set(t.lower() for t in task.correct_diagnostic_tools)
    tools_used = set(
        a.tool_used.lower()
        for a in action_history
        if a.tool_used and a.action_type.lower() in (
            "diagnose", "check_metrics", "check_connectivity", "query_logs",
        )
    )
    if correct_tools:
        tool_accuracy = len(tools_used & correct_tools) / len(correct_tools)
        score += 0.20 * tool_accuracy

    # Remediation success (30%)
    all_healthy = all(v == "healthy" for v in service_health.values()) if service_health else False
    if all_healthy:
        score += 0.30
    else:
        healthy_count = sum(1 for v in service_health.values() if v == "healthy")
        total_count = len(service_health)
        if total_count > 0:
            score += 0.30 * (healthy_count / total_count)

    # Efficiency (15%)
    if max_steps > 0:
        efficiency = max(0.0, 1.0 - (steps_taken / max_steps))
        score += 0.15 * efficiency

    # Action quality — penalize destructive actions (deducted from total)
    destructive_count = 0
    for a in action_history:
        if a.action_type.lower() in ("rollback", "restart_service"):
            if "postgres" in a.target.lower() and task.severity == "P1":
                destructive_count += 1
    score -= 0.10 * destructive_count

    # Unnecessary escalations
    escalation_count = sum(
        1 for a in action_history if a.action_type.lower() == "escalate"
    )
    score -= 0.05 * escalation_count

    return _clamp_reward(score)


# ---------------------------------------------------------------------------
# Internal State
# ---------------------------------------------------------------------------


@dataclass
class IncidentEnvState:
    """Mutable state for the incident environment."""

    episode_id: str = ""
    task: Optional[TaskConfig] = None
    step_count: int = 0
    root_cause_identified: bool = False
    all_services_restored: bool = False
    service_health: Dict[str, str] = field(default_factory=dict)
    action_history: List[ResilienceOpsAction] = field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ResilienceOpsEnvironment(Environment):
    """IT Incident Response environment for OpenEnv.

    Agents triage, diagnose, and resolve simulated infrastructure incidents
    under time pressure. Three difficulty levels:
    - Easy: single-service outage
    - Medium: cascading database failure
    - Hard: multi-region network partition
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = "easy"):
        self._task_name = task_name if task_name in TASKS else "easy"
        self._state: IncidentEnvState = IncidentEnvState()
        self._reset()

    # -- OpenEnv interface ---------------------------------------------------

    def reset(self) -> ResilienceOpsObservation:
        self._reset()
        return self._build_observation("Environment reset. New incident received.")

    def step(self, action: ResilienceOpsAction) -> ResilienceOpsObservation:  # type: ignore[override]
        task = self._state.task
        if task is None:
            return self._build_observation("No active incident. Call reset() first.")

        if self._state.done:
            return self._build_observation("Episode already finished. Call reset() for a new incident.")

        self._state.step_count += 1
        self._state.action_history.append(action)

        reward_delta, result_desc = compute_reward(action, task, self._state)
        self._state.cumulative_reward += reward_delta

        # Check termination
        done = False
        if self._state.all_services_restored:
            done = True
        elif self._state.step_count >= task.max_steps:
            done = True

        self._state.done = done

        obs = self._build_observation(result_desc)
        obs.reward = reward_delta
        obs.done = done
        return obs

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
        )

    # -- Internal helpers ----------------------------------------------------

    def _reset(self) -> None:
        task = TASKS[self._task_name]
        self._state = IncidentEnvState(
            episode_id=str(uuid4()),
            task=task,
            step_count=0,
            root_cause_identified=False,
            all_services_restored=False,
            service_health=dict(task.initial_service_health),
            action_history=[],
            cumulative_reward=0.0,
            done=False,
        )

    def _build_observation(self, previous_action_result: str) -> ResilienceOpsObservation:
        task = self._state.task
        if task is None:
            return ResilienceOpsObservation(
                previous_action_result=previous_action_result,
                done=True,
            )

        steps_remaining = max(0, task.max_steps - self._state.step_count)

        return ResilienceOpsObservation(
            incident_id=self._state.episode_id[:8],
            incident_title=task.incident_title,
            severity=task.severity,
            affected_services=list(task.affected_services),
            alert_signals=list(task.initial_alerts),
            log_snippet=task.initial_log_snippet,
            available_tools=list(task.available_tools),
            steps_remaining=steps_remaining,
            step_count=self._state.step_count,
            previous_action_result=previous_action_result,
            service_health=dict(self._state.service_health),
            task_name=task.difficulty,
            root_cause_identified=self._state.root_cause_identified,
            hints=list(task.hints),
            done=self._state.done,
            reward=self._state.cumulative_reward,
        )

    def get_final_grade(self) -> float:
        """Return the final graded score for the episode (0.0–1.0)."""
        if self._state.task is None:
            return 0.0
        return grade_episode(
            task=self._state.task,
            action_history=self._state.action_history,
            service_health=self._state.service_health,
            root_cause_identified=self._state.root_cause_identified,
            steps_taken=self._state.step_count,
            max_steps=self._state.task.max_steps,
        )

    def set_task(self, task_name: str) -> None:
        """Switch the active task difficulty."""
        if task_name in TASKS:
            self._task_name = task_name
