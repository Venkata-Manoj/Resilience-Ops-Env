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

import random
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator


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
        max_length=128,
        description=(
            "Type of action to take. One of: "
            "diagnose, remediate, escalate, query_logs, check_metrics, "
            "restart_service, rollback, scale_up, check_connectivity, analyze_root_cause"
        ),
    )
    target: str = Field(
        default="",
        max_length=256,
        description="Service/component to act on (e.g. 'api-gateway', 'postgres-primary')",
    )
    tool_used: str = Field(
        default="",
        max_length=256,
        description="Specific diagnostic or remediation tool (e.g. 'top', 'pg_isready', 'kubectl')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters as key-value pairs",
    )

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters to prevent injection and DoS."""
        if len(v) > 10:
            raise ValueError("parameters dict must have at most 10 keys")
        sanitized = {}
        for key, val in v.items():
            k = str(key)[:64]
            sanitized[k] = str(val)[:256] if val is not None else ""
        return sanitized


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
    # v3.0 New Tasks
    "k8s_crashloop": TaskConfig(
        name="k8s_crashloop",
        difficulty="medium+",
        incident_title="Kubernetes Pod CrashLoopBackOff",
        severity="P2",
        affected_services=["payment-service", "payment-worker", "redis-cache"],
        initial_alerts=[
            "pod_crashloop_backoff",
            "memory_pressure",
            "oom_kills_detected",
        ],
        initial_log_snippet=(
            "payment-worker[abc12]: FATAL: OutOfMemoryError: Java heap space\n"
            "kubelet[node-3]: WARN: Pod payment-worker exceeded memory limit (1Gi/1Gi)\n"
            "payment-service[def34]: ERROR: Connection refused to payment-worker\n"
            "prometheus: ALERT: container_memory_working_set_bytes > 95% limit"
        ),
        root_cause="payment-worker memory leak causing OOM kills and cascading failures",
        correct_diagnostic_tools=["kubectl", "prometheus", "docker_stats", "top"],
        correct_remediation_sequence=[
            {"action_type": "diagnose", "target": "payment-worker", "tool_used": "kubectl"},
            {"action_type": "check_metrics", "target": "payment-worker", "tool_used": "prometheus"},
            {"action_type": "analyze_root_cause", "target": "payment-worker", "tool_used": "docker_stats"},
            {"action_type": "remediate", "target": "payment-worker", "tool_used": "kubectl", "parameters": {"action": "increase_memory_limit"}},
            {"action_type": "restart_service", "target": "payment-worker", "tool_used": "kubectl"},
            {"action_type": "check_metrics", "target": "payment-service", "tool_used": "prometheus"},
        ],
        available_tools=[
            "kubectl", "prometheus", "docker_stats", "top", "journalctl",
            "netstat", "heap_dump", "curl",
        ],
        max_steps=15,
        hints=[
            "Look for OOM kills in the pod events.",
            "Check memory usage trends before the crashes.",
            "The worker pod may need more memory allocated.",
        ],
        initial_service_health={
            "payment-service": "degraded",
            "payment-worker": "down",
            "redis-cache": "healthy",
        },
        service_health_after_remediation={
            "payment-service": "healthy",
            "payment-worker": "healthy",
            "redis-cache": "healthy",
        },
    ),
    "security_lateral": TaskConfig(
        name="security_lateral",
        difficulty="hard",
        incident_title="Security Incident: Lateral Movement Detection",
        severity="P1",
        affected_services=["auth-service", "api-gateway", "internal-db", "jump-host"],
        initial_alerts=[
            "suspicious_ssh_connections",
            "auth_anomaly_detected",
            "privilege_escalation_attempt",
            "unusual_data_access_pattern",
        ],
        initial_log_snippet=(
            "auth-audit[1234]: ALERT: Multiple failed sudo attempts from 10.0.3.45\n"
            "auth-service[5678]: WARN: Login from unknown IP range (10.0.3.x)\n"
            "api-gateway[9012]: ERROR: Unusual API key usage pattern detected\n"
            "jump-host[3456]: CRITICAL: SSH session from auth-service to internal-db\n"
            "internal-db[7890]: WARN: SELECT * FROM users executed by non-app user"
        ),
        root_cause="compromised jump-host being used for lateral movement to internal systems",
        correct_diagnostic_tools=["audit_log", "netstat", "ss", "last"],
        correct_remediation_sequence=[
            {"action_type": "check_connectivity", "target": "jump-host", "tool_used": "netstat"},
            {"action_type": "query_logs", "target": "auth-service", "tool_used": "audit_log"},
            {"action_type": "analyze_root_cause", "target": "jump-host", "tool_used": "last"},
            {"action_type": "remediate", "target": "jump-host", "tool_used": "ss", "parameters": {"action": "isolate_node"}},
            {"action_type": "query_logs", "target": "internal-db", "tool_used": "audit_log"},
            {"action_type": "remediate", "target": "auth-service", "tool_used": "kubectl", "parameters": {"action": "revoke_sessions"}},
            {"action_type": "check_metrics", "target": "api-gateway", "tool_used": "prometheus"},
        ],
        available_tools=[
            "audit_log", "netstat", "ss", "last", "kubectl", "prometheus",
            "journalctl", "lsof", "tcpdump", "iptables",
        ],
        max_steps=18,
        hints=[
            "SSH connections between internal services are suspicious.",
            "Check audit logs for privilege escalation attempts.",
            "Isolate compromised nodes before investigating further.",
            "Verify legitimate traffic is not disrupted.",
        ],
        initial_service_health={
            "auth-service": "degraded",
            "api-gateway": "degraded",
            "internal-db": "degraded",
            "jump-host": "down",
        },
        service_health_after_remediation={
            "auth-service": "healthy",
            "api-gateway": "healthy",
            "internal-db": "healthy",
            "jump-host": "healthy",
        },
    ),
    "dns_failure": TaskConfig(
        name="dns_failure",
        difficulty="easy+",
        incident_title="DNS Resolution Failure Chain",
        severity="P3",
        affected_services=["frontend-app", "api-gateway", "dns-resolver-us", "dns-resolver-eu"],
        initial_alerts=[
            "dns_resolution_timeout",
            "intermittent_5xx_errors",
            "cache_poisoning_detected",
        ],
        initial_log_snippet=(
            "frontend-app[1234]: ERROR: getaddrinfo EAI_AGAIN api.example.com\n"
            "dns-resolver-us[5678]: WARN: Stale entry for api.example.com (TTL exceeded by 3600s)\n"
            "api-gateway[9012]: ERROR: Unable to resolve auth-service.internal\n"
            "dns-resolver-eu[3456]: INFO: Serving stale entry for api.example.com\n"
            "frontend-app[7890]: WARN: Fallback DNS also failing"
        ),
        root_cause="DNS cache poisoning with stale entries causing intermittent resolution failures",
        correct_diagnostic_tools=["dig", "nslookup", "tcpdump", "systemctl"],
        correct_remediation_sequence=[
            {"action_type": "diagnose", "target": "dns-resolver-us", "tool_used": "dig"},
            {"action_type": "check_connectivity", "target": "api-gateway", "tool_used": "nslookup"},
            {"action_type": "analyze_root_cause", "target": "dns-resolver-us", "tool_used": "tcpdump"},
            {"action_type": "remediate", "target": "dns-resolver-us", "tool_used": "systemctl", "parameters": {"action": "flush_cache"}},
            {"action_type": "remediate", "target": "dns-resolver-eu", "tool_used": "systemctl", "parameters": {"action": "flush_cache"}},
            {"action_type": "check_metrics", "target": "frontend-app", "tool_used": "curl"},
        ],
        available_tools=[
            "dig", "nslookup", "tcpdump", "systemctl", "journalctl",
            "ping", "curl", "netstat", "ss",
        ],
        max_steps=12,
        hints=[
            "DNS issues often cause intermittent failures.",
            "Check both regional DNS resolvers.",
            "Flush DNS caches after identifying stale entries.",
        ],
        initial_service_health={
            "frontend-app": "degraded",
            "api-gateway": "degraded",
            "dns-resolver-us": "degraded",
            "dns-resolver-eu": "degraded",
        },
        service_health_after_remediation={
            "frontend-app": "healthy",
            "api-gateway": "healthy",
            "dns-resolver-us": "healthy",
            "dns-resolver-eu": "healthy",
        },
    ),
    "circuit_breaker_storm": TaskConfig(
        name="circuit_breaker_storm",
        difficulty="expert",
        incident_title="Cascading Microservice Circuit Breaker Storm",
        severity="P1",
        affected_services=[
            "inventory-service", "order-service", "payment-service",
            "notification-service", "slow-legacy-api",
        ],
        initial_alerts=[
            "circuit_breaker_open",
            "cascade_failure_detected",
            "high_error_rate",
            "thread_pool_exhaustion",
            "timeout_cascade",
        ],
        initial_log_snippet=(
            "inventory-service[1234]: ERROR: Circuit breaker OPEN for slow-legacy-api\n"
            "order-service[5678]: ERROR: Fallback failed, circuit breaker OPEN\n"
            "payment-service[9012]: ERROR: Timeout waiting for inventory-service\n"
            "notification-service[3456]: WARN: Thread pool saturated (500/500 active)\n"
            "slow-legacy-api[7890]: CRITICAL: Response time p99=45s (SLA=2s)\n"
            "hystrix: ALERT: Multiple services tripping circuit breakers simultaneously"
        ),
        root_cause="slow-legacy-api bottleneck causing cascading circuit breaker failures across microservices",
        correct_diagnostic_tools=["prometheus", "hystrix_dashboard", "thread_dump", "tcpdump"],
        correct_remediation_sequence=[
            {"action_type": "check_metrics", "target": "slow-legacy-api", "tool_used": "prometheus"},
            {"action_type": "diagnose", "target": "inventory-service", "tool_used": "hystrix_dashboard"},
            {"action_type": "query_logs", "target": "slow-legacy-api", "tool_used": "thread_dump"},
            {"action_type": "analyze_root_cause", "target": "slow-legacy-api", "tool_used": "tcpdump"},
            {"action_type": "remediate", "target": "slow-legacy-api", "tool_used": "kubectl", "parameters": {"action": "apply_backpressure"}},
            {"action_type": "remediate", "target": "inventory-service", "tool_used": "kubectl", "parameters": {"action": "increase_timeout"}},
            {"action_type": "remediate", "target": "order-service", "tool_used": "kubectl", "parameters": {"action": "reset_circuit_breaker"}},
            {"action_type": "scale_up", "target": "slow-legacy-api", "tool_used": "kubectl"},
            {"action_type": "check_metrics", "target": "payment-service", "tool_used": "prometheus"},
        ],
        available_tools=[
            "prometheus", "hystrix_dashboard", "thread_dump", "tcpdump",
            "kubectl", "curl", "netstat", "jstack", "lsof",
        ],
        max_steps=25,
        hints=[
            "Multiple circuit breakers opening simultaneously indicates a common bottleneck.",
            "Identify the slowest downstream dependency first.",
            "Apply backpressure before resetting circuit breakers.",
            "Scale the bottleneck service after applying fixes.",
            "Verify all services recover before completing.",
        ],
        initial_service_health={
            "inventory-service": "down",
            "order-service": "down",
            "payment-service": "degraded",
            "notification-service": "degraded",
            "slow-legacy-api": "degraded",
        },
        service_health_after_remediation={
            "inventory-service": "healthy",
            "order-service": "healthy",
            "payment-service": "healthy",
            "notification-service": "healthy",
            "slow-legacy-api": "healthy",
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
    """Clamp to strict (0, 1) — evaluator rejects exactly 0.0 and 1.0."""
    return max(0.01, min(0.99, value))


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

    # Validate action_type
    action_type = action.action_type.lower()
    if action_type not in [t.lower() for t in VALID_ACTION_TYPES]:
        reward += REWARD_TABLE["wasted_step"]
        return _clamp_reward(reward), f"Invalid action type: {action.action_type}"
    target = action.target.lower()
    tool = action.tool_used.lower()

    # --- Correct tool selection for diagnosis ---
    diagnostic_actions = ("diagnose", "check_metrics", "check_connectivity", "query_logs")
    if action_type in diagnostic_actions:
        state.diagnostic_count += 1
        if tool in [t.lower() for t in task.correct_diagnostic_tools]:
            reward += REWARD_TABLE["correct_tool_selection"]
            result_parts.append(f"Correct diagnostic tool '{tool}' selected")
            # Award correct_diagnostic_sequence for multiple correct diagnostics
            if state.diagnostic_count >= 2 and not state.diagnostic_sequence_rewarded:
                state.diagnostic_sequence_rewarded = True
                reward += REWARD_TABLE["correct_diagnostic_sequence"]
                result_parts.append("Correct diagnostic sequence")
        elif tool:
            reward += REWARD_TABLE["wasted_step"]
            result_parts.append(f"Suboptimal tool '{tool}' for diagnosis")

    # --- Severity classification (implicit from correct prioritization) ---
    if action_type == "diagnose" and not state.severity_classified:
        # Award for correctly identifying severity through prioritization
        if target in [s.lower() for s in task.affected_services[:1]]:
            state.severity_classified = True
            reward += REWARD_TABLE["correct_severity_classification"]
            result_parts.append("Severity correctly classified through prioritization")

    # --- Root cause identification ---
    if action_type == "analyze_root_cause" and not state.root_cause_identified:
        # Require at least 1 diagnostic before root cause analysis
        if state.diagnostic_count < 1:
            reward += REWARD_TABLE["wasted_step"]
            result_parts.append("Insufficient diagnostics before root cause analysis")
        elif target in [s.lower() for s in task.affected_services]:
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
            reward += REWARD_TABLE["successful_remediation"]
            result_parts.append(f"Valid remediation on '{target}'")
            if target in state.service_health:
                state.service_health[target] = "healthy"
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
            reward += REWARD_TABLE["successful_remediation"]
            result_parts.append(f"Scale-up initiated for '{target}'")
            if target in state.service_health:
                state.service_health[target] = "healthy"
        else:
            reward += REWARD_TABLE["wasted_step"]
            result_parts.append(f"Scale-up target '{target}' not applicable")

    # --- Escalate ---
    if action_type == "escalate":
        if task.severity == "P3":
            # P3 is low priority - escalation is unnecessary
            reward += REWARD_TABLE["unnecessary_escalation"]
            result_parts.append("Unnecessary escalation for P3 — agent should attempt remediation first")
        elif task.severity == "P1":
            # P1 is critical - escalation is appropriate
            result_parts.append("Appropriate escalation for critical P1 incident")
        else:
            # P2 - neutral, wasted step but not heavily penalized
            reward += REWARD_TABLE["wasted_step"]
            result_parts.append("Escalation for P2 - consider remediation first")

    # --- Rollback ---
    if action_type == "rollback":
        # Validate rollback target is in affected services
        if target not in [s.lower() for s in task.affected_services]:
            reward += REWARD_TABLE["wasted_step"]
            result_parts.append(f"Cannot rollback '{target}': not in affected services")
        elif task.severity == "P1" and "postgres" in target:
            reward += REWARD_TABLE["destructive_action_penalty"]
            result_parts.append(
                "DANGEROUS: Rollback on database during P1 — potential data loss"
            )
        else:
            reward += REWARD_TABLE["partial_resolution"]
            result_parts.append(f"Rollback on '{target}'")

    # --- Verification step (mutually exclusive with correct_tool_selection) ---
    if action_type == "check_metrics" and state.root_cause_identified:
        # Only give verification reward — do NOT also give tool selection
        # (tool selection was already conditionally awarded above)
        reward += REWARD_TABLE["verification_step"]
        result_parts.append("Verification step after root cause identification")

    # --- Safe action on P1 ---
    if task.severity == "P1" and action_type in (
        "check_connectivity", "diagnose", "analyze_root_cause", "check_metrics"
    ):
        # Only award if correct diagnostic tool is used
        if tool in [t.lower() for t in task.correct_diagnostic_tools]:
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
    return _clamp_reward(reward), result_str


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

    # Correct tool usage (20%) - use Jaccard similarity
    correct_tools = set(t.lower() for t in task.correct_diagnostic_tools)
    diagnostic_actions = ("diagnose", "check_metrics", "check_connectivity", "query_logs")
    tools_used = set(
        a.tool_used.lower()
        for a in action_history
        if a.tool_used and a.action_type.lower() in diagnostic_actions
    )
    if correct_tools or tools_used:
        union = correct_tools | tools_used
        if union:
            jaccard = len(correct_tools & tools_used) / len(union)
            score += 0.20 * jaccard

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

    # Severity awareness (10%) - NEW
    if task.severity == "P1":
        # P1: should check_connectivity first, avoid destructive actions
        action_types_used = [a.action_type.lower() for a in action_history]
        if "check_connectivity" in action_types_used[:3]:
            score += 0.05
        if not any(a.action_type == "restart_service" and "postgres" in a.target for a in action_history):
            score += 0.05
    elif task.severity == "P2":
        # P2: should identify root cause before remediation
        if root_cause_identified:
            score += 0.05
        # Reward for systematic diagnosis (at least 2 diagnostic actions)
        diag_actions = [a for a in action_history if a.action_type.lower() in
                       ("diagnose", "check_metrics", "check_connectivity", "query_logs")]
        if len(diag_actions) >= 2:
            score += 0.05
    elif task.severity == "P3":
        # P3: should be quick, direct remediation
        if steps_taken <= 4:
            score += 0.10

    # Action quality — penalize destructive actions (capped at -0.30 total)
    destructive_count = 0
    for a in action_history:
        if a.action_type.lower() in ("rollback", "restart_service"):
            if "postgres" in a.target.lower() and task.severity == "P1":
                destructive_count += 1
    escalation_count = sum(
        1 for a in action_history if a.action_type.lower() == "escalate"
    )
    penalty = 0.10 * destructive_count + 0.05 * escalation_count
    score -= min(penalty, 0.30)  # Cap total penalties at -0.30

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
    # Tracking for reward logic
    diagnostic_count: int = 0
    diagnostic_sequence_rewarded: bool = False
    severity_classified: bool = False


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

        # Dynamic alerts based on service health (EC-10)
        dynamic_alerts = []
        for service, health in self._state.service_health.items():
            if health == "healthy":
                dynamic_alerts.append(f"RESOLVED: {service} is healthy")
            elif health == "degraded":
                dynamic_alerts.append(f"WARNING: {service} is degraded")
            elif health == "down":
                dynamic_alerts.append(f"CRITICAL: {service} is DOWN")

        # Add original alerts for services not yet resolved
        for alert in task.initial_alerts:
            # Extract service name from alert (simple heuristic)
            alert_lower = alert.lower()
            matching_service = None
            for svc in task.affected_services:
                if svc.lower().replace("-", "_") in alert_lower or svc.lower().replace("-", "") in alert_lower:
                    matching_service = svc
                    break
            if matching_service and self._state.service_health.get(matching_service) != "healthy":
                dynamic_alerts.append(alert)

        # Stochastic elements for grading diversity (EC-16)
        rng = random.Random(self._state.episode_id)  # Deterministic per episode
        alert_order = dynamic_alerts.copy()
        rng.shuffle(alert_order)

        # Maybe add red herring alerts for P3/P2 tasks
        red_herrings = [
            "Monitoring: cpu_usage_spike on monitoring-server",
            "NOTICE: scheduled_maintenance window started",
            "INFO: backup_job completed successfully",
        ]
        if task.severity in ("P2", "P3") and rng.random() < 0.3:
            alert_order.insert(rng.randint(0, len(alert_order)), rng.choice(red_herrings))

        return ResilienceOpsObservation(
            incident_id=self._state.episode_id[:8],
            incident_title=task.incident_title,
            severity=task.severity,
            affected_services=list(task.affected_services),
            alert_signals=alert_order if alert_order else list(task.initial_alerts),
            log_snippet=task.initial_log_snippet,
            available_tools=list(task.available_tools),
            steps_remaining=steps_remaining,
            step_count=self._state.step_count,
            previous_action_result=previous_action_result,
            service_health=dict(self._state.service_health),
            task_name=task.name,
            root_cause_identified=self._state.root_cause_identified,
            hints=list(task.hints),
            done=self._state.done,
            reward=self._state.cumulative_reward,
        )

    def get_final_grade(self) -> float:
        """Return the final graded score for the episode (0.0–1.0)."""
        if self._state.task is None:
            return 0.01
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


# ---------------------------------------------------------------------------
# v3.0 Features — Dynamic Environments + Multi-Agent + Observability
# ---------------------------------------------------------------------------

@dataclass
class DynamicTaskConfig(TaskConfig):
    """Tasks with randomized parameters for grading diversity."""

    alert_noise: float = 0.2  # Probability of false-positive alerts
    log_corruption: float = 0.1  # Probability of truncated/garbled logs
    service_health_noise: float = 0.1  # Probability of flapping health status
    red_herring_services: List[str] = field(default_factory=list)
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
            lines = logs.split("\n")
            if len(lines) > 2 and rng.random() < 0.5:
                lines = lines[:-1]
            if rng.random() < 0.5:
                lines.insert(rng.randint(0, len(lines)), "[CORRUPTED LOG ENTRY]")
            logs = "\n".join(lines)
        return logs

    def _randomize_health(self, rng: random.Random) -> Dict[str, str]:
        """Randomize initial service health with potential flapping."""
        health = dict(self.initial_service_health)
        if rng.random() < self.service_health_noise:
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
    agent_actions: Dict[str, List[ResilienceOpsAction]] = field(
        default_factory=lambda: defaultdict(list)
    )


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
        self.shared_incident = SharedIncidentState(
            service_health=dict(task.initial_service_health),
            global_alerts=list(task.initial_alerts),
        )
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
            obs = self._build_observation_for_agent(
                agent_id, "Environment reset. New incident received."
            )
            observations[agent_id] = obs
        return observations

    def multi_agent_step(
        self, agent_id: str, action: ResilienceOpsAction
    ) -> ResilienceOpsObservation:
        """Execute action from one agent, affecting shared state visible to all."""
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        state = self.agents[agent_id]
        task = state.task
        if task is None:
            return ResilienceOpsObservation(
                previous_action_result="No active incident.", done=True
            )
        if state.done:
            return ResilienceOpsObservation(
                previous_action_result="Episode finished.", done=True
            )
        state.step_count += 1
        state.action_history.append(action)
        self.shared_incident.agent_actions[agent_id].append(action)
        reward_delta, result_desc = compute_reward(action, task, state)
        state.cumulative_reward += reward_delta
        for svc, health in state.service_health.items():
            if health == "healthy":
                self.shared_incident.service_health[svc] = health
        if state.root_cause_identified:
            self.shared_incident.root_cause_identified = True
        all_healthy = all(
            v == "healthy" for v in self.shared_incident.service_health.values()
        )
        if all_healthy:
            self.shared_incident.all_services_restored = True
            state.all_services_restored = True
            state.done = True
        elif state.step_count >= task.max_steps:
            state.done = True
        return self._build_observation_for_agent(agent_id, result_desc)

    def _build_observation_for_agent(
        self, agent_id: str, previous_action_result: str
    ) -> ResilienceOpsObservation:
        """Build observation for a specific agent with shared state."""
        state = self.agents.get(agent_id)
        task = state.task if state else None
        if task is None:
            return ResilienceOpsObservation(
                previous_action_result=previous_action_result, done=True
            )
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
            task_name=task.name,
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

    episodes_completed: int = 0
    episodes_failed: int = 0
    total_rewards: List[float] = field(default_factory=list)
    mean_reward_per_step: List[float] = field(default_factory=list)
    action_type_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    root_cause_identification_count: int = 0
    total_episodes: int = 0
    time_to_resolution: List[int] = field(default_factory=list)
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
        if rewards:
            self.total_rewards.append(sum(rewards))
            self.mean_reward_per_step.append(sum(rewards) / len(rewards))
        for action in action_history:
            self.action_type_counts[action.action_type] += 1
        if root_cause_identified:
            self.root_cause_identification_count += 1
        if success:
            self.time_to_resolution.append(steps_taken)
        for action in action_history:
            if action.action_type in ("rollback", "restart_service") and "postgres" in action.target.lower():
                self.destructive_action_count += 1
            if action.action_type == "escalate":
                self.escalation_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        import statistics

        summary: Dict[str, Any] = {
            "episodes_completed": self.episodes_completed,
            "episodes_failed": self.episodes_failed,
            "root_cause_identification_rate": (
                self.root_cause_identification_count / self.total_episodes
                if self.total_episodes > 0
                else 0.0
            ),
            "action_type_distribution": dict(self.action_type_counts),
            "destructive_action_rate": (
                self.destructive_action_count / sum(self.action_type_counts.values())
                if self.action_type_counts
                else 0.0
            ),
            "escalation_rate": (
                self.escalation_count / sum(self.action_type_counts.values())
                if self.action_type_counts
                else 0.0
            ),
        }
        if self.total_rewards:
            summary["mean_total_reward"] = statistics.mean(self.total_rewards)
            summary["std_total_reward"] = (
                statistics.stdev(self.total_rewards)
                if len(self.total_rewards) > 1
                else 0.0
            )
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
        lines.append(
            f'resilience_ops_episodes_total{{status="completed"}} {self.episodes_completed}'
        )
        lines.append(
            f'resilience_ops_episodes_total{{status="failed"}} {self.episodes_failed}'
        )
        lines.append(
            "# HELP resilience_ops_root_cause_identified_total Total root cause identifications"
        )
        lines.append("# TYPE resilience_ops_root_cause_identified_total counter")
        lines.append(
            f"resilience_ops_root_cause_identified_total {self.root_cause_identification_count}"
        )
        lines.append("# HELP resilience_ops_action_total Total actions by type")
        lines.append("# TYPE resilience_ops_action_total counter")
        for action_type, count in self.action_type_counts.items():
            lines.append(f'resilience_ops_action_total{{type="{action_type}"}} {count}')
        lines.append("# HELP resilience_ops_time_to_resolution Steps to resolution")
        lines.append("# TYPE resilience_ops_time_to_resolution histogram")
        for ttr in self.time_to_resolution:
            lines.append(
                f'resilience_ops_time_to_resolution_bucket{{le="{ttr}"}} 1'
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-as-Judge Grading
# ---------------------------------------------------------------------------

class LLMJudgeGrader:
    """Supplement rule-based grading with LLM evaluation of reasoning quality."""

    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name

    def grade_reasoning(
        self,
        action_history: List[ResilienceOpsAction],
        task: TaskConfig,
    ) -> float:
        """Evaluate whether the agent's diagnostic reasoning was sound."""
        action_summary = self._format_actions(action_history)
        prompt = (
            f"Evaluate this SRE agent's incident response reasoning:\n\n"
            f"Incident: {task.incident_title} (Severity: {task.severity})\n"
            f"Root Cause: {task.root_cause}\n"
            f"Affected Services: {', '.join(task.affected_services)}\n\n"
            f"Agent Actions:\n{action_summary}\n\n"
            f"Score the reasoning quality from 0.0 to 1.0 based on:\n"
            f"1. Did the agent follow a logical diagnostic sequence? (0.25)\n"
            f"2. Did the agent prioritize correctly given the severity? (0.25)\n"
            f"3. Did the agent avoid destructive actions? (0.25)\n"
            f"4. Was the remediation sequence appropriate? (0.25)\n\n"
            f"Respond with ONLY a float between 0.0 and 1.0."
        )
        try:
            return self._call_judge_model(prompt)
        except Exception:
            return 0.5

    def _format_actions(self, action_history: List[ResilienceOpsAction]) -> str:
        """Format action history for the judge prompt."""
        lines = []
        for i, action in enumerate(action_history, 1):
            lines.append(
                f"  {i}. {action.action_type}(target={action.target}, tool={action.tool_used})"
            )
        return "\n".join(lines) if lines else "  (no actions taken)"

    def _call_judge_model(self, prompt: str) -> float:
        """Call the LLM judge model."""
        import re
        try:
            import os
            from openai import OpenAI

            api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
            base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
            client = OpenAI(api_key=api_key, base_url=base_url)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SRE evaluator. Score reasoning quality objectively.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            content = response.choices[0].message.content.strip()
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

    def compute_rewards(
        self, prompts: List[str], completions: List[str]
    ) -> List[float]:
        """Given (prompt, completion) pairs, return rewards by executing completions."""
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                action = self._parse_completion_to_action(completion)
                obs = self.env.step(action)
                if obs.done:
                    reward = self.env.get_final_grade()
                else:
                    reward = obs.reward
                rewards.append(reward)
            except Exception:
                rewards.append(0.0)
        return rewards

    def _parse_completion_to_action(self, completion: str) -> ResilienceOpsAction:
        """Parse a text completion into a ResilienceOpsAction."""
        import json

        try:
            data = json.loads(completion)
            return ResilienceOpsAction(
                action_type=data.get("action_type", "diagnose"),
                target=data.get("target", ""),
                tool_used=data.get("tool_used", ""),
                parameters=data.get("parameters", {}),
            )
        except json.JSONDecodeError:
            return ResilienceOpsAction(
                action_type="diagnose",
                target="",
                tool_used="",
                parameters={},
            )

