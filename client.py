# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resilience Ops Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ResilienceOpsAction, ResilienceOpsObservation


class ResilienceOpsEnv(
    EnvClient[ResilienceOpsAction, ResilienceOpsObservation, State]
):
    """
    Client for the Resilience Ops Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with ResilienceOpsEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.incident_title)
        ...
        ...     result = client.step(ResilienceOpsAction(
        ...         action_type="diagnose",
        ...         target="api-gateway",
        ...         tool_used="top",
        ...     ))
        ...     print(result.observation.previous_action_result)
    """

    def _step_payload(self, action: ResilienceOpsAction) -> Dict:
        return {
            "action_type": action.action_type,
            "target": action.target,
            "tool_used": action.tool_used,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ResilienceOpsObservation]:
        obs_data = payload.get("observation", {})
        observation = ResilienceOpsObservation(
            incident_id=obs_data.get("incident_id", ""),
            incident_title=obs_data.get("incident_title", ""),
            severity=obs_data.get("severity", ""),
            affected_services=obs_data.get("affected_services", []),
            alert_signals=obs_data.get("alert_signals", []),
            log_snippet=obs_data.get("log_snippet", ""),
            available_tools=obs_data.get("available_tools", []),
            steps_remaining=obs_data.get("steps_remaining", 0),
            step_count=obs_data.get("step_count", 0),
            previous_action_result=obs_data.get("previous_action_result", ""),
            service_health=obs_data.get("service_health", {}),
            task_name=obs_data.get("task_name", ""),
            root_cause_identified=obs_data.get("root_cause_identified", False),
            hints=obs_data.get("hints", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
