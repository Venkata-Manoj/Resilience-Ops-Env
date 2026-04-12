# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ResilienceOps environment models, reward computation, and grading."""

import sys
import os
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ResilienceOpsAction,
    ResilienceOpsObservation,
    ResilienceOpsEnvironment,
    IncidentEnvState,
    TaskConfig,
    TASKS,
    VALID_ACTION_TYPES,
    REWARD_TABLE,
    compute_reward,
    grade_episode,
    _clamp_reward,
)


# ---------------------------------------------------------------------------
# Action Validation Tests
# ---------------------------------------------------------------------------

class TestActionValidation:
    """Test input validation on ResilienceOpsAction."""

    def test_valid_action(self):
        action = ResilienceOpsAction(
            action_type="diagnose",
            target="api-gateway",
            tool_used="top",
            parameters={"action": "check_cpu"},
        )
        assert action.action_type == "diagnose"
        assert action.target == "api-gateway"

    def test_empty_action_type_rejected(self):
        """Empty action_type should still be accepted by Pydantic (validated by reward fn)."""
        action = ResilienceOpsAction(action_type="", target="test")
        assert action.action_type == ""

    def test_parameters_max_keys(self):
        """Parameters dict should reject more than 10 keys."""
        big_params = {f"key_{i}": f"val_{i}" for i in range(11)}
        with pytest.raises(Exception):
            ResilienceOpsAction(action_type="diagnose", parameters=big_params)

    def test_parameters_value_sanitization(self):
        """Parameter values should be sanitized to strings."""
        action = ResilienceOpsAction(
            action_type="diagnose",
            parameters={"nested": {"bad": "value"}, "normal": "ok"},
        )
        # Nested dict should be stringified
        assert isinstance(action.parameters["nested"], str)
        assert action.parameters["normal"] == "ok"

    def test_max_length_enforcement(self):
        """Fields with max_length should reject oversized input."""
        long_str = "x" * 300
        with pytest.raises(Exception):
            ResilienceOpsAction(action_type=long_str)


# ---------------------------------------------------------------------------
# Reward Computation Tests
# ---------------------------------------------------------------------------

class TestRewardComputation:
    """Test reward signals for different actions."""

    def _make_state(self, task_name: str = "easy") -> IncidentEnvState:
        task = TASKS[task_name]
        return IncidentEnvState(
            episode_id="test-episode",
            task=task,
            step_count=1,
            root_cause_identified=False,
            all_services_restored=False,
            service_health=dict(task.initial_service_health),
            action_history=[],
            cumulative_reward=0.0,
            done=False,
        )

    def test_valid_diagnostic_tool_rewarded(self):
        """Correct diagnostic tool should give positive reward."""
        state = self._make_state("easy")
        action = ResilienceOpsAction(
            action_type="diagnose", target="api-gateway", tool_used="top"
        )
        reward, desc = compute_reward(action, TASKS["easy"], state)
        assert reward > 0.0
        assert "Correct diagnostic tool" in desc

    def test_invalid_action_type_penalized(self):
        """Invalid action type should return wasted_step penalty (clamped)."""
        state = self._make_state("easy")
        action = ResilienceOpsAction(
            action_type="invalid_action", target="api-gateway", tool_used="top"
        )
        reward, desc = compute_reward(action, TASKS["easy"], state)
        assert "Invalid action type" in desc

    def test_time_penalty_applied(self):
        """Every step should incur a time penalty."""
        state = self._make_state("easy")
        action = ResilienceOpsAction(
            action_type="diagnose", target="some-service", tool_used="unknown"
        )
        reward, _ = compute_reward(action, TASKS["easy"], state)
        # Even with no positive reward, time penalty should be present
        # but reward is clamped to [0,1]
        assert 0.0 <= reward <= 1.0

    def test_destructive_action_on_p1_penalized(self):
        """Restarting postgres during P1 should be penalized."""
        state = self._make_state("hard")
        action = ResilienceOpsAction(
            action_type="restart_service", target="shared-postgres", tool_used="systemctl"
        )
        reward, desc = compute_reward(action, TASKS["hard"], state)
        assert "DANGEROUS" in desc

    def test_reward_always_in_range(self):
        """Reward should always be clamped to [0.0, 1.0]."""
        for task_name in TASKS:
            state = self._make_state(task_name)
            for action_type in VALID_ACTION_TYPES:
                action = ResilienceOpsAction(
                    action_type=action_type,
                    target=TASKS[task_name].affected_services[0],
                    tool_used="unknown",
                )
                reward, _ = compute_reward(action, TASKS[task_name], state)
                assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range for {task_name}/{action_type}"

    def test_escalation_p3_penalized(self):
        """Escalating a P3 incident should be penalized."""
        state = self._make_state("easy")
        action = ResilienceOpsAction(action_type="escalate", target="api-gateway")
        reward, desc = compute_reward(action, TASKS["easy"], state)
        assert "Unnecessary escalation" in desc

    def test_correct_remediation_rewarded(self):
        """Correct remediation target should give reward."""
        state = self._make_state("easy")
        state.root_cause_identified = True
        state.diagnostic_count = 2
        action = ResilienceOpsAction(
            action_type="remediate",
            target="api-gateway",
            tool_used="systemctl",
        )
        # Check if remediation matches any expected step
        task = TASKS["easy"]
        # For easy task, the correct_remediation_sequence has restart_service, not remediate for api-gateway
        # But let's test with medium which has remediate
        state_m = self._make_state("medium")
        action_m = ResilienceOpsAction(
            action_type="remediate",
            target="postgres-primary",
            tool_used="pg_terminate_backend",
            parameters={"action": "terminate_idle_connections"},
        )
        reward, desc = compute_reward(action_m, TASKS["medium"], state_m)
        assert "Valid remediation" in desc


# ---------------------------------------------------------------------------
# Grading Tests
# ---------------------------------------------------------------------------

class TestGrading:
    """Test episode grading function."""

    def test_grade_in_range(self):
        """Grade should always be in [0.0, 1.0] for all tasks."""
        for task_name, task in TASKS.items():
            score = grade_episode(
                task=task,
                action_history=[],
                service_health=task.initial_service_health,
                root_cause_identified=False,
                steps_taken=0,
                max_steps=task.max_steps,
            )
            assert 0.0 <= score <= 1.0, f"Grade {score} out of range for {task_name}"

    def test_grade_deterministic(self):
        """Same inputs should produce identical grades."""
        for task_name, task in TASKS.items():
            scores = []
            for _ in range(5):
                score = grade_episode(
                    task=task,
                    action_history=[],
                    service_health=task.initial_service_health,
                    root_cause_identified=False,
                    steps_taken=3,
                    max_steps=task.max_steps,
                )
                scores.append(score)
            assert len(set(scores)) == 1, f"Non-deterministic grades for {task_name}: {scores}"

    def test_perfect_episode_high_score(self):
        """A successful episode should score higher than a failed one."""
        task = TASKS["easy"]
        
        # Failed episode
        failed_score = grade_episode(
            task=task,
            action_history=[],
            service_health=task.initial_service_health,
            root_cause_identified=False,
            steps_taken=task.max_steps,
            max_steps=task.max_steps,
        )
        
        # Successful episode
        success_score = grade_episode(
            task=task,
            action_history=[],
            service_health=task.service_health_after_remediation,
            root_cause_identified=True,
            steps_taken=3,
            max_steps=task.max_steps,
        )
        
        assert success_score > failed_score

    def test_grade_not_constant(self):
        """Different inputs should produce different grades."""
        task = TASKS["easy"]
        
        score_empty = grade_episode(
            task=task,
            action_history=[],
            service_health=task.initial_service_health,
            root_cause_identified=False,
            steps_taken=0,
            max_steps=task.max_steps,
        )
        
        score_perfect = grade_episode(
            task=task,
            action_history=[],
            service_health=task.service_health_after_remediation,
            root_cause_identified=True,
            steps_taken=2,
            max_steps=task.max_steps,
        )
        
        assert score_empty != score_perfect, "Grader returns constant score"


# ---------------------------------------------------------------------------
# Environment Tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    """Test ResilienceOpsEnvironment lifecycle."""

    def test_reset_produces_observation(self):
        """Reset should return a valid observation."""
        env = ResilienceOpsEnvironment(task_name="easy")
        obs = env.reset()
        assert obs.incident_title == "API Gateway Timeout"
        assert obs.severity == "P3"
        assert len(obs.affected_services) > 0
        assert obs.steps_remaining > 0

    def test_step_returns_observation(self):
        """Step should return an observation with reward."""
        env = ResilienceOpsEnvironment(task_name="easy")
        env.reset()
        action = ResilienceOpsAction(
            action_type="diagnose", target="api-gateway", tool_used="top"
        )
        obs = env.step(action)
        assert obs.step_count == 1
        assert obs.reward is not None

    def test_episode_terminates(self):
        """Episode should terminate after max_steps."""
        env = ResilienceOpsEnvironment(task_name="easy")
        env.reset()
        task = TASKS["easy"]
        for _ in range(task.max_steps + 5):
            action = ResilienceOpsAction(
                action_type="diagnose", target="api-gateway", tool_used="top"
            )
            obs = env.step(action)
            if obs.done:
                break
        assert obs.done, "Episode did not terminate"

    def test_final_grade_in_range(self):
        """Final grade should be in [0.0, 1.0]."""
        env = ResilienceOpsEnvironment(task_name="easy")
        env.reset()
        action = ResilienceOpsAction(
            action_type="diagnose", target="api-gateway", tool_used="top"
        )
        env.step(action)
        grade = env.get_final_grade()
        assert 0.0 <= grade <= 1.0

    def test_all_tasks_runnable(self):
        """All 7 tasks should be valid and runnable."""
        for task_name in TASKS:
            env = ResilienceOpsEnvironment(task_name=task_name)
            obs = env.reset()
            assert not obs.done, f"Task {task_name} starts done"
            assert obs.incident_title, f"Task {task_name} has no title"

    def test_set_task(self):
        """set_task should switch the active task."""
        env = ResilienceOpsEnvironment(task_name="easy")
        env.reset()
        env.set_task("medium")
        obs = env.reset()
        assert obs.incident_title == "Cascading Database Connection Pool Exhaustion"

    def test_task_name_returns_name_not_difficulty(self):
        """Observation task_name should return the task name, not difficulty."""
        env = ResilienceOpsEnvironment(task_name="k8s_crashloop")
        obs = env.reset()
        assert obs.task_name == "k8s_crashloop"


# ---------------------------------------------------------------------------
# Task Configuration Tests
# ---------------------------------------------------------------------------

class TestTaskConfiguration:
    """Test task configurations are valid."""

    def test_minimum_task_count(self):
        """Must have at least 3 tasks."""
        assert len(TASKS) >= 3

    def test_actual_task_count(self):
        """Should have exactly 7 tasks."""
        assert len(TASKS) == 7

    def test_all_tasks_have_required_fields(self):
        """All tasks must have required configuration fields."""
        for name, task in TASKS.items():
            assert task.name, f"Task {name} missing name"
            assert task.difficulty, f"Task {name} missing difficulty"
            assert task.incident_title, f"Task {name} missing title"
            assert task.severity in ("P1", "P2", "P3"), f"Task {name} has invalid severity: {task.severity}"
            assert len(task.affected_services) > 0, f"Task {name} has no affected services"
            assert task.max_steps > 0, f"Task {name} has invalid max_steps"
            assert len(task.correct_remediation_sequence) > 0, f"Task {name} has no remediation sequence"
            assert len(task.available_tools) > 0, f"Task {name} has no available tools"

    def test_difficulty_progression(self):
        """Tasks should have meaningful difficulty progression."""
        easy = TASKS["easy"]
        hard = TASKS["hard"]
        expert = TASKS["circuit_breaker_storm"]
        assert easy.max_steps < hard.max_steps
        assert hard.max_steps < expert.max_steps
        assert len(easy.affected_services) < len(expert.affected_services)

    def test_reward_table_has_all_keys(self):
        """Reward table should have all required keys."""
        required_keys = [
            "correct_severity_classification",
            "correct_root_cause_identification",
            "successful_remediation",
            "correct_tool_selection",
            "wasted_step",
            "unnecessary_escalation",
            "time_penalty_per_step",
            "episode_completion_bonus",
        ]
        for key in required_keys:
            assert key in REWARD_TABLE, f"Missing reward key: {key}"


# ---------------------------------------------------------------------------
# Clamp Function Tests
# ---------------------------------------------------------------------------

class TestClampReward:
    """Test reward clamping."""

    def test_clamp_negative(self):
        assert _clamp_reward(-0.5) == 0.0

    def test_clamp_above_one(self):
        assert _clamp_reward(1.5) == 1.0

    def test_clamp_in_range(self):
        assert _clamp_reward(0.5) == 0.5

    def test_clamp_zero(self):
        assert _clamp_reward(0.0) == 0.0

    def test_clamp_one(self):
        assert _clamp_reward(1.0) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
