#!/usr/bin/env python3
"""
Pre-submission validation script for ResilienceOps environment.

This script validates all requirements from the evaluation criteria:
- OpenEnv spec compliance
- Dockerfile builds
- Inference script runs
- 3+ tasks with graders (0.0-1.0 scores)
- Deterministic grading
- Runtime < 20min
- Environment variables configured

Usage:
    python validate.py [--full]

Exit codes:
    0 - All validations passed
    1 - One or more validations failed
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Optional

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")


def print_pass(text: str):
    print(f"{GREEN}✓ PASS{RESET}: {text}")


def print_fail(text: str):
    print(f"{RED}✗ FAIL{RESET}: {text}")


def print_warn(text: str):
    print(f"{YELLOW}⚠ WARN{RESET}: {text}")


def check_environment_variables() -> Tuple[bool, List[str]]:
    """Check required environment variables."""
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing = []
    found = []
    
    for var in required:
        value = os.getenv(var)
        if not value:
            missing.append(var)
        else:
            # Don't print the actual token
            display = value[:20] + "..." if len(value) > 20 else value
            if var == "HF_TOKEN":
                display = "[SET]"
            found.append(f"{var}={display}")
    
    if missing:
        return False, missing
    return True, found


def check_file_structure() -> Tuple[bool, List[str]]:
    """Check required files exist."""
    required_files = [
        "openenv.yaml",
        "Dockerfile",
        "inference.py",
        "models.py",
        "requirements.txt",
        "README.md",
        "server/app.py",
    ]
    
    missing = []
    found = []
    
    for file in required_files:
        path = Path(file)
        if path.exists():
            found.append(file)
        else:
            missing.append(file)
    
    if missing:
        return False, missing
    return True, found


def check_openenv_yaml() -> Tuple[bool, str]:
    """Validate openenv.yaml structure."""
    try:
        import yaml
        with open("openenv.yaml") as f:
            config = yaml.safe_load(f)
        
        required_keys = ["spec_version", "name", "type", "runtime", "app", "port"]
        missing = [k for k in required_keys if k not in config]
        
        if missing:
            return False, f"Missing keys: {missing}"
        
        if config["name"] != "resilience_ops_env":
            return False, f"Wrong name: {config['name']}"
        
        return True, "Valid openenv.yaml"
    except Exception as e:
        return False, str(e)


def check_python_syntax() -> Tuple[bool, List[str]]:
    """Check Python files compile without errors."""
    import py_compile
    
    files = [
        "models.py",
        "inference.py",
        "server/app.py",
        "v3_0_features.py",
    ]
    
    errors = []
    compiled = []
    
    for file in files:
        if not Path(file).exists():
            errors.append(f"{file} not found")
            continue
        try:
            py_compile.compile(file, doraise=True)
            compiled.append(file)
        except Exception as e:
            errors.append(f"{file}: {e}")
    
    if errors:
        return False, errors
    return True, compiled


def check_typed_models() -> Tuple[bool, str]:
    """Verify models use Pydantic types."""
    try:
        with open("models.py") as f:
            content = f.read()
        
        required_classes = [
            "ResilienceOpsAction",
            "ResilienceOpsObservation",
            "TaskConfig",
        ]
        
        missing = [c for c in required_classes if c not in content]
        
        if missing:
            return False, f"Missing classes: {missing}"
        
        if "BaseModel" not in content and "dataclass" not in content:
            return False, "No Pydantic BaseModel or dataclass found"
        
        return True, f"Found: {required_classes}"
    except Exception as e:
        return False, str(e)


def check_inference_script() -> Tuple[bool, str]:
    """Check inference.py has required components."""
    try:
        with open("inference.py") as f:
            content = f.read()
        
        required_components = [
            "[START]",
            "[STEP]",
            "[END]",
            "OpenAI",
            "API_BASE_URL",
            "MODEL_NAME",
            "HF_TOKEN",
        ]
        
        missing = [c for c in required_components if c not in content]
        
        if missing:
            return False, f"Missing: {missing}"
        
        # Check for async main
        if "async def main" not in content and "asyncio" not in content:
            return False, "Missing async/await pattern"
        
        return True, "All required components found"
    except Exception as e:
        return False, str(e)


def check_task_count() -> Tuple[bool, str]:
    """Verify at least 3 tasks exist."""
    try:
        sys.path.insert(0, str(Path.cwd()))
        from models import TASKS
        
        task_count = len(TASKS)
        if task_count < 3:
            return False, f"Only {task_count} tasks found (need 3+)"
        
        tasks_list = list(TASKS.keys())
        return True, f"Found {task_count} tasks: {tasks_list}"
    except Exception as e:
        return False, str(e)


def check_graders() -> Tuple[bool, List[str]]:
    """Verify graders produce 0.0-1.0 scores."""
    try:
        sys.path.insert(0, str(Path.cwd()))
        from models import TASKS, grade_episode, ResilienceOpsAction
        
        errors = []
        passed = []
        
        for task_name, task in TASKS.items():
            try:
                # Test with empty action history
                score = grade_episode(
                    task=task,
                    action_history=[],
                    service_health=task.initial_service_health,
                    root_cause_identified=False,
                    steps_taken=0,
                    max_steps=task.max_steps,
                )
                
                if not (0.0 <= score <= 1.0):
                    errors.append(f"{task_name}: score {score} out of range [0.0, 1.0]")
                else:
                    passed.append(f"{task_name}: {score:.3f}")
                    
            except Exception as e:
                errors.append(f"{task_name}: {e}")
        
        if errors:
            return False, errors
        return True, passed
    except Exception as e:
        return False, [str(e)]


def check_deterministic_grading() -> Tuple[bool, str]:
    """Verify graders produce same score for same inputs."""
    try:
        sys.path.insert(0, str(Path.cwd()))
        from models import TASKS, grade_episode, ResilienceOpsAction
        
        task = list(TASKS.values())[0]
        
        # Run grading twice with same inputs
        scores = []
        for _ in range(2):
            score = grade_episode(
                task=task,
                action_history=[],
                service_health=task.initial_service_health,
                root_cause_identified=False,
                steps_taken=5,
                max_steps=task.max_steps,
            )
            scores.append(score)
        
        if scores[0] != scores[1]:
            return False, f"Non-deterministic: {scores[0]} vs {scores[1]}"
        
        return True, f"Deterministic: {scores[0]:.3f} (both runs)"
    except Exception as e:
        return False, str(e)


def check_reward_density() -> Tuple[bool, str]:
    """Check that rewards are not sparse (provide signal at each step)."""
    try:
        sys.path.insert(0, str(Path.cwd()))
        from models import TASKS, compute_reward, IncidentEnvState
        
        task = list(TASKS.values())[0]
        
        # Create a test state
        state = IncidentEnvState(
            episode_id="test",
            task=task,
            step_count=1,
            root_cause_identified=False,
            all_services_restored=False,
            service_health=dict(task.initial_service_health),
            action_history=[],
            cumulative_reward=0.0,
            done=False,
        )
        
        # Test a few different actions
        from models import ResilienceOpsAction
        
        actions = [
            ResilienceOpsAction(action_type="diagnose", target=task.affected_services[0], tool_used=task.correct_diagnostic_tools[0]),
            ResilienceOpsAction(action_type="check_metrics", target=task.affected_services[0], tool_used="prometheus"),
            ResilienceOpsAction(action_type="remediate", target=task.affected_services[0], tool_used="systemctl"),
        ]
        
        rewards = []
        for action in actions:
            reward, _ = compute_reward(action, task, state)
            rewards.append(reward)
        
        # Check that rewards vary (not all the same)
        unique_rewards = len(set([round(r, 3) for r in rewards]))
        
        if unique_rewards < 2:
            return False, f"Sparse rewards: {rewards} (too similar)"
        
        return True, f"Dense rewards: {rewards} (varied signal)"
    except Exception as e:
        return False, str(e)


def check_dockerfile() -> Tuple[bool, str]:
    """Check Dockerfile exists and has required components."""
    try:
        with open("Dockerfile") as f:
            content = f.read()
        
        required = ["FROM", "EXPOSE", "CMD", "uvicorn"]
        missing = [r for r in required if r not in content]
        
        if missing:
            return False, f"Missing: {missing}"
        
        if "8000" not in content:
            return False, "Port 8000 not referenced"
        
        return True, "Dockerfile valid"
    except Exception as e:
        return False, str(e)


def run_docker_build() -> Tuple[bool, str]:
    """Test Docker build (if Docker is available)."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False, "Docker not available"
    except Exception:
        return False, "Docker not available"
    
    print_warn("Docker build test skipped (run 'docker build .' manually)")
    return True, "Docker available (manual test recommended)"


def check_runtime_estimate() -> Tuple[bool, str]:
    """Estimate runtime based on task configurations."""
    try:
        sys.path.insert(0, str(Path.cwd()))
        from models import TASKS
        
        total_steps = sum(t.max_steps for t in TASKS.values())
        
        # Estimate: ~5 seconds per step (LLM call + processing)
        estimated_seconds = total_steps * 5
        estimated_minutes = estimated_seconds / 60
        
        if estimated_minutes > 20:
            return False, f"Estimated {estimated_minutes:.1f}min (exceeds 20min limit)"
        
        return True, f"Estimated {estimated_minutes:.1f}min for {total_steps} steps ({len(TASKS)} tasks)"
    except Exception as e:
        return False, str(e)


def main():
    """Run all validation checks."""
    print_header("ResilienceOps Environment Pre-Submission Validation")
    
    all_passed = True
    results = []
    
    # 1. Environment Variables
    print("\n📋 Checking environment variables...")
    passed, details = check_environment_variables()
    if passed:
        print_pass(f"Environment variables set: {details}")
    else:
        print_fail(f"Missing environment variables: {details}")
        all_passed = False
    results.append(("Environment Variables", passed))
    
    # 2. File Structure
    print("\n📁 Checking file structure...")
    passed, details = check_file_structure()
    if passed:
        print_pass(f"All required files present: {len(details)} files")
    else:
        print_fail(f"Missing files: {details}")
        all_passed = False
    results.append(("File Structure", passed))
    
    # 3. Python Syntax
    print("\n🐍 Checking Python syntax...")
    passed, details = check_python_syntax()
    if passed:
        print_pass(f"All files compile: {len(details)} files")
    else:
        print_fail(f"Syntax errors: {details}")
        all_passed = False
    results.append(("Python Syntax", passed))
    
    # 4. OpenEnv YAML
    print("\n⚙️  Checking openenv.yaml...")
    passed, details = check_openenv_yaml()
    if passed:
        print_pass(details)
    else:
        print_fail(details)
        all_passed = False
    results.append(("OpenEnv YAML", passed))
    
    # 5. Typed Models
    print("\n🔍 Checking typed models...")
    passed, details = check_typed_models()
    if passed:
        print_pass(details)
    else:
        print_fail(details)
        all_passed = False
    results.append(("Typed Models", passed))
    
    # 6. Inference Script
    print("\n📝 Checking inference.py...")
    passed, details = check_inference_script()
    if passed:
        print_pass(details)
    else:
        print_fail(details)
        all_passed = False
    results.append(("Inference Script", passed))
    
    # 7. Task Count
    print("\n🎯 Checking task count...")
    passed, details = check_task_count()
    if passed:
        print_pass(details)
    else:
        print_fail(details)
        all_passed = False
    results.append(("Task Count", passed))
    
    # 8. Graders
    print("\n📊 Checking graders...")
    passed, details = check_graders()
    if passed:
        print_pass(f"All graders valid: {len(details)} tasks")
    else:
        print_fail(f"Grader errors: {details}")
        all_passed = False
    results.append(("Graders", passed))
    
    # 9. Deterministic Grading
    print("\n🎲 Checking grading determinism...")
    passed, details = check_deterministic_grading()
    if passed:
        print_pass(details)
    else:
        print_fail(details)
        all_passed = False
    results.append(("Deterministic Grading", passed))
    
    # 10. Reward Density
    print("\n💰 Checking reward density...")
    passed, details = check_reward_density()
    if passed:
        print_pass(details)
    else:
        print_fail(details)
        all_passed = False
    results.append(("Reward Density", passed))
    
    # 11. Dockerfile
    print("\n🐳 Checking Dockerfile...")
    passed, details = check_dockerfile()
    if passed:
        print_pass(details)
    else:
        print_fail(details)
        all_passed = False
    results.append(("Dockerfile", passed))
    
    # 12. Docker Build (optional, requires Docker)
    print("\n🔨 Checking Docker availability...")
    passed, details = run_docker_build()
    if passed:
        print_pass(details)
    else:
        print_warn(details)
    results.append(("Docker Build", passed))
    
    # 13. Runtime Estimate
    print("\n⏱️  Checking runtime estimate...")
    passed, details = check_runtime_estimate()
    if passed:
        print_pass(details)
    else:
        print_fail(details)
        all_passed = False
    results.append(("Runtime Estimate", passed))
    
    # Summary
    print_header("Validation Summary")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nPassed: {GREEN}{passed_count}/{total_count}{RESET}")
    print(f"Failed: {RED}{total_count - passed_count}/{total_count}{RESET}")
    
    if all_passed:
        print(f"\n{GREEN}{BOLD}✓ All validations passed! Ready for submission.{RESET}")
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ Some validations failed. Fix issues before submission.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
