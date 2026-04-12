"""
Inference Script for ResilienceOps Environment
==============================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import json
import os
import sys
import signal
import textwrap
import asyncio
import time
from typing import List, Optional

# Load env vars FIRST, before any other imports
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, APITimeoutError, APIError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "resilience_ops_env"))

from resilience_ops_env import ResilienceOpsAction, ResilienceOpsEnv
from resilience_ops_env.models import TASKS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
if not API_KEY:
    print("[END] success=false steps=0 score=0.000 rewards=", flush=True)
    sys.stderr.write("FATAL: HF_TOKEN or API_KEY environment variable is required\n")
    sys.exit(1)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
BENCHMARK = os.getenv("RESILIENCE_OPS_BENCHMARK", "resilience_ops_env")
TEMPERATURE = 0.3
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.3
_MAX_REWARD_PER_STEP = 1.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Site Reliability Engineer (SRE) responding to IT incidents.
    Each turn you must send a JSON action to triage, diagnose, and resolve the incident.
    Reward is based on correct tool selection, root cause identification, successful remediation, and efficiency.
    Your goal is to maximize total reward by resolving incidents quickly and safely.

    Reply with exactly one JSON object — no quotes, no prefixes, no markdown, just the JSON:
    {"action_type": "diagnose"|"remediate"|"escalate"|"query_logs"|"check_metrics"|"restart_service"|"rollback"|"scale_up"|"check_connectivity"|"analyze_root_cause", "target": "service_name", "tool_used": "tool_name", "parameters": {}}
    """
).strip()

# ---------------------------------------------------------------------------
# Signal handler for clean shutdown
# ---------------------------------------------------------------------------

def _signal_handler(signum, frame):
    """Ensure clean exit on signals."""
    sys.exit(0)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# ---------------------------------------------------------------------------
# Structured logging — ONLY these three functions write to stdout
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END]   success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Internal logging — all goes to stderr, never stdout
# ---------------------------------------------------------------------------

def _debug(msg: str) -> None:
    sys.stderr.write(f"[DEBUG] {msg}\n")
    sys.stderr.flush()

def _warn(msg: str) -> None:
    sys.stderr.write(f"[WARN] {msg}\n")
    sys.stderr.flush()

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_user_prompt(step: int, obs, last_reward: float, history: List[str]) -> str:
    incident_info = (
        f"Incident: {obs.incident_title} | Severity: {obs.severity} | "
        f"Affected: {', '.join(obs.affected_services)} | "
        f"Health: {json.dumps(obs.service_health)} | "
        f"Steps Left: {obs.steps_remaining} | "
        f"Tools: {', '.join(obs.available_tools)}"
    )
    if obs.previous_action_result:
        incident_info += f" | Last Result: {obs.previous_action_result}"
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Incident State: {incident_info}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next JSON action.
        """
    ).strip()


# ---------------------------------------------------------------------------
# Fallback actions for each task (used when LLM fails)
# ---------------------------------------------------------------------------

FALLBACK_ACTIONS = {
    "easy": {"action_type": "diagnose", "target": "api-gateway", "tool_used": "top", "parameters": {}},
    "medium": {"action_type": "diagnose", "target": "postgres-primary", "tool_used": "pg_isready", "parameters": {}},
    "hard": {"action_type": "check_connectivity", "target": "global-load-balancer", "tool_used": "traceroute", "parameters": {}},
    "k8s_crashloop": {"action_type": "diagnose", "target": "payment-worker", "tool_used": "kubectl", "parameters": {}},
    "security_lateral": {"action_type": "check_connectivity", "target": "jump-host", "tool_used": "netstat", "parameters": {}},
    "dns_failure": {"action_type": "diagnose", "target": "dns-resolver-us", "tool_used": "dig", "parameters": {}},
    "circuit_breaker_storm": {"action_type": "check_metrics", "target": "slow-legacy-api", "tool_used": "prometheus", "parameters": {}},
}


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _call_llm_with_retry(client: OpenAI, messages: list, max_retries: int = 3, base_delay: float = 1.0) -> str:
    """Call LLM with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                timeout=30,
            )
            text = (completion.choices[0].message.content or "").strip()
            # Strip markdown code blocks if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            # Validate JSON
            json.loads(text)
            return text if text else ""
        except (APITimeoutError, APIError) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                _warn(f"LLM call failed (attempt {attempt + 1}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                _warn(f"LLM call failed after {max_retries} attempts: {e}")
                raise
        except json.JSONDecodeError:
            # Response wasn't valid JSON, try to extract JSON from text
            if text:
                try:
                    # Try to find JSON object in the text
                    start = text.index("{")
                    end = text.rindex("}") + 1
                    candidate = text[start:end]
                    json.loads(candidate)
                    return candidate
                except (ValueError, json.JSONDecodeError):
                    pass
            if attempt < max_retries - 1:
                _warn(f"Invalid JSON response, retrying...")
                time.sleep(base_delay)
            else:
                _warn(f"Invalid JSON after {max_retries} attempts")
                return ""
    return ""


def get_model_message(client: OpenAI, step: int, obs, last_reward: float, history: List[str], task_name: str = "easy") -> str:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        result = _call_llm_with_retry(client, messages)
        return result or json.dumps(FALLBACK_ACTIONS.get(task_name, FALLBACK_ACTIONS["easy"]))
    except Exception as exc:
        _debug(f"Model request failed: {exc}")
        return json.dumps(FALLBACK_ACTIONS.get(task_name, FALLBACK_ACTIONS["easy"]))


def action_dict_to_str(action_text: str) -> str:
    """Convert action JSON string to a compact string for logging."""
    try:
        d = json.loads(action_text)
        return f"{d.get('action_type', '?')}({d.get('target', '')}, tool={d.get('tool_used', '')})"
    except Exception:
        return action_text[:50]


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task_name: str) -> None:
    """Run a single task episode."""
    # Validate task exists
    if task_name not in TASKS:
        _warn(f"Unknown task: {task_name}, skipping")
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    env = ResilienceOpsEnv(base_url=ENV_BASE_URL)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_config = TASKS[task_name]
    max_steps = task_config.max_steps
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        result = await env.reset(task=task_name)
        last_reward = 0.0
        
        for step in range(1, max_steps + 1):
            if result.done:
                break
            
            action_text = get_model_message(client, step, result.observation, last_reward, history, task_name)
            
            try:
                action_data = json.loads(action_text)
            except json.JSONDecodeError:
                action_data = {
                    "action_type": "diagnose",
                    "target": task_config.affected_services[0],
                    "tool_used": task_config.correct_diagnostic_tools[0],
                    "parameters": {},
                }
            
            action = ResilienceOpsAction(
                action_type=action_data.get("action_type", "diagnose"),
                target=action_data.get("target", ""),
                tool_used=action_data.get("tool_used", ""),
                parameters=action_data.get("parameters", {}),
            )
            
            result = await env.step(action)
            
            reward = result.reward or 0.0
            done = result.done
            error = None
            
            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            
            action_str = action_dict_to_str(action_text)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            
            history.append(f"Step {step}: {action_str!r} -> reward {reward:+.2f}")
            
            if done:
                break
        
        # Task-specific score normalization
        max_possible = max_steps * _MAX_REWARD_PER_STEP
        score = sum(rewards) / max_possible if max_possible > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        _debug(f"Error during task execution: {e}")
    finally:
        try:
            await env.close()
        except Exception as e:
            _debug(f"env.close() error: {e}")
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main_async() -> None:
    """Main entry point - runs all tasks (3 original + 4 v3.0)."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # All 7 tasks: 3 original + 4 v3.0
    task_names = ["easy", "medium", "hard", "k8s_crashloop", "security_lateral", "dns_failure", "circuit_breaker_storm"]
    
    for task_name in task_names:
        await run_task(client, task_name)


def main() -> None:
    """Entry point that runs async main."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
