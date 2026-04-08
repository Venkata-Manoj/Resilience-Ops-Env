---
title: ResilienceOps — IT Incident Response RL Environment
emoji: 🚨
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - sre
  - incident-response
---

# 🚨 ResilienceOps Environment

An [OpenEnv](https://github.com/openai/openenv) reinforcement learning environment where AI agents learn to triage, diagnose, and resolve simulated IT infrastructure incidents under time pressure and resource constraints.

---

## 📋 Environment Overview

ResilienceOps simulates the real-world workflow of Site Reliability Engineers (SREs) during incident response. Agents must process alerts, analyze logs, identify root causes, and execute remediation steps — all while avoiding destructive actions that could worsen the situation.

### Motivation

IT incident response costs enterprises billions annually in downtime. SRE teams spend hours manually triaging alerts, correlating logs, and executing runbooks. Training AI agents to autonomously handle incidents — prioritizing severity, selecting diagnostic tools, and applying correct remediation — is a critical real-world problem that companies like Google, Netflix, and AWS are actively solving.

This environment addresses that need by providing:

- **Realistic scenarios**: Database failures, network partitions, and cascading outages
- **Dense rewards**: Continuous feedback throughout episodes, not just binary outcomes
- **Safety constraints**: Penalizes dangerous actions like restarting databases during split-brain scenarios
- **Progressive complexity**: Three difficulty levels from P3 (simple) to P1 (critical)

---

## 🎮 Action Space

Actions are defined by the `ResilienceOpsAction` Pydantic model:

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | One of: `diagnose`, `remediate`, `escalate`, `query_logs`, `check_metrics`, `restart_service`, `rollback`, `scale_up`, `check_connectivity`, `analyze_root_cause` |
| `target` | `str` | Service/component to act on (e.g., `api-gateway`, `postgres-primary`) |
| `tool_used` | `str` | Specific diagnostic or remediation tool (e.g., `top`, `pg_isready`, `kubectl`) |
| `parameters` | `Dict[str, str]` | Additional action-specific parameters |

### Action Types Explained

- **diagnose**: Run diagnostic tools on a service
- **remediate**: Apply a fix to resolve an issue
- **escalate**: Escalate to human operators
- **query_logs**: Examine log files for diagnostic information
- **check_metrics**: Review performance metrics
- **restart_service**: Restart a failing service
- **rollback**: Revert to previous state
- **scale_up**: Increase capacity/resources
- **check_connectivity**: Verify network connectivity
- **analyze_root_cause**: Identify the root cause of the incident

---

## 👁️ Observation Space

Observations are returned as `ResilienceOpsObservation` objects:

| Field | Type | Description |
|-------|------|-------------|
| `incident_id` | `str` | Unique identifier for the current incident |
| `incident_title` | `str` | Human-readable title describing the incident |
| `severity` | `str` | Incident priority: P1 (critical), P2 (high), or P3 (low) |
| `affected_services` | `List[str]` | Services currently impacted by the incident |
| `alert_signals` | `List[str]` | Active monitoring alerts (e.g., `high_cpu`, `5xx_spike`) |
| `log_snippet` | `str` | Relevant log excerpt to aid diagnosis |
| `available_tools` | `List[str]` | Tools available for the agent to use |
| `steps_remaining` | `int` | Number of steps left in the current episode |
| `step_count` | `int` | Current step number |
| `previous_action_result` | `str` | Description of the outcome from the last action |
| `service_health` | `Dict[str, str]` | Health status per service: `healthy` / `degraded` / `down` |
| `task_name` | `str` | Current task difficulty: `easy` / `medium` / `hard` |
| `root_cause_identified` | `bool` | Whether the agent has identified the root cause |
| `hints` | `List[str]` | Contextual hints to guide the agent |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Cumulative reward earned so far |

---

## 📝 Tasks

The environment provides three tasks with increasing difficulty:

### Task 1: Easy — Single-Service Outage (P3)

**Scenario**: API Gateway Timeout

An API gateway is experiencing timeouts due to worker pool exhaustion from a memory leak. Only one service is affected.

**Difficulty**: Easy

**Agent Objectives**:
1. Identify the severity (P3 - low priority)
2. Run correct diagnostic (check CPU/memory with `top`)
3. Apply appropriate fix (restart the service)

**Max Steps**: 10

**Expected Agent Behavior**: Quick identification and remediation with minimal steps.

---

### Task 2: Medium — Cascading Database Failure (P2)

**Scenario**: PostgreSQL Connection Pool Exhaustion

A database connection pool has reached maximum capacity, causing cascading failures in downstream services (API gateway and web frontend).

**Difficulty**: Medium

**Agent Objectives**:
1. Prioritize multiple correlated alerts
2. Identify the root cause (database, not symptoms)
3. Execute correct remediation sequence:
   - Check database readiness (`pg_isready`)
   - Query active connections (`pg_stat_activity`)
   - Terminate idle connections
   - Scale up connection pool

**Max Steps**: 15

**Expected Agent Behavior**: Avoid treating symptoms (API timeouts) and focus on the root database issue.

---

### Task 3: Hard — Multi-Region Network Partition (P1)

**Scenario**: Split-Brain Network Partition

A network partition between US-East and EU-West regions has caused a split-brain scenario in a shared PostgreSQL database. Both regions are accepting writes, risking data inconsistency.

**Difficulty**: Hard

**Agent Objectives**:
1. Correlate cross-region alert signals
2. Identify the network partition as root cause
3. Choose **safe** remediation (avoid data loss):
   - Verify connectivity (`traceroute`)
   - Check replication status
   - Set EU region to read-only
   - Drain traffic from affected region
   - Resync replication
4. Verify recovery before completing

**Max Steps**: 20

**Expected Agent Behavior**: Exercise extreme caution. Destructive actions (restarting database) during split-brain will be heavily penalized.

---

## 🚀 Setup and Usage Instructions

### Prerequisites

- Python 3.10+
- Docker (for deployment)
- Hugging Face account with API token

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd resilience_ops_env

# Install dependencies
pip install -r requirements.txt
```

### Local Development

```bash
# Run the server locally using uv
uv run server

# Or using uvicorn directly
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Docker Build and Run

```bash
# Build the Docker image
docker build -t resilience-ops-env:latest .

# Run the container
docker run -p 8000:8000 resilience-ops-env:latest
```

The web interface will be available at `http://localhost:8000/web/`

### Deploy to Hugging Face Spaces

```bash
# Push to Hugging Face Spaces
openenv push --repo-id your-username/resilience-ops-env
```

Ensure your Space is tagged with `openenv` for discoverability.

### Using the Python Client

```python
from resilience_ops_env import ResilienceOpsEnv, ResilienceOpsAction

# Connect to the environment
with ResilienceOpsEnv(base_url="http://localhost:8000") as env:
    # Start a new episode (easy task)
    result = env.reset(task="easy")
    obs = result.observation
    print(f"Incident: {obs.incident_title}")
    print(f"Severity: {obs.severity}")
    print(f"Affected: {obs.affected_services}")

    # Take an action
    result = env.step(ResilienceOpsAction(
        action_type="diagnose",
        target="api-gateway",
        tool_used="top",
    ))
    print(f"Reward: {result.observation.reward}")
    print(f"Result: {result.observation.previous_action_result}")
```

---

## 📊 Baseline Performance Scores

The `inference.py` script provides baseline scores using a Hugging Face LLM (default: Qwen/Qwen2.5-72B-Instruct).

### Running Baseline Evaluation

```bash
# Set required environment variables
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Run inference across all three tasks
python inference.py
```

### Expected Baseline Scores

Based on preliminary testing, expected scores for frontier models are:

| Task | Difficulty | Expected Score Range | Notes |
|------|------------|---------------------|-------|
| Easy (P3) | Low | 0.70 - 0.85 | Single-service issues are straightforward |
| Medium (P2) | Medium | 0.50 - 0.70 | Requires distinguishing root cause from symptoms |
| Hard (P1) | High | 0.30 - 0.55 | Complex multi-region scenarios with safety constraints |

### Grading Rubric

Final episode scores (0.0 - 1.0) are computed as:

| Component | Weight | Description |
|-----------|--------|-------------|
| Root cause identification | 25% | Did agent correctly identify the root cause? |
| Correct tool usage | 20% | Were appropriate diagnostic tools selected? |
| Remediation success | 30% | Were all services restored to healthy state? |
| Efficiency | 15% | Fewer steps = higher score |
| Penalties | Variable | Deductions for destructive actions and unnecessary escalations |

### Output Format

The inference script emits structured output:

```
[START] task=easy env=resilience_ops_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=diagnose(target='api-gateway', tool='top') reward=0.08 done=false error=null
[STEP] step=2 action=restart_service(target='api-gateway', tool='systemctl') reward=0.43 done=true error=null
[END] success=true steps=2 rewards=0.08,0.43
```

---

## 🏗️ Project Structure

```
resilience_ops_env/
├── Dockerfile                      # Docker configuration
├── README.md                       # This file
├── LICENSE                         # License
├── openenv.yaml                    # OpenEnv manifest
├── requirements.txt                # Dependencies
├── models.py                       # Types, tasks, reward, grader
├── client.py                       # Environment client
├── inference.py                    # Baseline inference script
├── server/
│   ├── __init__.py
│   ├── app.py                      # FastAPI application
│   ├── requirements.txt
│   └── resilience_ops_env_environment.py
└── .agents/skills/                 # Agent skills
```

---

## 🧪 Validation

Run OpenEnv validation to ensure compliance:

```bash
openenv validate
```

---

## 📄 License

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
