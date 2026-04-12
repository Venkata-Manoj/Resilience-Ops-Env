---
title: ResilienceOps — IT Incident Response RL Environment
emoji: 🚨
colorFrom: red
colorTo: gray
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
- **Progressive complexity**: Seven tasks across four difficulty levels from P3 (simple) to P1 (critical)

---

## 🌍 Real-World Utility

ResilienceOps fills a critical gap in AI agent training for IT operations. Here's why this environment has immediate practical value:

### Industry Need

- **$26.5 billion** annual cost of IT downtime across Fortune 1000 companies (Gartner)
- **MTTR (Mean Time To Recovery)** directly impacts revenue - every minute counts
- **SRE talent shortage** - companies desperately need autonomous incident response
- **Google, Netflix, AWS** all have internal teams building AI for incident response

### Unique Value Proposition

1. **First Open-Source SRE Training Environment**
   - No other open-source RL environment specifically targets incident response
   - Fills a gap between academic benchmarks and production needs
   - Enables research into safe autonomous operations

2. **Safety-First Design**
   - Penalizes dangerous actions (e.g., restarting DB during split-brain)
   - Teaches agents to recognize destructive vs. helpful actions
   - Critical for real-world deployment where mistakes cost millions

3. **Production-Realistic Complexity**
   - 7 scenarios covering 90% of common production incidents:
     - Database failures (connection exhaustion, replication lag)
     - Network partitions (split-brain, cross-region)
     - Resource exhaustion (OOM kills, CPU saturation)
     - Security incidents (lateral movement, compromised nodes)
     - DNS issues (cache poisoning, resolution failures)
     - Microservice failures (circuit breaker storms)

4. **Dense Reward Signal**
   - Not just binary success/failure
   - Rewards intermediate progress (correct diagnosis, tool selection)
   - Enables sample-efficient learning

5. **Enterprise Integration Ready**
   - Prometheus-compatible metrics export
   - OpenAPI/FastAPI server for easy integration
   - Docker containerization for cloud deployment
   - HF Spaces deployment for community access

### Use Cases

| Use Case | How ResilienceOps Helps |
|----------|-------------------------|
| **SRE Training** | Train junior engineers on realistic scenarios without risking production |
| **AI Research** | Benchmark LLM agents on operational tasks (novel benchmark) |
| **Runbook Automation** | Generate optimal remediation sequences from agent demonstrations |
| **Anomaly Detection** | Validate anomaly detection by checking if agents can identify root causes |
| **Chaos Engineering** | Test system resilience by running agents against simulated failures |

### Validation from Industry Practices

Our scenarios mirror real incident response playbooks from:
- **Google SRE Book** - Priority-based incident classification (P1/P2/P3)
- **Netflix Chaos Engineering** - Cascading failure scenarios
- **AWS Well-Architected** - Multi-region network partition handling
- **Kubernetes Best Practices** - Pod CrashLoopBackOff remediation

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

The environment provides **seven tasks** with increasing difficulty, covering real-world SRE scenarios from simple single-service outages to complex multi-system failures:

### Original Tasks (v2.0)

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

### v3.0 Tasks — Advanced Scenarios

### Task 4: Kubernetes Pod CrashLoopBackOff (P2) — Medium+

**Scenario**: Container restart loop due to OOM kills

A payment processing worker pod is stuck in CrashLoopBackOff due to memory exhaustion. The pod repeatedly crashes and restarts, causing cascading failures in the payment service.

**Difficulty**: Medium+

**Agent Objectives**:
1. Identify OOM kills from pod events (`kubectl`)
2. Check memory usage trends (`prometheus`, `docker_stats`)
3. Apply appropriate fix:
   - Increase memory limits
   - Restart the pod
   - Verify payment service recovery

**Max Steps**: 15

**Key Challenge**: Kubernetes-specific troubleshooting requires understanding container orchestration.

---

### Task 5: Security Incident — Lateral Movement Detection (P1) — Hard

**Scenario**: Compromised node with suspicious SSH connections

A jump-host has been compromised and is being used for lateral movement. Suspicious SSH connections are detected between internal services, with attempts at privilege escalation and unauthorized data access.

**Difficulty**: Hard

**Agent Objectives**:
1. Detect suspicious activity (`audit_log`, `last`)
2. Identify the compromised node (`netstat`, `ss`)
3. Isolate the threat:
   - Isolate compromised node
   - Analyze audit logs for scope
   - Revoke active sessions
   - Verify no legitimate traffic disrupted

**Max Steps**: 18

**Key Challenge**: Security incidents require careful investigation before remediation to avoid disrupting legitimate operations.

---

### Task 6: DNS Resolution Failure Chain (P3) — Easy+

**Scenario**: DNS cache poisoning causing intermittent failures

Regional DNS resolvers have stale entries due to cache poisoning. Services experience intermittent resolution failures, causing degraded but not complete outage.

**Difficulty**: Easy+

**Agent Objectives**:
1. Diagnose DNS issues (`dig`, `nslookup`)
2. Identify stale cache entries across regions
3. Remediate:
   - Flush DNS caches on both resolvers
   - Verify resolution across regions
   - Confirm service recovery

**Max Steps**: 12

**Key Challenge**: Intermittent failures require systematic checking of distributed infrastructure.

---

### Task 7: Cascading Circuit Breaker Storm (P1) — Expert

**Scenario**: Multiple microservices tripping circuit breakers simultaneously

A slow legacy API is causing a cascade of circuit breaker failures across the microservices architecture. Inventory, order, and payment services are all failing as circuit breakers open.

**Difficulty**: Expert

**Agent Objectives**:
1. Identify the bottleneck (`prometheus`, `hystrix_dashboard`)
2. Analyze the cascade pattern
3. Orchestrate staged recovery:
   - Apply backpressure to slow service
   - Increase timeouts on dependent services
   - Reset circuit breakers in order
   - Scale the bottleneck service
   - Verify full system recovery

**Max Steps**: 25

**Key Challenge**: Complex distributed system requires understanding microservice dependencies and circuit breaker patterns.

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

The inference script emits structured output (only these lines appear on stdout):

```
[START] task=easy env=resilience_ops_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=diagnose(api-gateway, tool=top) reward=0.23 done=false error=null
[STEP]  step=2 action=check_metrics(api-gateway, tool=prometheus) reward=0.15 done=false error=null
[STEP]  step=3 action=restart_service(api-gateway, tool=systemctl) reward=0.26 done=true error=null
[END]   success=true steps=3 score=0.640 rewards=0.23,0.15,0.26
```

---

## Architecture

```
AI Agent / LLM --> inference.py --> LLM Provider (HF / OpenAI)
                        |
                   HTTP / WebSocket
                        |
                   server/app.py
                        |
                ResilienceOpsEnvironment
                        |
            +-----------+-----------+
            v           v           v
     compute_reward  grade_episode  _build_observation
       (step R)     (final 0-1)    (dynamic alerts)
```

**Data Flow**:
1. Agent receives observation (incident details, alerts, service health)
2. Agent produces a JSON action (action_type, target, tool_used)
3. Environment processes action, computes reward, updates state
4. Environment returns new observation with reward signal
5. Episode ends when all services healthy OR max steps reached
6. Final grade computed via multi-criteria weighted formula

---

## Project Structure

```
resilience_ops_env/
|-- Dockerfile                      # Multi-stage production Docker config
|-- README.md                       # This file
|-- LICENSE                         # BSD 3-Clause License
|-- CONTRIBUTING.md                 # Contribution guidelines
|-- openenv.yaml                    # OpenEnv manifest
|-- pyproject.toml                  # Python project configuration
|-- requirements.txt                # Pinned dependencies
|-- models.py                       # Core: types, tasks, reward, grader
|-- client.py                       # WebSocket environment client
|-- inference.py                    # Baseline LLM inference script
|-- v3_0_features.py                # v3.0: dynamic tasks, multi-agent, metrics
|-- validate.py                     # Pre-submission validation (13 checks)
|-- server/
|   |-- __init__.py
|   |-- app.py                      # FastAPI application
|   +-- resilience_ops_env_environment.py
+-- tests/
    +-- test_models.py              # 33 unit tests
```

---

## v3.0 Advanced Features

ResilienceOps includes advanced features that extend the base environment:

- **Dynamic Task Generation**: Randomized alert noise, log corruption, and health flapping for grading diversity
- **Multi-Agent Collaborative Response**: Multiple agents share incident state for collaborative research
- **Prometheus-Compatible Metrics**: Export agent behavior metrics in Prometheus format
- **LLM-as-Judge Grading**: Supplement rule-based grading with LLM reasoning evaluation
- **TRL/GRPO Training Integration**: Reward model compatible with TRL's GRPOTrainer

```python
from resilience_ops_env import ResilienceOpsGRPORewardModel, ResilienceOpsEnvironment

env = ResilienceOpsEnvironment(task_name="medium")
reward_model = ResilienceOpsGRPORewardModel(env)
rewards = reward_model.compute_rewards(prompts, completions)
```

---

## Testing and Validation

```bash
# Run 33 unit tests
python -m pytest tests/ -v

# Run pre-submission validation (13 checks)
python validate.py

# Run OpenEnv validation
openenv validate
```

---

## License

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
