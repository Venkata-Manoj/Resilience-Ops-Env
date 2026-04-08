# MetaXHF - OpenEnv Reinforcement Learning Environment

## Project Structure Rules

- Dockerfile must be placed in root directory
- `models.py` consists of all project functions and definitions
- Inference script must be named `inference.py` and placed in root directory

## Docker Configuration

- `ENV ENABLE_WEB_INTERFACE=true` — place in Dockerfile after `ENV PYTHONPATH` & below `HEALTH CHECK`
- Follow the docker commands shown in the `openenv init` command (2 commands)
- Change docker file from server folder to root folder
- Run docker in background

## Setup Commands

```bash
pip install "openenv-core[cli]"

openenv init <your_project_name>

# Change docker file from server folder to root folder
# Initialise the inference.py file
# Run docker in background
# Run two commands of docker
# Then run inference.py file
# Check the web at 0.0.0.0/8000/web/
```

## Local Testing

```bash
uv run server
```

## Deployment

```bash
openenv push --repo-id your-username/my-env
```

## Key Requirements

- Use any LLMs at Hugging Face
- Project must function at all 3 states, run and validate at HF Spaces
- Frameworks: only OpenEnv environment by Meta and Hugging Face

### Criteria

- Must have correct runtime without any errors
- Follow OpenEnv standard inference compliance
- Clear, realistic, testable task designs
- Gradient system has the reward system makes sense

### Must Submit

- Public GitHub repo with environment code
- `requirements.txt`
- Demo script
- README file
- Deployed HF Space URL to showcase the demo

## OpenEnv Specification

Implement the full OpenEnv interface
- Typed observation
- Action
- Reward Pydantic models
- `step(action)` → returns observation, reward, done, info
- `reset()` → returns initial observation
- `state()` → returns current state
- `openenv.yaml` with metadata
- Tested via `openenv validate`

## Task Requirements

- Minimum 3 tasks with agent graders (easy → medium → hard, scores/reward 0.0–1.0)
- Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0–1.0)
- Tasks should range: easy → medium → hard
- Graders must have clear, deterministic success/failure criteria
- Must simulate a real-world task (not games or toys)

## Reward Function

- Provides signal over the full trajectory (not just binary end-of-episode)
- Rewards partial progress toward task completion
- Penalizes clearly undesirable behavior (e.g. infinite loops, destructive actions)

## Baseline Inference

- Uses the OpenAI API client to run a model against the environment
- Reads API credentials from environment variables (`HF_TOKEN`)
- Produces a reproducible baseline score on all 3 tasks

## Docker Requirements

- Must include a working Dockerfile
- Environment should start cleanly with `docker build` + `docker run`

## README Requirements

- Environment description and motivation
- Action and observation space definitions
- Task descriptions with expected difficulty
- Setup and usage instructions
- Baseline scores

## Evaluation Rubrics

### Real-world utility (30%)

- 0–5: Toy/artificial problem with no practical application
- 6–15: Valid domain but shallow modeling of the real task
- 16–25: Good domain modeling, would be useful for agent evaluation
- 26–30: Excellent — fills a real gap, immediate value for the RL/agent community

### Task & grader quality (25%)

- 3+ tasks with difficulty range?
- Graders produce scores between 0.0–1.0?
- Graders deterministic and reproducible?
- Hard task genuinely challenges frontier models?

### Environment design (20%)

- `reset()` produces clean state?
- Action/observation types well-designed and documented?
- Reward function provides useful varying signal (not just sparse)?
- Episode boundaries sensible?

### Code quality & spec compliance (15%)

- `openenv validate` passes?
- `docker build` && `docker run` works?
- HF Space deploys and responds?
- Baseline script runs and reproduces scores?

### Creativity & novelty (10%)

- Domain we haven't seen in OpenEnv before?
- Reward design has interesting properties?
- Clever mechanics that make the environment engaging?

## Evaluation Phases

### Phase 1: Automated Validation

Pass/fail gate — HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.

### Phase 2: Agentic Evaluation

Scored — baseline agent re-run, standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments, score variance check.

### Phase 3: Human Review

Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checks.

### Disqualification Criteria

- Environment does not deploy or respond
- Plagiarized or trivially modified existing environments
- Graders that always return the same score
- No baseline inference script

## Pre-Submission Checklist

All must pass or you're disqualified:

- [ ] HF Space deploys
- [ ] Automated ping to the Space URL — must return 200 and respond to `reset()`
- [ ] OpenEnv spec compliance — Validate `openenv.yaml`, typed models, `step()`/`reset()`/`state()` endpoints
- [ ] Dockerfile builds — Automated docker build on the submitted repo
- [ ] Baseline reproduces — Run the submitted inference script — must complete without error and produce scores
- [ ] 3+ tasks with graders — Enumerate tasks, run each grader, verify scores/reward in 0.0–1.0 range

## Environment Configuration

Before submitting, ensure the following variables are defined in your environment configuration:

- `API_BASE_URL` — The API endpoint for the LLM
- `MODEL_NAME` — The model identifier to use for inference
- `HF_TOKEN` — Your Hugging Face / API key

## Inference Script Requirements

- Must be named `inference.py` and placed in the root directory of the project
- Must use OpenAI Client for all LLM calls using above variables
- Must emit structured stdout logs strictly following the `[START]`, `[STEP]`, and `[END]` format
- Any deviation in field names, ordering, or formatting will result in incorrect evaluation scoring

## Infrastructure Restrictions

- Runtime of inference script should be less than 20min
- Make sure your env and inference can run on a machine with vcpu=2, memory=8gb

## Validator

Run the pre-submission validation script before submitting.


### User Instructions

- from now on use the current directory as the working directory
- kill the servers if the work is completed
- always write the strong and functional code
- ask the user before making any changes
- as you work, keep updating this file with your progress
- as the user if anything manually needs to be done
- if you encounter any issues, ask the user for help if it wants to do it manually
