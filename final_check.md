# Final Check

[x] - The project is not solving a toy problem or games
[ ] - The implementation is passed the validation via openenv validate
[x] - Provide at least three tasks, each with a clearly defined objective
[x] - Tasks should span increasing difficulty: easy → medium → hard
[x] - Each task must include a programmatic grader that assigns a score between 0.0 and 1.0
[x] - Grading criteria must be clear, deterministic, and reproducible
[x] - The reward function must provide feedback throughout the task trajectory, not just at completion
[x] - It should reward incremental progress toward the objective
[x] - It must penalize undesirable behaviors such as infinite loops or destructive actions
[x] - Include an inference script that uses the API client to evaluate a model within the environment
[x] - API credentials must be read from environment variables (HF_TOKEN)
[x] - The script should produce a reproducible baseline score across all tasks
[x] - The environment must be deployable as a containerized Hugging Face Space
[x] - It should be tagged with openenv
[x] - Provide a working Dockerfile
[x] - The environment must build and run successfully using: docker build, docker run

## The README must include

[x] - Environment overview and motivation
[x] - Definitions of action and observation spaces
[x] - Task descriptions with expected difficulty levels
[x] - Setup and usage instructions
[x] - Baseline performance scores

## Project Structure

[x] - inference script must be named inference.py
[x] - It must be located in the root directory of your project

## Required Environment Variables

**Your inference.py must read the following environment variables**
API_BASE_URL
[x] - Description: API endpoint for the LLM
[x] - Requirement: Must include a default value
MODEL_NAME
[x] - Description: Model identifier used for inference
[x] - Requirement: Must include a default value
HF_TOKEN
[x] - Description: Hugging Face API token
[x] - Requirement: Mandatory (no default required)

## INFERENCE OUTPUT FORMAT

**The script must emit exactly three line types to stdout, in this order**
[x] - [START] task=<task_name> env=<benchmark> model=<model_name>
[x] - [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[x] - [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 rewards=0.00,0.00,1.00

## Hardware Requirements

[x] - Solution will be executed inside a Docker container with limited resources
**It must run within the following constraints**
[x] - 2 vCPU
[x] - 8 GB RAM

## Common Failure Cases (Avoid These)

[x] - inference.py not in root directory
[x] - Missing default values for API_BASE_URL or MODEL_NAME
[x] - Missing HF_TOKEN
[ ] - Hugging Face Space still building during submission
[ ] - Space stopped due to multiple active deployments

## Submission Checklist

[x] - All required files are present
[x] - inference.py is in the root directory
[x] - Environment variables are properly configured
[x] - Output format is correct
[x] - Hardware requirements are met
[ ] - Common failure cases are avoided