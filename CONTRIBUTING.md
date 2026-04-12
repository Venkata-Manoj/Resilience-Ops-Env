# Contributing to ResilienceOps

Thank you for your interest in contributing to the ResilienceOps environment!

## Getting Started

1. **Fork** the repository
2. **Clone** your fork locally
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest  # for development
   ```

## Development

### Running the Server

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
python -m pytest tests/ -v
```

### Running Validation

```bash
python validate.py
```

## Adding New Tasks

1. Define a new `TaskConfig` in `models.py` under the `TASKS` dictionary
2. Ensure the task has:
   - Unique `name` and `difficulty`
   - `severity` (P1, P2, or P3)
   - `affected_services` list
   - `correct_diagnostic_tools` and `correct_remediation_sequence`
   - `max_steps` limit
   - `hints` for guiding agents
   - `initial_service_health` and `service_health_after_remediation`
3. Add corresponding fallback action in `inference.py`
4. Add tests in `tests/test_models.py`

## Code Style

- Use type hints on all functions
- Follow Pydantic patterns for data models
- Ensure all graders produce scores in [0.0, 1.0]
- Maintain deterministic grading (no global mutable state)

## Pull Request Guidelines

- Include tests for new functionality
- Run `python validate.py` and ensure all checks pass
- Keep changes focused — one feature per PR

## License

By contributing, you agree that your contributions will be licensed under the BSD 3-Clause License.
