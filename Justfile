set shell := ["bash", "-cu"]
set windows-shell := ["powershell", "-NoProfile", "-Command"]

# ------------------------
# Test tasks
# ------------------------

test:
    uv run pytest tests/

test-coverage:
    uv run pytest tests/ --cov=notebooks/utils --cov-report=html

test-ci:
    uv run pytest tests/ --cov=notebooks/utils --cov-report=term-missing

# Smoke test with argument
# Usage: just smoke-test smoke_bao
smoke-test file:
    uv run python -m tests.{{file}}

# ------------------------
# Lint / format / typing
# ------------------------

lint:
    uv run ruff check .

lint-fix:
    uv run ruff check --fix .

format:
    uv run ruff format .

typecheck:
    uv run mypy .