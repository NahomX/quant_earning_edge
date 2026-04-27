.PHONY: install lint format type test test-property cov mlflow clean

install:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

type:
	uv run mypy src tests

test:
	uv run pytest

test-property:
	uv run pytest -m property -v

cov:
	uv run pytest --cov=src/quant_earning_edge --cov-report=term-missing --cov-report=html

mlflow:
	docker compose up -d mlflow

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
