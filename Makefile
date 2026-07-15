.PHONY: sync lint typecheck test check install-local

sync:
	uv sync --dev

lint:
	uv run ruff check .

typecheck:
	uv run mypy src/mailweek

test:
	uv run pytest

check: lint typecheck test

install-local:
	uv tool install --force .
