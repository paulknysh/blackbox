.PHONY: lint format test sure

RUN := $(if $(shell command -v uv >/dev/null 2>&1 && echo yes),uv run,)

lint: ## Run ruff linter
	$(RUN) ruff check .

format: ## Apply ruff formatting
	$(RUN) ruff format .

test: ## Run the test suite
	$(RUN) pytest -q

sure: lint format test ## lint, format, test