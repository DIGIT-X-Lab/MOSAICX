# MOSAICX Development Makefile
# ============================

.DEFAULT_GOAL := help

# ── Setup ──────────────────────────────────────────────────

.PHONY: install
install: ## Install in editable mode with dev dependencies
	pip install -e ".[dev]"

.PHONY: install-all
install-all: ## Install with all optional dependencies
	pip install -e ".[all,dev]"

# ── Quality ────────────────────────────────────────────────

.PHONY: lint
lint: ## Run ruff linter (check + fix)
	ruff check --fix mosaicx/ tests/
	ruff format mosaicx/ tests/

.PHONY: lint-check
lint-check: ## Run ruff without auto-fixing
	ruff check mosaicx/ tests/
	ruff format --check mosaicx/ tests/

.PHONY: typecheck
typecheck: ## Run mypy type checker
	mypy mosaicx/

# ── Testing ────────────────────────────────────────────────

.PHONY: test
test: ## Run all tests
	pytest tests/ -q

.PHONY: test-unit
test-unit: ## Run unit tests only
	pytest tests/ -q -m unit

.PHONY: test-integration
test-integration: ## Run integration tests only
	pytest tests/ -q -m integration

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	pytest tests/ --cov=mosaicx --cov-report=term-missing

# ── Combined ───────────────────────────────────────────────

.PHONY: check
check: lint-check typecheck test ## Run all quality checks (lint + types + tests)

.PHONY: validate
validate: ## Quick smoke test (CLI + imports)
	@mosaicx --help > /dev/null && echo "CLI: ok"
	@python -c "import mosaicx; print('Import: ok')"

# ── Cleanup ────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Help ───────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@echo "MOSAICX Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
