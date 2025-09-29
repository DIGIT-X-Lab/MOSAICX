# MOSAICX Development Makefile
# ============================
# Convenient commands for MOSAICX development and validation

.PHONY: test pre-push-test validate install clean help

# Default target
help:
	@echo "🧬 MOSAICX Development Commands"
	@echo "=============================="
	@echo ""
	@echo "📋 Available Commands:"
	@echo "  make test           - Run all pre-push validation tests"
	@echo "  make pre-push-test  - Comprehensive pre-push validation (alias for test)"
	@echo "  make validate       - Quick validation (CLI + API checks)"
	@echo "  make install        - Install MOSAICX in development mode"
	@echo "  make clean          - Clean generated files and artifacts"
	@echo "  make help           - Show this help message"
	@echo ""
	@echo "🚀 Before pushing to production, always run: make test"

# Run comprehensive pre-push validation tests
test: pre-push-test

pre-push-test:
	@echo "🧪 Running comprehensive pre-push validation..."
	@bash scripts/pre-push-validation.sh

# Quick validation (faster subset of tests)
validate:
	@echo "⚡ Running quick validation..."
	@echo "Checking CLI health..."
	@mosaicx --help > /dev/null
	@echo "✅ CLI functional"
	@echo "Checking Python API imports..."
	@python -c "from mosaicx import generate_schema, extract_pdf, summarize_reports; print('✅ API imports successful')"
	@echo "🎉 Quick validation passed!"

# Install in development mode
install:
	@echo "📦 Installing MOSAICX in development mode..."
	pip install -e .
	@echo "✅ Installation complete"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
	@rm -f output/summary_*.json 2>/dev/null || true
	@rm -f schemas/generated_*.py 2>/dev/null || true
	@rm -f /tmp/mosaicx_test_*.log 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Test individual components
test-cli:
	@echo "🧪 Testing CLI functionality..."
	@mosaicx generate --desc "Test schema generation" --model mistral:latest
	@echo "✅ CLI test passed"

test-api:
	@echo "🧪 Testing Python API..."
	@python -c "
from mosaicx import generate_schema;
schema = generate_schema('Test API', model='mistral:latest');
print('✅ API test passed')
"

# Development workflow helpers
dev-setup: install
	@echo "🛠️  Development environment setup complete!"
	@echo "Run 'make test' to validate everything is working"

# Pre-commit hook (can be used with git hooks)
pre-commit: validate
	@echo "✅ Pre-commit validation passed"