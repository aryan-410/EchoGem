.PHONY: help install install-dev test lint format clean build publish

help: ## Show this help message
	@echo "EchoGem Development Commands"
	@echo "============================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install EchoGem in development mode
	pip install -e .

install-dev: ## Install EchoGem with development dependencies
	pip install -e .[dev]

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=echogem --cov-report=html

lint: ## Run linting checks
	flake8 echogem/ tests/
	mypy echogem/

format: ## Format code with black and isort
	black echogem/ tests/
	isort echogem/ tests/

format-check: ## Check if code is properly formatted
	black --check echogem/ tests/
	isort --check-only echogem/ tests/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build distribution packages
	python setup.py sdist bdist_wheel

publish: ## Publish to PyPI (requires twine)
	twine upload dist/*

check-dist: ## Check distribution packages
	check-manifest
	twine check dist/*

docs: ## Build documentation
	cd docs && make html

install-deps: ## Install all dependencies
	pip install -r requirements.txt
	pip install -e .[dev,docs]

setup-dev: ## Complete development setup
	@echo "Setting up development environment..."
	pip install --upgrade pip
	make install-deps
	python -m spacy download en_core_web_sm
	@echo "Development environment ready!"

check-env: ## Check environment variables
	@echo "Checking environment variables..."
	@if [ -z "$$GOOGLE_API_KEY" ]; then echo "❌ GOOGLE_API_KEY not set"; else echo "✅ GOOGLE_API_KEY is set"; fi
	@if [ -z "$$PINECONE_API_KEY" ]; then echo "❌ PINECONE_API_KEY not set"; else echo "✅ PINECONE_API_KEY is set"; fi

run-example: ## Run basic example
	@echo "Running basic example..."
	python examples/basic_usage.py

run-advanced: ## Run advanced example
	@echo "Running advanced example..."
	python examples/advanced_usage.py

run-graph: ## Run graph visualization example
	@echo "Running graph visualization example..."
	python examples/graph_visualization.py

cli-test: ## Test CLI functionality
	@echo "Testing CLI..."
	py -m echogem.cli --help
	py -m echogem.cli stats

all: format lint test ## Run all checks (format, lint, test)

pre-commit: format lint test ## Run pre-commit checks

ci: format-check lint test ## Run CI checks
