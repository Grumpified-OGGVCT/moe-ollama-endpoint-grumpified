# Makefile for MoE Ollama Endpoint

.PHONY: help install dev-install test lint format run build docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make dev-install  - Install development dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make run          - Run the application"
	@echo "  make build        - Build with Podman"
	@echo "  make docker-run   - Run with Podman Compose"
	@echo "  make clean        - Clean temporary files"

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio black ruff mypy

test:
	pytest -v

lint:
	ruff check app/ tests/
	mypy app/

format:
	black app/ tests/ examples/

run:
	python -m app.main

build:
	podman build -t moe-ollama-endpoint -f Containerfile .

docker-run:
	podman-compose up -d

docker-stop:
	podman-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
