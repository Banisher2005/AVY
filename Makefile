.PHONY: help install ingest build-index test eval serve lint clean

VENV ?= .venv
PYTHON = $(VENV)/bin/python
UV = /home/abhinav/.local/bin/uv

help:
	@echo "AVY — Streaming Live RAG Assistant (Samsung PRISM Hackathon)"
	@echo "Available commands:"
	@echo "  make install      - Create virtualenv and install dependencies"
	@echo "  make ingest       - Ingest corpus markdown documents"
	@echo "  make build-index  - Build FAISS vector index from corpus"
	@echo "  make test         - Run all test suites"
	@echo "  make eval         - Run automated evaluation harness"
	@echo "  make serve        - Start Web UI & Streaming SSE server"
	@echo "  make demo         - Run interactive CLI demo scenarios"
	@echo "  make lint         - Run ruff code linter"
	@echo "  make clean        - Remove caches and build artifacts"

install:
	$(UV) venv $(VENV) --python 3.12
	$(UV) pip install -e . --python $(PYTHON)
	$(UV) pip install pytest pytest-asyncio ruff --python $(PYTHON)

ingest:
	$(PYTHON) scripts/ingest.py

build-index:
	$(PYTHON) scripts/build_index.py

test:
	$(PYTHON) -m pytest tests/ -v

eval:
	$(PYTHON) scripts/evaluate.py

serve:
	$(PYTHON) scripts/demo_server.py

demo:
	$(PYTHON) scripts/run_demo.py

lint:
	$(PYTHON) -m ruff check src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache dist/ build/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
