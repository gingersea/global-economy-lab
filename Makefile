.PHONY: install update-data test notebook dashboard clean lint format

# ── Install dependencies ─────────────────────────────────────
install:
	pip install -r requirements.txt

install-poetry:
	poetry install

# ── Data pipeline ────────────────────────────────────────────
update-data:
	python scripts/update_data.py

# ── Tests ─────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Notebooks ────────────────────────────────────────────────
notebook:
	jupyter notebook notebooks/

# ── Dashboard ────────────────────────────────────────────────
dashboard:
	python scripts/run_dashboard.py

# ── Code quality ─────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/ config/

format:
	black src/ tests/ scripts/ config/

# ── Clean ─────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ .coverage htmlcov/ .ruff_cache/ .mypy_cache/
	@echo "Cleaned up cache and build artifacts."
