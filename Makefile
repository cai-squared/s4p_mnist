.PHONY: install dev data train predict test lint format clean docker_build docker_run docs

# Note: 'uv' is a faster alternative to pip. Install with: pip install uv
# Then replace 'pip install' with 'uv pip install' in the commands below.

install:
	pip install -U pip
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install -r requirements_dev.txt
	pre-commit install

data:
	python -m S4P_MNIST.data.make_dataset

train:
	python -m S4P_MNIST.train_model

predict:
	python -m S4P_MNIST.predict_model

test:
	pytest tests/

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name build -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

docker_build:
	docker build -t S4P_MNIST -f dockerfiles/Dockerfile .

docker_run:
	docker run --rm S4P_MNIST

docs:
	mkdocs serve
