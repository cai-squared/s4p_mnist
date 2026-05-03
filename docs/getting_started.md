# Getting Started with S4P_MNIST

## Prerequisites

- Python 3.11 or higher
- pip or uv package manager
- Git (for version control)
- Docker (optional, for containerized execution)

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd S4P_MNIST
```

### Step 2: Create a Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n S4P_MNIST python=3.11
conda activate S4P_MNIST
```

### Step 3: Install Dependencies

**Using uv (recommended):**
```bash
pip install uv
uv pip install -r requirements.txt
```

**Using pip:**
```bash
pip install -U pip
pip install -r requirements.txt
```

### Step 4: Set Up Development Environment

```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

## Running the Project

### Data Processing

Prepare your data for model training:

```bash
make data
```

Or directly:

```bash
python -m S4P_MNIST.data.make_dataset
```

### Model Training

Train the machine learning model:

```bash
make train
```

With custom parameters:

```bash
python -m S4P_MNIST.train_model --epochs 100 --batch-size 64
```

### Model Prediction

Generate predictions on new data:

```bash
make predict
```

With custom inputs:

```bash
python -m S4P_MNIST.predict_model --model-path models/model.pkl --input data/test.csv
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
pytest tests/ --cov=S4P_MNIST

# Run specific test file
pytest tests/test_model.py -v
```

### Code Quality

```bash
# Check for linting issues
make lint

# Auto-format and fix issues
make format

# Type checking
mypy src
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

```bash
# Manually run pre-commit checks
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## Docker Usage

### Build Docker Image

```bash
make docker_build
```

Or manually:

```bash
docker build -t S4P_MNIST -f dockerfiles/Dockerfile .
```

### Run with Docker Compose

```bash
docker-compose up
```

### Run Single Container

```bash
make docker_run
```

## Project Structure

```
S4P_MNIST/                  # Repository root
├── src/
│   └── S4P_MNIST/          # Importable package (src/ layout)
│       ├── config.py                  # Paths & typed config
│       ├── logging_config.py
│       ├── data/                      # Loaders + raw→processed pipeline
│       ├── features/                  # Feature engineering
│       ├── models/                    # BaseModel + concrete Model
│       ├── evaluation/                # Metrics
│       ├── visualization/
│       ├── utils/                     # seed, io
│       ├── train_model.py
│       └── predict_model.py
├── tests/                             # Unit tests
├── data/                              # raw/ and processed/
├── models/                            # Trained model artifacts
├── docs/                              # MkDocs documentation
├── configs/                           # Hydra configuration (optional)
├── pyproject.toml
├── requirements.txt
└── Makefile
```

## Configuration

### Using Hydra Configuration
Configuration is managed via Hydra. Edit `configs/config.yaml`:

```yaml
model:
  name: my_model
  type: sklearn
training:
  epochs: 100
  batch_size: 32
```

Override at runtime:

```bash
python -m S4P_MNIST.train_model \
  model.name=custom_model \
  training.epochs=200
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure:
1. Virtual environment is activated
2. Dependencies are installed: `pip install -r requirements.txt`
3. Package is installed in editable mode: `pip install -e .`

### CUDA/GPU Issues

If using PyTorch with GPU:

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only version if needed
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Pre-commit Hook Failures

If pre-commit hooks fail:

```bash
# See what failed
pre-commit run --all-files

# Fix issues manually or with auto-fix
make format

# Try committing again
```

## Next Steps

1. Review the [documentation](index.md)
2. Start with [Phase 1](PHASE1.md) - Data Exploration
3. Check the [API Reference](api.md)

## Support

For issues and questions:
- Check existing [documentation](index.md)
- Review [Phase deliverables](PHASE1.md)
- Contact Cindy Cai (ccai5@depaul.edu)
