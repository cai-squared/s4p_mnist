# PROJECT - S4P MNIST

## 👥 Team Information

- **Team Name:** S4P
- **Team Members:** *Cai Cindy (ccai5@depaul.edu)*
                    *Riffa Hammed (rriffaha@depaul.edu)*
                    *Sai Subodh Gundam Raju (sgundamr@depaul.edu)*
                    *Saumyaa Kannan (skannan3@depaul.edu)*
- **Course & Section:** SE489 ML ENGINEERING FOR PRODUCTION
                        Section:(930 Online: Sync - 910 Online: Async)

## 🧠 Project Overview

🚀 S4P MNIST is an end-to-end machine learning engineering project that designs, 
trains, evaluates, and deploys a handwritten digit classifier on the MNIST dataset - 70,000 grayscale images of handwritten digits around 10 classes (0-9) across 60,000 training 
and 10,000 test samples. The project goes beyond model accuracy, emphasizing 
production-grade MLOps practices: reproducible data pipelines, experiment 
tracking, containerization, continuous integration, and a live deployed 
interface for real-time predictions. Built across three phases, S4P MNIST 
demonstrates a complete ML project lifecycle from raw data ingestion through 
to a scalable, monitored, and user-accessible deployment.

🎯 **Key Objectives:**
- [ ] 🔬 Design and train a high-accuracy digit classification model with fully 
      reproducible data processing and experiment tracking
- [ ] 🐳 Containerize and automate the ML pipeline using Docker and CI/CD tools 
      to ensure consistent, scalable execution
- [ ] 🌐 Deploy the trained model as a live, user-accessible application capable 
      of making real-time predictions on new handwritten digit inputs

## Architecture Diagram

```
[Placeholder for architecture diagram]

Insert your system architecture diagram here, showing data flow, components,
and interactions between different parts of the system.
```

## Phase Deliverables

### Phase 1: Project Design & Model Development
- See [PHASE1.md](PHASE1.md) for detailed checklist
- See [docs/PHASE1.md](docs/PHASE1.md) for phase documentation.

### Phase 2: Containerization & Monitoring
- See [PHASE2.md](PHASE2.md) for detailed checklist

### Phase 3: CI/CD & Deployment
- See [PHASE3.md](PHASE3.md) for detailed checklist

## Setup Instructions

### Prerequisites
- Python 3.11+ installed
- Git installed
- (Optional) Docker and Docker Compose

### Installation

**Option 1: Using uv (recommended - faster)**
```bash
pip install uv
uv pip install -r requirements.txt
```

**Option 2: Using pip**
```bash
pip install -U pip
pip install -r requirements.txt
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

### Running the Pipeline

```bash
# Prepare data
make data

# Train the model
make train

# Generate predictions
make predict

# See all available commands
make help
```

## Technology Stack

### Core Dependencies
- **numpy** >= 1.26.0 - Numerical computing
- **pandas** >= 2.2.0 - Data manipulation
- **scikit-learn** >= 1.5.0 - Machine learning algorithms
- **matplotlib** >= 3.9.0 - Visualization
- **tqdm** >= 4.66.0 - Progress bars
- **pyyaml** >= 6.0 - Configuration files
### Experiment Tracking
- **mlflow** >= 2.16.0 - MLflow experiment tracking
### Configuration Management
- **hydra-core** >= 1.3.0 - Hydra configuration framework
- **omegaconf** >= 2.3.0 - Hierarchical configuration
### Data Version Control
- **dvc** >= 3.55.0 - Data Version Control

### Development Tools
- **pytest** >= 8.0 - Testing framework
- **pytest-cov** >= 5.0 - Code coverage
- **ruff** >= 0.6.0 - Linting and formatting
- **mypy** >= 1.11 - Static type checking
- **pre-commit** >= 3.8 - Git hooks framework

## Project Structure

This template uses the modern **`src/` layout** — the importable package lives in `src/S4P_MNIST/`, decoupled from the repository root. That forces `pip install -e .` before imports work, which catches packaging bugs early.

```
S4P_MNIST/                  # Repository root
├── src/
│   └── S4P_MNIST/          # Importable Python package
│       ├── __init__.py                # Version + package metadata
│       ├── config.py                  # Paths & typed config (PROJECT_ROOT, TrainingConfig, ...)
│       ├── logging_config.py          # setup_logging() + get_logger()
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py             # load_raw / load_processed / save_processed
│       │   └── make_dataset.py        # Raw → processed pipeline CLI
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py      # Feature engineering
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py                # BaseModel ABC (fit/predict/save/load)
│       │   └── model.py               # Concrete Model scaffold
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py             # classification_report, regression_report
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── visualize.py           # Plot helpers
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── io.py                  # JSON helpers
│       │   └── seed.py                # set_seed for reproducibility
│       ├── train_model.py             # Training CLI
│       └── predict_model.py           # Inference CLI
├── tests/                             # Unit and integration tests
│   ├── conftest.py
│   └── test_model.py
├── data/
│   ├── raw/                           # Immutable raw data
│   └── processed/                     # Cleaned, transformed data
├── models/                            # Trained model artifacts (.joblib)
├── notebooks/                         # Jupyter notebooks for exploration
├── reports/
│   └── figures/                       # Generated analysis and figures
├── docs/                              # MkDocs documentation
│   ├── mkdocs.yml
│   ├── index.md
│   ├── getting_started.md
│   └── api.md
├── dockerfiles/                       # Docker configuration
│   └── Dockerfile
├── configs/                           # Hydra configuration (if selected)
│   └── config.yaml
├── api/                               # FastAPI service (if selected)
├── .github/workflows/                 # GitHub Actions CI/CD
│   └── ci.yml
├── PHASE1.md                          # Phase 1 deliverables checklist
├── PHASE2.md                          # Phase 2 deliverables checklist
├── PHASE3.md                          # Phase 3 deliverables checklist
├── .pre-commit-config.yaml            # Pre-commit hooks (Ruff, mypy)
├── Makefile                           # Common commands
├── docker-compose.yaml                # Docker Compose setup
├── pyproject.toml                     # Project config & dependencies
├── requirements.txt                   # Runtime dependencies
├── requirements_dev.txt               # Development dependencies
├── LICENSE
└── README.md
```

### Why `src/` layout?

| | `src/` layout (this template) | Flat layout |
|---|---|---|
| Forces `pip install -e .` before import | ✅ | ❌ |
| Catches packaging bugs early | ✅ | ❌ |
| Adopted by | attrs, httpx, pydantic, flask, sqlalchemy | Older data-science templates |

Data and model artifacts are accessed via the constants in `S4P_MNIST.config` (`PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, …) rather than relative paths — code is independent of where you invoke it from.

## Common Commands

```bash
# Install package + runtime dependencies (editable install)
make install

# Install dev tools + pre-commit hooks
make dev

# Run linting and formatting checks
make lint

# Auto-format code
make format

# Run tests
make test

# Clean up build artifacts
make clean

# Docker operations
make docker_build
make docker_run

# Serve documentation locally
make docs
```

## Contribution Summary

- **Cindy Cai** — Data exploration, EDA notebook, code review
- **Riffa Hammed** — Data pipeline (raw MNIST IDX files → processed .npy arrays)
- **Subodh Gundam Raju** — Model development and training (6 algorithms including CNN, 96% accuracy)
- **Saumyaa Kannan** — Project documentation (README, PHASE1.md, project description)

## References

- [Project Documentation](docs/index.md)
- [Phase 1 — Project Design & Model Development](PHASE1.md)
- [Phase 2 — Containerization & Monitoring](PHASE2.md)
- [Phase 3 — CI/CD & Deployment](PHASE3.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
