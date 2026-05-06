# PHASE 1: Project Design & Model Development

## Overview
Phase 1 establishes the foundation for your MLOps project. This phase covers project planning, initial code organization, team collaboration setup, data handling, baseline model development, and comprehensive documentation. By the end of this phase, you should have a well-organized repository with a trained baseline model and clear documentation for future team members.

---

## 1. Project Proposal

- [ ] **Scope & Objectives**: Define the problem statement, goals, and success metrics for s4p_mnist
- [ ] **Detailed Description**: Write a 300+ word project description covering the business context, technical approach, and expected outcomes
- [ ] **Dataset Selection**: Choose appropriate dataset(s) and document the selection justification
- [ ] **Dataset Description**: Document dataset characteristics (size, features, format, sources)
- [ ] **Model Considerations**: Identify initial model architectures and algorithms suitable for your problem
- [ ] **Open-Source Tools**: Document and justify the selection of open-source tools and libraries for the project

---

## 2. Code Organization & Setup

- [ ] **GitHub Repository**: Create repository with cookiecutter MLOps structure
- [ ] **Environment Setup**: Configure Python virtual environment (venv or conda)
- [ ] **Dependency Management**: Create and maintain requirements.txt or pyproject.toml
- [ ] **Project Structure**: Organize code with clear separation of concerns (src/, tests/, data/, etc.)
- [ ] **Version Pinning**: Pin all critical dependencies to specific versions
- [ ] **Installation Documentation**: Document how to set up the development environment

---

## 3. Version Control & Collaboration

- [ ] **Regular Commits**: Establish commit discipline with descriptive, atomic commits
- [ ] **Branching Strategy**: Implement feature branching (e.g., git-flow or GitHub Flow)
- [ ] **Pull Request Process**: Establish PR template and review requirements
- [ ] **Team Roles**: Clearly define responsibilities (author: Cindy Cai, team members, reviewers)
- [ ] **Code Review Guidelines**: Document code review expectations and checklist
- [ ] **Commit History**: Maintain clean, readable git history for project traceability

---

## 4. Data Handling

- [x] **Data Cleaning Scripts**: `src/s4p_mnist/data/make_dataset.py` reads raw MNIST IDX binary files using a custom `MnistDataloader` class (validates magic numbers via `struct.unpack`: 2049 for labels, 2051 for images) and converts them to `.npy` arrays stored in `data/processed/`
- [x] **Normalization**: Pixel values divided by 255.0 to normalize to float32 [0.0, 1.0]. For the PyTorch CNN, additional channel-wise standardization is applied using MNIST mean (0.1307) and std (0.3081). For all sklearn models, `StandardScaler` is applied as the first step of the pipeline.
- [ ] **Data Augmentation**: Develop and document data augmentation strategies if applicable
- [x] **Data Documentation**: Dataset documented in README - 70,000 images (60k train / 10k test), 28×28 grayscale pixels, 10 digit classes, IDX binary format. EDA notebook includes a training label distribution bar chart confirming balanced class counts.
- [x] **Data Splits**: Standard MNIST split: 60,000 training / 10,000 test. CNN training additionally holds out 10% of training data as a validation split (`val_fraction=0.1`, configurable via Hydra config). Train/val split is reproducible via `torch.Generator().manual_seed(seed)` with seed=42.
- [x] **Data Validation**: `MnistDataloader` validates IDX magic numbers and raises `ValueError` on mismatch. Array shapes are immplicitly validated through pipeline structure.
- [x] **DVC Setup (Optional)**: DVC initialized - `.dvc/` directory and `.dvcignore` present in repository root for versioning raw and processed data files.

---

## 5. Model Training

- [x] **Training Environment**: Local CPU for all sklearn models. PyTorch CNN auto-detects CUDA via `torch.cuda.is_available()`. Fixed random seed (default: 42) via `src/s4p_mnist/utils/seed.py` for full reproducibility.
- [x] **Baseline Model**: Logistic Regression (`max_iter=2000`) with `StandardScaler + PCA(100)` pipeline. Achieves 92% test accuracy - serves as the primary baseline.
- [x] **Hyperparameter Configuration**: All hyperparameters managed via Hydra config in `configs/config.yaml`. CNN-specific: `epochs=15`, `learning_rate=1.2e-3`, `weight_decay=1e-4`, `batch_size=128`, `dropout=0.3`, `val_fraction=0.1`. sklearn models: `PCA(n_components=100)` + `StandardScaler` for all.
- [x] **Evaluation Metrics**: Accuracy, precision, recall, F1-score (macro and weighted averages) computed via `src/s4p_mnist/evaluation/metrics.py`. Confusion matrices plotted for all 6 models using `ConfusionMatrixDisplay`. Results: LR 92%, KNN 96%, SVM 97%, RF 94%, MLP 98%, CNN ~96%.
- [x] **Model Persistence**: sklearn models saved via `joblib`. CNN saved as a versioned bundle dict `{"kind": "s4p_mnist_torch_cnn_v1", "state_dict": ..., "config": ...}` using `joblib.dump`. All models saved to `models/` via `BaseModel.save(path)` / `BaseModel.load(path)` interface.
- [x] **Training Reproducibility**: Global seeds set via `src/s4p_mnist/utils/seed.py`. CNN train/val split uses `torch.Generator().manual_seed(seed)`. All hyperparameters tracked through Hydra configuration files and logged to WandB.
- [x] **Performance Baseline**:
  - Logistic Regression: **92%** test accuracy (baseline)
  - KNN (k=5): **96%** test accuracy
  - SVM (RBF kernel): **97%** test accuracy
  - Random Forest (100 estimators): **94%** test accuracy
  - MLP (128→64 hidden layers): **98%** test accuracy ← best sklearn model
  - CNN (3 conv blocks, PyTorch): **~96%** test accuracy

---

## 6. Documentation & Reporting

- [x] **README**: Comprehensive README created with:
  - [x] Project overview and objectives
  - [x] Setup and installation instructions (uv option and pip option)
  - [x] Quick start guide (`make data`, `make train`, `make predict`)
  - [x] Dependencies and requirements (`requirements.txt`, `requirements_dev.txt`, `pyproject.toml`)
  - [ ] Contributing guidelines - **CONTRIBUTING.md not yet created**
  - [x] License information (MIT License)
- [x] **Code Docstrings**: NumPy/Google-style docstrings throughout `src/s4p_mnist/`. All CNN model methods include type annotations.
- [x] **Code Style**: Ruff configured in `pyproject.toml` and enforced via `.pre-commit-config.yaml` pre-commit hooks.
- [x] **Type Hints**: Used consistently throughout `src/s4p_mnist/` — `dict[str, Any]`, `np.ndarray`, `torch.Tensor`, `Path`. Uses `from __future__ import annotations`.
- [x] **Type Checking**: mypy configured via `.pre-commit-config.yaml`.
- [x] **Makefile**: Makefile created with all required commands:
  - [x] `make install` — pip + requirements.txt + editable install
  - [x] `make dev` — install + requirements_dev.txt + pre-commit
  - [x] `make data` — runs data pipeline
  - [x] `make train` - runs training
  - [x] `make predict` — runs inference
  - [x] `make test` - pytest
  - [x] `make lint` / `make format` — ruff
  - [x] `make clean`, `make docker_build`, `make docker_run`, `make docs`
- [ ] **CONTRIBUTING.md**: Document contribution guidelines and development workflow
- [x] **API Documentation**: Public APIs documented via docstrings. MkDocs config present in `docs/` directory.

---

> **Checklist:** Use this as a guide for documenting your Phase 1 deliverables.
