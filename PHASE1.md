# PHASE 1: Project Design & Model Development

## Overview
Phase 1 establishes the foundation for the S4P MNIST MLOps project. This phase covers problem scoping, repository setup, team collaboration workflows, MNIST data pipeline construction, baseline model development across six algorithms, and comprehensive documentation.

By the end of this phase, the S4P team has delivered:
- A clean, versioned data pipeline that ingests raw MNIST IDX binary files and produces processed `.npy` arrays
- Six trained and benchmarked classifiers (Logistic Regression, KNN, SVM, Random Forest, MLP, and a custom PyTorch CNN), with test accuracies ranging from 92% to 98%
- A saved model artifact ready for API serving in Phase 2
- Full project documentation enabling any team member to reproduce training end-to-end with a single `make train` command

---

## 1. Project Proposal

- [x] **Scope & Objectives**:
      
  **Problem Statement**: Handwritten digit recognition is a foundational computer vision problem with broad real-world applications in postal automation, bank cheque processing, form digitization, and accessibility tools. The MNIST dataset containing 70,000 labeled grayscale images of handwritten digits (0-9) serves as the standard benchmark for evaluating digit classification systems. The core problem is to build a classifier that maps a 28 x 28 pixel image to one of 10 digit classes with high accuracy and reliability.

  **Goals**:
  - Design and train a high-accuracy digit classification model achieving >=95% test accuracy on the MNIST held-out test set
  - Benchmark multiple ML algorithms (Logistic Regression, KNN, SVM, Random Forest, MLP, CNN) to identify the best-performing approach
  - Build a fully reproducible ML pipeline from raw data ingestion through to model prediction
  - Containerize and automate the pipeline using Docker and CI/CD for consistent, scalable execution
  - Deploy the trained model as a live, user-accessible API capable of making real-time predictions on new handwritten digit inputs

  **Success Metrics**:
  - Primary: >=95% classification accuracy on the 10,000 sample MNIST test set
  - Secondary: Per class F1-score >= 0.93 for all 10 digit classes
  - Operational: End-to-end pipeline runs reproducibly from a single `make train` command with fixed random seed
  - Deployment: Model accessible via a live FastAPI endpoint returning predictions in <100ms per request
- [x] **Detailed Description**:

  **Business Context**:
  Optical character recognition (OCR) and digit classification underpin critical systems across industries. Banks rely on automated cheque reading to process millions of transactions daily; postal services use handwritten digit recognition to sort mail by zip code; healthcare providers digitize handwritten patient forms to feed into electronic health records. Even as deep learning has advanced, robust, reproducible digit classifiers remain a core building block in production ML systems. Building such a system from scratch while adhering to MLOps best practices provides a direct analogue to real-world production ML workflows.

  **Technical Approach**:
  S4P MNIST takes an end-to-end ML engineering approach across three phases. In Phase 1, the focus is on data pipeline construction, model development, and baseline benchmarking. Raw MNIST data is ingested from IDX binary files using a custom dataloader that validates file integrity via magic number checks, then converted to NumPy arrays stored in a versioned `data/processed/` directory tracked by DVC.

  Six classification algorithms are systematically evaluated: Logistic Regression as a linear baseline, K-Nearest Neighbors as a non-parametric approach, Support Vector Machine with RBF kernel for margin-based classification, Random Forest as an ensemble tree method, Multi-Layer Perceptron as a shallow neural network, and a custom PyTorch Convolutional Neural Network as the deep learning approach. All scikit-learn models share a common preprocessing pipeline (StandardScaler + PCA with 100 components) to reduce dimensionality from 784 to 100 features while retaining >95% variance. The PyTorch CNN processes raw 28 x 28 pixels directly, using three convolutional blocks with BatchNorm, MaxPool, and AdaptiveAvgPool, trained with AdamW and a cosine learning rate schedule.

  Configuration is managed through Hydra, enabling reproducible hyperparameter sweeps. Experiment metrics (accuracy, F1-score, confusion matrices) are logged to Weights & Biases for tracking and comparison across runs.

  **Expected Outcomes**:
  By the end of Phase 1, the project delivers: a clean, versioned data pipeline; six trained and evaluated models with benchmarked accuracies (ranging from 92% for Logistic Regression to 98% for MLP); a saved model artifact compatible with the FastAPI serving layer; and full documentation enabling any team member to reproduce training from scratch in a single command. Phases 2 and 3 build on this foundation to deliver containerized deployment, monitoring, and CI/CD automation.
- [x] **Dataset Selection**: MNIST - 70,000 grayscale 28 x 28 images across 10 digit classes. Chosen for its standardized IDX binary format, balanced class distribution (~6,000 samples per class), and widespread use as an ML engineering benchmark.
- [x] **Dataset Description**: 60,000 training + 10,000 test images. Pixel values are uint8 in range [0, 255]. Labels are integers 0-9. Raw data stored as IDX binary files in `data/raw/`. Class distribution verified as roughly balanced via EDA notebook bar chart.
- [x] **Model Considerations**: Evaluated 6 algorithms - all sklearn models use a StandardScaler + PCA(100 components) preprocessing pipeline. Best sklearn model: MLP with hidden layers [128, 64] at 98% accuracy. CNN uses PyTorch with 3 convolutional blocks (32->64->128 channels), BatchNorm, MaxPool, AdaptiveAvgPool, dropout=0.3, AdamW optimizer, CosineAnnealingLR over 15 epochs.
- [x] **Open-Source Tools**:
  - **PyTorch 2.3.0** - Selected as the deep learning framework for the CNN model due to its dynamic computation graph, strong GPU support, and active research community. PyTorch's `nn.Module` API provides clean architectural abstractions (Conv2d, BatchNorm2d, AdaptiveAvgPool2d), and `DataLoader` handles batching and shuffling. Chosen over TensorFlow for its more Pythonic interface and better alignment with research reproducibility standards.

  - **scikit-learn 1.5.0** - Used for all traditional ML baselines (LR, KNN, SVM, RF, MLP). Selected for its unified `Pipeline` API, which cleanly chains preprocessing (StandardScaler, PCA) with classification, preventing data leakage. Its `classification_report` and `ConfusionMatrixDisplay` utilities provide standardized evaluation across all models.

  - **Weights & Biases (wandb) 0.18.0** - Selected for experiment tracking because it automatically logs metrics, hyperparameters, and model artifacts with minimal code changes. Enables direct comparison of all six model runs in a shared dashboard, which is critical for communicating results across team members.

  - **DVC 3.55.0** - Selected for data version control to ensure the raw and processed MNIST data files are tracked alongside code. DVC integrates with Git to allow reproducible data snapshots without storing large binaries in the repository, following ML engineering best practices.

  - **Hydra + OmegaConf 1.3.0 / 2.3.0** - Selected for configuration management because Hydra allows hyperparameters (learning rate, batch size, epochs, dropout) to be defined in YAML files and overridden from the command line. This eliminates hardcoded values and makes hyperparameter sweeps reproducible and auditable.

  - **FastAPI** - Selected for model serving due to its async support, automatic OpenAPI documentation generation, and Pydantic-based request validation. Significantly faster to develop and deploy than Flask while maintaining production-grade performance.

  - **Docker** - Selected for containerization to ensure the full pipeline (data processing, training, serving) runs identically across developer machines and cloud environments, eliminating "works on my machine" failures.

  - **Ruff** - Selected as the linter and formatter (replacing flake8 + black + isort) because it is 10-100x faster than equivalent tools and handles all three responsibilities in a single tool, simplifying the pre-commit configuration.

  - **mypy** - Selected for static type checking to catch type errors before runtime. Particularly important for the model interface (BaseModel subclass) where incorrect types (e.g., passing a list instead of np.ndarray) could cause silent failures in production.

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
- [x] **Data Documentation**: Dataset documented in README - 70,000 images (60k train / 10k test), 28 x 28 grayscale pixels, 10 digit classes, IDX binary format. EDA notebook includes a training label distribution bar chart confirming balanced class counts.
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
