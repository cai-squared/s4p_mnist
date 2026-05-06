# PHASE 1: Project Design & Model Development

## Overview

Phase 1 establishes the foundation for the S4P MNIST MLOps project. This phase covers problem scoping, repository setup, team collaboration workflows, MNIST data pipeline construction, baseline model development across six algorithms, and comprehensive documentation.

By the end of this phase, the S4P team has delivered:
- A clean, versioned data pipeline that ingests raw MNIST IDX binary files and produces processed `.npy` arrays
- Six trained and benchmarked classifiers (Logistic Regression, KNN, SVM, Random Forest, MLP, and a custom PyTorch CNN), with test accuracies ranging from 92% to >99%
- A saved model artifact ready for API serving in Phase 2
- Full project documentation enabling any team member to reproduce training end-to-end with a single `make train` command

## 1. Project Proposal

### 1.1 Project Scope and Objectives

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

**Detailed Description**:

Optical character recognition (OCR) and digit classification underpin critical systems across industries. Banks rely on automated cheque reading to process millions of transactions daily; postal services use handwritten digit recognition to sort mail by zip code; healthcare providers digitize handwritten patient forms to feed into electronic health records. Even as deep learning has advanced, robust, reproducible digit classifiers remain a core building block in production ML systems. Building such a system from scratch while adhering to MLOps best practices provides a direct analogue to real-world production ML workflows.

S4P MNIST takes an end-to-end ML engineering approach across three phases. In Phase 1, the focus is on data pipeline construction, model development, and baseline benchmarking. Raw MNIST data is ingested from IDX binary files using a custom dataloader that validates file integrity via magic number checks, then converted to NumPy arrays stored in a versioned `data/processed/` directory tracked by DVC.

Six classification algorithms are systematically evaluated: Logistic Regression as a linear baseline, K-Nearest Neighbors as a non-parametric approach, Support Vector Machine with RBF kernel for margin-based classification, Random Forest as an ensemble tree method, Multi-Layer Perceptron as a shallow neural network, and a custom PyTorch Convolutional Neural Network as the deep learning approach. All scikit-learn models share a common preprocessing pipeline (StandardScaler + PCA with 100 components) to reduce dimensionality from 784 to 100 features while retaining >95% variance. The PyTorch CNN processes raw 28 x 28 pixels directly, using three convolutional blocks with BatchNorm, MaxPool, and AdaptiveAvgPool, trained with AdamW and a cosine learning rate schedule.

Configuration is managed through Hydra, enabling reproducible hyperparameter sweeps. Experiment metrics (accuracy, F1-score, confusion matrices) are logged to Weights & Biases for tracking and comparison across runs.

By the end of Phase 1, the project delivers: a clean, versioned data pipeline; six trained and evaluated models with benchmarked accuracies (ranging from 92% for Logistic Regression to >99% for CNN); a saved model artifact compatible with the FastAPI serving layer; and full documentation enabling any team member to reproduce training from scratch in a single command. Phases 2 and 3 build on this foundation to deliver containerized deployment, monitoring, and CI/CD automation.

### 1.2 Selection of Data

**Dataset Selection**: MNIST - 70,000 grayscale 28 x 28 images across 10 digit classes. Chosen for its standardized IDX binary format, balanced class distribution (~6,000 samples per class), and widespread use as an ML engineering benchmark.

**Dataset Description**: 60,000 training + 10,000 test images. Pixel values are uint8 in range [0, 255]. Labels are integers 0-9. Raw data stored as IDX binary files in `data/raw/`. Class distribution verified as roughly balanced via EDA notebook bar chart.

### 1.3 Model Considerations

6 algorithms were evaluated - all sklearn models use a StandardScaler + PCA(100 components) preprocessing pipeline. Best sklearn model: MLP with hidden layers [128, 64] at 98% accuracy. CNN uses PyTorch with 3 convolutional blocks (32->64->128 channels), BatchNorm, MaxPool, AdaptiveAvgPool, dropout=0.3, AdamW optimizer, CosineAnnealingLR over 15 epochs. The CNN model achieved 99.5% accuracy.

### 1.4 Open-Source Tools

- **PyTorch 2.3.0** - Selected as the deep learning framework for the CNN model due to its dynamic computation graph, strong GPU support, and active research community. PyTorch's `nn.Module` API provides clean architectural abstractions (Conv2d, BatchNorm2d, AdaptiveAvgPool2d), and `DataLoader` handles batching and shuffling. Chosen over TensorFlow for its more Pythonic interface and better alignment with research reproducibility standards.

- **scikit-learn 1.5.0** - Used for all traditional ML baselines (LR, KNN, SVM, RF, MLP). Selected for its unified `Pipeline` API, which cleanly chains preprocessing (StandardScaler, PCA) with classification, preventing data leakage. Its `classification_report` and `ConfusionMatrixDisplay` utilities provide standardized evaluation across all models.

- **Weights & Biases (wandb) 0.18.0** - Selected for experiment tracking because it automatically logs metrics, hyperparameters, and model artifacts with minimal code changes. Enables direct comparison of all six model runs in a shared dashboard, which is critical for communicating results across team members.

- **DVC 3.55.0** - Selected for data version control to ensure the raw and processed MNIST data files are tracked alongside code. DVC integrates with Git to allow reproducible data snapshots without storing large binaries in the repository, following ML engineering best practices.

- **Hydra + OmegaConf 1.3.0 / 2.3.0** - Selected for configuration management because Hydra allows hyperparameters (learning rate, batch size, epochs, dropout) to be defined in YAML files and overridden from the command line. This eliminates hardcoded values and makes hyperparameter sweeps reproducible and auditable.

- **FastAPI** - Selected for model serving due to its async support, automatic OpenAPI documentation generation, and Pydantic-based request validation. Significantly faster to develop and deploy than Flask while maintaining production-grade performance.

- **Docker** - Selected for containerization to ensure the full pipeline (data processing, training, serving) runs identically across developer machines and cloud environments, eliminating "works on my machine" failures.

- **Ruff** - Selected as the linter and formatter (replacing flake8 + black + isort) because it is 10-100x faster than equivalent tools and handles all three responsibilities in a single tool, simplifying the pre-commit configuration.

- **mypy** - Selected for static type checking to catch type errors before runtime. Particularly important for the model interface (BaseModel subclass) where incorrect types (e.g., passing a list instead of np.ndarray) could cause silent failures in production.

## 2. Code Organization and Setup

### 2.1 Repository Setup

The s4p_mnist Github repository was created from a Cookiecutter template. The repository structure is detailed in the README.md under the Project Structure section.

### 2.2 Environment Setup

For this project, we'll create a conda environment. To do so, we run

```
conda create -n s4p_environment python=3.11
```

To install the required packages, use the `make install` or `make dev` commands. This will install the packages described by requirements.txt and/or requirements_dev.txt.

## 3. Version Control and Collaboration

### 3.1 Git Usage

Since Cindy, Subodh, and Riffa are primarily the programmers for the project, 3 GitHub branches were created from main for each of them to work on. This allows them to work individually on their own parts without conflicts, and when they are satisfied with their work they can create pull requests asking to be merged to main. Since Saumyaa is working on the documentation, it was decided that she could simply work on main.

One can clearly trace the progress of the project through the commit history. Pre-commit hooks and GitHub Actions ensure that changes made comply with Python standards

### 3.2 Team Collaboration

Besides GitHub, the S4P team regularly communicates over Discord about their work. They also have weekly meetings over Zoom to delegate work and finalize code.

## 4. Data Handling

### 4.1 Data Preparation

**Data Cleaning Scripts**: `src/s4p_mnist/data/make_dataset.py` reads raw MNIST IDX binary files using a custom `MnistDataloader` class (validates magic numbers via `struct.unpack`: 2049 for labels, 2051 for images) and converts them to `.npy` arrays stored in `data/processed/`

**Normalization**: Pixel values divided by 255.0 to normalize to float32 [0.0, 1.0]. For the PyTorch CNN, additional channel-wise standardization is applied using MNIST mean (0.1307) and std (0.3081). For all sklearn models, `StandardScaler` is applied as the first step of the pipeline.

**Data Splits**: Standard MNIST split: 60,000 training / 10,000 test. CNN training additionally holds out 10% of training data as a validation split (`val_fraction=0.1`, configurable via Hydra config). Train/val split is reproducible via `torch.Generator().manual_seed(seed)` with seed=42.

**Data Validation**: `MnistDataloader` validates IDX magic numbers and raises `ValueError` on mismatch. Array shapes are immplicitly validated through pipeline structure.

**DVC Setup**: DVC initialized - `.dvc/` directory and `.dvcignore` present in repository root for versioning raw and processed data files.

### 4.2 Data Documentation

Dataset is documented in README - 70,000 images (60k train / 10k test), 28 x 28 grayscale pixels, 10 digit classes, IDX binary format. EDA notebook includes a training label distribution bar chart confirming balanced class counts.

## 5. Model Training

### 5.1 Training Infrastructure

Local CPU is used for all sklearn models. PyTorch CNN auto-detects CUDA via `torch.cuda.is_available()`. Fixed random seed (default: 42) from `src/s4p_mnist/config.py` for full reproducibility.

### 5.2 Initial Training & Evaluation

- Logistic Regression: **92%** test accuracy
- KNN (k=5): **96%** test accuracy
- SVM (RBF kernel): **97%** test accuracy
- Random Forest (100 estimators): **94%** test accuracy
- MLP (128→64 hidden layers): **98%** test accuracy ← best sklearn model
- CNN (3 conv blocks, PyTorch): **>99%** test accuracy ← best model

The CNN model is chosen as the baseline model due to its high accuracy. The model is saved to `models/models.joblib`.

Currently, the hyperparameters for the CNN model are managed by the default configuration file `src/s4p_mnist/config.py`. In the following phase, hyperparameters will be managed by Hydra, and different experiments will be tracked by WandB.

## 6. Documentation & Reporting

### 6.1 Project README

A comprehensive README was created with a summary of the project and a system architecture diagram.

### 6.2 Code Documentation

**Code Docstrings**: NumPy/Google-style docstrings throughout `src/s4p_mnist/`. All CNN model methods include type annotations.

**Code Style**: Ruff configured in `pyproject.toml` and enforced via `.pre-commit-config.yaml` pre-commit hooks.

**Type Hints**: Used consistently throughout `src/s4p_mnist/` — `dict[str, Any]`, `np.ndarray`, `torch.Tensor`, `Path`. Uses `from __future__ import annotations`.

**Type Checking**: mypy configured via `.pre-commit-config.yaml`.

**MakeFile**:

- `make install` - installs packages in requirements.txt
- `make dev` — make dev + installs additional packages in requirements_dev.txt
- `make data` — runs data pipeline from raw -> processed
- `make train` - trains the model
- `make predict` — runs inference on test dataset

## Key Findings & Challenges

Key findings were that the CNN model could achieve near-human accuracy, at 99.5%. In the following phase, hyperparameters will be tweaked to see if accuracy can be pushed even higher.

Challenges were following Python formatting standards as well as coordinating different parts of the project. In the following phase we will try to improve our communication.
