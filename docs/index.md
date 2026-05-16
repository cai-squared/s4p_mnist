# S4P_MNIST

machine learning for MNIST

## Overview

Welcome to s4p_mnist! This project is designed to provide a scalable, production-ready machine learning pipeline.

## Quick Start

### Installation

```bash
# Using pip
pip install -r requirements.txt

# Using uv (faster alternative)
uv pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Prepare data
make data

# Train the model
make train

# Generate predictions
make predict
```

## Documentation

- [Getting Started](getting_started.md)
- [API Reference](api.md)

## Project Structure

```
s4p_mnist/                  # Repository root
├── src/
│   └── s4p_mnist/          # Importable package (src/ layout)
│       ├── config.py                  # Paths + typed config
│       ├── logging_config.py
│       ├── data/                      # Loaders + raw→processed pipeline
│       ├── features/                  # Feature engineering
│       ├── models/                    # BaseModel ABC + concrete Model
│       ├── evaluation/                # Metric helpers
│       ├── visualization/             # Plot helpers
│       ├── utils/                     # seed, io
│       ├── train_model.py             # Training CLI
│       └── predict_model.py           # Inference CLI
├── data/                              # raw/ and processed/
├── models/                            # Trained artifacts
├── tests/                             # Unit tests
├── docs/                              # MkDocs docs
├── Makefile                           # Common commands
└── pyproject.toml                     # Packaging & deps
```

## License

This project is licensed under the MIT License. See LICENSE for details.
