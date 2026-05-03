# API Reference

The package is importable as `S4P_MNIST` after running `pip install -e .`.

## `S4P_MNIST.config`

Project-wide path constants and typed config dataclasses.

```python
from S4P_MNIST.config import (
    PROJECT_ROOT,
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    Config, TrainingConfig, DataConfig, DEFAULT_CONFIG,
)
```

Use these constants instead of hard-coded relative paths — they resolve against the repo root regardless of the current working directory.

## `S4P_MNIST.logging_config`

```python
from S4P_MNIST.logging_config import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)
```

## `S4P_MNIST.data`

| Function | Purpose |
|---|---|
| `load_raw(filename)` | Read CSV from `data/raw/` |
| `load_processed(filename)` | Read CSV from `data/processed/` |
| `save_processed(df, filename)` | Write CSV to `data/processed/` |
| `process_data(input_dir, output_dir)` | Raw → processed pipeline |

CLI: `python -m S4P_MNIST.data.make_dataset [--input PATH] [--output PATH]`

## `S4P_MNIST.features`

```python
from S4P_MNIST.features import build_features

df_features = build_features(df_processed)
```

## `S4P_MNIST.models`

### `BaseModel` (abstract)

Abstract interface with `fit`, `predict`, `save`, `load`. Extend this for any new estimator.

### `Model`

Reference implementation scaffold. Serializes via `joblib`.

```python
from pathlib import Path
from S4P_MNIST.models import Model

model = Model(config={"lr": 0.01})
# model.fit(X_train, y_train)
model.save(Path("models/model.joblib"))
reloaded = Model.load(Path("models/model.joblib"))
```

## `S4P_MNIST.evaluation`

```python
from S4P_MNIST.evaluation import classification_report, regression_report

metrics = classification_report(y_true, y_pred)
# -> {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
```

## `S4P_MNIST.visualization`

```python
from S4P_MNIST.visualization import plot_training_history, plot_confusion_matrix
```

## `S4P_MNIST.utils`

```python
from S4P_MNIST.utils import set_seed, save_json, load_json

set_seed(42)
```

## Training / Prediction CLIs

```bash
python -m S4P_MNIST.train_model --epochs 100 --batch-size 64
python -m S4P_MNIST.predict_model --model-path models/model.joblib --input data/processed/test.csv
```

## Hydra Configuration

Configuration is managed through Hydra — see `configs/config.yaml` for defaults and override at runtime:

```bash
python -m S4P_MNIST.train_model model.name=custom_model training.epochs=200
```

---

**S4P_MNIST** · Version see `S4P_MNIST.__version__`
