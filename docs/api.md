# API Reference

The package is importable as `s4p_mnist` after running `pip install -e .`.

## `s4p_mnist.config`

Project-wide path constants and typed config dataclasses.

```python
from s4p_mnist.config import (
    PROJECT_ROOT,
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    Config, TrainingConfig, DataConfig, DEFAULT_CONFIG,
)
```

Use these constants instead of hard-coded relative paths — they resolve against the repo root regardless of the current working directory.

## `s4p_mnist.logging_config`

```python
from s4p_mnist.logging_config import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)
```

## `s4p_mnist.data`

| Function | Purpose |
|---|---|
| `load_raw(filename)` | Read CSV from `data/raw/` |
| `load_processed(filename)` | Read CSV from `data/processed/` |
| `save_processed(df, filename)` | Write CSV to `data/processed/` |
| `process_data(input_dir, output_dir)` | Raw → processed pipeline |

CLI: `python -m s4p_mnist.data.make_dataset [--input PATH] [--output PATH]`

## `s4p_mnist.features`

```python
from s4p_mnist.features import build_features

df_features = build_features(df_processed)
```

## `s4p_mnist.models`

### `BaseModel` (abstract)

Abstract interface with `fit`, `predict`, `save`, `load`. Extend this for any new estimator.

### `Model`

Reference implementation scaffold. Serializes via `joblib`.

```python
from pathlib import Path
from s4p_mnist.models import Model

model = Model(config={"lr": 0.01})
# model.fit(X_train, y_train)
model.save(Path("models/model.joblib"))
reloaded = Model.load(Path("models/model.joblib"))
```

## `s4p_mnist.evaluation`

```python
from s4p_mnist.evaluation import classification_report, regression_report

metrics = classification_report(y_true, y_pred)
# -> {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
```

## `s4p_mnist.visualization`

```python
from s4p_mnist.visualization import plot_training_history, plot_confusion_matrix
```

## `s4p_mnist.utils`

```python
from s4p_mnist.utils import set_seed, save_json, load_json

set_seed(42)
```

## Training / Prediction CLIs

Both use Hydra with `configs/config.yaml`. You can override keys from the terminal:

```bash
python -m s4p_mnist.train_model training.epochs=12 training.batch_size=64
python -m s4p_mnist.predict_model predict.output_file=out/preds.csv
```

## Hydra Configuration

Training reads `paths`, `data`, and `training`. Prediction reads `predict`. Relative paths are resolved from `s4p_mnist.config.PROJECT_ROOT` so you do not have to cd into a specific folder first.

---

**s4p_mnist** · Version see `s4p_mnist.__version__`
