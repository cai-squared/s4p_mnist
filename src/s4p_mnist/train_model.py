from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import hydra
import matplotlib.pyplot as plt
import numpy as np
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torchvision import datasets

from s4p_mnist.config import DATA_DIR, PROJECT_ROOT
from s4p_mnist.data.loaders import load_processed
from s4p_mnist.logging_config import get_logger, setup_logging
from s4p_mnist.models.model import Model
from s4p_mnist.utils.seed import set_seed

logger = get_logger(__name__)


def _assert_no_nan(x: np.ndarray, name: str = "X") -> None:
    """Raise AssertionError if x contains NaN or Inf values."""
    if np.isnan(x).any():
        raise AssertionError(f"{name} contains NaN values — check your data pipeline.")
    if np.isinf(x).any():
        raise AssertionError(f"{name} contains Inf values — check normalization step.")


def _assert_shape(x: np.ndarray, y: np.ndarray) -> None:
    """Raise AssertionError if x/y shapes are incompatible with MNIST."""
    if x.ndim != 2 or x.shape[1] != 784:
        raise AssertionError(
            f"Expected x shape (N, 784), got {x.shape}. "
            "Images must be flattened 28x28 pixels."
        )
    if y.ndim != 1:
        raise AssertionError(f"Expected y shape (N,), got {y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise AssertionError(
            f"Sample count mismatch: x has {x.shape[0]} rows but y has {y.shape[0]}."
        )


def _check_train_cfg(cfg: DictConfig) -> None:
    if not OmegaConf.is_config(cfg):
        raise TypeError("cfg must be an OmegaConf config")
    t = cfg.training
    d = cfg.data
    p = cfg.paths
    keys = (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "dropout",
        "seed",
    )
    for key in keys:
        if key not in t:
            raise ValueError(f"training.{key} is required")
    if int(t.epochs) < 1:
        raise ValueError("training.epochs must be at least 1")
    if int(t.batch_size) < 1:
        raise ValueError("training.batch_size must be at least 1")
    if float(t.learning_rate) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if float(t.weight_decay) < 0:
        raise ValueError("training.weight_decay cannot be negative")
    if not 0 <= float(t.dropout) <= 1:
        raise ValueError("training.dropout must be between 0 and 1")
    if "val_fraction" not in d:
        raise ValueError("data.val_fraction is required")
    vf = float(d.val_fraction)
    if not 0 < vf < 1:
        raise ValueError("data.val_fraction must be strictly between 0 and 1")
    for key in ("data_processed", "models_dir"):
        if key not in p or not str(p[key]).strip():
            raise ValueError(f"paths.{key} must be a non-empty string")


def load_training_xy(
    data_path: Path,
    *,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        X_train_img, y_train, _, _ = load_processed(data_path)
        x_arr = X_train_img.reshape(X_train_img.shape[0], -1).astype(
            np.float32, copy=False
        )
        y_arr = y_train.astype(np.int64, copy=False)
        logger.info("Loaded training from processed .npy in %s", data_path)
        _assert_no_nan(x_arr, "X_train")
        _assert_shape(x_arr, y_arr)
        return x_arr, y_arr
    except FileNotFoundError:
        logger.info("Processed .npy not found under %s", data_path)

    logger.info(
        "Falling back to torchvision MNIST under %s",
        DATA_DIR,
    )
    raw = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=download,
        transform=None,
    )
    x_arr = raw.data.numpy().astype(np.float32).reshape(-1, 784) / 255.0
    y_arr = raw.targets.numpy().astype(np.int64, copy=False)
    _assert_no_nan(x_arr, "X_train")
    _assert_shape(x_arr, y_arr)
    return x_arr, y_arr


def train(
    data_path: Path,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    *,
    seed: int,
    val_fraction: float,
    weight_decay: float,
    dropout: float,
    out_dir: Path,
    use_wandb: bool = True,
) -> None:
    cfg: dict[str, Any] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "val_fraction": val_fraction,
        "seed": seed,
    }

    wandb_mode: Literal["online", "disabled"] = "online" if use_wandb else "disabled"
    run = wandb.init(
        entity="rriffaha-",
        project="s4p-mnist",
        config=cfg,
        job_type="train",
        mode=wandb_mode,
    )
    logger.info("W&B run initialized: mode=%s name=%s", wandb_mode, run.name)

    cfg["out_dir"] = str(out_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_training_xy(data_path, download=True)
    logger.info(
        "Training CNN on %d samples (epochs=%d batch_size=%d lr=%g)",
        len(x_train),
        epochs,
        batch_size,
        lr,
    )

    model = Model(cfg)
    model.fit(x_train, y_train)
    logger.info("Finished training model for %d epochs", epochs)

    out_path = model_dir / "model.joblib"
    model.save(out_path)
    logger.info("Saved trained model to %s", out_path)

    try:
        _, _, X_test_img, y_test_u8 = load_processed(data_path)
        x_test = X_test_img.reshape(X_test_img.shape[0], -1).astype(
            np.float32, copy=False
        )
        y_test = y_test_u8.astype(np.int64, copy=False)
        logger.info("Evaluating on processed test arrays from %s", data_path)
    except FileNotFoundError:
        logger.info(
            "Evaluating on torchvision MNIST test (root=%s)",
            DATA_DIR,
        )
        raw_test = datasets.MNIST(
            root=str(DATA_DIR),
            train=False,
            download=True,
            transform=None,
        )
        x_test = raw_test.data.numpy().astype(np.float32).reshape(-1, 784) / 255.0
        y_test = raw_test.targets.numpy().astype(np.int64, copy=False)
    pred = model.predict(x_test)
    acc = float((pred == y_test).mean())

    report = classification_report(y_test, pred, digits=3)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    fig.savefig("confusion_matrix.png", dpi=120)

    logger.info("Held-out MNIST test accuracy: %.6f", acc)
    print(f"cnn_mnist_test_accuracy={acc:.6f}")

    wandb.log({"test_accuracy": acc})
    wandb.summary["test_accuracy"] = acc

    artifact = wandb.Artifact(
        name="s4p-mnist-model",
        type="model",
        description="Trained CNN on MNIST",
        metadata={"test_accuracy": acc, **cfg},
    )
    artifact.add_file(str(out_path))
    run.log_artifact(artifact)

    run_url = getattr(run, "url", None)
    with open("wandb_report.md", "w", encoding="utf-8") as wb:
        wb.write("### W&B run\n")
        if run_url:
            wb.write(f"[Open the W&B run here]({run_url})\n\n")
        else:
            wb.write("W&B run URL unavailable.\n\n")
        wb.write("### W&B artifact\n")
        wb.write(f"- name: {artifact.name}\n")
        wb.write(f"- type: {artifact.type}\n")
        if artifact.aliases:
            wb.write(f"- aliases: {', '.join(artifact.aliases)}\n")

    run.finish()
    logger.info("W&B run finished")


def _resolve_under_root(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _check_train_cfg(cfg)
    setup_logging()

    t = cfg.training
    d = cfg.data

    set_seed(int(t.seed))

    data_path = _resolve_under_root(str(cfg.paths.data_processed))
    model_dir = _resolve_under_root(str(cfg.paths.models_dir))

    out_dir = Path(HydraConfig.get().runtime.output_dir)

    train(
        data_path,
        model_dir,
        int(t.epochs),
        int(t.batch_size),
        float(t.learning_rate),
        seed=int(t.seed),
        val_fraction=float(d.val_fraction),
        weight_decay=float(t.weight_decay),
        dropout=float(t.dropout),
        out_dir=out_dir,
        use_wandb=t.wandb,
    )
    logger.info("Training complete")


if __name__ == "__main__":
    main()
