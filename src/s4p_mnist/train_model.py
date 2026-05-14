from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from torchvision import datasets

from s4p_mnist.config import DATA_DIR, DEFAULT_CONFIG, MODELS_DIR, PROCESSED_DATA_DIR
from s4p_mnist.data.loaders import load_processed
from s4p_mnist.logging_config import get_logger, setup_logging
from s4p_mnist.models.model import Model
from s4p_mnist.utils.seed import set_seed

logger = get_logger(__name__)


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
    use_wandb: bool = True,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_training_xy(data_path, download=True)
    logger.info(
        "Training CNN on %d samples (epochs=%d batch_size=%d lr=%g)",
        len(x_train),
        epochs,
        batch_size,
        lr,
    )

    cfg: dict[str, Any] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "val_fraction": val_fraction,
        "seed": seed,
    }

    wandb_mode = "online" if use_wandb else "disabled"
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "s4p-mnist"),
        config=cfg,
        job_type="train",
        mode=wandb_mode,
    )
    logger.info("W&B run initialized: mode=%s name=%s", wandb_mode, run.name)

    model = Model(cfg)
    model.fit(x_train, y_train)

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

    run.finish()
    logger.info("W&B run finished")


def main() -> None:
    cfg = DEFAULT_CONFIG.training
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_CONFIG.data.val_split,
    )
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging for this run.",
    )
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    train(
        args.data_path,
        args.model_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        seed=args.seed,
        val_fraction=args.val_fraction,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        use_wandb=not args.no_wandb,
    )
    logger.info("Training complete")


if __name__ == "__main__":
    main()
