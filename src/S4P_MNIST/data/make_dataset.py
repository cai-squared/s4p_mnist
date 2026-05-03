"""Raw -> processed MNIST pipeline CLI.

Run via the project ``Makefile``::

    make data

or directly::

    python -m s4p_mnist.data.make_dataset
    python -m s4p_mnist.data.make_dataset --force

The pipeline reads the four IDX files from ``data/raw/``, performs minimal
validation, and writes ``X_train.npy``, ``y_train.npy``, ``X_test.npy``,
``y_test.npy`` into ``data/processed/``.

By default the pipeline is a no-op when the processed files already exist
on disk; pass ``--force`` to regenerate them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from s4p_mnist.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from s4p_mnist.data.loaders import (
    PROCESSED_X_TEST,
    PROCESSED_X_TRAIN,
    PROCESSED_Y_TEST,
    PROCESSED_Y_TRAIN,
    load_raw,
    save_processed,
)
from s4p_mnist.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _processed_files_exist(processed_dir: Path) -> bool:
    """Return ``True`` iff all four expected processed files are present."""
    return all(
        (processed_dir / name).is_file()
        for name in (
            PROCESSED_X_TRAIN,
            PROCESSED_Y_TRAIN,
            PROCESSED_X_TEST,
            PROCESSED_Y_TEST,
        )
    )


def make_dataset(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    force: bool = False,
) -> None:
    """Run the raw -> processed transformation.

    Parameters
    ----------
    raw_dir : Path, optional
        Directory containing the raw IDX files. Defaults to
        :data:`s4p_mnist.config.RAW_DATA_DIR`.
    processed_dir : Path, optional
        Output directory for processed ``.npy`` files. Defaults to
        :data:`s4p_mnist.config.PROCESSED_DATA_DIR`.
    force : bool, default False
        If ``False`` (the default) and all four processed files already
        exist in ``processed_dir``, the pipeline logs a message and
        returns without doing any work. If ``True``, processed files are
        regenerated from raw.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    processed_dir = (
        Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR
    )

    if not force and _processed_files_exist(processed_dir):
        logger.info(
            "Processed files already exist in %s. Use --force to regenerate.",
            processed_dir,
        )
        return

    logger.info("Building processed dataset: %s -> %s", raw_dir, processed_dir)
    X_train, y_train, X_test, y_test = load_raw(raw_dir=raw_dir)
    save_processed(
        X_train,
        y_train,
        X_test,
        y_test,
        processed_dir=processed_dir,
    )
    logger.info("Done. Processed dataset is ready in %s", processed_dir)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="s4p_mnist.data.make_dataset",
        description="Transform raw MNIST IDX files into processed NumPy arrays.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override the raw data directory (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Override the processed data directory (default: data/processed).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate processed files even if they already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str, optional
        Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` on a known failure (missing/corrupt
        input files). Unknown exceptions propagate.
    """
    setup_logging()
    args = _build_parser().parse_args(argv)
    try:
        make_dataset(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            force=args.force,
        )
    except FileNotFoundError as exc:
        logger.error("Input file missing: %s", exc)
        logger.error(
            "Place the four IDX files (train-images.idx3-ubyte, "
            "train-labels.idx1-ubyte, t10k-images.idx3-ubyte, "
            "t10k-labels.idx1-ubyte) in data/raw/ and try again."
        )
        return 1
    except ValueError as exc:
        logger.error("Invalid raw data: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
