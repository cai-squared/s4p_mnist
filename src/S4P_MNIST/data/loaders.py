"""Data loaders for the S4P MNIST project.

This module provides the canonical I/O layer for MNIST data:

* :func:`load_raw` parses the original IDX-format files shipped by Yann
  LeCun's MNIST distribution (``train-images.idx3-ubyte``,
  ``train-labels.idx1-ubyte``, ``t10k-images.idx3-ubyte``,
  ``t10k-labels.idx1-ubyte``) from ``data/raw/`` into NumPy arrays.
* :func:`save_processed` writes processed arrays as ``.npy`` files into
  ``data/processed/``.
* :func:`load_processed` reads those ``.npy`` files back.

Design notes
------------
The processed arrays produced by this module are kept as ``uint8`` with
pixel values in ``[0, 255]`` and image shape ``(N, 28, 28)``. Normalization
and reshaping/flattening are intentionally *not* performed here: those are
modeling/feature-engineering concerns and live in
``s4p_mnist.features.build_features``. Keeping the data layer
transformation-free means ``data/processed/`` remains a single,
unambiguous source of truth.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from s4p_mnist.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from s4p_mnist.logging_config import get_logger

logger = get_logger(__name__)

# IDX magic numbers (see http://yann.lecun.com/exdb/mnist/).
# The IDX format encodes the data type and rank in a 4-byte big-endian
# magic number. For unsigned-byte tensors, byte 3 is 0x08; byte 4 is the
# number of dimensions (1 for labels, 3 for image stacks).
_IDX_MAGIC_IMAGES: Final[int] = 2051  # 0x00000803 -> uint8, 3 dims
_IDX_MAGIC_LABELS: Final[int] = 2049  # 0x00000801 -> uint8, 1 dim

# Canonical filenames in data/raw/.
RAW_TRAIN_IMAGES: Final[str] = "train-images.idx3-ubyte"
RAW_TRAIN_LABELS: Final[str] = "train-labels.idx1-ubyte"
RAW_TEST_IMAGES: Final[str] = "t10k-images.idx3-ubyte"
RAW_TEST_LABELS: Final[str] = "t10k-labels.idx1-ubyte"

# Canonical filenames in data/processed/.
PROCESSED_X_TRAIN: Final[str] = "X_train.npy"
PROCESSED_Y_TRAIN: Final[str] = "y_train.npy"
PROCESSED_X_TEST: Final[str] = "X_test.npy"
PROCESSED_Y_TEST: Final[str] = "y_test.npy"

# Type aliases. ``npt.NDArray[np.uint8]`` keeps mypy honest about the dtype
# we emit: anyone downstream who needs floats must cast explicitly.
ImageArray = npt.NDArray[np.uint8]
LabelArray = npt.NDArray[np.uint8]


def _read_idx_images(path: Path) -> ImageArray:
    """Parse an IDX3 image file into a ``(N, rows, cols)`` ``uint8`` array.

    The IDX3 image format starts with a 16-byte big-endian header:
    ``magic (4) | count (4) | rows (4) | cols (4)``, followed by
    ``count * rows * cols`` raw pixel bytes.

    Parameters
    ----------
    path : Path
        Path to an ``.idx3-ubyte`` file.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(count, rows, cols)`` with dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the magic number does not match the IDX3 image format or the
        body length does not match the header.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Raw image file not found: {path}")

    with path.open("rb") as f:
        header = f.read(16)
        if len(header) < 16:
            raise ValueError(f"{path}: truncated IDX header ({len(header)} bytes)")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != _IDX_MAGIC_IMAGES:
            raise ValueError(
                f"{path}: bad magic number {magic} (expected {_IDX_MAGIC_IMAGES} "
                f"for IDX3 images)"
            )
        body = f.read()

    expected = count * rows * cols
    if len(body) != expected:
        raise ValueError(
            f"{path}: body has {len(body)} bytes, expected {expected} "
            f"(count={count}, rows={rows}, cols={cols})"
        )

    images: ImageArray = np.frombuffer(body, dtype=np.uint8).reshape(count, rows, cols)
    logger.debug("Parsed %d images of shape (%d, %d) from %s", count, rows, cols, path)
    return images


def _read_idx_labels(path: Path) -> LabelArray:
    """Parse an IDX1 label file into a ``(N,)`` ``uint8`` array.

    The IDX1 label format starts with an 8-byte big-endian header:
    ``magic (4) | count (4)``, followed by ``count`` raw label bytes.

    Parameters
    ----------
    path : Path
        Path to an ``.idx1-ubyte`` file.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(count,)`` with dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the magic number does not match the IDX1 label format or the
        body length does not match the header.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Raw label file not found: {path}")

    with path.open("rb") as f:
        header = f.read(8)
        if len(header) < 8:
            raise ValueError(f"{path}: truncated IDX header ({len(header)} bytes)")
        magic, count = struct.unpack(">II", header)
        if magic != _IDX_MAGIC_LABELS:
            raise ValueError(
                f"{path}: bad magic number {magic} (expected {_IDX_MAGIC_LABELS} "
                f"for IDX1 labels)"
            )
        body = f.read()

    if len(body) != count:
        raise ValueError(f"{path}: body has {len(body)} bytes, expected {count}")

    labels: LabelArray = np.frombuffer(body, dtype=np.uint8)
    logger.debug("Parsed %d labels from %s", count, path)
    return labels


def load_raw(
    raw_dir: Path | None = None,
) -> tuple[ImageArray, LabelArray, ImageArray, LabelArray]:
    """Load the raw MNIST IDX files into NumPy arrays.

    Reads the four canonical MNIST files from ``raw_dir`` and returns
    them as ``(X_train, y_train, X_test, y_test)``. Pixel values are
    preserved as ``uint8`` in ``[0, 255]``; images keep their native
    ``(N, 28, 28)`` shape.

    Parameters
    ----------
    raw_dir : Path, optional
        Directory containing the four IDX files. Defaults to
        :data:`s4p_mnist.config.RAW_DATA_DIR`.

    Returns
    -------
    X_train : numpy.ndarray
        Training images, shape ``(60000, 28, 28)``, dtype ``uint8``.
    y_train : numpy.ndarray
        Training labels, shape ``(60000,)``, dtype ``uint8``.
    X_test : numpy.ndarray
        Test images, shape ``(10000, 28, 28)``, dtype ``uint8``.
    y_test : numpy.ndarray
        Test labels, shape ``(10000,)``, dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If any of the four expected files is missing from ``raw_dir``.
    ValueError
        If any file is not a valid IDX file or its contents are malformed.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    logger.info("Loading raw MNIST IDX files from %s", raw_dir)

    X_train = _read_idx_images(raw_dir / RAW_TRAIN_IMAGES)
    y_train = _read_idx_labels(raw_dir / RAW_TRAIN_LABELS)
    X_test = _read_idx_images(raw_dir / RAW_TEST_IMAGES)
    y_test = _read_idx_labels(raw_dir / RAW_TEST_LABELS)

    # Sanity checks: catch corruption or mismatched files early.
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Train set size mismatch: {X_train.shape[0]} images vs "
            f"{y_train.shape[0]} labels"
        )
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"Test set size mismatch: {X_test.shape[0]} images vs "
            f"{y_test.shape[0]} labels"
        )
    if y_train.max() > 9 or y_test.max() > 9:
        raise ValueError("Found label outside [0, 9]; expected MNIST digit labels")

    logger.info("Loaded raw MNIST: train=%s, test=%s", X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test


def save_processed(
    X_train: ImageArray,
    y_train: LabelArray,
    X_test: ImageArray,
    y_test: LabelArray,
    processed_dir: Path | None = None,
) -> None:
    """Save processed MNIST arrays as ``.npy`` files.

    Writes four files to ``processed_dir``: ``X_train.npy``,
    ``y_train.npy``, ``X_test.npy``, ``y_test.npy``. The directory is
    created if it does not already exist.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : numpy.ndarray
        Arrays to persist.
    processed_dir : Path, optional
        Output directory. Defaults to
        :data:`s4p_mnist.config.PROCESSED_DATA_DIR`.
    """
    processed_dir = (
        Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving processed arrays to %s", processed_dir)

    np.save(processed_dir / PROCESSED_X_TRAIN, X_train)
    np.save(processed_dir / PROCESSED_Y_TRAIN, y_train)
    np.save(processed_dir / PROCESSED_X_TEST, X_test)
    np.save(processed_dir / PROCESSED_Y_TEST, y_test)

    logger.info(
        "Saved %s (%s), %s (%s), %s (%s), %s (%s)",
        PROCESSED_X_TRAIN,
        X_train.shape,
        PROCESSED_Y_TRAIN,
        y_train.shape,
        PROCESSED_X_TEST,
        X_test.shape,
        PROCESSED_Y_TEST,
        y_test.shape,
    )


def load_processed(
    processed_dir: Path | None = None,
) -> tuple[ImageArray, LabelArray, ImageArray, LabelArray]:
    """Load the processed MNIST arrays previously written by :func:`save_processed`.

    Parameters
    ----------
    processed_dir : Path, optional
        Directory containing the four ``.npy`` files. Defaults to
        :data:`s4p_mnist.config.PROCESSED_DATA_DIR`.

    Returns
    -------
    X_train, y_train, X_test, y_test : numpy.ndarray
        Arrays in the same shapes/dtypes that :func:`save_processed`
        wrote.

    Raises
    ------
    FileNotFoundError
        If any of the four ``.npy`` files is missing. The error message
        suggests running ``make data`` to regenerate them.
    """
    processed_dir = (
        Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR
    )
    logger.info("Loading processed arrays from %s", processed_dir)

    expected = [
        PROCESSED_X_TRAIN,
        PROCESSED_Y_TRAIN,
        PROCESSED_X_TEST,
        PROCESSED_Y_TEST,
    ]
    missing = [name for name in expected if not (processed_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed files in {processed_dir}: {missing}. "
            f"Run `make data` to regenerate them from data/raw/."
        )

    X_train: ImageArray = np.load(processed_dir / PROCESSED_X_TRAIN)
    y_train: LabelArray = np.load(processed_dir / PROCESSED_Y_TRAIN)
    X_test: ImageArray = np.load(processed_dir / PROCESSED_X_TEST)
    y_test: LabelArray = np.load(processed_dir / PROCESSED_Y_TEST)

    logger.info(
        "Loaded processed MNIST: train=%s, test=%s", X_train.shape, X_test.shape
    )
    return X_train, y_train, X_test, y_test
