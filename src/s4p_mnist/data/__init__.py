"""Data layer for the S4P MNIST project.

Public API:

* :func:`load_raw` -- parse raw IDX files from ``data/raw/``.
* :func:`load_processed` -- read processed ``.npy`` files from ``data/processed/``.
* :func:`save_processed` -- write processed ``.npy`` files to ``data/processed/``.

The CLI entry point lives in :mod:`s4p_mnist.data.make_dataset` and is
invoked by ``make data``.
"""

from s4p_mnist.data.loaders import (
    load_processed,
    load_raw,
    save_processed,
)

__all__ = [
    "load_processed",
    "load_raw",
    "save_processed",
]
