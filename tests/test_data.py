"""Unit tests for the S4P-MNIST data layer.

Covers:
* ``s4p_mnist.data.loaders``      - IDX binary parser, save/load round-trip
* ``s4p_mnist.data.make_dataset`` - idempotency, --force flag, CLI exit codes

All tests are self-contained: synthetic IDX binary files and temporary
directories are created on the fly via pytest's ``tmp_path`` fixture so the
real MNIST data files are never required.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from s4p_mnist.data.loaders import (
    _IDX_MAGIC_IMAGES,
    _IDX_MAGIC_LABELS,
    RAW_TEST_IMAGES,
    RAW_TEST_LABELS,
    RAW_TRAIN_IMAGES,
    RAW_TRAIN_LABELS,
    _read_idx_images,
    _read_idx_labels,
    load_processed,
    load_raw,
    save_processed,
)
from s4p_mnist.data.make_dataset import (
    _processed_files_exist,
    main,
    make_dataset,
)

# ---------------------------------------------------------------------------
# Helpers – write minimal synthetic IDX binary files
# ---------------------------------------------------------------------------


def _write_idx_images(
    path: Path,
    n: int = 5,
    rows: int = 28,
    cols: int = 28,
    seed: int = 42,
) -> np.ndarray:
    """Write a minimal IDX3 image file; return the pixel array (N, rows, cols)."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(n, rows, cols), dtype=np.uint8)
    with path.open("wb") as f:
        f.write(struct.pack(">IIII", _IDX_MAGIC_IMAGES, n, rows, cols))
        f.write(pixels.tobytes())
    return pixels


def _write_idx_labels(path: Path, n: int = 5) -> np.ndarray:
    """Write a minimal IDX1 label file; return the labels array (N,).

    Labels are ``0..n-1 mod 10`` so they always stay inside the MNIST
    digit range [0, 9].
    """
    labels = np.arange(n, dtype=np.uint8) % 10
    with path.open("wb") as f:
        f.write(struct.pack(">II", _IDX_MAGIC_LABELS, n))
        f.write(labels.tobytes())
    return labels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Temp directory populated with four synthetic IDX files (10 train, 4 test)."""
    _write_idx_images(tmp_path / RAW_TRAIN_IMAGES, n=10)
    _write_idx_labels(tmp_path / RAW_TRAIN_LABELS, n=10)
    _write_idx_images(tmp_path / RAW_TEST_IMAGES, n=4, seed=99)
    _write_idx_labels(tmp_path / RAW_TEST_LABELS, n=4)
    return tmp_path


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    """Temp directory pre-populated with four .npy files via save_processed."""
    out = tmp_path / "processed"
    out.mkdir()
    X_train = np.zeros((10, 28, 28), dtype=np.uint8)  # noqa: N806
    y_train = np.arange(10, dtype=np.uint8)
    X_test = np.ones((4, 28, 28), dtype=np.uint8)  # noqa: N806
    y_test = np.arange(4, dtype=np.uint8)
    save_processed(X_train, y_train, X_test, y_test, processed_dir=out)
    return out


# ---------------------------------------------------------------------------
# _read_idx_images
# ---------------------------------------------------------------------------


class TestReadIdxImages:
    def test_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "images.idx3-ubyte"
        _write_idx_images(path, n=5)
        assert _read_idx_images(path).shape == (5, 28, 28)

    def test_dtype(self, tmp_path: Path) -> None:
        path = tmp_path / "images.idx3-ubyte"
        _write_idx_images(path, n=3)
        assert _read_idx_images(path).dtype == np.uint8

    def test_values_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "images.idx3-ubyte"
        expected = _write_idx_images(path, n=5)
        np.testing.assert_array_equal(_read_idx_images(path), expected)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _read_idx_images(tmp_path / "nonexistent.idx3-ubyte")

    def test_wrong_magic_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_magic.idx3-ubyte"
        with path.open("wb") as f:
            f.write(struct.pack(">IIII", 9999, 5, 28, 28))
            f.write(b"\x00" * (5 * 28 * 28))
        with pytest.raises(ValueError, match="magic"):
            _read_idx_images(path)

    def test_truncated_header_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "truncated.idx3-ubyte"
        path.write_bytes(b"\x00\x00\x08\x03")  # 4 bytes; needs 16
        with pytest.raises(ValueError, match="truncated"):
            _read_idx_images(path)

    def test_body_length_mismatch_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "mismatch.idx3-ubyte"
        with path.open("wb") as f:
            f.write(struct.pack(">IIII", _IDX_MAGIC_IMAGES, 5, 28, 28))
            f.write(b"\x00" * 10)  # far too short
        with pytest.raises(ValueError):
            _read_idx_images(path)


# ---------------------------------------------------------------------------
# _read_idx_labels
# ---------------------------------------------------------------------------


class TestReadIdxLabels:
    def test_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.idx1-ubyte"
        _write_idx_labels(path, n=7)
        assert _read_idx_labels(path).shape == (7,)

    def test_dtype(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.idx1-ubyte"
        _write_idx_labels(path, n=3)
        assert _read_idx_labels(path).dtype == np.uint8

    def test_values_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.idx1-ubyte"
        expected = _write_idx_labels(path, n=5)
        np.testing.assert_array_equal(_read_idx_labels(path), expected)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _read_idx_labels(tmp_path / "nonexistent.idx1-ubyte")

    def test_wrong_magic_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_magic.idx1-ubyte"
        with path.open("wb") as f:
            f.write(struct.pack(">II", 9999, 5))
            f.write(b"\x00" * 5)
        with pytest.raises(ValueError, match="magic"):
            _read_idx_labels(path)

    def test_body_length_mismatch_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "mismatch.idx1-ubyte"
        with path.open("wb") as f:
            f.write(struct.pack(">II", _IDX_MAGIC_LABELS, 10))
            f.write(b"\x00" * 3)  # only 3 bytes, header says 10
        with pytest.raises(ValueError):
            _read_idx_labels(path)


# ---------------------------------------------------------------------------
# load_raw
# ---------------------------------------------------------------------------


class TestLoadRaw:
    def test_shapes(self, raw_dir: Path) -> None:
        X_train, y_train, X_test, y_test = load_raw(raw_dir=raw_dir)  # noqa: N806
        assert X_train.shape == (10, 28, 28)
        assert y_train.shape == (10,)
        assert X_test.shape == (4, 28, 28)
        assert y_test.shape == (4,)

    def test_dtypes(self, raw_dir: Path) -> None:
        X_train, y_train, X_test, y_test = load_raw(raw_dir=raw_dir)  # noqa: N806
        assert X_train.dtype == np.uint8
        assert y_train.dtype == np.uint8
        assert X_test.dtype == np.uint8
        assert y_test.dtype == np.uint8

    def test_train_size_mismatch_raises(self, tmp_path: Path) -> None:
        _write_idx_images(tmp_path / RAW_TRAIN_IMAGES, n=10)
        _write_idx_labels(tmp_path / RAW_TRAIN_LABELS, n=8)  # intentional mismatch
        _write_idx_images(tmp_path / RAW_TEST_IMAGES, n=4)
        _write_idx_labels(tmp_path / RAW_TEST_LABELS, n=4)
        with pytest.raises(ValueError, match="Train set size mismatch"):
            load_raw(raw_dir=tmp_path)

    def test_test_size_mismatch_raises(self, tmp_path: Path) -> None:
        _write_idx_images(tmp_path / RAW_TRAIN_IMAGES, n=10)
        _write_idx_labels(tmp_path / RAW_TRAIN_LABELS, n=10)
        _write_idx_images(tmp_path / RAW_TEST_IMAGES, n=4)
        _write_idx_labels(tmp_path / RAW_TEST_LABELS, n=2)  # intentional mismatch
        with pytest.raises(ValueError, match="Test set size mismatch"):
            load_raw(raw_dir=tmp_path)

    def test_label_out_of_range_raises(self, tmp_path: Path) -> None:
        _write_idx_images(tmp_path / RAW_TRAIN_IMAGES, n=3)
        bad_labels = np.full(3, 255, dtype=np.uint8)
        with (tmp_path / RAW_TRAIN_LABELS).open("wb") as f:
            f.write(struct.pack(">II", _IDX_MAGIC_LABELS, 3))
            f.write(bad_labels.tobytes())
        _write_idx_images(tmp_path / RAW_TEST_IMAGES, n=2)
        _write_idx_labels(tmp_path / RAW_TEST_LABELS, n=2)
        with pytest.raises(ValueError, match="label"):
            load_raw(raw_dir=tmp_path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        _write_idx_images(tmp_path / RAW_TRAIN_IMAGES, n=5)
        _write_idx_labels(tmp_path / RAW_TRAIN_LABELS, n=5)
        _write_idx_images(tmp_path / RAW_TEST_IMAGES, n=3)
        # RAW_TEST_LABELS intentionally absent
        with pytest.raises(FileNotFoundError):
            load_raw(raw_dir=tmp_path)


# ---------------------------------------------------------------------------
# save_processed + load_processed
# ---------------------------------------------------------------------------


class TestSaveAndLoadProcessed:
    def test_all_four_files_created(self, tmp_path: Path) -> None:
        out = tmp_path / "processed"
        X_train = np.zeros((5, 28, 28), dtype=np.uint8)  # noqa: N806
        save_processed(
            X_train,
            np.arange(5, dtype=np.uint8),
            X_train,
            np.arange(5, dtype=np.uint8),
            processed_dir=out,
        )
        for name in ("X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"):
            assert (out / name).exists(), f"{name} missing after save_processed"

    def test_roundtrip_values(self, tmp_path: Path) -> None:
        out = tmp_path / "processed"
        rng = np.random.default_rng(0)
        X_train = rng.integers(0, 256, (8, 28, 28), dtype=np.uint8)  # noqa: N806
        y_train = rng.integers(0, 10, (8,), dtype=np.uint8)
        X_test = rng.integers(0, 256, (3, 28, 28), dtype=np.uint8)  # noqa: N806
        y_test = rng.integers(0, 10, (3,), dtype=np.uint8)
        save_processed(X_train, y_train, X_test, y_test, processed_dir=out)
        X_tr, y_tr, X_te, y_te = load_processed(processed_dir=out)  # noqa: N806
        np.testing.assert_array_equal(X_tr, X_train)
        np.testing.assert_array_equal(y_tr, y_train)
        np.testing.assert_array_equal(X_te, X_test)
        np.testing.assert_array_equal(y_te, y_test)

    def test_roundtrip_dtypes(self, tmp_path: Path) -> None:
        out = tmp_path / "processed"
        X = np.zeros((5, 28, 28), dtype=np.uint8)  # noqa: N806
        y = np.arange(5, dtype=np.uint8)
        save_processed(X, y, X, y, processed_dir=out)
        X_tr, y_tr, X_te, y_te = load_processed(processed_dir=out)  # noqa: N806
        for arr in (X_tr, y_tr, X_te, y_te):
            assert arr.dtype == np.uint8

    def test_load_missing_files_raises_with_hint(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="make data"):
            load_processed(processed_dir=empty)

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "does" / "not" / "exist"
        assert not out.exists()
        X = np.zeros((2, 28, 28), dtype=np.uint8)  # noqa: N806
        y = np.zeros(2, dtype=np.uint8)
        save_processed(X, y, X, y, processed_dir=out)
        assert out.is_dir()


# ---------------------------------------------------------------------------
# _processed_files_exist
# ---------------------------------------------------------------------------


class TestProcessedFilesExist:
    def test_returns_true_when_all_present(self, processed_dir: Path) -> None:
        assert _processed_files_exist(processed_dir) is True

    def test_returns_false_when_directory_empty(self, tmp_path: Path) -> None:
        assert _processed_files_exist(tmp_path) is False

    def test_returns_false_when_partially_present(self, tmp_path: Path) -> None:
        np.save(tmp_path / "X_train.npy", np.zeros(1))
        np.save(tmp_path / "y_train.npy", np.zeros(1))
        # X_test.npy and y_test.npy are absent
        assert _processed_files_exist(tmp_path) is False


# ---------------------------------------------------------------------------
# make_dataset – idempotency and --force
# ---------------------------------------------------------------------------


class TestMakeDataset:
    def test_creates_processed_files(self, raw_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "processed"
        make_dataset(raw_dir=raw_dir, processed_dir=out)
        assert _processed_files_exist(out)

    def test_idempotent_skips_second_run(
        self, raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without --force, make_dataset must not call save_processed again."""
        out = tmp_path / "processed"
        make_dataset(raw_dir=raw_dir, processed_dir=out)

        call_count: dict[str, int] = {"n": 0}
        original = save_processed

        def counting_save(*args, **kwargs):  # type: ignore[no-untyped-def]
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr("s4p_mnist.data.make_dataset.save_processed", counting_save)
        make_dataset(raw_dir=raw_dir, processed_dir=out, force=False)
        assert (
            call_count["n"] == 0
        ), "save_processed called on second run without --force"

    def test_force_triggers_reprocessing(
        self, raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With --force, save_processed must be called even when outputs exist."""
        out = tmp_path / "processed"
        make_dataset(raw_dir=raw_dir, processed_dir=out)

        call_count: dict[str, int] = {"n": 0}
        original = save_processed

        def counting_save(*args, **kwargs):  # type: ignore[no-untyped-def]
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr("s4p_mnist.data.make_dataset.save_processed", counting_save)
        make_dataset(raw_dir=raw_dir, processed_dir=out, force=True)
        assert (
            call_count["n"] == 1
        ), "save_processed called on second run without --force"

    def test_missing_raw_raises(self, tmp_path: Path) -> None:
        empty_raw = tmp_path / "raw"
        empty_raw.mkdir()
        with pytest.raises(FileNotFoundError):
            make_dataset(raw_dir=empty_raw, processed_dir=tmp_path / "processed")


# ---------------------------------------------------------------------------
# main() – CLI exit codes
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_zero_on_success(self, raw_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "processed"
        assert main(["--raw-dir", str(raw_dir), "--processed-dir", str(out)]) == 0

    def test_returns_one_on_missing_raw(self, tmp_path: Path) -> None:
        empty_raw = tmp_path / "raw"
        empty_raw.mkdir()
        result = main(
            [
                "--raw-dir",
                str(empty_raw),
                "--processed-dir",
                str(tmp_path / "processed"),
            ]
        )
        assert result == 1

    def test_force_flag_accepted(self, raw_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "processed"
        result = main(
            [
                "--raw-dir",
                str(raw_dir),
                "--processed-dir",
                str(out),
                "--force",
            ]
        )
        assert result == 0
